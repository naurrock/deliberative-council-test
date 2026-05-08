"""
Model registry and LiteLLM wrapper for Deliberative Council.

Handles model selection with family diversity, tier matching, round-robin
distribution, failure tracking, fallback chains, and token counting.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import litellm

from council.types import (
    HealthCheckResult,
    ModelInfo,
    ModelTier,
    ModelUsage,
    RoleSpec,
)

from enum import Enum

logger = logging.getLogger(__name__)

# Suppress litellm's verbose logging
litellm.suppress_debug_info = True


# ── Token Budget Tracker ───────────────────────────────────────────────


@dataclass
class TokenBudget:
    """Tracks token consumption against a budget."""

    total: int = 0
    used: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.used)

    @property
    def is_exhausted(self) -> bool:
        return self.used >= self.total

    def consume(self, tokens: int) -> None:
        """Record token consumption."""
        self.used += tokens

    def can_allocate(self, tokens: int) -> bool:
        """Check if we can allocate this many tokens."""
        return self.used + tokens <= self.total


# ── Failure Severity ───────────────────────────────────────────────────


class FailureSeverity(str, Enum):
    """Failure severity levels with different cooldown durations.

    TRANSIENT  — First failure of any kind. Could be RPM throttle, network
                 blip, timeout, or even a 403/401. Cools down in ~60 seconds.
                 On the second failure while still in cooldown, escalates to DAILY.

    DAILY      — Second consecutive failure for the same model. Strong evidence
                 of RPD quota exhaustion, daily budget depletion, or a persistent
                 issue (revoked key, geo-block discovered at runtime). Cools down
                 at midnight UTC (next daily reset).

    There is NO "permanent" severity. Models known to be unreachable (geo-blocked)
    are filtered out at selection time via the `geo_blocked` config flag — they
    never reach FailureTracker at all. Runtime-discovered permanent failures
    (e.g., revoked API key) will simply re-escalate to DAILY each day, costing
    at most one attempt per day — negligible waste.
    """
    TRANSIENT = "transient"    # ~60s cooldown (first strike)
    DAILY = "daily"            # cooldown until midnight UTC (second strike)


# Default cooldown per severity (seconds)
_SEVERITY_COOLDOWNS: dict[str, float] = {
    FailureSeverity.TRANSIENT: 60.0,
    FailureSeverity.DAILY: 86400.0,       # 24h — recalculated to midnight
}


# ── Failure Tracker ────────────────────────────────────────────────────


class FailureRecord:
    """A single failure record with severity and expiry."""

    __slots__ = ("severity", "expires_at", "error_hint", "provider")

    def __init__(
        self,
        severity: str,
        provider: str = "",
        error_hint: str = "",
        expires_at: float | None = None,
    ):
        self.severity = severity
        self.provider = provider
        self.error_hint = error_hint
        if expires_at is not None:
            self.expires_at = expires_at
        elif severity == FailureSeverity.DAILY:
            # Expire at next midnight UTC
            import datetime
            now = datetime.datetime.now(datetime.timezone.utc)
            tomorrow = (now + datetime.timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            self.expires_at = tomorrow.timestamp()
        else:
            self.expires_at = time.monotonic() + _SEVERITY_COOLDOWNS[severity]

    @property
    def is_expired(self) -> bool:
        # For DAILY records, use wall-clock time (monotonic doesn't work
        # across midnight in a meaningful way for "expire at midnight UTC")
        if self.severity == FailureSeverity.DAILY:
            import datetime
            return datetime.datetime.now(datetime.timezone.utc).timestamp() > self.expires_at
        return time.monotonic() > self.expires_at


class FailureTracker:
    """Tracks model failures with two-strikes escalation.

    Design principle: Don't try to classify errors by parsing messages or
    naming providers. Just track the pattern:

    - First failure (any error) → TRANSIENT (60s cooldown)
    - Second failure while TRANSIENT is still active → DAILY (until midnight)

    This naturally distinguishes RPM throttling (one 429, then it works
    after 60s) from RPD exhaustion (429, wait 60s, still 429 → DAILY).

    Models known to be unreachable (geo_blocked=True in config) are
    filtered out at selection time — they never reach this tracker.
    """

    def __init__(self):
        # model_id → list of FailureRecord
        self._model_failures: dict[str, list[FailureRecord]] = defaultdict(list)
        # model_id → consecutive failure count (reset on success)
        self._strike_count: dict[str, int] = defaultdict(int)

    def record_failure(
        self,
        model_id: str,
        provider: str = "",
        error_hint: str = "",
    ) -> str:
        """Record a failure for a model. Returns the severity assigned.

        Two-strikes escalation:
        - Strike 1: TRANSIENT (60s cooldown)
        - Strike 2+: DAILY (until midnight UTC)

        A success resets the strike count.
        """
        self._strike_count[model_id] += 1
        strike = self._strike_count[model_id]

        if strike >= 2:
            severity = FailureSeverity.DAILY
        else:
            severity = FailureSeverity.TRANSIENT

        record = FailureRecord(
            severity=severity,
            provider=provider,
            error_hint=error_hint,
        )
        self._model_failures[model_id].append(record)

        # Prune expired records for this model
        self._model_failures[model_id] = [
            r for r in self._model_failures[model_id] if not r.is_expired
        ]

        logger.debug(
            f"Failure recorded for {model_id}: "
            f"strike {strike}, severity={severity}, hint={error_hint[:60]}"
        )

        return severity

    def record_success(self, model_id: str, provider: str = "") -> None:
        """Record a successful call — resets strike count and clears transient failures.

        DAILY failures are NOT cleared (they need midnight reset).
        But the strike count resets, so the next failure starts fresh.
        """
        # Reset strike count
        self._strike_count[model_id] = 0

        if model_id in self._model_failures:
            # Only clear transient failures
            self._model_failures[model_id] = [
                r for r in self._model_failures[model_id]
                if r.severity != FailureSeverity.TRANSIENT
            ]
            if not self._model_failures[model_id]:
                del self._model_failures[model_id]

    def is_cooling_down(self, model_id: str) -> bool:
        """Check if a model has any active (non-expired) failure records."""
        if model_id not in self._model_failures:
            return False
        active = [r for r in self._model_failures[model_id] if not r.is_expired]
        return len(active) > 0

    def is_daily_exhausted(self, model_id: str) -> bool:
        """Check if a model has an active DAILY failure."""
        if model_id not in self._model_failures:
            return False
        return any(
            r.severity == FailureSeverity.DAILY and not r.is_expired
            for r in self._model_failures[model_id]
        )

    def failure_count(self, model_id: str) -> int:
        """Number of active (non-expired) failure records for a model."""
        if model_id not in self._model_failures:
            return 0
        return len([r for r in self._model_failures[model_id] if not r.is_expired])

    def strike_count(self, model_id: str) -> int:
        """Current consecutive failure count for a model."""
        return self._strike_count.get(model_id, 0)

    def get_failure_summary(self) -> dict[str, dict]:
        """Get a summary of all active failures for debugging/logging."""
        summary = {}
        for model_id, records in self._model_failures.items():
            active = [r for r in records if not r.is_expired]
            if active:
                worst = max(
                    active,
                    key=lambda r: {
                        FailureSeverity.TRANSIENT: 1,
                        FailureSeverity.DAILY: 2,
                    }.get(r.severity, 0),
                )
                summary[model_id] = {
                    "severity": worst.severity,
                    "provider": worst.provider,
                    "hint": worst.error_hint,
                    "active_failures": len(active),
                    "strike_count": self._strike_count.get(model_id, 0),
                }
        return summary


# ── Model Registry ─────────────────────────────────────────────────────


class ModelRegistry:
    """Registry of available models with health status and selection logic.

    Uses round-robin within tiers to distribute load across models,
    preventing a single model from being hammered while others sit idle.
    Family diversity is enforced for debater assignment.

    Geo-blocked models (geo_blocked=True in config) are filtered out at
    selection time — they're never tried, never tracked in FailureTracker.

    Provider metadata (from the providers: section of providers.yaml) is
    stored for cross-provider fallback awareness. When a model fails, the
    registry can prefer the same underlying model (same canonical_id) from
    a different provider — e.g., llama-3.3-70b on Cloudflare fails, try
    the same model on OpenRouter.
    """

    def __init__(
        self,
        models: list[ModelInfo],
        fallback_chains: dict[str, list[str]] | None = None,
        providers: dict | None = None,
    ):
        self._models: dict[str, ModelInfo] = {}
        self._fallback_chains: dict[str, list[str]] = fallback_chains or {}
        self._token_usage: dict[str, int] = defaultdict(int)

        # Round-robin counters: tier_key → last-used index
        # Key format: "{tier}:{family}" or "{tier}:__all__"
        self._rr_counters: dict[str, int] = defaultdict(int)

        # Failure tracking for health-aware selection
        self.failures = FailureTracker()

        # Index: canonical_id → [model_ids] for cross-provider dedup
        self._canonical_index: dict[str, list[str]] = defaultdict(list)

        # Provider metadata for cross-provider awareness
        self._providers: dict = providers or {}

        for m in models:
            self._models[m.model_id] = m
            if m.canonical_id:
                self._canonical_index[m.canonical_id].append(m.model_id)

    @classmethod
    def from_config(cls, config: Any) -> "ModelRegistry":
        """Create a registry from a CouncilConfig object."""
        models = [
            ModelInfo(
                model_id=mc.model_id,
                family=mc.family,
                tier=mc.tier,
                context_window=mc.context_window,
                input_cost_per_m=mc.input_cost_per_m,
                output_cost_per_m=mc.output_cost_per_m,
                supports_local=mc.supports_local,
                is_available=mc.enabled,
                provider=mc.provider,
                api_base=mc.api_base,
                api_key=mc.api_key,
                conserve=mc.conserve,
                geo_blocked=mc.geo_blocked,
                canonical_id=mc.canonical_id,
                rpm=mc.rpm,
                daily_quota=mc.daily_quota,
                regenerates=mc.regenerates,
            )
            for mc in config.models
        ]
        chains = {fc.role_type: fc.chain for fc in config.fallback_chains}
        # Pass provider metadata for cross-provider fallback awareness
        providers = {}
        if hasattr(config, 'providers') and config.providers:
            providers = {name: info.model_dump() for name, info in config.providers.items()}
        registry = cls(models=models, fallback_chains=chains, providers=providers)

        # Log geo-blocked models (filtered at selection time, NOT tracked)
        geo_blocked = [m for m in models if m.geo_blocked]
        if geo_blocked:
            providers = sorted({m.provider for m in geo_blocked if m.provider})
            logger.info(
                f"Geo-blocked models (skipped at selection): "
                f"{len(geo_blocked)} models from providers {providers}"
            )

        return registry

    def get(self, model_id: str) -> ModelInfo | None:
        """Get a model by ID, or None if not found."""
        return self._models.get(model_id)

    def all_models(self) -> list[ModelInfo]:
        """Return all registered models (including geo-blocked)."""
        return list(self._models.values())

    def available_models(self) -> list[ModelInfo]:
        """Return models that are available and not geo-blocked."""
        return [m for m in self._models.values() if m.is_available and not m.geo_blocked]

    def selectable_models(self) -> list[ModelInfo]:
        """Return models eligible for selection: available, not geo-blocked,
        not daily-exhausted.

        This is the pool used by selection algorithms.
        """
        return [
            m for m in self._models.values()
            if m.is_available
            and not m.geo_blocked
            and not self.failures.is_daily_exhausted(m.model_id)
        ]

    def models_by_family(self, family: str) -> list[ModelInfo]:
        """Get available models in a family."""
        return [m for m in self.available_models() if m.family == family]

    def models_by_tier(self, tier: ModelTier) -> list[ModelInfo]:
        """Get available models at a tier."""
        return [m for m in self.available_models() if m.tier == tier]

    def available_families(self) -> list[str]:
        """Get distinct families with at least one available model."""
        families = sorted({m.family for m in self.available_models()})
        return families

    def provider_availability(self) -> dict[str, dict[str, list[str]]]:
        """Get a map of family → provider → [model_ids] for cross-provider awareness.

        Returns:
            {family: {provider: [model_id, ...], ...}, ...}
        """
        result: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        for m in self.available_models():
            if m.provider:
                result[m.family][m.provider].append(m.model_id)
        return dict(result)

    def models_by_provider(self, provider: str) -> list[ModelInfo]:
        """Get available models from a specific provider."""
        return [m for m in self.available_models() if m.provider == provider]

    def models_by_canonical(self, canonical_id: str) -> list[ModelInfo]:
        """Get all models sharing a canonical ID (same model, different providers)."""
        ids = self._canonical_index.get(canonical_id, [])
        return [self._models[mid] for mid in ids if mid in self._models]

    def cheapest_in_tier(self, tier: ModelTier, family: str | None = None) -> ModelInfo | None:
        """Find the cheapest available model at a given tier, optionally in a family."""
        candidates = self.models_by_tier(tier)
        if family:
            candidates = [m for m in candidates if m.family == family]
        if not candidates:
            return None
        return min(candidates, key=lambda m: m.input_cost_per_m)

    def record_usage(self, model_id: str, tokens: int) -> None:
        """Record token usage for a model."""
        self._token_usage[model_id] += tokens

    def get_usage(self) -> dict[str, ModelUsage]:
        """Get accumulated token usage per model."""
        usage = {}
        for model_id, tokens in self._token_usage.items():
            model = self._models.get(model_id)
            if model:
                usage[model_id] = ModelUsage(
                    model=model_id,
                    family=model.family,
                    tokens=tokens,
                )
        return usage

    # ── Round-Robin Selection ──────────────────────────────────────────

    def _rr_select(
        self,
        candidates: list[ModelInfo],
        rr_key: str,
    ) -> ModelInfo | None:
        """Select a model using round-robin within a candidate pool.

        Skips models that are:
        1. Currently cooling down (recent failure)
        2. Daily-exhausted (two-strikes escalated)
        3. Flagged as conserve (non-regenerating credits) unless all are conserved

        Geo-blocked models are already filtered out before reaching this method.

        The round-robin counter ensures we cycle through available models
        instead of always hitting the same one.
        """
        if not candidates:
            return None

        # Sort candidates by health: prefer non-cooling, non-daily, non-conserved
        def sort_key(m: ModelInfo) -> tuple:
            daily = self.failures.is_daily_exhausted(m.model_id)
            cooling = self.failures.is_cooling_down(m.model_id)
            conserved = m.conserve
            # Lower is better: (daily, cooling, conserved)
            return (daily, cooling, conserved)

        sorted_candidates = sorted(candidates, key=sort_key)

        # If ALL candidates are cooling down, pick least-bad option
        all_cooling = all(
            self.failures.is_cooling_down(m.model_id)
            for m in sorted_candidates
        )
        if all_cooling:
            return min(sorted_candidates, key=lambda m: (
                self.failures.is_daily_exhausted(m.model_id),
                self.failures.failure_count(m.model_id),
            ))

        # Filter out daily-exhausted and cooling-down models (if alternatives exist)
        active = [
            m for m in sorted_candidates
            if not self.failures.is_daily_exhausted(m.model_id)
            and not self.failures.is_cooling_down(m.model_id)
        ]
        if not active:
            # Fall back to cooling-down but not daily-exhausted (they might recover)
            active = [
                m for m in sorted_candidates
                if not self.failures.is_daily_exhausted(m.model_id)
            ]
        if not active:
            # Last resort: daily-exhausted too
            active = sorted_candidates

        # Filter out conserved models (if we have non-conserved alternatives)
        non_conserved = [m for m in active if not m.conserve]
        if non_conserved:
            pool = non_conserved
        else:
            pool = active

        # Round-robin within the pool
        idx = self._rr_counters[rr_key] % len(pool)
        selected = pool[idx]
        self._rr_counters[rr_key] = idx + 1
        return selected

    # ── Model Selection Algorithm ──────────────────────────────────────

    def select_model_for_role(
        self,
        role: RoleSpec,
        *,
        family: str | None = None,
        exclude_families: list[str] | None = None,
        local_only: bool = False,
        api_only: bool = False,
        model_override: str | None = None,
        already_assigned_families: list[str] | None = None,
    ) -> ModelInfo | None:
        """Select the best model for a given role.

        Uses round-robin within tiers to distribute load, with family
        diversity as a hard constraint for debaters.

        Selection algorithm:
        1. Filter out geo-blocked models (never selected)
        2. Apply user constraints (family, exclude, local_only, api_only)
        3. Maximize family diversity: prefer unassigned families for debaters
        4. Select tier based on role importance
        5. Round-robin within the filtered pool
        6. Fallback to adjacent tier if no matches
        """
        exclude = set(exclude_families or [])
        assigned = already_assigned_families or []

        # If explicit override, use it
        if model_override:
            m = self.get(model_override)
            if m and m.is_available and not m.geo_blocked:
                return m
            logger.warning(f"Model override {model_override} not available, falling back")

        # Determine target tier from role type
        target_tier = self._tier_for_role(role)

        # Determine target family
        target_family = family or role.suggested_model or None

        # Get candidate pool — already excludes geo-blocked
        candidates = self.available_models()

        # Apply local/API constraint
        if local_only:
            candidates = [m for m in candidates if m.supports_local]
        if api_only:
            candidates = [m for m in candidates if not m.supports_local]

        # Apply family constraint
        if target_family:
            candidates = [m for m in candidates if m.family == target_family]
        candidates = [m for m in candidates if m.family not in exclude]

        # Prefer family diversity: avoid families already assigned
        if assigned and not target_family:
            # Try to find a model from an unassigned family first
            unassigned = [m for m in candidates if m.family not in assigned]
            if unassigned:
                candidates = unassigned
            # If all families are taken, pick from least-used family
            else:
                family_counts = defaultdict(int)
                for f in assigned:
                    family_counts[f] += 1
                # Sort candidates by their family's assignment count (ascending)
                candidates = sorted(
                    candidates,
                    key=lambda m: family_counts.get(m.family, 0),
                )

        # Filter by tier
        tier_matches = [m for m in candidates if m.tier == target_tier]
        if tier_matches:
            # Build round-robin key from constraints
            rr_key = f"{target_tier.value}:{target_family or '__all__'}"
            selected = self._rr_select(tier_matches, rr_key)
            if selected:
                return selected
            # Fallback to cheapest if round-robin somehow fails
            return min(tier_matches, key=lambda m: m.input_cost_per_m)

        # Fallback: try adjacent tier
        if target_tier == ModelTier.PREMIUM:
            fallback_tier = ModelTier.MID
        elif target_tier == ModelTier.MID:
            fallback_tier = ModelTier.CHEAP
        else:
            # Already at cheapest, try any tier
            if candidates:
                rr_key = f"any:{target_family or '__all__'}"
                return self._rr_select(candidates, rr_key) or min(
                    candidates, key=lambda m: m.input_cost_per_m
                )
            return None

        tier_matches = [m for m in candidates if m.tier == fallback_tier]
        if tier_matches:
            rr_key = f"{fallback_tier.value}:{target_family or '__all__'}"
            return self._rr_select(tier_matches, rr_key) or min(
                tier_matches, key=lambda m: m.input_cost_per_m
            )

        # Last resort: any available candidate
        if candidates:
            return self._rr_select(candidates, "last_resort") or min(
                candidates, key=lambda m: m.input_cost_per_m
            )
        return None

    def resolve_fallback(self, model_id: str, role_type: str) -> ModelInfo | None:
        """Resolve a failed model through its fallback chain.

        Strategy (in order):
        1. Cross-provider same-model fallback: If the failed model has a
           canonical_id, try the same model on a different provider first.
           This is the highest-value fallback — same reasoning capability,
           different route.
        2. Explicit fallback chain: Walk the configured chain for the role.
        3. Same-tier round-robin: Any available model at the same tier.

        Skips geo-blocked models, daily-exhausted models, and models
        currently in cooldown.
        """
        failed_model = self.get(model_id)

        # Strategy 1: Cross-provider same-model fallback
        if failed_model and failed_model.canonical_id:
            cross_provider = self._cross_provider_fallback(
                model_id, failed_model.canonical_id
            )
            if cross_provider:
                logger.info(
                    f"Cross-provider fallback: {model_id} → "
                    f"{cross_provider.model_id} (same canonical_id: "
                    f"{failed_model.canonical_id})"
                )
                return cross_provider

        # Strategy 2: Explicit fallback chain
        chain = self._fallback_chains.get(role_type, [])
        for fallback_id in chain:
            if fallback_id == model_id:
                continue  # Skip the one that already failed
            m = self.get(fallback_id)
            if not m or not m.is_available:
                continue
            if m.geo_blocked:
                continue  # Skip geo-blocked
            if self.failures.is_daily_exhausted(fallback_id):
                continue  # Skip daily-exhausted
            if self.failures.is_cooling_down(fallback_id):
                continue  # Skip in cooldown
            return m

        # Strategy 3: Same-tier round-robin among all available models
        if failed_model:
            tier_models = [
                m for m in self.available_models()
                if m.tier == failed_model.tier
                and m.model_id != model_id
                and not self.failures.is_daily_exhausted(m.model_id)
                and not self.failures.is_cooling_down(m.model_id)
            ]
            if tier_models:
                return self._rr_select(tier_models, f"fallback:{role_type}")
        return None

    def _cross_provider_fallback(
        self, failed_model_id: str, canonical_id: str
    ) -> ModelInfo | None:
        """Find an alternative provider for the same model (same canonical_id).

        Prefers providers by priority (from providers: section), and skips
        models that are geo-blocked, daily-exhausted, or in cooldown.
        """
        alternatives = self.models_by_canonical(canonical_id)
        if not alternatives:
            return None

        # Filter out the failed model, geo-blocked, and unhealthy models
        viable = []
        for m in alternatives:
            if m.model_id == failed_model_id:
                continue
            if m.geo_blocked or not m.is_available:
                continue
            if self.failures.is_daily_exhausted(m.model_id):
                continue
            if self.failures.is_cooling_down(m.model_id):
                continue
            viable.append(m)

        if not viable:
            return None

        # Sort by provider priority (lower = higher priority)
        def provider_priority(m: ModelInfo) -> int:
            if m.provider and m.provider in self._providers:
                return self._providers[m.provider].get("priority", 99)
            return 99

        viable.sort(key=provider_priority)

        # Filter conserved providers (prefer non-conserved)
        non_conserved = [m for m in viable if not m.conserve]
        return non_conserved[0] if non_conserved else viable[0]

    @staticmethod
    def _tier_for_role(role: RoleSpec) -> ModelTier:
        """Determine the appropriate tier for a role."""
        if role.is_research:
            return ModelTier.CHEAP
        # Debaters default to MID; synthesizer will be handled separately
        return ModelTier.MID

    # ── Health Check ───────────────────────────────────────────────────

    async def health_check(self, timeout: float = 10.0) -> list[HealthCheckResult]:
        """Ping each non-geo-blocked model with a trivial request.

        Returns health check results. Models that fail are tracked by
        FailureTracker (two-strikes escalation applies).
        """
        results = []
        # Only check non-geo-blocked models — geo-blocked are known failures
        models_to_check = [m for m in self._models.values() if not m.geo_blocked]

        async def check_one(model: ModelInfo) -> HealthCheckResult:
            try:
                start = time.monotonic()
                kwargs: dict[str, Any] = {
                    "model": model.model_id,
                    "messages": [{"role": "user", "content": "Say OK"}],
                    "max_tokens": 5,
                    "timeout": timeout,
                }
                # Pass custom api_base/api_key if configured
                if model.api_base:
                    kwargs["api_base"] = model.api_base
                if model.api_key:
                    kwargs["api_key"] = model.api_key

                response = await litellm.acompletion(**kwargs)
                latency = (time.monotonic() - start) * 1000
                model.is_available = True
                self.failures.record_success(model.model_id, provider=model.provider)
                return HealthCheckResult(
                    model_id=model.model_id,
                    is_healthy=True,
                    latency_ms=round(latency, 1),
                )
            except Exception as e:
                model.is_available = False
                self.failures.record_failure(
                    model.model_id, provider=model.provider,
                    error_hint=str(e)[:80],
                )
                return HealthCheckResult(
                    model_id=model.model_id,
                    is_healthy=False,
                    error=str(e)[:200],
                )

        # Run health checks concurrently with a semaphore
        sem = asyncio.Semaphore(5)

        async def limited_check(model: ModelInfo) -> HealthCheckResult:
            async with sem:
                return await check_one(model)

        results = await asyncio.gather(
            *[limited_check(m) for m in models_to_check],
            return_exceptions=False,
        )

        healthy = sum(1 for r in results if r.is_healthy)
        total = len(results)
        logger.info(f"Health check: {healthy}/{total} models healthy")

        return list(results)


# ── LiteLLM Wrapper ────────────────────────────────────────────────────


class LLMClient:
    """Wrapper around LiteLLM for making completion calls with budget tracking."""

    def __init__(self, registry: ModelRegistry, budget: TokenBudget | None = None):
        self.registry = registry
        self.budget = budget

    async def complete(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        response_format: dict | None = None,
        tools: list[dict] | None = None,
        budget_override: TokenBudget | None = None,
        **kwargs,
    ) -> tuple[str, int]:
        """Make a completion call and return (response_text, tokens_used).

        Two-strikes failure tracking:
        - First failure → TRANSIENT (60s cooldown), retry with backoff
        - Second failure → DAILY (until midnight), skip to fallback

        Geo-blocked models are filtered at selection time and should
        never reach this method.
        """
        # Use the call-level budget if provided, otherwise fall back to the client-level one
        active_budget = budget_override or self.budget

        # Check budget before making the call
        if active_budget and active_budget.is_exhausted:
            raise RuntimeError("Token budget exhausted")

        # Get model info for custom api_base/api_key
        model_info = self.registry.get(model_id)

        retries = 2
        backoff_times = [1, 4]

        for attempt in range(retries + 1):
            try:
                call_kwargs: dict[str, Any] = {
                    "model": model_id,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "response_format": response_format,
                    "tools": tools,
                    **kwargs,
                }

                # Pass custom api_base/api_key if configured (for Cloudflare etc.)
                if model_info and model_info.api_base:
                    call_kwargs["api_base"] = model_info.api_base
                if model_info and model_info.api_key:
                    call_kwargs["api_key"] = model_info.api_key

                response = await litellm.acompletion(**call_kwargs)

                # Extract response text — handle reasoning models
                content = response.choices[0].message.content or ""
                # Some reasoning models put content in reasoning_content
                if not content:
                    reasoning = getattr(
                        response.choices[0].message, "reasoning_content", None
                    )
                    if reasoning:
                        content = str(reasoning)

                # Track token usage
                usage = response.usage
                tokens_used = usage.total_tokens if usage else 0

                if active_budget:
                    active_budget.consume(tokens_used)
                self.registry.record_usage(model_id, tokens_used)
                self.registry.failures.record_success(
                    model_id, provider=model_info.provider if model_info else ""
                )

                return content, tokens_used

            except Exception as e:
                provider = model_info.provider if model_info else ""
                error_str = str(e)

                # Record failure — two-strikes escalation happens inside
                severity = self.registry.failures.record_failure(
                    model_id, provider=provider,
                    error_hint=error_str[:80],
                )

                # If escalated to DAILY, skip remaining retries and go to fallback
                if severity == FailureSeverity.DAILY:
                    logger.warning(
                        f"Daily exhaustion for {model_id} (strike "
                        f"{self.registry.failures.strike_count(model_id)}): "
                        f"{error_str[:100]}"
                    )
                    # Immediately try fallback — no point retrying
                    fallback = self.registry.resolve_fallback(
                        model_id, self._role_type_for_model(model_id)
                    )
                    if fallback:
                        logger.warning(
                            f"Daily exhaustion on {model_id}, "
                            f"falling back to {fallback.model_id}"
                        )
                        return await self.complete(
                            model_id=fallback.model_id,
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            response_format=response_format,
                            tools=tools,
                            budget_override=budget_override,
                            **kwargs,
                        )
                    raise RuntimeError(
                        f"Daily exhaustion for {model_id} and no fallback: {e}"
                    )

                # TRANSIENT severity — retry with backoff
                if attempt < retries:
                    wait = backoff_times[attempt]
                    logger.warning(
                        f"Transient failure for {model_id} (strike 1): "
                        f"{error_str[:100]}, retrying in {wait}s"
                    )
                    await asyncio.sleep(wait)
                else:
                    # All retries exhausted, try fallback
                    fallback = self.registry.resolve_fallback(
                        model_id, self._role_type_for_model(model_id)
                    )
                    if fallback:
                        logger.warning(
                            f"All retries failed for {model_id}, "
                            f"falling back to {fallback.model_id}"
                        )
                        return await self.complete(
                            model_id=fallback.model_id,
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            response_format=response_format,
                            tools=tools,
                            budget_override=budget_override,
                            **kwargs,
                        )
                    raise RuntimeError(
                        f"Failed to get response from {model_id} after {retries} retries: {e}"
                    )

        # Should never reach here, but just in case
        raise RuntimeError(f"Unexpected error calling {model_id}")

    def _role_type_for_model(self, model_id: str) -> str:
        """Determine role type for a model by checking fallback chains."""
        for role_type, chain in self.registry._fallback_chains.items():
            if model_id in chain:
                return role_type
        return "debater"  # Default fallback chain
