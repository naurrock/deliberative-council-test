"""
Model registry and LiteLLM wrapper for Deliberative Council.

Handles model selection with family diversity, tier matching, fallback
chains, provider health checks, and token counting.
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


# ── Model Registry ─────────────────────────────────────────────────────


class ModelRegistry:
    """Registry of available models with health status and selection logic."""

    def __init__(self, models: list[ModelInfo], fallback_chains: dict[str, list[str]] | None = None):
        self._models: dict[str, ModelInfo] = {}
        self._fallback_chains: dict[str, list[str]] = fallback_chains or {}
        self._token_usage: dict[str, int] = defaultdict(int)

        for m in models:
            self._models[m.model_id] = m

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
            )
            for mc in config.models
        ]
        chains = {fc.role_type: fc.chain for fc in config.fallback_chains}
        return cls(models=models, fallback_chains=chains)

    def get(self, model_id: str) -> ModelInfo | None:
        """Get a model by ID, or None if not found."""
        return self._models.get(model_id)

    def all_models(self) -> list[ModelInfo]:
        """Return all registered models."""
        return list(self._models.values())

    def available_models(self) -> list[ModelInfo]:
        """Return models that are currently available (healthy and enabled)."""
        return [m for m in self._models.values() if m.is_available]

    def models_by_family(self, family: str) -> list[ModelInfo]:
        """Get all available models in a family."""
        return [m for m in self.available_models() if m.family == family]

    def models_by_tier(self, tier: ModelTier) -> list[ModelInfo]:
        """Get all available models at a tier."""
        return [m for m in self.available_models() if m.tier == tier]

    def available_families(self) -> list[str]:
        """Get distinct families with at least one available model."""
        families = sorted({m.family for m in self.available_models()})
        return families

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

        Follows the Scout model selection algorithm:
        1. Apply user constraints (family, exclude, local_only, api_only)
        2. Maximize family diversity across debaters
        3. Select tier based on role importance
        4. Pick cheapest model meeting tier within filtered set
        """
        exclude = set(exclude_families or [])
        assigned = already_assigned_families or []

        # If explicit override, use it
        if model_override:
            m = self.get(model_override)
            if m and m.is_available:
                return m
            logger.warning(f"Model override {model_override} not available, falling back")

        # Determine target tier from role type
        target_tier = self._tier_for_role(role)

        # Determine target family
        target_family = family or role.suggested_model or None

        # Get candidate pool
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
            # If all families are taken, use round-robin (pick least-used family)

        # Filter by tier
        tier_matches = [m for m in candidates if m.tier == target_tier]
        if tier_matches:
            return min(tier_matches, key=lambda m: m.input_cost_per_m)

        # Fallback: try adjacent tier
        if target_tier == ModelTier.PREMIUM:
            fallback_tier = ModelTier.MID
        elif target_tier == ModelTier.MID:
            fallback_tier = ModelTier.CHEAP
        else:
            # Already at cheapest, try any tier
            if candidates:
                return min(candidates, key=lambda m: m.input_cost_per_m)
            return None

        tier_matches = [m for m in candidates if m.tier == fallback_tier]
        if tier_matches:
            return min(tier_matches, key=lambda m: m.input_cost_per_m)

        # Last resort: any available candidate
        if candidates:
            return min(candidates, key=lambda m: m.input_cost_per_m)
        return None

    def resolve_fallback(self, model_id: str, role_type: str) -> ModelInfo | None:
        """Resolve a failed model through its fallback chain."""
        chain = self._fallback_chains.get(role_type, [])
        for fallback_id in chain:
            if fallback_id == model_id:
                continue  # Skip the one that already failed
            m = self.get(fallback_id)
            if m and m.is_available:
                return m
        # Last resort: cheapest available from any family at same tier
        failed_model = self.get(model_id)
        if failed_model:
            cheapest = self.cheapest_in_tier(failed_model.tier)
            if cheapest and cheapest.model_id != model_id:
                return cheapest
        return None

    @staticmethod
    def _tier_for_role(role: RoleSpec) -> ModelTier:
        """Determine the appropriate tier for a role."""
        if role.is_research:
            return ModelTier.CHEAP
        # Debaters default to MID; synthesizer will be handled separately
        return ModelTier.MID

    # ── Health Check ───────────────────────────────────────────────────

    async def health_check(self, timeout: float = 10.0) -> list[HealthCheckResult]:
        """Ping each configured model with a trivial request.

        Returns health check results. Models that fail are marked unavailable.
        """
        results = []
        models_to_check = list(self._models.values())

        async def check_one(model: ModelInfo) -> HealthCheckResult:
            try:
                start = time.monotonic()
                response = await litellm.acompletion(
                    model=model.model_id,
                    messages=[{"role": "user", "content": "Say OK"}],
                    max_tokens=5,
                    timeout=timeout,
                )
                latency = (time.monotonic() - start) * 1000
                model.is_available = True
                return HealthCheckResult(
                    model_id=model.model_id,
                    is_healthy=True,
                    latency_ms=round(latency, 1),
                )
            except Exception as e:
                model.is_available = False
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
        **kwargs,
    ) -> tuple[str, int]:
        """Make a completion call and return (response_text, tokens_used).

        Raises RuntimeError if the call fails after retries.
        """
        # Check budget before making the call
        if self.budget and self.budget.is_exhausted:
            raise RuntimeError("Token budget exhausted")

        retries = 2
        backoff_times = [1, 4]

        for attempt in range(retries + 1):
            try:
                response = await litellm.acompletion(
                    model=model_id,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_format=response_format,
                    tools=tools,
                    **kwargs,
                )

                # Extract response text
                content = response.choices[0].message.content or ""

                # Track token usage
                usage = response.usage
                tokens_used = usage.total_tokens if usage else 0

                if self.budget:
                    self.budget.consume(tokens_used)
                self.registry.record_usage(model_id, tokens_used)

                return content, tokens_used

            except litellm.Timeout as e:
                if attempt < retries:
                    wait = backoff_times[attempt]
                    logger.warning(
                        f"Timeout calling {model_id}, retrying in {wait}s "
                        f"(attempt {attempt + 1}/{retries})"
                    )
                    await asyncio.sleep(wait)
                else:
                    # Try fallback
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
                            **kwargs,
                        )
                    raise RuntimeError(
                        f"Failed to get response from {model_id} after {retries} retries: {e}"
                    )

            except Exception as e:
                if attempt < retries:
                    wait = backoff_times[attempt]
                    logger.warning(
                        f"Error calling {model_id}: {e}, retrying in {wait}s"
                    )
                    await asyncio.sleep(wait)
                else:
                    raise RuntimeError(
                        f"Failed to get response from {model_id} after {retries} retries: {e}"
                    )

        # Should never reach here, but just in case
        raise RuntimeError(f"Unexpected error calling {model_id}")

    def _role_type_for_model(self, model_id: str) -> str:
        """Heuristic role type for fallback resolution."""
        model = self.registry.get(model_id)
        if model:
            return "debater"  # Default fallback chain
        return "debater"

    async def complete_structured(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        schema: type,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        **kwargs,
    ) -> tuple[Any, int]:
        """Make a completion call expecting JSON output that conforms to a Pydantic schema.

        Returns (parsed_model_instance, tokens_used).
        Falls back to manual parsing if structured output not supported.
        """
        # Try with JSON response format
        import json

        schema_json = schema.model_json_schema()
        system_hint = (
            f"You must respond with valid JSON conforming to this schema:\n"
            f"{json.dumps(schema_json, indent=2)}\n\n"
            f"Respond ONLY with the JSON object, no other text."
        )

        augmented_messages = list(messages)
        # Add schema hint to the last user message or as a system message
        if augmented_messages and augmented_messages[-1]["role"] == "user":
            augmented_messages[-1]["content"] += f"\n\n{system_hint}"
        else:
            augmented_messages.append({"role": "system", "content": system_hint})

        try:
            content, tokens = await self.complete(
                model_id=model_id,
                messages=augmented_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "json_object"},
                **kwargs,
            )
        except Exception:
            # Fallback: try without response_format
            content, tokens = await self.complete(
                model_id=model_id,
                messages=augmented_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )

        # Parse JSON from response
        try:
            # Strip markdown code fences if present
            cleaned = content.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                # Remove first and last lines (code fences)
                cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned

            data = json.loads(cleaned)
            return schema(**data), tokens
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse structured output: {e}")
            # Return raw content in a minimal wrapper
            raise ValueError(f"Could not parse structured response: {e}\nRaw: {content[:500]}")
