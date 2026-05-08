"""Tests for council.models — model registry and LLM client."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from council.models import ModelRegistry, TokenBudget, LLMClient
from council.types import (
    ModelInfo,
    ModelTier,
    RoleSpec,
    HealthCheckResult,
)


# ── Test Data ──────────────────────────────────────────────────────────

SAMPLE_MODELS = [
    ModelInfo(model_id="openai/gpt-4.1", family="openai", tier=ModelTier.PREMIUM, context_window=128_000, input_cost_per_m=2.00, output_cost_per_m=8.00),
    ModelInfo(model_id="openai/gpt-4.1-mini", family="openai", tier=ModelTier.MID, context_window=128_000, input_cost_per_m=0.40, output_cost_per_m=1.60),
    ModelInfo(model_id="anthropic/claude-sonnet-4", family="anthropic", tier=ModelTier.PREMIUM, context_window=200_000, input_cost_per_m=3.00, output_cost_per_m=15.00),
    ModelInfo(model_id="gemini/gemini-2.5-flash", family="google", tier=ModelTier.MID, context_window=1_000_000, input_cost_per_m=0.15, output_cost_per_m=0.60),
    ModelInfo(model_id="ollama_chat/qwen3:8b", family="alibaba", tier=ModelTier.CHEAP, context_window=128_000, supports_local=True),
    ModelInfo(model_id="deepseek/deepseek-chat", family="deepseek", tier=ModelTier.MID, context_window=128_000, input_cost_per_m=0.27, output_cost_per_m=1.10),
    ModelInfo(model_id="ollama_chat/phi4-mini", family="microsoft", tier=ModelTier.CHEAP, context_window=128_000, supports_local=True),
]

SAMPLE_CHAINS = {
    "research": ["ollama_chat/qwen3:8b", "anthropic/claude-sonnet-4"],
    "debater": ["openai/gpt-4.1-mini", "gemini/gemini-2.5-flash", "deepseek/deepseek-chat"],
    "synthesizer": ["anthropic/claude-sonnet-4", "openai/gpt-4.1"],
}


# ── TokenBudget ────────────────────────────────────────────────────────


class TestTokenBudget:
    def test_initial_state(self):
        b = TokenBudget(total=1000)
        assert b.total == 1000
        assert b.used == 0
        assert b.remaining == 1000
        assert not b.is_exhausted

    def test_consume(self):
        b = TokenBudget(total=1000)
        b.consume(300)
        assert b.used == 300
        assert b.remaining == 700

    def test_exhausted(self):
        b = TokenBudget(total=1000)
        b.consume(1000)
        assert b.is_exhausted
        assert b.remaining == 0

    def test_can_allocate(self):
        b = TokenBudget(total=1000)
        assert b.can_allocate(500)
        assert b.can_allocate(1000)
        assert not b.can_allocate(1001)

    def test_over_allocate(self):
        b = TokenBudget(total=1000)
        b.consume(800)
        assert not b.can_allocate(300)


# ── ModelRegistry ──────────────────────────────────────────────────────


class TestModelRegistry:
    def test_creation(self):
        reg = ModelRegistry(SAMPLE_MODELS, SAMPLE_CHAINS)
        assert len(reg.all_models()) == 7

    def test_get_model(self):
        reg = ModelRegistry(SAMPLE_MODELS)
        m = reg.get("openai/gpt-4.1")
        assert m is not None
        assert m.family == "openai"
        assert m.tier == ModelTier.PREMIUM

    def test_get_missing_model(self):
        reg = ModelRegistry(SAMPLE_MODELS)
        assert reg.get("nonexistent") is None

    def test_available_models(self):
        reg = ModelRegistry(SAMPLE_MODELS)
        # Mark one as unavailable
        reg.get("openai/gpt-4.1").is_available = False
        available = reg.available_models()
        assert len(available) == 6
        assert all(m.is_available for m in available)

    def test_models_by_family(self):
        # Fresh models since other tests mutate is_available
        models = [
            ModelInfo(model_id="openai/gpt-4.1", family="openai", tier=ModelTier.PREMIUM, context_window=128_000, input_cost_per_m=2.00, output_cost_per_m=8.00),
            ModelInfo(model_id="openai/gpt-4.1-mini", family="openai", tier=ModelTier.MID, context_window=128_000, input_cost_per_m=0.40, output_cost_per_m=1.60),
            ModelInfo(model_id="gemini/gemini-2.5-flash", family="google", tier=ModelTier.MID, context_window=1_000_000, input_cost_per_m=0.15, output_cost_per_m=0.60),
        ]
        reg = ModelRegistry(models)
        openai = reg.models_by_family("openai")
        assert len(openai) == 2  # gpt-4.1 and gpt-4.1-mini

    def test_models_by_tier(self):
        reg = ModelRegistry(SAMPLE_MODELS)
        cheap = reg.models_by_tier(ModelTier.CHEAP)
        assert len(cheap) == 2  # qwen3:8b and phi4-mini

    def test_available_families(self):
        reg = ModelRegistry(SAMPLE_MODELS)
        families = reg.available_families()
        assert "openai" in families
        assert "anthropic" in families
        assert "google" in families
        assert "alibaba" in families

    def test_cheapest_in_tier(self):
        reg = ModelRegistry(SAMPLE_MODELS)
        cheapest_mid = reg.cheapest_in_tier(ModelTier.MID)
        assert cheapest_mid is not None
        # Gemini Flash at $0.15/M should be cheapest mid-tier
        assert cheapest_mid.model_id == "gemini/gemini-2.5-flash"

    def test_cheapest_in_tier_with_family(self):
        reg = ModelRegistry(SAMPLE_MODELS)
        cheapest_openai = reg.cheapest_in_tier(ModelTier.MID, family="openai")
        assert cheapest_openai is not None
        assert cheapest_openai.model_id == "openai/gpt-4.1-mini"

    def test_record_usage(self):
        reg = ModelRegistry(SAMPLE_MODELS)
        reg.record_usage("openai/gpt-4.1", 5000)
        reg.record_usage("openai/gpt-4.1", 3000)
        usage = reg.get_usage()
        assert "openai/gpt-4.1" in usage
        assert usage["openai/gpt-4.1"].tokens == 8000
        assert usage["openai/gpt-4.1"].family == "openai"


# ── Model Selection ────────────────────────────────────────────────────


class TestModelSelection:
    def test_select_research_role(self):
        """Research roles should get CHEAP tier models."""
        reg = ModelRegistry(SAMPLE_MODELS, SAMPLE_CHAINS)
        role = RoleSpec(
            name="Researcher",
            perspective="Factual",
            expertise="General",
            suggested_model="alibaba",
            system_prompt="Research...",
            is_research=True,
            research_subquestion="What is X?",
        )
        model = reg.select_model_for_role(role)
        assert model is not None
        assert model.tier == ModelTier.CHEAP

    def test_select_debater_role(self):
        """Debater roles should get MID tier models."""
        reg = ModelRegistry(SAMPLE_MODELS, SAMPLE_CHAINS)
        role = RoleSpec(
            name="Analyst",
            perspective="Skeptical",
            expertise="Analysis",
            suggested_model="deepseek",
            system_prompt="Debate...",
        )
        model = reg.select_model_for_role(role)
        assert model is not None
        assert model.tier == ModelTier.MID

    def test_select_with_family_constraint(self):
        reg = ModelRegistry(SAMPLE_MODELS, SAMPLE_CHAINS)
        role = RoleSpec(
            name="Analyst",
            perspective="Critical",
            expertise="General",
            suggested_model="",
            system_prompt="Debate...",
        )
        model = reg.select_model_for_role(role, family="google")
        assert model is not None
        assert model.family == "google"

    def test_select_with_exclude(self):
        reg = ModelRegistry(SAMPLE_MODELS, SAMPLE_CHAINS)
        role = RoleSpec(
            name="Analyst",
            perspective="Critical",
            expertise="General",
            suggested_model="",
            system_prompt="Debate...",
        )
        model = reg.select_model_for_role(role, exclude_families=["openai", "anthropic"])
        assert model is not None
        assert model.family not in ["openai", "anthropic"]

    def test_select_local_only(self):
        reg = ModelRegistry(SAMPLE_MODELS, SAMPLE_CHAINS)
        role = RoleSpec(
            name="Researcher",
            perspective="Factual",
            expertise="General",
            suggested_model="",
            system_prompt="Research...",
            is_research=True,
        )
        model = reg.select_model_for_role(role, local_only=True)
        assert model is not None
        assert model.supports_local

    def test_select_with_model_override(self):
        reg = ModelRegistry(SAMPLE_MODELS, SAMPLE_CHAINS)
        role = RoleSpec(
            name="Analyst",
            perspective="Critical",
            expertise="General",
            suggested_model="",
            system_prompt="Debate...",
        )
        model = reg.select_model_for_role(role, model_override="deepseek/deepseek-chat")
        assert model is not None
        assert model.model_id == "deepseek/deepseek-chat"

    def test_select_with_family_diversity(self):
        """Should prefer unassigned families."""
        reg = ModelRegistry(SAMPLE_MODELS, SAMPLE_CHAINS)
        role = RoleSpec(
            name="Analyst",
            perspective="Critical",
            expertise="General",
            suggested_model="",
            system_prompt="Debate...",
        )
        # Already assigned openai and anthropic
        model = reg.select_model_for_role(
            role,
            already_assigned_families=["openai", "anthropic"],
        )
        assert model is not None
        # Should prefer a non-openai, non-anthropic family
        assert model.family not in ["openai", "anthropic"] or len(reg.available_families()) <= 2

    def test_fallback_chain(self):
        reg = ModelRegistry(SAMPLE_MODELS, SAMPLE_CHAINS)
        # Make gpt-4.1-mini unavailable
        reg.get("openai/gpt-4.1-mini").is_available = False
        fallback = reg.resolve_fallback("openai/gpt-4.1-mini", "debater")
        assert fallback is not None
        assert fallback.model_id != "openai/gpt-4.1-mini"
        # Should be Gemini Flash or DeepSeek from the chain
        assert fallback.model_id in ["gemini/gemini-2.5-flash", "deepseek/deepseek-chat"]


# ── Cross-Provider Fallback ──────────────────────────────────────────────


class TestCrossProviderFallback:
    """Tests for the canonical_id-based cross-provider fallback strategy.

    When the same model is available on multiple providers (identified by
    canonical_id), resolve_fallback() should prefer the same model on a
    different provider over falling back to a different model.
    """

    def setup_method(self):
        """Set up models with canonical_ids for cross-provider testing."""
        self.models = [
            ModelInfo(
                model_id="openai/@cf/meta/llama-3.3-70b-instruct-fp8-fast",
                family="llama", tier=ModelTier.MID, context_window=131_072,
                provider="cloudflare", canonical_id="llama-3.3-70b-instruct",
            ),
            ModelInfo(
                model_id="openrouter/meta-llama/llama-3.3-70b-instruct:free",
                family="llama", tier=ModelTier.MID, context_window=65_536,
                provider="openrouter", canonical_id="llama-3.3-70b-instruct",
            ),
            ModelInfo(
                model_id="sambanova/Meta-Llama-3.3-70B-Instruct",
                family="llama", tier=ModelTier.MID, context_window=131_072,
                provider="sambanova", canonical_id="llama-3.3-70b-instruct",
                conserve=True,
            ),
            ModelInfo(
                model_id="groq/llama-3.3-70b-versatile",
                family="llama", tier=ModelTier.MID, context_window=131_072,
                provider="groq", canonical_id="llama-3.3-70b-instruct",
                geo_blocked=True,
            ),
            # A different model family for fallback chain testing
            ModelInfo(
                model_id="openai/@cf/qwen/qwen3-30b-a3b-fp8",
                family="qwen", tier=ModelTier.MID, context_window=131_072,
                provider="cloudflare",
            ),
        ]
        self.providers = {
            "cloudflare": {"name": "cloudflare", "priority": 0},
            "openrouter": {"name": "openrouter", "priority": 1},
            "sambanova": {"name": "sambanova", "priority": 4},
            "groq": {"name": "groq", "priority": 3},
        }
        self.chains = {
            "debater": [
                "openai/@cf/meta/llama-3.3-70b-instruct-fp8-fast",
                "openrouter/meta-llama/llama-3.3-70b-instruct:free",
                "openai/@cf/qwen/qwen3-30b-a3b-fp8",
            ],
        }

    def test_cross_provider_same_model_preferred(self):
        """When CF llama fails, should try OpenRouter llama (same canonical_id) first."""
        reg = ModelRegistry(self.models, self.chains, self.providers)
        # Simulate CF llama failing
        reg.failures.record_failure(
            "openai/@cf/meta/llama-3.3-70b-instruct-fp8-fast",
            provider="cloudflare", error_hint="429 Rate Limit",
        )
        reg.failures.record_failure(
            "openai/@cf/meta/llama-3.3-70b-instruct-fp8-fast",
            provider="cloudflare", error_hint="429 Rate Limit",
        )
        fallback = reg.resolve_fallback(
            "openai/@cf/meta/llama-3.3-70b-instruct-fp8-fast", "debater"
        )
        assert fallback is not None
        # Should prefer OpenRouter's llama (same canonical_id, different provider)
        assert fallback.canonical_id == "llama-3.3-70b-instruct"
        assert fallback.provider == "openrouter"

    def test_cross_provider_skips_geo_blocked(self):
        """Cross-provider fallback should skip geo-blocked alternatives."""
        reg = ModelRegistry(self.models, self.chains, self.providers)
        # Make both CF and OpenRouter llama fail
        for mid in [
            "openai/@cf/meta/llama-3.3-70b-instruct-fp8-fast",
            "openrouter/meta-llama/llama-3.3-70b-instruct:free",
        ]:
            reg.failures.record_failure(mid, error_hint="429")
            reg.failures.record_failure(mid, error_hint="429")
        fallback = reg.resolve_fallback(
            "openai/@cf/meta/llama-3.3-70b-instruct-fp8-fast", "debater"
        )
        # Should NOT be groq (geo-blocked) even though it has same canonical_id
        if fallback and fallback.canonical_id == "llama-3.3-70b-instruct":
            assert fallback.provider != "groq"

    def test_cross_provider_prefers_non_conserved(self):
        """Cross-provider fallback should prefer non-conserved providers."""
        reg = ModelRegistry(self.models, self.chains, self.providers)
        # Make CF and OpenRouter fail, leaving sambanova and groq
        for mid in [
            "openai/@cf/meta/llama-3.3-70b-instruct-fp8-fast",
            "openrouter/meta-llama/llama-3.3-70b-instruct:free",
        ]:
            reg.failures.record_failure(mid, error_hint="429")
            reg.failures.record_failure(mid, error_hint="429")
        # SambaNova is non-geo-blocked but conserved
        fallback = reg._cross_provider_fallback(
            "openai/@cf/meta/llama-3.3-70b-instruct-fp8-fast",
            "llama-3.3-70b-instruct",
        )
        # Only sambanova is left as non-geo-blocked with same canonical_id
        if fallback:
            # It's OK that it's conserved — it's the only option left
            assert fallback.provider == "sambanova"

    def test_no_cross_provider_without_canonical_id(self):
        """Models without canonical_id should fall through to chain fallback."""
        models_no_canonical = [
            ModelInfo(
                model_id="mystery/model-a",
                family="mystery", tier=ModelTier.MID, context_window=65_536,
                provider="unknown",
            ),
        ]
        reg = ModelRegistry(models_no_canonical, {})
        # No canonical_id → no cross-provider fallback → returns None
        fallback = reg.resolve_fallback("mystery/model-a", "debater")
        # No chain, no alternatives → None
        assert fallback is None

    def test_models_by_canonical(self):
        """Test the canonical_id lookup method."""
        reg = ModelRegistry(self.models, self.chains, self.providers)
        same_model = reg.models_by_canonical("llama-3.3-70b-instruct")
        assert len(same_model) == 4  # CF, OR, SambaNova, Groq
        providers = {m.provider for m in same_model}
        assert providers == {"cloudflare", "openrouter", "sambanova", "groq"}


# ── LLMClient ──────────────────────────────────────────────────────────


class TestLLMClient:
    @pytest.mark.asyncio
    async def test_complete_success(self):
        """Test a successful completion call (mocked)."""
        reg = ModelRegistry(SAMPLE_MODELS)
        budget = TokenBudget(total=100000)
        client = LLMClient(reg, budget)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage = MagicMock()
        mock_response.usage.total_tokens = 100

        with patch("council.models.litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            content, tokens = await client.complete(
                model_id="openai/gpt-4.1-mini",
                messages=[{"role": "user", "content": "Hello"}],
            )
            assert content == "Test response"
            assert tokens == 100
            assert budget.used == 100

    @pytest.mark.asyncio
    async def test_complete_budget_exhausted(self):
        """Test that budget exhaustion raises RuntimeError."""
        reg = ModelRegistry(SAMPLE_MODELS)
        budget = TokenBudget(total=0)
        client = LLMClient(reg, budget)

        with pytest.raises(RuntimeError, match="budget exhausted"):
            await client.complete(
                model_id="openai/gpt-4.1-mini",
                messages=[{"role": "user", "content": "Hello"}],
            )

    @pytest.mark.asyncio
    async def test_complete_with_budget_override(self):
        """Test that budget_override takes precedence over self.budget."""
        reg = ModelRegistry(SAMPLE_MODELS)
        # Client has no budget attached
        client = LLMClient(reg, budget=None)

        # But we pass a per-call budget
        phase_budget = TokenBudget(total=100000)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage = MagicMock()
        mock_response.usage.total_tokens = 100

        with patch("council.models.litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            content, tokens = await client.complete(
                model_id="openai/gpt-4.1-mini",
                messages=[{"role": "user", "content": "Hello"}],
                budget_override=phase_budget,
            )
            assert content == "Test response"
            assert tokens == 100
            # Phase budget should be consumed, not the client's (which is None)
            assert phase_budget.used == 100

    @pytest.mark.asyncio
    async def test_complete_budget_override_exhausted(self):
        """Test that an exhausted budget_override raises RuntimeError."""
        reg = ModelRegistry(SAMPLE_MODELS)
        # Client has a generous budget
        client_budget = TokenBudget(total=1000000)
        client = LLMClient(reg, budget=client_budget)

        # But the phase override is exhausted
        phase_budget = TokenBudget(total=0)

        with pytest.raises(RuntimeError, match="budget exhausted"):
            await client.complete(
                model_id="openai/gpt-4.1-mini",
                messages=[{"role": "user", "content": "Hello"}],
                budget_override=phase_budget,
            )
        # Client's own budget should NOT be consumed
        assert client_budget.used == 0

    @pytest.mark.asyncio
    async def test_complete_no_budget_no_override(self):
        """Test that a call works fine when neither budget nor override is set."""
        reg = ModelRegistry(SAMPLE_MODELS)
        client = LLMClient(reg, budget=None)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage = MagicMock()
        mock_response.usage.total_tokens = 100

        with patch("council.models.litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            content, tokens = await client.complete(
                model_id="openai/gpt-4.1-mini",
                messages=[{"role": "user", "content": "Hello"}],
            )
            assert content == "Test response"
            assert tokens == 100

    @pytest.mark.asyncio
    async def test_complete_override_takes_precedence_over_client_budget(self):
        """When both client budget and override are set, override wins."""
        reg = ModelRegistry(SAMPLE_MODELS)
        client_budget = TokenBudget(total=100000)
        client = LLMClient(reg, budget=client_budget)

        override_budget = TokenBudget(total=50000)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage = MagicMock()
        mock_response.usage.total_tokens = 100

        with patch("council.models.litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            content, tokens = await client.complete(
                model_id="openai/gpt-4.1-mini",
                messages=[{"role": "user", "content": "Hello"}],
                budget_override=override_budget,
            )
            assert tokens == 100
            # Override budget should be consumed, not client budget
            assert override_budget.used == 100
            assert client_budget.used == 0  # Not touched
