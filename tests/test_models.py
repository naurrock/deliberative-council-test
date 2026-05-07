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


# ── LLMClient ──────────────────────────────────────────────────────────


class TestLLLClient:
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
