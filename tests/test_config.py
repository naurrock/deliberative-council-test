"""Tests for council.config — configuration loading and validation."""

import tempfile
from pathlib import Path

import pytest
import yaml

from council.config import (
    BudgetConfig,
    CouncilConfig,
    DebateConfig,
    FallbackChain,
    ModelConfig,
    NLIConfig,
    ResearchConfig,
    load_config,
    load_default_config,
    DEFAULT_MODELS,
    DEFAULT_FALLBACK_CHAINS,
)
from council.types import Complexity, ModelTier, ProviderInfo, ResearchMode


# ── ModelConfig ────────────────────────────────────────────────────────


class TestModelConfig:
    def test_basic_model(self):
        mc = ModelConfig(
            model_id="openai/gpt-4.1",
            family="openai",
            tier=ModelTier.PREMIUM,
        )
        assert mc.family == "openai"
        assert mc.tier == ModelTier.PREMIUM
        assert mc.enabled is True
        assert mc.context_window == 128_000

    def test_local_model(self):
        mc = ModelConfig(
            model_id="ollama_chat/qwen3:8b",
            family="alibaba",
            tier=ModelTier.CHEAP,
            supports_local=True,
        )
        assert mc.supports_local is True
        assert mc.input_cost_per_m == 0.0


# ── NLIConfig ──────────────────────────────────────────────────────────


class TestNLIConfig:
    def test_defaults(self):
        nli = NLIConfig()
        assert nli.convergence_threshold == 0.75
        assert nli.position_stability_threshold == 0.80
        assert nli.position_stability_threshold > nli.convergence_threshold
        assert nli.convergence_rounds == 2
        assert nli.position_stability_rounds == 2

    def test_custom_thresholds(self):
        nli = NLIConfig(
            convergence_threshold=0.70,
            position_stability_threshold=0.85,
        )
        assert nli.convergence_threshold == 0.70
        assert nli.position_stability_threshold == 0.85


# ── BudgetConfig ───────────────────────────────────────────────────────


class TestBudgetConfig:
    def test_defaults(self):
        b = BudgetConfig()
        assert b.default_budget == 500_000
        total_share = b.research_share + b.debate_share + b.synthesis_share
        assert abs(total_share - 1.0) < 0.01  # Should roughly sum to 1.0

    def test_invalid_share(self):
        with pytest.raises(Exception):
            BudgetConfig(research_share=-0.1)


# ── DebateConfig ───────────────────────────────────────────────────────


class TestDebateConfig:
    def test_defaults(self):
        d = DebateConfig()
        assert d.default_rounds["trivial"] == 0
        assert d.default_rounds["deep"] == 3
        assert d.graph_strategy == "full"
        assert d.debate_strategy == "none"
        assert d.context_strategy == "full"


# ── ResearchConfig ─────────────────────────────────────────────────────


class TestResearchConfig:
    def test_defaults(self):
        r = ResearchConfig()
        assert r.mode == ResearchMode.STRICT
        assert r.max_concurrent_agents == 3
        assert "jina.ai" in r.jina_search_url


# ── CouncilConfig ──────────────────────────────────────────────────────


class TestCouncilConfig:
    def test_default_config(self):
        config = load_default_config()
        assert len(config.models) > 0
        assert len(config.fallback_chains) > 0
        assert config.nli.convergence_threshold == 0.75
        assert config.format == "markdown"

    def test_complexity_override(self):
        config = load_default_config()
        config.complexity_override = Complexity.DEEP
        assert config.complexity_override == Complexity.DEEP

    def test_family_constraint(self):
        config = load_default_config()
        config.family = "openai"
        assert config.family == "openai"

    def test_exclude_families(self):
        config = load_default_config()
        config.exclude_families = ["moonshot"]
        assert "moonshot" in config.exclude_families

    def test_model_overrides(self):
        config = load_default_config()
        config.model_overrides = {"debater_0": "deepseek/deepseek-chat"}
        assert config.model_overrides["debater_0"] == "deepseek/deepseek-chat"


# ── Load Config from YAML ─────────────────────────────────────────────


class TestLoadConfig:
    def test_load_from_yaml(self):
        yaml_content = {
            "nli": {
                "convergence_threshold": 0.80,
                "position_stability_threshold": 0.85,
            },
            "budget": {
                "default_budget": 1000000,
            },
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(yaml_content, f)
            f.flush()
            config = load_config(f.name)

        assert config.nli.convergence_threshold == 0.80
        assert config.nli.position_stability_threshold == 0.85
        assert config.budget.default_budget == 1000000

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")

    def test_load_with_overrides(self):
        config = load_config(
            complexity_override=Complexity.COMPLEX,
            verbose=True,
        )
        assert config.complexity_override == Complexity.COMPLEX
        assert config.verbose is True

    def test_default_models_in_config(self):
        config = load_default_config()
        model_ids = [m.model_id for m in config.models]
        # Check key free-tier models exist (Cloudflare + OpenRouter)
        assert any("cloudflare" in m or "@cf/" in m for m in model_ids)
        assert any("openrouter" in m for m in model_ids)
        assert any("llama" in m for m in model_ids)

    def test_default_fallback_chains(self):
        config = load_default_config()
        chains = {fc.role_type: fc.chain for fc in config.fallback_chains}
        assert "research" in chains
        assert "debater" in chains
        assert "synthesizer" in chains
        assert len(chains["research"]) > 0


# ── Default model list completeness ────────────────────────────────────


class TestDefaultModels:
    def test_covers_multiple_families(self):
        families = {m.family for m in DEFAULT_MODELS}
        # Default models cover at least llama and qwen families
        assert "llama" in families
        assert "qwen" in families

    def test_covers_all_tiers(self):
        tiers = {m.tier for m in DEFAULT_MODELS}
        assert ModelTier.MID in tiers
        assert ModelTier.CHEAP in tiers

    def test_all_free_tier(self):
        """All default models should be free-tier (zero cost)."""
        for m in DEFAULT_MODELS:
            assert m.input_cost_per_m == 0.0, f"{m.model_id} has non-zero cost"
            assert m.output_cost_per_m == 0.0, f"{m.model_id} has non-zero cost"

    def test_includes_cloudflare_models(self):
        cf_models = [m for m in DEFAULT_MODELS if m.provider == "cloudflare"]
        assert len(cf_models) > 0, "Should have Cloudflare models in defaults"

    def test_includes_openrouter_models(self):
        or_models = [m for m in DEFAULT_MODELS if m.provider == "openrouter"]
        assert len(or_models) > 0, "Should have OpenRouter models in defaults"


# ── ProviderInfo Parsing ────────────────────────────────────────────────


class TestProviderInfo:
    """Tests for the providers: section parsing from providers.yaml."""

    def test_provider_info_creation(self):
        pi = ProviderInfo(
            name="cloudflare",
            priority=0,
            env_key="CLOUDFLARE_API_KEY",
            regenerates=True,
            daily_quota=10000,
            rpm=50,
            notes="55 LLM models",
        )
        assert pi.name == "cloudflare"
        assert pi.priority == 0
        assert pi.regenerates is True
        assert pi.daily_quota == 10000

    def test_provider_info_geo_blocked(self):
        pi = ProviderInfo(
            name="groq",
            priority=3,
            env_key="GROQ_API_KEY",
            regenerates=True,
            geo_blocked=True,
        )
        assert pi.geo_blocked is True

    def test_provider_info_conserve(self):
        pi = ProviderInfo(
            name="sambanova",
            priority=4,
            env_key="SAMBANOVA_API_KEY",
            regenerates=False,
            conserve=True,
        )
        assert pi.conserve is True
        assert pi.regenerates is False

    def test_providers_parsed_from_yaml(self):
        """Test that the providers: section of providers.yaml is parsed."""
        yaml_content = {
            "providers": {
                "cloudflare": {
                    "priority": 0,
                    "env_key": "CLOUDFLARE_API_KEY",
                    "regenerates": True,
                    "daily_quota": 10000,
                    "rpm": 50,
                },
                "openrouter": {
                    "priority": 1,
                    "env_key": "OPENROUTER_API_KEY",
                    "regenerates": True,
                    "daily_quota": 50,
                },
            },
            "models": [
                {
                    "model_id": "test/model",
                    "family": "test",
                    "tier": "cheap",
                },
            ],
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(yaml_content, f)
            f.flush()
            config = load_config(f.name)

        assert "cloudflare" in config.providers
        assert "openrouter" in config.providers
        assert config.providers["cloudflare"].priority == 0
        assert config.providers["openrouter"].priority == 1
        assert config.providers["cloudflare"].daily_quota == 10000
        assert config.providers["openrouter"].regenerates is True
