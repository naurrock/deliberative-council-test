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
from council.types import Complexity, ModelTier, ResearchMode


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
        # Check some key models exist
        assert any("gpt-4.1" in mid for mid in model_ids)
        assert any("qwen3" in mid for mid in model_ids)
        assert any("deepseek" in mid for mid in model_ids)

    def test_default_fallback_chains(self):
        config = load_default_config()
        chains = {fc.role_type: fc.chain for fc in config.fallback_chains}
        assert "research" in chains
        assert "debater" in chains
        assert "synthesizer" in chains
        assert len(chains["research"]) > 0


# ── Default model list completeness ────────────────────────────────────


class TestDefaultModels:
    def test_covers_all_families(self):
        families = {m.family for m in DEFAULT_MODELS}
        expected = {"openai", "anthropic", "google", "deepseek", "alibaba", "meta", "mistral", "microsoft", "moonshot"}
        assert expected.issubset(families), f"Missing families: {expected - families}"

    def test_covers_all_tiers(self):
        tiers = {m.tier for m in DEFAULT_MODELS}
        assert ModelTier.PREMIUM in tiers
        assert ModelTier.MID in tiers
        assert ModelTier.CHEAP in tiers

    def test_has_local_models(self):
        local_models = [m for m in DEFAULT_MODELS if m.supports_local]
        assert len(local_models) > 0, "Should have at least some local models"
