"""
Configuration loading and validation for Deliberative Council.

Uses YAML for human-readable config files and Pydantic for programmatic
validation. Pydantic models serve as living documentation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from council.types import Complexity, ModelTier, ResearchMode


# ── Model Config ───────────────────────────────────────────────────────


class ModelConfig(BaseModel):
    """Configuration for a specific model in the registry."""

    model_id: str = Field(description="LiteLLM model identifier")
    family: str = Field(description="Model family")
    tier: ModelTier = Field(description="Capability tier")
    context_window: int = Field(default=128_000, description="Max context tokens")
    input_cost_per_m: float = Field(default=0.0, description="Cost per M input tokens")
    output_cost_per_m: float = Field(default=0.0, description="Cost per M output tokens")
    supports_local: bool = Field(default=False)
    enabled: bool = Field(default=True)


class FallbackChain(BaseModel):
    """Fallback chain for a role type."""

    role_type: Literal["research", "debater", "synthesizer"]
    chain: list[str] = Field(
        description="Ordered list of model IDs to try"
    )


# ── NLI Config ─────────────────────────────────────────────────────────


class NLIConfig(BaseModel):
    """Configuration for the two-tier NLI system."""

    tier1_model: str = Field(
        default="cross-encoder/nli-deberta-v3-large",
        description="HuggingFace model ID for DeBERTa Tier 1",
    )
    tier1_uncertain_low: float = Field(
        default=0.25,
        description="Below this, Tier 1 is confident about disagreement",
    )
    tier1_uncertain_high: float = Field(
        default=0.65,
        description="Above this, Tier 1 is confident about agreement",
    )
    convergence_threshold: float = Field(
        default=0.75,
        description="NLI agreement threshold for convergence detection",
    )
    convergence_rounds: int = Field(
        default=2,
        description="Number of consecutive rounds above threshold for convergence",
    )
    position_stability_threshold: float = Field(
        default=0.80,
        description="Position stability threshold for novelty injection (slightly higher than convergence)",
    )
    position_stability_rounds: int = Field(
        default=2,
        description="Number of consecutive rounds above threshold for novelty injection",
    )
    tier2_model: str = Field(
        default="openai/gpt-4.1-mini",
        description="LLM model for Tier 2 agreement analysis",
    )
    graceful_degradation: bool = Field(
        default=True,
        description="If DeBERTa fails to load, fall back to cheap LLM for Tier 1",
    )


# ── Budget Config ──────────────────────────────────────────────────────


class BudgetConfig(BaseModel):
    """Token budget configuration."""

    default_budget: int = Field(
        default=500_000,
        description="Default token budget for Research+Debate+Synthesis",
    )
    max_per_agent: int = Field(
        default=100_000,
        description="Maximum tokens per agent call",
    )
    research_share: float = Field(
        default=0.3,
        description="Fraction of budget allocated to research",
    )
    debate_share: float = Field(
        default=0.5,
        description="Fraction of budget allocated to debate",
    )
    synthesis_share: float = Field(
        default=0.2,
        description="Fraction of budget allocated to synthesis",
    )

    @field_validator("research_share", "debate_share", "synthesis_share")
    @classmethod
    def shares_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Budget shares must be positive")
        return v


# ── Debate Config ──────────────────────────────────────────────────────


class DebateConfig(BaseModel):
    """Debate phase configuration."""

    default_rounds: dict[str, int] = Field(
        default={
            "trivial": 0,
            "moderate": 1,
            "complex": 2,
            "deep": 3,
        },
        description="Default rounds per complexity level",
    )
    max_concurrent_agents: int = Field(
        default=3,
        description="Maximum concurrent API calls",
    )
    graph_strategy: Literal["full"] = Field(
        default="full",
        description="Communication graph strategy (sparse is extension)",
    )
    debate_strategy: Literal["none"] = Field(
        default="none",
        description="Devil's advocate strategy (rotate/weakest/random are extensions)",
    )
    context_strategy: Literal["full"] = Field(
        default="full",
        description="Context window strategy (progressive is extension)",
    )


# ── Research Config ────────────────────────────────────────────────────


class ResearchConfig(BaseModel):
    """Research phase configuration."""

    mode: ResearchMode = Field(
        default=ResearchMode.STRICT,
        description="strict = only sourced findings; augmented = all tags allowed",
    )
    max_concurrent_agents: int = Field(
        default=3,
        description="Maximum concurrent research agents",
    )
    jina_search_url: str = Field(
        default="https://s.jina.ai",
        description="Jina.ai search endpoint",
    )
    jina_extract_url: str = Field(
        default="https://r.jina.ai",
        description="Jina.ai content extraction endpoint",
    )
    max_search_results: int = Field(
        default=5,
        description="Maximum search results per query",
    )


# ── Top-Level Config ───────────────────────────────────────────────────


class CouncilConfig(BaseModel):
    """Top-level configuration for a Deliberative Council run."""

    # Model registry
    models: list[ModelConfig] = Field(
        default_factory=list,
        description="Available models for the registry",
    )
    fallback_chains: list[FallbackChain] = Field(
        default_factory=list,
        description="Fallback chains per role type",
    )

    # Subsystem configs
    nli: NLIConfig = Field(default_factory=NLIConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    debate: DebateConfig = Field(default_factory=DebateConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)

    # Overrides
    complexity_override: Complexity | None = Field(
        default=None,
        description="Override Scout's complexity classification",
    )
    budget_override: int | None = Field(
        default=None,
        description="Override the token budget",
    )
    format: Literal["markdown", "pdf", "docx", "json", "text"] = Field(
        default="markdown",
        description="Output format",
    )
    output_path: str | None = Field(
        default=None,
        description="Output file path (default: current directory)",
    )

    # Model selection constraints
    family: str | None = Field(
        default=None,
        description="Restrict to a specific model family",
    )
    exclude_families: list[str] = Field(
        default_factory=list,
        description="Families to exclude",
    )
    local_only: bool = Field(
        default=False,
        description="Only use Ollama/local models",
    )
    api_only: bool = Field(
        default=False,
        description="Only use cloud API models",
    )
    model_overrides: dict[str, str] = Field(
        default_factory=dict,
        description="Override model for specific roles, e.g. {'debater_0': 'deepseek/deepseek-v3'}",
    )

    # Runtime
    verbose: bool = Field(default=False)
    dry_run: bool = Field(default=False)
    resume: bool = Field(default=False)
    config_path: str | None = Field(default=None)


# ── Loading ────────────────────────────────────────────────────────────


# Default model configuration
DEFAULT_MODELS: list[ModelConfig] = [
    # OpenAI
    ModelConfig(model_id="openai/gpt-4.1", family="openai", tier=ModelTier.PREMIUM, context_window=128_000, input_cost_per_m=2.00, output_cost_per_m=8.00),
    ModelConfig(model_id="openai/gpt-4.1-mini", family="openai", tier=ModelTier.MID, context_window=128_000, input_cost_per_m=0.40, output_cost_per_m=1.60),
    # Anthropic
    ModelConfig(model_id="anthropic/claude-sonnet-4-20250514", family="anthropic", tier=ModelTier.PREMIUM, context_window=200_000, input_cost_per_m=3.00, output_cost_per_m=15.00),
    ModelConfig(model_id="anthropic/claude-haiku-3-5-20241022", family="anthropic", tier=ModelTier.CHEAP, context_window=200_000, input_cost_per_m=0.80, output_cost_per_m=4.00),
    # Google
    ModelConfig(model_id="gemini/gemini-2.5-pro", family="google", tier=ModelTier.PREMIUM, context_window=1_000_000, input_cost_per_m=1.25, output_cost_per_m=10.00),
    ModelConfig(model_id="gemini/gemini-2.5-flash", family="google", tier=ModelTier.MID, context_window=1_000_000, input_cost_per_m=0.15, output_cost_per_m=0.60),
    # DeepSeek
    ModelConfig(model_id="deepseek/deepseek-chat", family="deepseek", tier=ModelTier.MID, context_window=128_000, input_cost_per_m=0.27, output_cost_per_m=1.10, supports_local=True),
    ModelConfig(model_id="deepseek/deepseek-reasoner", family="deepseek", tier=ModelTier.PREMIUM, context_window=128_000, input_cost_per_m=0.55, output_cost_per_m=2.19, supports_local=True),
    # Alibaba
    ModelConfig(model_id="dashscope/qwen3-235b", family="alibaba", tier=ModelTier.PREMIUM, context_window=128_000, input_cost_per_m=0.40, output_cost_per_m=1.20),
    ModelConfig(model_id="dashscope/qwen3-32b", family="alibaba", tier=ModelTier.MID, context_window=128_000),
    ModelConfig(model_id="ollama_chat/qwen3:8b", family="alibaba", tier=ModelTier.CHEAP, context_window=128_000, supports_local=True),
    ModelConfig(model_id="dashscope/qwq-32b", family="alibaba", tier=ModelTier.MID, context_window=128_000),
    # Meta
    ModelConfig(model_id="meta_llama/llama-4-maverick", family="meta", tier=ModelTier.MID, context_window=1_000_000, input_cost_per_m=0.20, output_cost_per_m=0.80),
    ModelConfig(model_id="ollama_chat/llama4-scout", family="meta", tier=ModelTier.CHEAP, context_window=1_000_000, supports_local=True),
    # Mistral
    ModelConfig(model_id="mistral/mistral-large", family="mistral", tier=ModelTier.PREMIUM, context_window=128_000, input_cost_per_m=2.00, output_cost_per_m=6.00),
    ModelConfig(model_id="mistral/mistral-small", family="mistral", tier=ModelTier.CHEAP, context_window=128_000, input_cost_per_m=0.20, output_cost_per_m=0.60, supports_local=True),
    # Microsoft
    ModelConfig(model_id="ollama_chat/phi4-mini", family="microsoft", tier=ModelTier.CHEAP, context_window=128_000, input_cost_per_m=0.10, output_cost_per_m=0.40, supports_local=True),
    # Moonshot
    ModelConfig(model_id="moonshot/kimi-k2", family="moonshot", tier=ModelTier.PREMIUM, context_window=128_000, input_cost_per_m=1.50, output_cost_per_m=6.00),
]

DEFAULT_FALLBACK_CHAINS: list[FallbackChain] = [
    FallbackChain(role_type="research", chain=["ollama_chat/qwen3:8b", "anthropic/claude-haiku-3-5-20241022", "ollama_chat/llama4-scout"]),
    FallbackChain(role_type="debater", chain=["openai/gpt-4.1-mini", "gemini/gemini-2.5-flash", "deepseek/deepseek-chat"]),
    FallbackChain(role_type="synthesizer", chain=["anthropic/claude-sonnet-4-20250514", "openai/gpt-4.1", "gemini/gemini-2.5-pro"]),
]


def load_config(config_path: str | Path | None = None, **overrides) -> CouncilConfig:
    """Load configuration from a YAML file with optional overrides.

    Priority (highest to lowest):
    1. Explicit keyword arguments (from CLI flags)
    2. YAML file values
    3. Built-in defaults
    """
    config_data: dict = {}

    if config_path is not None:
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                config_data = yaml.safe_load(f) or {}
        else:
            raise FileNotFoundError(f"Config file not found: {path}")

    # Merge overrides on top of file values
    for key, value in overrides.items():
        if value is not None:
            config_data[key] = value

    # Ensure default models if none specified
    if "models" not in config_data:
        config_data["models"] = [m.model_dump() for m in DEFAULT_MODELS]
    if "fallback_chains" not in config_data:
        config_data["fallback_chains"] = [f.model_dump() for f in DEFAULT_FALLBACK_CHAINS]

    return CouncilConfig(**config_data)


def load_default_config() -> CouncilConfig:
    """Load configuration with all built-in defaults."""
    return CouncilConfig(
        models=DEFAULT_MODELS,
        fallback_chains=DEFAULT_FALLBACK_CHAINS,
    )


# ── Default YAML ───────────────────────────────────────────────────────

DEFAULT_YAML = """\
# Deliberative Council Default Configuration
# Override any setting via CLI flags or a custom YAML file.

# NLI agreement tracking
nli:
  tier1_model: "cross-encoder/nli-deberta-v3-large"
  tier1_uncertain_low: 0.25
  tier1_uncertain_high: 0.65
  convergence_threshold: 0.75
  convergence_rounds: 2
  position_stability_threshold: 0.80
  position_stability_rounds: 2
  tier2_model: "openai/gpt-4.1-mini"
  graceful_degradation: true

# Token budgets
budget:
  default_budget: 500000
  max_per_agent: 100000
  research_share: 0.3
  debate_share: 0.5
  synthesis_share: 0.2

# Debate settings
debate:
  default_rounds:
    trivial: 0
    moderate: 1
    complex: 2
    deep: 3
  max_concurrent_agents: 3
  graph_strategy: "full"
  debate_strategy: "none"
  context_strategy: "full"

# Research settings
research:
  mode: "strict"
  max_concurrent_agents: 3
  jina_search_url: "https://s.jina.ai"
  jina_extract_url: "https://r.jina.ai"
  max_search_results: 5
"""
