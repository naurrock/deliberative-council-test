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

from council.types import Complexity, ModelTier, ProviderInfo, ResearchMode


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

    # Provider routing
    provider: str = Field(default="", description="Provider name")
    api_base: str = Field(default="", description="Custom API base (may contain ${ENV_VAR})")
    api_key: str = Field(default="", description="API key (may contain ${ENV_VAR})")
    conserve: bool = Field(default=False, description="Non-regenerating credits — use sparingly")
    geo_blocked: bool = Field(default=False, description="Provider is geo-blocked from this server")
    canonical_id: str = Field(default="", description="Canonical model ID for cross-provider dedup")
    rpm: int = Field(default=20, description="Requests per minute limit")
    daily_quota: int = Field(default=0, description="Daily request quota (0=unlimited)")
    regenerates: bool = Field(default=True, description="Whether daily quota regenerates")


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
        default="openrouter/meta-llama/llama-3.3-70b-instruct:free",
        description="LLM model for Tier 2 agreement analysis (must be in providers.yaml)",
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
    providers: dict[str, ProviderInfo] = Field(
        default_factory=dict,
        description="Provider metadata (from providers: section). Key = provider name.",
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


# Default model configuration — used only when no providers.yaml is found.
# These are FREE-TIER models that require no payment. When providers.yaml
# is present (the normal case), these defaults are ignored entirely.
DEFAULT_MODELS: list[ModelConfig] = [
    # Cloudflare Workers AI — Priority 0, daily regenerating
    ModelConfig(model_id="openai/@cf/meta/llama-3.3-70b-instruct-fp8-fast", family="llama", tier=ModelTier.MID, context_window=131_072, provider="cloudflare"),
    ModelConfig(model_id="openai/@cf/qwen/qwen3-30b-a3b-fp8", family="qwen", tier=ModelTier.MID, context_window=131_072, provider="cloudflare"),
    ModelConfig(model_id="openai/@cf/meta/llama-3.2-3b-instruct", family="llama", tier=ModelTier.CHEAP, context_window=131_072, provider="cloudflare"),
    # OpenRouter — Priority 1, 50 RPD free tier
    ModelConfig(model_id="openrouter/meta-llama/llama-3.3-70b-instruct:free", family="llama", tier=ModelTier.MID, context_window=65_536, provider="openrouter"),
    ModelConfig(model_id="openrouter/qwen/qwen3-coder:free", family="qwen", tier=ModelTier.MID, context_window=262_000, provider="openrouter"),
    ModelConfig(model_id="openrouter/meta-llama/llama-3.2-3b-instruct:free", family="llama", tier=ModelTier.CHEAP, context_window=131_072, provider="openrouter"),
]

DEFAULT_FALLBACK_CHAINS: list[FallbackChain] = [
    FallbackChain(role_type="research", chain=[
        "openai/@cf/meta/llama-3.2-3b-instruct",
        "openrouter/meta-llama/llama-3.2-3b-instruct:free",
    ]),
    FallbackChain(role_type="debater", chain=[
        "openai/@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        "openai/@cf/qwen/qwen3-30b-a3b-fp8",
        "openrouter/meta-llama/llama-3.3-70b-instruct:free",
    ]),
    FallbackChain(role_type="synthesizer", chain=[
        "openai/@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        "openrouter/meta-llama/llama-3.3-70b-instruct:free",
    ]),
]


def _expand_env_vars(value: str) -> str:
    """Expand ${VAR_NAME} references in a string with environment variables."""
    import re
    def replacer(match):
        var_name = match.group(1)
        return os.environ.get(var_name, match.group(0))
    return re.sub(r'\$\{(\w+)\}', replacer, value)


def _auto_find_config() -> Path | None:
    """Find the providers.yaml config file automatically.

    Search order:
    1. ./config/providers.yaml (relative to CWD)
    2. ~/.config/deliberative-council/providers.yaml
    """
    candidates = [
        Path.cwd() / "config" / "providers.yaml",
        Path.home() / ".config" / "deliberative-council" / "providers.yaml",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_config(config_path: str | Path | None = None, **overrides) -> CouncilConfig:
    """Load configuration from a YAML file with optional overrides.

    Priority (highest to lowest):
    1. Explicit keyword arguments (from CLI flags)
    2. YAML file values
    3. Auto-discovered config/providers.yaml
    4. Built-in defaults
    """
    config_data: dict = {}

    if config_path is not None:
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                config_data = yaml.safe_load(f) or {}
        else:
            raise FileNotFoundError(f"Config file not found: {path}")
    else:
        # Auto-discover providers.yaml
        auto_path = _auto_find_config()
        if auto_path is not None:
            with open(auto_path) as f:
                config_data = yaml.safe_load(f) or {}

    # Expand env vars in model api_base and api_key fields
    if "models" in config_data:
        for m in config_data["models"]:
            if isinstance(m, dict):
                for key in ("api_base", "api_key"):
                    if key in m and isinstance(m[key], str):
                        m[key] = _expand_env_vars(m[key])

    # Parse the providers: section into ProviderInfo objects
    if "providers" in config_data:
        raw_providers = config_data.pop("providers")
        if isinstance(raw_providers, dict):
            parsed_providers = {}
            for name, info in raw_providers.items():
                if isinstance(info, dict):
                    info["name"] = name
                    # Expand env vars in api_base_template
                    if "api_base_template" in info and isinstance(info["api_base_template"], str):
                        info["api_base_template"] = _expand_env_vars(info["api_base_template"])
                    # Map provider-level geo_blocked to all its models
                    # (models already read this from their own geo_blocked field,
                    # but we also store it at the provider level for awareness)
                    parsed_providers[name] = ProviderInfo(**info)
            config_data["providers"] = parsed_providers

    # Normalize fallback_chains: providers.yaml uses dict format
    # {role_type: [model_ids]}, but CouncilConfig expects list of
    # FallbackChain(role_type=..., chain=[...]).
    if "fallback_chains" in config_data:
        fc = config_data["fallback_chains"]
        if isinstance(fc, dict):
            config_data["fallback_chains"] = [
                {"role_type": role_type, "chain": chain}
                for role_type, chain in fc.items()
            ]

    # Merge overrides on top of file values
    for key, value in overrides.items():
        if value is not None:
            config_data[key] = value

    # Ensure default models if none specified
    if "models" not in config_data:
        config_data["models"] = [m.model_dump() for m in DEFAULT_MODELS]
    if "fallback_chains" not in config_data:
        config_data["fallback_chains"] = [f.model_dump() for f in DEFAULT_FALLBACK_CHAINS]

    # Strip non-ModelConfig fields from model dicts (providers.yaml includes
    # extra metadata like 'provider' at the top level that isn't a ModelConfig field)
    if "models" in config_data:
        valid_fields = set(ModelConfig.model_fields.keys())
        for m in config_data["models"]:
            if isinstance(m, dict):
                # Remove keys that aren't valid ModelConfig fields
                extra_keys = [k for k in m if k not in valid_fields]
                for k in extra_keys:
                    del m[k]

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
  tier2_model: "openrouter/meta-llama/llama-3.3-70b-instruct:free"
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
