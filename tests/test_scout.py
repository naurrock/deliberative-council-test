"""Comprehensive tests for council.scout — Scout phase."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from council.config import CouncilConfig, load_default_config
from council.models import LLMClient, ModelRegistry, TokenBudget
from council.scout import (
    SCOUT_SYSTEM_PROMPT,
    VERIFIER_SYSTEM_PROMPT,
    _add_default_roles,
    _parse_mission_brief,
    _select_scout_model,
    _select_verifier_model,
    run_scout,
)
from council.types import (
    Complexity,
    MissionBrief,
    ModelInfo,
    ModelTier,
    RoleSpec,
)


# ── Test Fixtures ──────────────────────────────────────────────────────


def make_registry() -> ModelRegistry:
    """Create a test model registry."""
    models = [
        ModelInfo(model_id="ollama_chat/qwen3:8b", family="alibaba", tier=ModelTier.CHEAP, context_window=128_000, supports_local=True),
        ModelInfo(model_id="ollama_chat/phi4-mini", family="microsoft", tier=ModelTier.CHEAP, context_window=128_000, supports_local=True),
        ModelInfo(model_id="openai/gpt-4.1-mini", family="openai", tier=ModelTier.MID, context_window=128_000, input_cost_per_m=0.40, output_cost_per_m=1.60),
        ModelInfo(model_id="gemini/gemini-2.5-flash", family="google", tier=ModelTier.MID, context_window=1_000_000, input_cost_per_m=0.15, output_cost_per_m=0.60),
        ModelInfo(model_id="openai/gpt-4.1", family="openai", tier=ModelTier.PREMIUM, context_window=128_000, input_cost_per_m=2.00, output_cost_per_m=8.00),
        ModelInfo(model_id="anthropic/claude-sonnet-4", family="anthropic", tier=ModelTier.PREMIUM, context_window=200_000, input_cost_per_m=3.00, output_cost_per_m=15.00),
    ]
    return ModelRegistry(models)


# ── _parse_mission_brief Tests ─────────────────────────────────────────


class TestParseMissionBrief:
    def test_valid_json_brief(self):
        """Test parsing a well-formed Mission Brief JSON."""
        brief_json = json.dumps({
            "question": "Is AI dangerous?",
            "complexity": "complex",
            "domain_tags": ["technology", "ethics"],
            "is_likely_solvable": False,
            "why_might_be_hard": "Subjective question with no objective answer",
            "suggested_roles": [
                {
                    "name": "AI Safety Expert",
                    "perspective": "Cautious about AI risks",
                    "expertise": "AI safety",
                    "suggested_model": "anthropic",
                    "system_prompt": "You are an AI safety expert...",
                    "is_research": False,
                    "research_subquestion": None,
                },
                {
                    "name": "Tech Optimist",
                    "perspective": "AI will benefit humanity",
                    "expertise": "Technology development",
                    "suggested_model": "openai",
                    "system_prompt": "You are a tech optimist...",
                    "is_research": False,
                    "research_subquestion": None,
                },
            ],
            "research_needed": True,
            "research_subquestions": ["Historical AI incidents"],
            "debate_rounds": 2,
            "token_budget": 250000,
            "human_checkpoints": [],
            "reasoning": "Complex question with ethical dimensions",
            "verification_notes": "Good classification",
        })

        brief = _parse_mission_brief(brief_json, "Is AI dangerous?")
        assert brief.question == "Is AI dangerous?"
        assert brief.complexity == Complexity.COMPLEX
        assert "technology" in brief.domain_tags
        assert len(brief.suggested_roles) == 2
        assert brief.research_needed is True
        assert brief.debate_rounds == 2
        assert brief.token_budget == 250000

    def test_fallback_on_invalid_json(self):
        """Test graceful fallback when JSON is invalid."""
        brief = _parse_mission_brief("Not JSON at all", "What is 2+2?")
        assert brief.question == "What is 2+2?"
        assert brief.complexity == Complexity.MODERATE  # Safe default
        assert len(brief.suggested_roles) >= 2  # Default roles added
        assert "Failed to parse" in brief.scout_reasoning

    def test_json_with_markdown_fences(self):
        """Test parsing JSON wrapped in markdown code fences."""
        inner = json.dumps({
            "question": "Test",
            "complexity": "trivial",
            "is_likely_solvable": True,
            "why_might_be_hard": "",
            "suggested_roles": [],
            "research_needed": False,
            "debate_rounds": 0,
            "token_budget": 10000,
            "reasoning": "Simple",
        })
        wrapped = f"```json\n{inner}\n```"
        brief = _parse_mission_brief(wrapped, "Test")
        assert brief.complexity == Complexity.TRIVIAL

    def test_unknown_complexity_defaults_to_moderate(self):
        """Unknown complexity strings should default to MODERATE."""
        brief_json = json.dumps({
            "complexity": "super_hard",
            "is_likely_solvable": True,
            "why_might_be_hard": "",
            "suggested_roles": [],
            "research_needed": False,
            "debate_rounds": 1,
            "token_budget": 50000,
            "reasoning": "Test",
        })
        brief = _parse_mission_brief(brief_json, "Test")
        assert brief.complexity == Complexity.MODERATE

    def test_missing_fields_use_defaults(self):
        """Missing JSON fields should use safe defaults."""
        brief_json = json.dumps({
            "complexity": "moderate",
            "is_likely_solvable": True,
            "why_might_be_hard": "",
            "suggested_roles": [],
            "research_needed": False,
            "debate_rounds": 1,
            "token_budget": 100000,
            "reasoning": "Test",
        })
        brief = _parse_mission_brief(brief_json, "Original question")
        assert brief.domain_tags == []
        assert brief.research_subquestions == []
        assert brief.human_checkpoints == []
        assert brief.verification_notes == ""

    def test_non_trivial_gets_default_roles(self):
        """Non-trivial questions with fewer than 2 roles get default roles."""
        brief_json = json.dumps({
            "complexity": "moderate",
            "is_likely_solvable": True,
            "why_might_be_hard": "",
            "suggested_roles": [
                {
                    "name": "Only Role",
                    "perspective": "Single view",
                    "expertise": "General",
                    "suggested_model": "openai",
                    "system_prompt": "Think carefully",
                    "is_research": False,
                }
            ],
            "research_needed": False,
            "debate_rounds": 1,
            "token_budget": 100000,
            "reasoning": "Test",
        })
        brief = _parse_mission_brief(brief_json, "Test question")
        assert len(brief.suggested_roles) >= 2  # Default roles should be added

    def test_trivial_keeps_zero_roles(self):
        """Trivial questions don't need default debate roles."""
        brief_json = json.dumps({
            "complexity": "trivial",
            "is_likely_solvable": True,
            "why_might_be_hard": "",
            "suggested_roles": [],
            "research_needed": False,
            "debate_rounds": 0,
            "token_budget": 10000,
            "reasoning": "Simple",
        })
        brief = _parse_mission_brief(brief_json, "What is 2+2?")
        assert len(brief.suggested_roles) == 0  # No default roles for trivial


# ── _add_default_roles Tests ────────────────────────────────────────────


class TestAddDefaultRoles:
    def test_empty_roles_moderate(self):
        """Moderate questions should get at least 2 roles."""
        roles = _add_default_roles([], Complexity.MODERATE)
        assert len(roles) >= 2
        # Should have diverse model families
        families = {r.suggested_model for r in roles}
        assert len(families) >= 2  # At least 2 different families

    def test_empty_roles_complex(self):
        """Complex questions should get at least 3 roles."""
        roles = _add_default_roles([], Complexity.COMPLEX)
        assert len(roles) >= 3

    def test_empty_roles_deep(self):
        """Deep questions should get at least 3 roles."""
        roles = _add_default_roles([], Complexity.DEEP)
        assert len(roles) >= 3

    def test_no_duplicate_names(self):
        """Default roles should not duplicate existing role names."""
        existing = [
            RoleSpec(
                name="Analytical Expert",
                perspective="Already exists",
                expertise="General",
                suggested_model="openai",
                system_prompt="Test",
            )
        ]
        roles = _add_default_roles(existing, Complexity.MODERATE)
        names = [r.name for r in roles]
        assert names.count("Analytical Expert") == 1  # No duplicates

    def test_preserves_existing_roles(self):
        """Existing roles should be preserved."""
        existing = [
            RoleSpec(
                name="Custom Role",
                perspective="Unique",
                expertise="Special",
                suggested_model="google",
                system_prompt="Custom prompt",
            )
        ]
        roles = _add_default_roles(existing, Complexity.MODERATE)
        custom = [r for r in roles if r.name == "Custom Role"]
        assert len(custom) == 1


# ── Model Selection Tests ──────────────────────────────────────────────


class TestModelSelection:
    def test_scout_model_cheapest(self):
        """Scout model selection should prefer cheap tier."""
        reg = make_registry()
        config = CouncilConfig()
        model = _select_scout_model(reg, config)
        # Should be one of the cheap models
        m = reg.get(model)
        assert m is not None
        assert m.tier == ModelTier.CHEAP

    def test_scout_model_local_only(self):
        """With local_only, scout should use a local model."""
        reg = make_registry()
        config = CouncilConfig(local_only=True)
        model = _select_scout_model(reg, config)
        m = reg.get(model)
        assert m is not None
        assert m.supports_local is True

    def test_scout_model_family_constraint(self):
        """With family constraint, scout should use that family."""
        reg = make_registry()
        config = CouncilConfig(family="microsoft")
        model = _select_scout_model(reg, config)
        m = reg.get(model)
        assert m is not None
        assert m.family == "microsoft"

    def test_verifier_model_mid_tier(self):
        """Verifier model selection should prefer mid tier."""
        reg = make_registry()
        config = CouncilConfig()
        model = _select_verifier_model(reg, config)
        m = reg.get(model)
        assert m is not None
        assert m.tier == ModelTier.MID

    def test_verifier_falls_back_to_premium(self):
        """If no mid-tier models available, verifier should use premium."""
        # Create registry with only cheap and premium models
        models = [
            ModelInfo(model_id="cheap/model", family="test", tier=ModelTier.CHEAP, context_window=128_000),
            ModelInfo(model_id="premium/model", family="test", tier=ModelTier.PREMIUM, context_window=128_000),
        ]
        reg = ModelRegistry(models)
        config = CouncilConfig()
        model = _select_verifier_model(reg, config)
        m = reg.get(model)
        assert m is not None
        assert m.tier == ModelTier.PREMIUM


# ── run_scout Integration Tests ─────────────────────────────────────────


class TestRunScout:
    @pytest.mark.asyncio
    async def test_scout_produces_mission_brief(self):
        """Full scout run should produce a valid MissionBrief."""
        reg = make_registry()
        budget = TokenBudget(total=100000)
        client = LLMClient(reg, budget)
        config = CouncilConfig()

        # Mock the LLM calls
        scout_json = json.dumps({
            "question": "Is democracy the best?",
            "complexity": "complex",
            "domain_tags": ["politics"],
            "is_likely_solvable": False,
            "why_might_be_hard": "No objective answer",
            "suggested_roles": [
                {
                    "name": "Democracy Supporter",
                    "perspective": "Pro-democracy",
                    "expertise": "Political theory",
                    "suggested_model": "anthropic",
                    "system_prompt": "Argue for democracy",
                    "is_research": False,
                },
                {
                    "name": "Critique",
                    "perspective": "Anti-democracy",
                    "expertise": "Political critique",
                    "suggested_model": "deepseek",
                    "system_prompt": "Critique democracy",
                    "is_research": False,
                },
            ],
            "research_needed": True,
            "research_subquestions": ["Historical outcomes"],
            "debate_rounds": 2,
            "token_budget": 200000,
            "reasoning": "Complex question",
        })

        verifier_json = json.dumps({
            "question": "Is democracy the best?",
            "complexity": "complex",
            "domain_tags": ["politics", "philosophy"],
            "is_likely_solvable": False,
            "why_might_be_hard": "No objective answer",
            "suggested_roles": [
                {
                    "name": "Democracy Supporter",
                    "perspective": "Pro-democracy",
                    "expertise": "Political theory",
                    "suggested_model": "anthropic",
                    "system_prompt": "Argue for democracy",
                    "is_research": False,
                },
                {
                    "name": "Critique",
                    "perspective": "Anti-democracy",
                    "expertise": "Political critique",
                    "suggested_model": "deepseek",
                    "system_prompt": "Critique democracy",
                    "is_research": False,
                },
                {
                    "name": "Pragmatist",
                    "perspective": "Real-world outcomes",
                    "expertise": "Comparative politics",
                    "suggested_model": "google",
                    "system_prompt": "Focus on outcomes",
                    "is_research": False,
                },
            ],
            "research_needed": True,
            "research_subquestions": ["Historical outcomes of democracy"],
            "debate_rounds": 2,
            "token_budget": 250000,
            "reasoning": "Verifier added pragmatist role",
            "verification_notes": "Added third role for diversity",
        })

        call_count = 0
        async def mock_complete(model_id, messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return (scout_json, 500)
            else:
                return (verifier_json, 600)

        with patch.object(client, 'complete', side_effect=mock_complete):
            brief = await run_scout("Is democracy the best?", client, reg, config=config)

        assert isinstance(brief, MissionBrief)
        assert brief.complexity == Complexity.COMPLEX
        assert len(brief.suggested_roles) >= 2
        assert brief.research_needed is True

    @pytest.mark.asyncio
    async def test_scout_with_complexity_override(self):
        """Complexity override should override the scout's classification."""
        reg = make_registry()
        budget = TokenBudget(total=100000)
        client = LLMClient(reg, budget)
        config = CouncilConfig(complexity_override=Complexity.TRIVIAL)

        scout_json = json.dumps({
            "complexity": "deep",
            "is_likely_solvable": True,
            "why_might_be_hard": "",
            "suggested_roles": [],
            "research_needed": False,
            "debate_rounds": 3,
            "token_budget": 500000,
            "reasoning": "Test",
        })
        verifier_json = scout_json

        call_count = 0
        async def mock_complete(model_id, messages, **kwargs):
            nonlocal call_count
            call_count += 1
            return (scout_json if call_count == 1 else verifier_json, 500)

        with patch.object(client, 'complete', side_effect=mock_complete):
            brief = await run_scout("What is 2+2?", client, reg, config=config)

        assert brief.complexity == Complexity.TRIVIAL  # Override takes effect

    @pytest.mark.asyncio
    async def test_scout_with_budget_override(self):
        """Budget override should override the scout's budget."""
        reg = make_registry()
        budget = TokenBudget(total=100000)
        client = LLMClient(reg, budget)
        config = CouncilConfig(budget_override=999999)

        brief_json = json.dumps({
            "complexity": "moderate",
            "is_likely_solvable": True,
            "why_might_be_hard": "",
            "suggested_roles": [],
            "research_needed": False,
            "debate_rounds": 1,
            "token_budget": 50000,
            "reasoning": "Test",
        })

        call_count = 0
        async def mock_complete(model_id, messages, **kwargs):
            nonlocal call_count
            call_count += 1
            return (brief_json, 500)

        with patch.object(client, 'complete', side_effect=mock_complete):
            brief = await run_scout("Test", client, reg, config=config)

        assert brief.token_budget == 999999


# ── Prompt Template Tests ──────────────────────────────────────────────


class TestPrompts:
    def test_scout_system_prompt_is_comprehensive(self):
        """Scout system prompt should mention all key elements."""
        assert "complexity" in SCOUT_SYSTEM_PROMPT.lower()
        assert "trivial" in SCOUT_SYSTEM_PROMPT.lower()
        assert "moderate" in SCOUT_SYSTEM_PROMPT.lower()
        assert "complex" in SCOUT_SYSTEM_PROMPT.lower()
        assert "deep" in SCOUT_SYSTEM_PROMPT.lower()
        assert "JSON" in SCOUT_SYSTEM_PROMPT

    def test_verifier_system_prompt_checks_errors(self):
        """Verifier system prompt should mention common Scout errors."""
        assert "trick" in VERIFIER_SYSTEM_PROMPT.lower()
        assert "budget" in VERIFIER_SYSTEM_PROMPT.lower()
        assert "verification_notes" in VERIFIER_SYSTEM_PROMPT
