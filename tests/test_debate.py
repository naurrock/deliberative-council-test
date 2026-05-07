"""Comprehensive tests for council.debate — Debate phase."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from council.config import CouncilConfig, NLIConfig
from council.models import LLMClient, ModelRegistry, TokenBudget
from council.debate import (
    _assign_models_to_agents,
    _collect_prior_positions,
    _find_previous_position,
    _parse_position,
    run_debate,
)
from council.types import (
    BehavioralSignals,
    Complexity,
    ConvergenceReason,
    Critique,
    DebateState,
    EvidenceReport,
    MissionBrief,
    ModelInfo,
    ModelTier,
    Position,
    RoleSpec,
    RoundResult,
)


# ── Test Fixtures ──────────────────────────────────────────────────────


def make_brief(rounds: int = 2, num_roles: int = 2) -> MissionBrief:
    roles = []
    for i in range(num_roles):
        roles.append(RoleSpec(
            name=f"Role_{i}",
            perspective=f"Perspective {i}",
            expertise="General",
            suggested_model=["openai", "anthropic", "google", "deepseek"][i % 4],
            system_prompt=f"You are debater {i}",
            is_research=False,
        ))
    return MissionBrief(
        question="Is democracy the best form of government?",
        complexity=Complexity.COMPLEX,
        is_likely_solvable=False,
        why_might_be_hard="No objective answer",
        suggested_roles=roles,
        research_needed=True,
        research_subquestions=["Historical outcomes"],
        debate_rounds=rounds,
        token_budget=200000,
        scout_reasoning="Complex question",
        verification_notes="",
    )


def make_registry() -> ModelRegistry:
    models = [
        ModelInfo(model_id="openai/gpt-4.1-mini", family="openai", tier=ModelTier.MID, context_window=128_000, input_cost_per_m=0.40, output_cost_per_m=1.60),
        ModelInfo(model_id="gemini/gemini-2.5-flash", family="google", tier=ModelTier.MID, context_window=1_000_000, input_cost_per_m=0.15, output_cost_per_m=0.60),
        ModelInfo(model_id="deepseek/deepseek-chat", family="deepseek", tier=ModelTier.MID, context_window=128_000, input_cost_per_m=0.27, output_cost_per_m=1.10),
        ModelInfo(model_id="ollama_chat/qwen3:8b", family="alibaba", tier=ModelTier.CHEAP, context_window=128_000, supports_local=True),
        ModelInfo(model_id="anthropic/claude-sonnet-4", family="anthropic", tier=ModelTier.PREMIUM, context_window=200_000, input_cost_per_m=3.00, output_cost_per_m=15.00),
    ]
    return ModelRegistry(models)


# ── _parse_position Tests ──────────────────────────────────────────────


class TestParsePosition:
    def test_valid_json_position(self):
        """Parse a well-formed position JSON."""
        position_json = json.dumps({
            "argument": "Democracy is the best because it protects individual rights",
            "supporting_evidence": ["Historical data shows democracies are more stable"],
            "self_confidence": 0.85,
            "metacognitive_notes": "I would change my position if shown evidence of democratic failures",
        })
        pos = _parse_position(position_json, "debater_0", "Analyst")
        assert pos.agent_id == "debater_0"
        assert pos.role_name == "Analyst"
        assert "individual rights" in pos.argument
        assert len(pos.supporting_evidence) == 1
        assert pos.self_confidence == 0.85
        assert "democratic failures" in pos.metacognitive_notes

    def test_invalid_json_uses_raw_text(self):
        """Invalid JSON should fall back to raw text as argument."""
        raw = "I believe democracy is flawed because..."
        pos = _parse_position(raw, "debater_0", "Analyst")
        assert pos.argument == raw
        assert pos.self_confidence == 0.5  # Default

    def test_json_with_markdown_fences(self):
        """Position JSON wrapped in code fences should parse correctly."""
        inner = json.dumps({
            "argument": "Test argument",
            "self_confidence": 0.7,
        })
        wrapped = f"```json\n{inner}\n```"
        pos = _parse_position(wrapped, "d0", "R")
        assert pos.argument == "Test argument"

    def test_confidence_out_of_bounds_clamped(self):
        """Confidence values outside [0,1] should be clamped."""
        position_json = json.dumps({
            "argument": "Test",
            "self_confidence": 1.5,
        })
        pos = _parse_position(position_json, "d0", "R")
        assert pos.self_confidence == 1.0

    def test_missing_confidence_defaults(self):
        """Missing confidence should default to 0.5."""
        position_json = json.dumps({
            "argument": "Test argument",
        })
        pos = _parse_position(position_json, "d0", "R")
        assert pos.self_confidence == 0.5

    def test_missing_evidence_defaults_to_empty(self):
        """Missing evidence should default to empty list."""
        position_json = json.dumps({
            "argument": "Test",
            "self_confidence": 0.8,
        })
        pos = _parse_position(position_json, "d0", "R")
        assert pos.supporting_evidence == []

    def test_non_numeric_confidence_defaults(self):
        """Non-numeric confidence should default to 0.5."""
        position_json = json.dumps({
            "argument": "Test",
            "self_confidence": "very confident",
        })
        pos = _parse_position(position_json, "d0", "R")
        assert pos.self_confidence == 0.5


# ── _assign_models_to_agents Tests ─────────────────────────────────────


class TestAssignModelsToAgents:
    def test_assigns_models_to_roles(self):
        """Each role should get a model assigned."""
        reg = make_registry()
        config = CouncilConfig()
        roles = [
            RoleSpec(name="R1", perspective="P1", expertise="E1", suggested_model="openai", system_prompt="S1"),
            RoleSpec(name="R2", perspective="P2", expertise="E2", suggested_model="google", system_prompt="S2"),
        ]
        models = _assign_models_to_agents(roles, reg, config)
        assert "debater_0" in models
        assert "debater_1" in models
        # Should be valid model IDs from registry
        assert reg.get(models["debater_0"]) is not None
        assert reg.get(models["debater_1"]) is not None

    def test_family_diversity_preferred(self):
        """Models from different families should be preferred."""
        reg = make_registry()
        config = CouncilConfig()
        roles = [
            RoleSpec(name="R1", perspective="P1", expertise="E1", suggested_model="", system_prompt="S1"),
            RoleSpec(name="R2", perspective="P2", expertise="E2", suggested_model="", system_prompt="S2"),
            RoleSpec(name="R3", perspective="P3", expertise="E3", suggested_model="", system_prompt="S3"),
        ]
        models = _assign_models_to_agents(roles, reg, config)
        families = set()
        for agent_id, model_id in models.items():
            m = reg.get(model_id)
            if m:
                families.add(m.family)
        # Should have at least 2 different families for 3 debaters
        assert len(families) >= 2

    def test_local_only_constraint(self):
        """With local_only, only local models should be assigned."""
        reg = make_registry()
        config = CouncilConfig(local_only=True)
        roles = [
            RoleSpec(name="R1", perspective="P1", expertise="E1", suggested_model="", system_prompt="S1"),
        ]
        models = _assign_models_to_agents(roles, reg, config)
        for agent_id, model_id in models.items():
            m = reg.get(model_id)
            if m:
                assert m.supports_local


# ── _collect_prior_positions Tests ─────────────────────────────────────


class TestCollectPriorPositions:
    def test_no_prior_rounds(self):
        """With no prior rounds, should return empty dict."""
        brief = make_brief()
        state = DebateState(mission_brief=brief)
        result = _collect_prior_positions(state)
        assert result == {}

    def test_collects_from_single_round(self):
        """Should collect positions from a single round."""
        brief = make_brief()
        pos_a = Position(agent_id="debater_0", role_name="A", argument="Arg A", self_confidence=0.8)
        pos_b = Position(agent_id="debater_1", role_name="B", argument="Arg B", self_confidence=0.7)
        round = RoundResult(round_number=1, positions=[pos_a, pos_b])
        state = DebateState(mission_brief=brief, rounds=[round])

        result = _collect_prior_positions(state)
        assert "debater_0" in result
        assert "debater_1" in result
        assert result["debater_0"][0]["argument"] == "Arg A"

    def test_collects_from_multiple_rounds(self):
        """Should collect positions from multiple rounds in order."""
        brief = make_brief()
        pos_r1 = Position(agent_id="debater_0", role_name="A", argument="Round 1", self_confidence=0.8)
        pos_r2 = Position(agent_id="debater_0", role_name="A", argument="Round 2", self_confidence=0.7)
        state = DebateState(
            mission_brief=brief,
            rounds=[
                RoundResult(round_number=1, positions=[pos_r1]),
                RoundResult(round_number=2, positions=[pos_r2]),
            ],
        )
        result = _collect_prior_positions(state)
        assert len(result["debater_0"]) == 2
        assert result["debater_0"][0]["argument"] == "Round 1"
        assert result["debater_0"][1]["argument"] == "Round 2"


# ── _find_previous_position Tests ──────────────────────────────────────


class TestFindPreviousPosition:
    def test_finds_existing_position(self):
        """Should find the position for a given agent."""
        pos = Position(agent_id="debater_0", role_name="A", argument="Found!", self_confidence=0.9)
        round = RoundResult(round_number=1, positions=[pos])
        result = _find_previous_position("debater_0", round)
        assert result is not None
        assert result.argument == "Found!"

    def test_returns_none_for_missing_agent(self):
        """Should return None if agent not in round."""
        round = RoundResult(
            round_number=1,
            positions=[Position(agent_id="debater_0", role_name="A", argument="X", self_confidence=0.5)],
        )
        result = _find_previous_position("debater_1", round)
        assert result is None


# ── Debate Integration Tests ───────────────────────────────────────────


class TestRunDebate:
    @pytest.mark.asyncio
    async def test_debate_runs_to_round_limit(self):
        """Debate should run up to the configured round limit."""
        brief = make_brief(rounds=1, num_roles=2)
        evidence = []
        reg = make_registry()
        budget = TokenBudget(total=100000)
        client = LLMClient(reg, budget)
        config = CouncilConfig()

        # Mock NLI to always return confident disagreement (avoid Tier 2)
        position_json = json.dumps({
            "argument": "I believe X is true",
            "supporting_evidence": ["Evidence A"],
            "self_confidence": 0.8,
            "metacognitive_notes": "Would change if proven wrong",
        })

        critique_json = json.dumps({
            "points_of_agreement": ["Some point"],
            "points_of_disagreement": ["Disagree on X"],
            "counter_arguments": ["Counter argument"],
            "new_evidence": [],
        })

        async def mock_complete(model_id, messages, **kwargs):
            # Return position or critique depending on prompt content
            if "critique" in messages[0].get("content", messages[-1].get("content", "")).lower():
                return (critique_json, 200)
            return (position_json, 300)

        with patch.object(client, 'complete', side_effect=mock_complete), \
             patch("council.debate.compute_agreement", new_callable=AsyncMock, return_value=(0.7, False)), \
             patch("council.debate.compute_position_stability", return_value=None):
            state = await run_debate(
                brief=brief,
                evidence=evidence,
                client=client,
                registry=reg,
                budget=budget,
                config=config,
            )

        assert isinstance(state, DebateState)
        assert state.is_resolved is True
        assert len(state.rounds) >= 1

    @pytest.mark.asyncio
    async def test_debate_with_existing_state(self):
        """Debate should respect prior state for resume."""
        brief = make_brief(rounds=2, num_roles=2)
        reg = make_registry()
        budget = TokenBudget(total=100000)
        client = LLMClient(reg, budget)
        config = CouncilConfig()

        # Start with a state that already has 1 round
        existing_round = RoundResult(
            round_number=1,
            positions=[
                Position(agent_id="debater_0", role_name="A", argument="X", self_confidence=0.8),
                Position(agent_id="debater_1", role_name="B", argument="Y", self_confidence=0.7),
            ],
            agreement_matrix={"debater_0": {"debater_1": 0.8}},
        )
        prior_state = DebateState(
            mission_brief=brief,
            rounds=[existing_round],
        )

        position_json = json.dumps({
            "argument": "Round 2 argument",
            "self_confidence": 0.8,
            "metacognitive_notes": "Refined position",
        })

        critique_json = json.dumps({
            "points_of_agreement": [],
            "points_of_disagreement": ["Still disagree"],
            "counter_arguments": [],
            "new_evidence": [],
        })

        async def mock_complete(model_id, messages, **kwargs):
            if "critique" in messages[0].get("content", messages[-1].get("content", "")).lower():
                return (critique_json, 200)
            return (position_json, 300)

        with patch.object(client, 'complete', side_effect=mock_complete), \
             patch("council.debate.compute_agreement", new_callable=AsyncMock, return_value=(0.8, False)), \
             patch("council.debate.compute_position_stability", return_value=0.75):
            state = await run_debate(
                brief=brief,
                evidence=[],
                client=client,
                registry=reg,
                budget=budget,
                config=config,
                prior_state=prior_state,
            )

        # Should have at least 2 rounds now (the existing + new)
        assert len(state.rounds) >= 2
