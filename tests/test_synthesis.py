"""Comprehensive tests for council.synthesis — Synthesis phase."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from council.config import CouncilConfig
from council.models import LLMClient, ModelRegistry, TokenBudget
from council.synthesis import (
    _build_debate_summary,
    _build_evidence_summary,
    _fallback_report,
    _generate_markdown,
    _parse_synthesis,
    _select_synthesizer,
    run_synthesis,
)
from council.types import (
    BehavioralSignals,
    Complexity,
    ConsensusLevel,
    ConvergenceReason,
    DebateState,
    EpistemicTag,
    EvidenceReport,
    EvidenceSource,
    FinalReport,
    KeyPoint,
    MissionBrief,
    ModelInfo,
    ModelTier,
    PipelineTrace,
    Position,
    ResearchFinding,
    RoundResult,
)


# ── Test Fixtures ──────────────────────────────────────────────────────


def make_brief() -> MissionBrief:
    return MissionBrief(
        question="Is democracy the best?",
        complexity=Complexity.COMPLEX,
        is_likely_solvable=False,
        why_might_be_hard="No objective answer",
        suggested_roles=[],
        research_needed=True,
        research_subquestions=["Historical outcomes"],
        debate_rounds=2,
        token_budget=200000,
        scout_reasoning="Complex question",
        verification_notes="",
    )


def make_debate_state() -> DebateState:
    brief = make_brief()
    pos_a = Position(
        agent_id="debater_0",
        role_name="Democrat",
        argument="Democracy protects individual rights and promotes freedom",
        self_confidence=0.85,
        position_stability=0.75,
    )
    pos_b = Position(
        agent_id="debater_1",
        role_name="Critic",
        argument="Democracy can lead to tyranny of the majority",
        self_confidence=0.70,
        position_stability=0.60,
    )
    round = RoundResult(
        round_number=1,
        positions=[pos_a, pos_b],
        agreement_matrix={"debater_0": {"debater_1": 0.4}},
    )
    return DebateState(
        mission_brief=brief,
        rounds=[round],
        is_resolved=True,
        resolution_reason=ConvergenceReason.ROUND_LIMIT_REACHED.value,
    )


def make_evidence() -> list[EvidenceReport]:
    return [
        EvidenceReport(
            agent_id="research_0",
            sub_question="Historical outcomes of democracy",
            key_findings=[
                ResearchFinding(
                    claim="Democracies have lower war participation rates",
                    sources=[EvidenceSource(url="https://example.com/war", snippet="Data shows...", title="War Study")],
                    epistemic_tag=EpistemicTag.SOURCED,
                    relevance=0.9,
                ),
                ResearchFinding(
                    claim="Democratic institutions promote economic growth",
                    sources=[],
                    epistemic_tag=EpistemicTag.INFERRED,
                    relevance=0.7,
                ),
            ],
            gaps="Limited data on newer democracies",
        )
    ]


def make_registry() -> ModelRegistry:
    models = [
        ModelInfo(model_id="anthropic/claude-sonnet-4", family="anthropic", tier=ModelTier.PREMIUM, context_window=200_000, input_cost_per_m=3.00, output_cost_per_m=15.00),
        ModelInfo(model_id="openai/gpt-4.1", family="openai", tier=ModelTier.PREMIUM, context_window=128_000, input_cost_per_m=2.00, output_cost_per_m=8.00),
        ModelInfo(model_id="openai/gpt-4.1-mini", family="openai", tier=ModelTier.MID, context_window=128_000, input_cost_per_m=0.40, output_cost_per_m=1.60),
        ModelInfo(model_id="gemini/gemini-2.5-flash", family="google", tier=ModelTier.MID, context_window=1_000_000, input_cost_per_m=0.15, output_cost_per_m=0.60),
    ]
    return ModelRegistry(models)


# ── _build_debate_summary Tests ────────────────────────────────────────


class TestBuildDebateSummary:
    def test_includes_round_count(self):
        state = make_debate_state()
        summary = _build_debate_summary(state)
        assert "Total rounds: 1" in summary

    def test_includes_positions(self):
        state = make_debate_state()
        summary = _build_debate_summary(state)
        assert "Democrat" in summary
        assert "Critic" in summary

    def test_includes_position_stability(self):
        state = make_debate_state()
        summary = _build_debate_summary(state)
        assert "Position stability" in summary

    def test_includes_resolution(self):
        state = make_debate_state()
        summary = _build_debate_summary(state)
        assert "round_limit_reached" in summary

    def test_includes_futility_flags(self):
        brief = make_brief()
        state = DebateState(
            mission_brief=brief,
            rounds=[RoundResult(round_number=1)],
            futility_flags=["Futility detected in round 1"],
        )
        summary = _build_debate_summary(state)
        assert "Futility Flags" in summary


# ── _build_evidence_summary Tests ──────────────────────────────────────


class TestBuildEvidenceSummary:
    def test_includes_findings(self):
        evidence = make_evidence()
        summary = _build_evidence_summary(evidence)
        assert "Historical outcomes" in summary
        assert "lower war participation" in summary

    def test_includes_epistemic_tags(self):
        evidence = make_evidence()
        summary = _build_evidence_summary(evidence)
        assert "sourced" in summary
        assert "inferred" in summary

    def test_includes_gaps(self):
        evidence = make_evidence()
        summary = _build_evidence_summary(evidence)
        assert "Limited data" in summary

    def test_empty_evidence(self):
        summary = _build_evidence_summary([])
        assert "No research evidence" in summary


# ── _parse_synthesis Tests ─────────────────────────────────────────────


class TestParseSynthesis:
    def test_valid_synthesis_json(self):
        synthesis_json = json.dumps({
            "answer": "Democracy has strengths and weaknesses.",
            "key_points": [
                {
                    "point": "Democracy protects rights",
                    "consensus": "strong",
                    "evidence": ["https://example.com"],
                    "dissent": None,
                },
                {
                    "point": "Majority tyranny is a risk",
                    "consensus": "contested",
                    "evidence": [],
                    "dissent": "Critic argues checks exist",
                },
            ],
            "dissenting_views": ["Tyranny of the majority concern"],
            "research_sources": ["https://example.com/war"],
            "futility_notes": None,
        })

        brief = make_brief()
        state = make_debate_state()
        evidence = make_evidence()

        report = _parse_synthesis(synthesis_json, brief, state, evidence)
        assert isinstance(report, FinalReport)
        assert "strengths and weaknesses" in report.answer
        assert len(report.key_points) == 2
        assert report.key_points[0].consensus == ConsensusLevel.STRONG
        assert report.key_points[1].consensus == ConsensusLevel.CONTESTED
        assert report.key_points[1].dissent is not None
        assert len(report.dissenting_views) == 1

    def test_invalid_json_creates_fallback(self):
        """Invalid JSON should create a fallback report."""
        brief = make_brief()
        state = make_debate_state()
        evidence = make_evidence()

        report = _parse_synthesis("Not JSON at all", brief, state, evidence)
        assert isinstance(report, FinalReport)
        assert "Fallback" in report.answer or "failed" in report.answer.lower()

    def test_json_with_markdown_fences(self):
        inner = json.dumps({
            "answer": "Test answer",
            "key_points": [],
            "dissenting_views": [],
            "research_sources": [],
        })
        wrapped = f"```json\n{inner}\n```"
        brief = make_brief()
        state = make_debate_state()
        report = _parse_synthesis(wrapped, brief, state, [])
        assert report.answer == "Test answer"

    def test_unknown_consensus_defaults_to_moderate(self):
        synthesis_json = json.dumps({
            "answer": "Test",
            "key_points": [
                {
                    "point": "A point",
                    "consensus": "super_strong",
                    "evidence": [],
                }
            ],
            "dissenting_views": [],
            "research_sources": [],
        })
        brief = make_brief()
        state = make_debate_state()
        report = _parse_synthesis(synthesis_json, brief, state, [])
        assert report.key_points[0].consensus == ConsensusLevel.MODERATE

    def test_convergence_score_computed(self):
        """Convergence score should be computed from last round's agreement matrix."""
        synthesis_json = json.dumps({
            "answer": "Test",
            "key_points": [],
            "dissenting_views": [],
            "research_sources": [],
        })
        brief = make_brief()
        state = make_debate_state()
        report = _parse_synthesis(synthesis_json, brief, state, [])
        # Our test state has agreement_matrix {"debater_0": {"debater_1": 0.4}}
        assert report.convergence_score == 0.4


# ── _fallback_report Tests ─────────────────────────────────────────────


class TestFallbackReport:
    def test_fallback_includes_positions(self):
        brief = make_brief()
        state = make_debate_state()
        evidence = make_evidence()
        report = _fallback_report(brief, state, evidence, "Test failure")
        assert "Fallback" in report.answer
        assert report.futility_notes == "Test failure"

    def test_fallback_collects_sources(self):
        brief = make_brief()
        state = make_debate_state()
        evidence = make_evidence()
        report = _fallback_report(brief, state, evidence, "Failed")
        assert "https://example.com/war" in report.research_sources


# ── _select_synthesizer Tests ──────────────────────────────────────────


class TestSelectSynthesizer:
    def test_selects_premium_model(self):
        """Should prefer premium models for synthesis."""
        reg = make_registry()
        state = make_debate_state()
        config = CouncilConfig()
        model = _select_synthesizer(state, reg, config)
        m = reg.get(model)
        assert m is not None
        assert m.tier == ModelTier.PREMIUM

    def test_model_override(self):
        """Model override for synthesizer should take priority."""
        reg = make_registry()
        state = make_debate_state()
        config = CouncilConfig(model_overrides={"synthesizer": "openai/gpt-4.1"})
        model = _select_synthesizer(state, reg, config)
        assert model == "openai/gpt-4.1"

    def test_fallback_to_mid_tier(self):
        """Should fall back to mid-tier if no premium available."""
        models = [
            ModelInfo(model_id="openai/gpt-4.1-mini", family="openai", tier=ModelTier.MID, context_window=128_000),
            ModelInfo(model_id="gemini/gemini-2.5-flash", family="google", tier=ModelTier.MID, context_window=1_000_000),
        ]
        reg = ModelRegistry(models)
        state = make_debate_state()
        config = CouncilConfig()
        model = _select_synthesizer(state, reg, config)
        m = reg.get(model)
        assert m is not None
        assert m.tier == ModelTier.MID

    def test_family_constraint(self):
        """Should respect family constraint."""
        reg = make_registry()
        state = make_debate_state()
        config = CouncilConfig(family="anthropic")
        model = _select_synthesizer(state, reg, config)
        m = reg.get(model)
        assert m is not None
        assert m.family == "anthropic"


# ── _generate_markdown Tests ───────────────────────────────────────────


class TestGenerateMarkdown:
    def test_includes_question(self):
        report = FinalReport(
            question="Is democracy best?",
            complexity=Complexity.COMPLEX,
            rounds_completed=2,
            convergence_score=0.65,
            answer="Democracy has trade-offs",
        )
        md = _generate_markdown(report)
        assert "Is democracy best?" in md

    def test_includes_key_points(self):
        report = FinalReport(
            question="Test",
            complexity=Complexity.MODERATE,
            rounds_completed=1,
            convergence_score=0.5,
            answer="Test answer",
            key_points=[
                KeyPoint(point="Point 1", consensus=ConsensusLevel.STRONG),
                KeyPoint(point="Point 2", consensus=ConsensusLevel.CONTESTED, dissent="Some dissent"),
            ],
        )
        md = _generate_markdown(report)
        assert "Point 1" in md
        assert "Point 2" in md
        assert "Some dissent" in md

    def test_includes_pipeline_trace(self):
        report = FinalReport(
            question="Test",
            complexity=Complexity.TRIVIAL,
            rounds_completed=0,
            convergence_score=1.0,
            answer="4",
            pipeline_trace=PipelineTrace(scout_tokens=1000, research_tokens=0, debate_tokens=0, synthesis_tokens=500),
        )
        md = _generate_markdown(report)
        assert "Scout" in md
        assert "1,000" in md or "1000" in md


# ── run_synthesis Integration Tests ────────────────────────────────────


class TestRunSynthesis:
    @pytest.mark.asyncio
    async def test_synthesis_produces_report(self):
        """Full synthesis should produce a valid FinalReport."""
        reg = make_registry()
        budget = TokenBudget(total=100000)
        client = LLMClient(reg, budget)
        config = CouncilConfig()
        brief = make_brief()
        state = make_debate_state()
        evidence = make_evidence()

        synthesis_json = json.dumps({
            "answer": "Democracy has strengths and weaknesses that must be balanced.",
            "key_points": [
                {
                    "point": "Democracy protects individual rights",
                    "consensus": "strong",
                    "evidence": ["https://example.com"],
                    "dissent": None,
                },
            ],
            "dissenting_views": ["Majority tyranny concern"],
            "research_sources": ["https://example.com/war"],
            "futility_notes": None,
        })

        async def mock_complete(model_id, messages, **kwargs):
            return (synthesis_json, 800)

        with patch.object(client, 'complete', side_effect=mock_complete):
            report = await run_synthesis(brief, evidence, state, client, reg, config)

        assert isinstance(report, FinalReport)
        assert report.question == "Is democracy the best?"
        assert report.complexity == Complexity.COMPLEX
        assert len(report.key_points) == 1
        assert report.raw_markdown  # Should have generated markdown
        assert report.pipeline_trace.synthesis_tokens > 0

    @pytest.mark.asyncio
    async def test_synthesis_fallback_on_failure(self):
        """Synthesis should produce a fallback report on LLM failure."""
        reg = make_registry()
        budget = TokenBudget(total=100000)
        client = LLMClient(reg, budget)
        config = CouncilConfig()
        brief = make_brief()
        state = make_debate_state()
        evidence = make_evidence()

        async def mock_complete_fails(model_id, messages, **kwargs):
            raise RuntimeError("LLM service unavailable")

        with patch.object(client, 'complete', side_effect=mock_complete_fails):
            report = await run_synthesis(brief, evidence, state, client, reg, config)

        assert isinstance(report, FinalReport)
        assert "Fallback" in report.answer or "failed" in report.answer.lower()
