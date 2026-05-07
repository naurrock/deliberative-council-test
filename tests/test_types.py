"""Tests for council.types — core type definitions."""

import pytest
from pydantic import ValidationError

from council.types import (
    BehavioralSignals,
    Complexity,
    ConvergenceReason,
    ConvergenceResult,
    ConsensusLevel,
    DebateState,
    EpistemicTag,
    ModelTier,
    Position,
    ResearchFinding,
    EvidenceSource,
    EvidenceReport,
    MissionBrief,
    RoleSpec,
    RoundResult,
    KeyPoint,
    ModelUsage,
    PipelineTrace,
    FinalReport,
    AgreementAnalysis,
    FutilityCheck,
    HealthCheckResult,
    ModelInfo,
)


# ── Complexity ─────────────────────────────────────────────────────────


class TestComplexity:
    def test_values(self):
        assert Complexity.TRIVIAL == "trivial"
        assert Complexity.MODERATE == "moderate"
        assert Complexity.COMPLEX == "complex"
        assert Complexity.DEEP == "deep"

    def test_from_string(self):
        assert Complexity("trivial") == Complexity.TRIVIAL
        assert Complexity("deep") == Complexity.DEEP


# ── Position ───────────────────────────────────────────────────────────


class TestPosition:
    def test_basic_position(self):
        p = Position(
            agent_id="agent_0",
            role_name="Skeptical Analyst",
            argument="I disagree because...",
            self_confidence=0.8,
        )
        assert p.agent_id == "agent_0"
        assert p.position_stability is None  # First round
        assert p.supporting_evidence == []
        assert p.metacognitive_notes == ""

    def test_position_with_stability(self):
        p = Position(
            agent_id="agent_0",
            role_name="Analyst",
            argument="Same point again",
            self_confidence=0.9,
            position_stability=0.85,
        )
        assert p.position_stability == 0.85

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            Position(
                agent_id="a",
                role_name="r",
                argument="x",
                self_confidence=1.5,  # Out of bounds
            )

    def test_stability_bounds(self):
        with pytest.raises(ValidationError):
            Position(
                agent_id="a",
                role_name="r",
                argument="x",
                self_confidence=0.5,
                position_stability=1.5,  # Out of bounds
            )


# ── MissionBrief ───────────────────────────────────────────────────────


class TestMissionBrief:
    def test_minimal_brief(self):
        brief = MissionBrief(
            question="What is 2+2?",
            complexity=Complexity.TRIVIAL,
            is_likely_solvable=True,
            why_might_be_hard="Not hard",
            suggested_roles=[],
            research_needed=False,
            debate_rounds=0,
            token_budget=10000,
            scout_reasoning="Simple arithmetic",
            verification_notes="Agreed",
        )
        assert brief.complexity == Complexity.TRIVIAL
        assert brief.domain_tags == []
        assert brief.research_subquestions == []

    def test_complex_brief(self):
        brief = MissionBrief(
            question="Is democracy the best form of government?",
            complexity=Complexity.DEEP,
            domain_tags=["politics", "philosophy"],
            is_likely_solvable=False,
            why_might_be_hard="No objective answer",
            suggested_roles=[
                RoleSpec(
                    name="Democratic Theorist",
                    perspective="Pro-democracy",
                    expertise="Political theory",
                    suggested_model="anthropic",
                    system_prompt="You argue for democracy...",
                ),
            ],
            research_needed=True,
            research_subquestions=["Historical outcomes of democracy"],
            debate_rounds=3,
            token_budget=500000,
            scout_reasoning="Complex philosophical question",
            verification_notes="No clear answer expected",
        )
        assert len(brief.suggested_roles) == 1
        assert brief.suggested_roles[0].is_research is False


# ── Research Types ─────────────────────────────────────────────────────


class TestResearchTypes:
    def test_evidence_source(self):
        src = EvidenceSource(
            url="https://example.com",
            snippet="Some evidence",
            title="Example Page",
        )
        assert src.url == "https://example.com"

    def test_research_finding(self):
        finding = ResearchFinding(
            claim="The sky is blue",
            sources=[EvidenceSource(url="https://sky.com", snippet="Blue sky", title="Sky Info")],
            epistemic_tag=EpistemicTag.SOURCED,
            relevance=0.95,
        )
        assert finding.epistemic_tag == EpistemicTag.SOURCED

    def test_relevance_bounds(self):
        with pytest.raises(ValidationError):
            ResearchFinding(
                claim="x",
                sources=[],
                epistemic_tag=EpistemicTag.JUDGMENT,
                relevance=1.5,
            )

    def test_evidence_report(self):
        report = EvidenceReport(
            agent_id="research_0",
            sub_question="What is quantum computing?",
            key_findings=[],
            gaps="No practical implementations found",
            tokens_used=500,
        )
        assert report.tokens_used == 500


# ── Debate Types ───────────────────────────────────────────────────────


class TestDebateTypes:
    def test_round_result(self):
        rr = RoundResult(
            round_number=1,
            positions=[],
            critiques=[],
            agreement_matrix={"a": {"b": 0.7}},
            behavioral_signals={},
        )
        assert rr.round_number == 1
        assert rr.agreement_matrix["a"]["b"] == 0.7

    def test_debate_state(self):
        brief = MissionBrief(
            question="Test",
            complexity=Complexity.MODERATE,
            is_likely_solvable=True,
            why_might_be_hard="",
            suggested_roles=[],
            research_needed=False,
            debate_rounds=1,
            token_budget=100000,
            scout_reasoning="",
            verification_notes="",
        )
        state = DebateState(mission_brief=brief)
        assert state.is_resolved is False
        assert state.rounds == []
        assert state.resolution_reason is None

    def test_convergence_result(self):
        cr = ConvergenceResult(
            converged=True,
            reason=ConvergenceReason.NLI_TIER1_CONVERGENCE,
        )
        assert cr.converged is True

    def test_futility_check(self):
        fc = FutilityCheck(is_futile=True, reason="All agents locked in", confidence=0.9)
        assert fc.is_futile is True


# ── FinalReport Types ──────────────────────────────────────────────────


class TestFinalReportTypes:
    def test_key_point(self):
        kp = KeyPoint(
            point="Democracy promotes freedom",
            consensus=ConsensusLevel.STRONG,
            evidence=["https://example.com"],
            dissent=None,
        )
        assert kp.consensus == ConsensusLevel.STRONG

    def test_model_usage(self):
        mu = ModelUsage(model="gpt-4.1-mini", family="openai", tokens=5000)
        assert mu.tokens == 5000

    def test_pipeline_trace_total(self):
        trace = PipelineTrace(
            scout_tokens=1000,
            research_tokens=5000,
            debate_tokens=10000,
            synthesis_tokens=2000,
        )
        assert trace.total_tokens == 18000

    def test_final_report(self):
        report = FinalReport(
            question="What is 2+2?",
            complexity=Complexity.TRIVIAL,
            rounds_completed=0,
            convergence_score=1.0,
            answer="4",
            raw_markdown="# Answer\n\n4",
        )
        assert report.answer == "4"
        assert report.pipeline_trace.total_tokens == 0
        assert report.futility_notes is None
