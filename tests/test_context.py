"""Comprehensive tests for council.context — context window management."""

import pytest

from council.context import build_agent_context, _build_full_context, _compress_context
from council.types import (
    EpistemicTag,
    EvidenceReport,
    EvidenceSource,
    ResearchFinding,
)


# ── Test Fixtures ──────────────────────────────────────────────────────


def make_evidence() -> list[EvidenceReport]:
    return [
        EvidenceReport(
            agent_id="research_0",
            sub_question="What is quantum computing?",
            key_findings=[
                ResearchFinding(
                    claim="Quantum computers use qubits",
                    sources=[EvidenceSource(url="https://example.com", snippet="Test", title="Test")],
                    epistemic_tag=EpistemicTag.SOURCED,
                    relevance=0.9,
                ),
                ResearchFinding(
                    claim="Quantum advantage is debated",
                    sources=[],
                    epistemic_tag=EpistemicTag.INFERRED,
                    relevance=0.6,
                ),
            ],
            gaps="Limited data on newer systems",
        )
    ]


# ── build_agent_context Tests ─────────────────────────────────────────


class TestBuildAgentContext:
    def test_full_strategy_with_no_prior(self):
        """Full context with no prior positions should only show research."""
        evidence = make_evidence()
        graph = {"agent_a": ["agent_b"], "agent_b": ["agent_a"]}
        context = build_agent_context(
            agent_id="agent_a",
            round_num=0,
            graph=graph,
            all_positions={},
            research_evidence=evidence,
            strategy="full",
        )
        assert "quantum" in context.lower()
        assert "Research Evidence" in context

    def test_full_strategy_with_prior_positions(self):
        """Full context should include prior positions from visible agents."""
        graph = {"agent_a": ["agent_b"], "agent_b": ["agent_a"]}
        all_positions = {
            "agent_a": [{"argument": "I believe X", "role_name": "Analyst", "self_confidence": 0.8}],
            "agent_b": [{"argument": "I believe Y", "role_name": "Critic", "self_confidence": 0.7}],
        }
        context = build_agent_context(
            agent_id="agent_a",
            round_num=1,
            graph=graph,
            all_positions=all_positions,
            research_evidence=[],
            strategy="full",
        )
        assert "Prior Debate Rounds" in context
        # Agent can see its own and visible agents' positions
        assert "Analyst" in context
        assert "Critic" in context

    def test_agent_sees_own_positions(self):
        """Agent should see its own prior positions even without graph visibility."""
        graph = {"agent_a": [], "agent_b": []}  # a sees nobody
        all_positions = {
            "agent_a": [{"argument": "My own argument", "role_name": "Me", "self_confidence": 0.9}],
        }
        context = build_agent_context(
            agent_id="agent_a",
            round_num=1,
            graph=graph,
            all_positions=all_positions,
            research_evidence=[],
            strategy="full",
        )
        assert "My own argument" in context

    def test_empty_context(self):
        """No evidence and no positions should produce minimal context."""
        context = build_agent_context(
            agent_id="agent_a",
            round_num=0,
            graph={"agent_a": []},
            all_positions={},
            research_evidence=[],
            strategy="full",
        )
        # Should be minimal/empty
        assert isinstance(context, str)

    def test_progressive_strategy_raises(self):
        """Progressive strategy should raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="extension not yet implemented"):
            build_agent_context(
                agent_id="a",
                round_num=0,
                graph={"a": []},
                all_positions={},
                research_evidence=[],
                strategy="progressive",
            )

    def test_unknown_strategy_raises(self):
        """Unknown strategy should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown context strategy"):
            build_agent_context(
                agent_id="a",
                round_num=0,
                graph={"a": []},
                all_positions={},
                research_evidence=[],
                strategy="invalid",
            )


# ── Context Compression Tests ─────────────────────────────────────────


class TestContextCompression:
    def test_short_context_not_compressed(self):
        """Short context should not be compressed."""
        text = "This is a short context"
        result = _compress_context(text, 1000)
        assert result == text

    def test_long_context_compressed(self):
        """Long context should be truncated with a note."""
        text = "x" * 10000
        result = _compress_context(text, 1000)
        assert len(result) <= 1100  # Some overhead for truncation note
        assert "truncated" in result.lower()

    def test_compression_preserves_recent_content(self):
        """Compression should preserve the most recent content (tail)."""
        text = "OLD_CONTENT_START" + "x" * 5000 + "RECENT_CONTENT_END"
        result = _compress_context(text, 200)
        assert "RECENT_CONTENT_END" in result
        assert "OLD_CONTENT_START" not in result


# ── Research Evidence Integration Tests ────────────────────────────────


class TestResearchEvidenceInContext:
    def test_evidence_includes_gaps(self):
        """Research gaps should be included in context."""
        evidence = [
            EvidenceReport(
                agent_id="r0",
                sub_question="Test",
                key_findings=[],
                gaps="Could not find information on X",
            )
        ]
        context = build_agent_context(
            agent_id="agent_a",
            round_num=0,
            graph={"agent_a": []},
            all_positions={},
            research_evidence=evidence,
            strategy="full",
        )
        assert "Could not find" in context

    def test_evidence_includes_source_urls(self):
        """Source URLs from evidence should be in context."""
        evidence = [
            EvidenceReport(
                agent_id="r0",
                sub_question="Test",
                key_findings=[
                    ResearchFinding(
                        claim="Test claim",
                        sources=[EvidenceSource(url="https://source.example.com/paper", snippet="Test", title="Paper")],
                        epistemic_tag=EpistemicTag.SOURCED,
                        relevance=0.8,
                    ),
                ],
            )
        ]
        context = build_agent_context(
            agent_id="agent_a",
            round_num=0,
            graph={"agent_a": []},
            all_positions={},
            research_evidence=evidence,
            strategy="full",
        )
        assert "https://source.example.com/paper" in context

    def test_epistemic_tags_in_context(self):
        """Epistemic tags should be visible in context."""
        evidence = [
            EvidenceReport(
                agent_id="r0",
                sub_question="Test",
                key_findings=[
                    ResearchFinding(
                        claim="Sourced claim",
                        sources=[EvidenceSource(url="https://example.com", snippet="Test", title="Test")],
                        epistemic_tag=EpistemicTag.SOURCED,
                        relevance=0.9,
                    ),
                    ResearchFinding(
                        claim="Inferred claim",
                        sources=[],
                        epistemic_tag=EpistemicTag.INFERRED,
                        relevance=0.5,
                    ),
                ],
            )
        ]
        context = build_agent_context(
            agent_id="agent_a",
            round_num=0,
            graph={"agent_a": []},
            all_positions={},
            research_evidence=evidence,
            strategy="full",
        )
        assert "sourced" in context
        assert "inferred" in context
