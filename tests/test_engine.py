"""Comprehensive tests for council.engine — pipeline orchestration."""

import json
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from council.config import CouncilConfig, load_default_config
from council.engine import CheckpointManager, run_council
from council.types import (
    Complexity,
    DebateState,
    EvidenceReport,
    FinalReport,
    MissionBrief,
    Position,
    RoundResult,
)


# ── CheckpointManager Tests ────────────────────────────────────────────


class TestCheckpointManager:
    def test_save_and_load_brief(self, tmp_path):
        """Should save and load a MissionBrief checkpoint."""
        db_path = str(tmp_path / "test_checkpoints.db")
        cp = CheckpointManager("test_run", db_path=db_path)

        brief = MissionBrief(
            question="What is 2+2?",
            complexity=Complexity.TRIVIAL,
            is_likely_solvable=True,
            why_might_be_hard="",
            suggested_roles=[],
            research_needed=False,
            debate_rounds=0,
            token_budget=10000,
            scout_reasoning="Simple",
            verification_notes="",
        )
        cp.save_brief(brief)

        loaded = cp.load_checkpoint()
        assert loaded is not None
        assert loaded["phase"] == "scout"
        assert loaded["brief_json"] is not None

    def test_save_evidence(self, tmp_path):
        """Should save research evidence checkpoint."""
        db_path = str(tmp_path / "test_checkpoints.db")
        cp = CheckpointManager("test_run", db_path=db_path)

        # Save brief first (needed for evidence update)
        brief = MissionBrief(
            question="Test",
            complexity=Complexity.MODERATE,
            is_likely_solvable=True,
            why_might_be_hard="",
            suggested_roles=[],
            research_needed=True,
            debate_rounds=1,
            token_budget=100000,
            scout_reasoning="",
            verification_notes="",
        )
        cp.save_brief(brief)

        evidence = [EvidenceReport(agent_id="r0", sub_question="Test", tokens_used=100)]
        cp.save_evidence(evidence)

        loaded = cp.load_checkpoint()
        assert loaded["phase"] == "research"
        assert loaded["evidence_json"] is not None

    def test_save_debate_state(self, tmp_path):
        """Should save debate state checkpoint."""
        db_path = str(tmp_path / "test_checkpoints.db")
        cp = CheckpointManager("test_run", db_path=db_path)

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
        cp.save_brief(brief)

        state = DebateState(mission_brief=brief, is_resolved=True, resolution_reason="test")
        cp.save_debate_state(state)

        loaded = cp.load_checkpoint()
        assert loaded["phase"] == "debate"

    def test_get_phase(self, tmp_path):
        """Should return the last completed phase."""
        db_path = str(tmp_path / "test_checkpoints.db")
        cp = CheckpointManager("test_run", db_path=db_path)

        brief = MissionBrief(
            question="Test",
            complexity=Complexity.TRIVIAL,
            is_likely_solvable=True,
            why_might_be_hard="",
            suggested_roles=[],
            research_needed=False,
            debate_rounds=0,
            token_budget=10000,
            scout_reasoning="",
            verification_notes="",
        )
        cp.save_brief(brief)
        assert cp.get_phase() == "scout"

    def test_load_nonexistent_checkpoint(self, tmp_path):
        """Loading a nonexistent checkpoint should return None."""
        db_path = str(tmp_path / "test_checkpoints.db")
        cp = CheckpointManager("nonexistent_run", db_path=db_path)
        assert cp.load_checkpoint() is None

    def test_creates_db_directory(self, tmp_path):
        """Should create the parent directory for the DB file."""
        db_path = str(tmp_path / "deep" / "nested" / "dir" / "checkpoints.db")
        cp = CheckpointManager("test_run", db_path=db_path)
        assert Path(db_path).parent.exists()


# ── run_council Integration Tests ──────────────────────────────────────


class TestRunCouncil:
    @pytest.mark.asyncio
    async def test_trivial_question_skips_to_synthesis(self):
        """Trivial questions should skip research and debate."""
        # Mock the entire pipeline to avoid real API calls
        with patch("council.engine.run_scout", new_callable=AsyncMock) as mock_scout, \
             patch("council.engine.run_synthesis", new_callable=AsyncMock) as mock_synthesis:

            mock_scout.return_value = MissionBrief(
                question="What is 2+2?",
                complexity=Complexity.TRIVIAL,
                is_likely_solvable=True,
                why_might_be_hard="",
                suggested_roles=[],
                research_needed=False,
                debate_rounds=0,
                token_budget=10000,
                scout_reasoning="Simple",
                verification_notes="",
            )

            mock_synthesis.return_value = FinalReport(
                question="What is 2+2?",
                complexity=Complexity.TRIVIAL,
                rounds_completed=0,
                convergence_score=1.0,
                answer="4",
            )

            config = CouncilConfig()
            report = await run_council("What is 2+2?", config)

            assert isinstance(report, FinalReport)
            assert report.answer == "4"
            mock_scout.assert_called_once()
            mock_synthesis.assert_called_once()
            # Should NOT have called run_research or run_debate

    @pytest.mark.asyncio
    async def test_complex_question_runs_full_pipeline(self):
        """Complex questions should run all phases."""
        with patch("council.engine.run_scout", new_callable=AsyncMock) as mock_scout, \
             patch("council.engine.run_research", new_callable=AsyncMock) as mock_research, \
             patch("council.engine.run_debate", new_callable=AsyncMock) as mock_debate, \
             patch("council.engine.run_synthesis", new_callable=AsyncMock) as mock_synthesis, \
             patch("council.engine.ToolRegistry") as mock_tools_cls:

            mock_tools = MagicMock()
            mock_tools.close = AsyncMock()
            mock_tools_cls.return_value = mock_tools

            mock_scout.return_value = MissionBrief(
                question="Is democracy best?",
                complexity=Complexity.COMPLEX,
                is_likely_solvable=False,
                why_might_be_hard="No objective answer",
                suggested_roles=[],
                research_needed=True,
                research_subquestions=["Historical outcomes"],
                debate_rounds=2,
                token_budget=200000,
                scout_reasoning="Complex",
                verification_notes="",
            )

            mock_research.return_value = [
                EvidenceReport(agent_id="r0", sub_question="Test", tokens_used=500)
            ]

            mock_debate.return_value = DebateState(
                mission_brief=mock_scout.return_value,
                is_resolved=True,
                resolution_reason="round_limit_reached",
            )

            mock_synthesis.return_value = FinalReport(
                question="Is democracy best?",
                complexity=Complexity.COMPLEX,
                rounds_completed=2,
                convergence_score=0.6,
                answer="Democracy has trade-offs",
            )

            config = CouncilConfig()
            report = await run_council("Is democracy best?", config)

            assert isinstance(report, FinalReport)
            mock_scout.assert_called_once()
            mock_research.assert_called_once()
            mock_debate.assert_called_once()
            mock_synthesis.assert_called_once()
