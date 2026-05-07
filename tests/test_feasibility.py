"""Tests for council.feasibility — futility detection."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from council.feasibility import check_futility_heuristic
from council.types import (
    BehavioralSignals,
    Complexity,
    ConvergenceReason,
    DebateState,
    MissionBrief,
    Position,
    RoundResult,
)


def make_brief(rounds: int = 3, budget: int = 100000) -> MissionBrief:
    return MissionBrief(
        question="Test",
        complexity=Complexity.MODERATE,
        is_likely_solvable=True,
        why_might_be_hard="",
        suggested_roles=[],
        research_needed=False,
        debate_rounds=rounds,
        token_budget=budget,
        scout_reasoning="",
        verification_notes="",
    )


class TestHeuristicFutility:
    def test_not_enough_rounds(self):
        brief = make_brief()
        state = DebateState(mission_brief=brief)
        result = check_futility_heuristic(state)
        assert not result.is_futile

    def test_all_agents_stuck(self):
        brief = make_brief(rounds=5)
        signals = {
            "a": BehavioralSignals(position_stability=0.90),
            "b": BehavioralSignals(position_stability=0.88),
        }
        rounds = [
            RoundResult(round_number=1, agreement_matrix={"a": {"b": 0.5}}),
            RoundResult(round_number=2, agreement_matrix={"a": {"b": 0.5}}),
            RoundResult(round_number=3, agreement_matrix={"a": {"b": 0.5}}, behavioral_signals=signals),
        ]
        state = DebateState(mission_brief=brief, rounds=rounds)
        result = check_futility_heuristic(state)
        assert result.is_futile

    def test_agents_making_progress(self):
        brief = make_brief(rounds=5)
        signals = {
            "a": BehavioralSignals(position_stability=0.60),
            "b": BehavioralSignals(position_stability=0.55),
        }
        rounds = [
            RoundResult(round_number=1, agreement_matrix={"a": {"b": 0.3}}),
            RoundResult(round_number=2, agreement_matrix={"a": {"b": 0.6}}, behavioral_signals=signals),
        ]
        state = DebateState(mission_brief=brief, rounds=rounds)
        result = check_futility_heuristic(state)
        assert not result.is_futile

    def test_frozen_agreement(self):
        brief = make_brief(rounds=5)
        rounds = [
            RoundResult(round_number=1, agreement_matrix={"a": {"b": 0.50}}),
            RoundResult(round_number=2, agreement_matrix={"a": {"b": 0.51}}),
            RoundResult(round_number=3, agreement_matrix={"a": {"b": 0.50}}),
        ]
        state = DebateState(mission_brief=brief, rounds=rounds)
        result = check_futility_heuristic(state)
        assert result.is_futile
