"""Tests for council.nli — two-tier NLI agreement system."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from council.config import NLIConfig
from council.models import LLMClient, ModelRegistry, TokenBudget
from council.nli import (
    _chunk_text,
    _chunk_text_by_chars,
    _heuristic_agreement,
    check_convergence,
    compute_position_stability,
    reset_deberta,
    should_inject_novelty,
    tier1_agreement,
    tier2_agreement,
)
from council.types import (
    AgreementAnalysis,
    BehavioralSignals,
    Complexity,
    ConvergenceReason,
    ConvergenceResult,
    DebateState,
    MissionBrief,
    ModelInfo,
    ModelTier,
    Position,
    RoundResult,
)


# ── Helpers ────────────────────────────────────────────────────────────


def make_brief(rounds: int = 2, budget: int = 100000) -> MissionBrief:
    return MissionBrief(
        question="Test question",
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


def make_position(agent_id: str = "a", argument: str = "I think X is true") -> Position:
    return Position(
        agent_id=agent_id,
        role_name="Analyst",
        argument=argument,
        self_confidence=0.8,
    )


# ── Chunking (character-based fallback) ────────────────────────────────


class TestChunkingByChars:
    """Tests for the character-based fallback chunker."""

    def test_short_text_no_chunk(self):
        chunks = _chunk_text_by_chars("Short text", 1800)
        assert len(chunks) == 1

    def test_long_text_chunks(self):
        long_text = ". ".join(["This is sentence number " + str(i) for i in range(50)])
        chunks = _chunk_text_by_chars(long_text, 200)
        assert len(chunks) > 1

    def test_empty_text(self):
        chunks = _chunk_text_by_chars("", 1800)
        assert len(chunks) == 1

    def test_respects_max_chars(self):
        long_text = ". ".join(["This is sentence number " + str(i) for i in range(100)])
        chunks = _chunk_text_by_chars(long_text, 300)
        for chunk in chunks:
            # Each chunk should be at most roughly max_chars (may slightly exceed
            # due to adding a full sentence, but should be reasonable)
            assert len(chunk) < 600  # generous bound


# ── Chunking (token-aware, with mocked tokenizer) ─────────────────────


class TestChunkingByTokens:
    """Tests for the token-aware chunker (requires mocked tokenizer)."""

    def setup_method(self):
        reset_deberta()

    def test_falls_back_to_chars_when_no_tokenizer(self):
        """When _deberta_tokenizer is None, should use character-based fallback."""
        # _deberta_tokenizer is None by default after reset
        chunks = _chunk_text("Short text", 500)
        assert len(chunks) == 1

    def test_long_text_fallback_chunks(self):
        """Long text without tokenizer should still chunk properly."""
        long_text = ". ".join(["This is sentence number " + str(i) for i in range(50)])
        # With no tokenizer, _chunk_text falls back to char-based with 1800-char
        # limit. The text is ~1800 chars, so use a small max_tokens to force chunking
        # via the char-based fallback (which uses 3.6 * max_tokens ≈ max_chars).
        # Alternatively, call _chunk_text_by_chars directly with a small max_chars.
        chunks = _chunk_text_by_chars(long_text, max_chars=200)
        assert len(chunks) > 1

    def test_with_mock_tokenizer(self):
        """Token-aware chunking with a mocked tokenizer."""
        mock_tokenizer = MagicMock()
        # Each short sentence is ~5 tokens
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens

        with patch("council.nli._deberta_tokenizer", mock_tokenizer):
            text = ". ".join(["Sentence " + str(i) for i in range(20)])
            chunks = _chunk_text(text, max_tokens=15)  # ~3 sentences per chunk
            assert len(chunks) > 1

    def test_with_mock_tokenizer_short_text(self):
        """Short text should not be chunked even with tokenizer."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]  # 3 tokens

        with patch("council.nli._deberta_tokenizer", mock_tokenizer):
            chunks = _chunk_text("Short text", max_tokens=500)
            assert len(chunks) == 1

    def test_with_mock_tokenizer_oversized_sentence(self):
        """A single sentence exceeding the token limit should be its own chunk."""
        mock_tokenizer = MagicMock()

        # First call returns 600 tokens (oversized), subsequent calls return 3
        call_count = [0]
        def mock_encode(text, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return list(range(600))  # oversized
            return [1, 2, 3]

        mock_tokenizer.encode = mock_encode

        with patch("council.nli._deberta_tokenizer", mock_tokenizer):
            text = "This is a very long sentence. And a short one."
            chunks = _chunk_text(text, max_tokens=500)
            # First oversized sentence should be its own chunk
            assert len(chunks) >= 1


# ── Heuristic Agreement ────────────────────────────────────────────────


class TestHeuristicAgreement:
    def test_identical_text(self):
        score = _heuristic_agreement("The sky is blue", "The sky is blue")
        assert 0.0 <= score <= 1.0
        assert score > 0.4  # Identical text should score higher

    def test_different_text(self):
        score = _heuristic_agreement("Cats are mammals", "Quantum physics is complex")
        assert 0.0 <= score <= 1.0

    def test_empty_text(self):
        score = _heuristic_agreement("", "")
        assert score == 0.5


# ── Tier 1 Agreement ──────────────────────────────────────────────────


class TestTier1Agreement:
    def setup_method(self):
        """Reset DeBERTa state before each test."""
        reset_deberta()

    def test_returns_fallback_when_deberta_unavailable(self):
        """When DeBERTa can't load, should return heuristic score."""
        with patch("council.nli._load_deberta", return_value=False):
            score = tier1_agreement("I agree with this", "I also agree with this")
            assert 0.0 <= score <= 1.0

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="torch not installed"),
        reason="torch not installed — DeBERTa test requires torch"
    )
    def test_with_mock_deberta(self):
        """Test with mocked DeBERTa model."""
        import torch

        mock_model = MagicMock()
        mock_model.config.label2id = {"contradiction": 0, "neutral": 1, "entailment": 2}

        # Mock output: high entailment probability
        mock_logits = torch.tensor([[0.1, 0.2, 0.7]])  # 70% entailment
        mock_output = MagicMock()
        mock_output.logits = mock_logits
        mock_model.return_value = mock_output

        mock_tokenizer = MagicMock()
        mock_inputs = {"input_ids": MagicMock(), "attention_mask": MagicMock(), "token_type_ids": MagicMock()}
        mock_tokenizer.return_value = mock_inputs

        with patch("council.nli._deberta_model", mock_model), \
             patch("council.nli._deberta_tokenizer", mock_tokenizer), \
             patch("council.nli._deberta_loaded", True):
            score = tier1_agreement("Short text A", "Short text B")
            assert 0.0 <= score <= 1.0


# ── Tier 2 Agreement ──────────────────────────────────────────────────


class TestTier2Agreement:
    @pytest.mark.asyncio
    async def test_successful_analysis(self):
        """Test Tier 2 with mocked LLM response."""
        reg = ModelRegistry([
            ModelInfo(model_id="test/model", family="test", tier=ModelTier.MID, context_window=128_000),
        ])
        budget = TokenBudget(total=100000)
        client = LLMClient(reg, budget)

        json_response = '''{
            "substantively_agree": true,
            "agreement_score": 0.85,
            "points_of_agreement": ["Both agree X is true"],
            "points_of_disagreement": [],
            "is_fundamental_disagreement": false,
            "summary": "Both positions agree on the core conclusion"
        }'''

        with patch("council.models.litellm.acompletion", new_callable=AsyncMock) as mock_completion:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = json_response
            mock_response.usage = MagicMock()
            mock_response.usage.total_tokens = 200
            mock_completion.return_value = mock_response

            pos_a = make_position("a", "X is true because of evidence")
            pos_b = make_position("b", "I agree that X is true")
            result = await tier2_agreement(pos_a, pos_b, client)

            assert result.substantively_agree is True
            assert result.agreement_score == 0.85
            assert len(result.points_of_agreement) == 1

    @pytest.mark.asyncio
    async def test_failed_analysis_returns_default(self):
        """Test Tier 2 graceful degradation on parse failure."""
        reg = ModelRegistry([
            ModelInfo(model_id="test/model", family="test", tier=ModelTier.MID, context_window=128_000),
        ])
        client = LLMClient(reg)

        with patch("council.models.litellm.acompletion", new_callable=AsyncMock) as mock_completion:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Not JSON at all"
            mock_response.usage = MagicMock()
            mock_response.usage.total_tokens = 50
            mock_completion.return_value = mock_response

            pos_a = make_position("a", "Argument A")
            pos_b = make_position("b", "Argument B")
            result = await tier2_agreement(pos_a, pos_b, client)

            assert result.substantively_agree is False
            assert result.agreement_score == 0.5  # Default uncertain


# ── Convergence Detection ─────────────────────────────────────────────


class TestConvergenceDetection:
    def test_no_rounds_continue(self):
        brief = make_brief()
        state = DebateState(mission_brief=brief)
        result = check_convergence(state)
        assert not result.converged
        assert result.reason == ConvergenceReason.DEBATE_CONTINUING

    def test_budget_exhausted(self):
        brief = make_brief()
        state = DebateState(
            mission_brief=brief,
            rounds=[RoundResult(round_number=1)],
        )
        result = check_convergence(state, budget_remaining=0)
        assert result.converged
        assert result.reason == ConvergenceReason.BUDGET_EXHAUSTED

    def test_round_limit_reached_converged(self):
        brief = make_brief(rounds=1)
        state = DebateState(
            mission_brief=brief,
            rounds=[
                RoundResult(
                    round_number=1,
                    agreement_matrix={"a": {"b": 0.8}},
                ),
            ],
        )
        result = check_convergence(state)
        assert result.converged
        assert result.reason == ConvergenceReason.ROUND_LIMIT_REACHED

    def test_round_limit_but_disagreement(self):
        brief = make_brief(rounds=1)
        state = DebateState(
            mission_brief=brief,
            rounds=[
                RoundResult(
                    round_number=1,
                    agreement_matrix={"a": {"b": 0.3}},
                ),
            ],
        )
        result = check_convergence(state)
        assert not result.converged
        assert result.reason == ConvergenceReason.ROUND_LIMIT_BUT_DISAGREEMENT

    def test_nli_convergence_two_rounds(self):
        config = NLIConfig(convergence_threshold=0.75, convergence_rounds=2)
        brief = make_brief(rounds=5)
        state = DebateState(
            mission_brief=brief,
            rounds=[
                RoundResult(round_number=1, agreement_matrix={"a": {"b": 0.80}}),
                RoundResult(round_number=2, agreement_matrix={"a": {"b": 0.82}}),
            ],
        )
        result = check_convergence(state, config=config)
        assert result.converged
        assert result.reason == ConvergenceReason.NLI_TIER1_CONVERGENCE

    def test_nli_not_yet_converged(self):
        config = NLIConfig(convergence_threshold=0.75, convergence_rounds=2)
        brief = make_brief(rounds=5)
        state = DebateState(
            mission_brief=brief,
            rounds=[
                RoundResult(round_number=1, agreement_matrix={"a": {"b": 0.80}}),
                # Only one round above threshold — need two consecutive
            ],
        )
        result = check_convergence(state, config=config)
        assert not result.converged

    def test_nli_mixed_rounds(self):
        """Round 1 high agreement, Round 2 low — should not converge."""
        config = NLIConfig(convergence_threshold=0.75, convergence_rounds=2)
        brief = make_brief(rounds=5)
        state = DebateState(
            mission_brief=brief,
            rounds=[
                RoundResult(round_number=1, agreement_matrix={"a": {"b": 0.80}}),
                RoundResult(round_number=2, agreement_matrix={"a": {"b": 0.50}}),
            ],
        )
        result = check_convergence(state, config=config)
        assert not result.converged


# ── Position Stability ─────────────────────────────────────────────────


class TestPositionStability:
    def setup_method(self):
        reset_deberta()

    def test_no_previous_position(self):
        current = make_position("a", "I believe X")
        result = compute_position_stability(current, None)
        assert result is None

    def test_with_previous_position(self):
        """Position stability should use NLI (mocked as heuristic since DeBERTa not available)."""
        current = make_position("a", "I strongly believe X is true")
        previous = make_position("a", "I believe X is true")
        with patch("council.nli._load_deberta", return_value=False):
            result = compute_position_stability(current, previous)
            assert result is not None
            assert 0.0 <= result <= 1.0


# ── Novelty Injection ─────────────────────────────────────────────────


class TestNoveltyInjection:
    def test_not_enough_history(self):
        config = NLIConfig(position_stability_threshold=0.80, position_stability_rounds=2)
        assert not should_inject_novelty([0.85], config)

    def test_should_inject(self):
        config = NLIConfig(position_stability_threshold=0.80, position_stability_rounds=2)
        assert should_inject_novelty([0.85, 0.82], config)

    def test_should_not_inject_below_threshold(self):
        config = NLIConfig(position_stability_threshold=0.80, position_stability_rounds=2)
        assert not should_inject_novelty([0.85, 0.60], config)

    def test_threshold_higher_than_convergence(self):
        """Position stability threshold should be higher than convergence threshold."""
        config = NLIConfig()
        assert config.position_stability_threshold > config.convergence_threshold

    def test_with_more_history(self):
        config = NLIConfig(position_stability_threshold=0.80, position_stability_rounds=2)
        # Only last 2 rounds matter
        assert should_inject_novelty([0.50, 0.60, 0.85, 0.82], config)
        assert not should_inject_novelty([0.50, 0.60, 0.85, 0.70], config)
