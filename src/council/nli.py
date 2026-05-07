"""
Two-tier NLI agreement system for Deliberative Council.

Tier 1: DeBERTa-v3-large-mnli — fast, free, deterministic for short text.
Tier 2: Cheap LLM — accurate analysis for long arguments, gated by Tier 1.

Also provides:
- Convergence detection (hybrid: round limit + NLI + budget)
- Position stability tracking (NLI self-comparison for DoT detection)
"""

from __future__ import annotations

import logging
import re
from typing import Any

from council.config import NLIConfig
from council.models import LLMClient, ModelRegistry
from council.types import (
    AgreementAnalysis,
    ConvergenceReason,
    ConvergenceResult,
    DebateState,
    Position,
)

logger = logging.getLogger(__name__)

# Lazy-loaded DeBERTa model and tokenizer
_deberta_model = None
_deberta_tokenizer = None
_deberta_loaded = False
_deberta_failed = False


def _load_deberta(model_name: str = "cross-encoder/nli-deberta-v3-large"):
    """Load DeBERTa model lazily (only when first needed)."""
    global _deberta_model, _deberta_tokenizer, _deberta_loaded, _deberta_failed

    if _deberta_loaded or _deberta_failed:
        return _deberta_loaded

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        logger.info(f"Loading DeBERTa model: {model_name}")
        _deberta_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _deberta_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        _deberta_model.eval()
        # Force CPU
        _deberta_model = _deberta_model.to("cpu")
        _deberta_loaded = True
        logger.info("DeBERTa model loaded successfully on CPU")
        return True
    except Exception as e:
        _deberta_failed = True
        logger.warning(f"Failed to load DeBERTa: {e}. Will use LLM fallback for Tier 1.")
        return False


def reset_deberta():
    """Reset DeBERTa state (useful for testing)."""
    global _deberta_model, _deberta_tokenizer, _deberta_loaded, _deberta_failed
    _deberta_model = None
    _deberta_tokenizer = None
    _deberta_loaded = False
    _deberta_failed = False


# ── Tier 1: DeBERTa ───────────────────────────────────────────────────


def tier1_agreement(text_a: str, text_b: str, config: NLIConfig | None = None) -> float:
    """DeBERTa-based fast agreement check. Returns score in [0, 1].

    Uses DeBERTa NLI to classify text_a (premise) vs text_b (hypothesis).
    The entailment probability serves as an agreement score.

    For long texts, chunks and averages scores.
    Falls back to a cheap LLM if DeBERTa is not available.
    """
    cfg = config or NLIConfig()

    if not _load_deberta(cfg.tier1_model):
        # Graceful degradation: return a score in the uncertain zone
        # so Tier 2 will be invoked
        logger.debug("DeBERTa unavailable, returning uncertain score for Tier 2 escalation")
        return _heuristic_agreement(text_a, text_b)

    try:
        return _deberta_score(text_a, text_b)
    except Exception as e:
        logger.warning(f"DeBERTa inference failed: {e}. Using heuristic fallback.")
        return _heuristic_agreement(text_a, text_b)


def _deberta_score(text_a: str, text_b: str) -> float:
    """Run DeBERTa inference on a pair of texts, chunking if needed."""
    import torch

    assert _deberta_model is not None
    assert _deberta_tokenizer is not None

    # Chunk long texts (token-aware, preserving sentence boundaries)
    max_tokens = 500  # Leave room for special tokens within 512 limit
    chunks_a = _chunk_text(text_a, max_tokens)
    chunks_b = _chunk_text(text_b, max_tokens)

    if len(chunks_a) == 1 and len(chunks_b) == 1:
        return _deberta_single(chunks_a[0], chunks_b[0])

    # Multiple chunks: average the scores
    scores = []
    for ca in chunks_a:
        for cb in chunks_b:
            scores.append(_deberta_single(ca, cb))

    return sum(scores) / len(scores) if scores else 0.5


def _deberta_single(text_a: str, text_b: str) -> float:
    """Run DeBERTa on a single pair of short texts."""
    import torch

    inputs = _deberta_tokenizer(
        text_a, text_b,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )

    with torch.no_grad():
        outputs = _deberta_model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)

    # NLI label ordering: [contradiction, neutral, entailment]
    # Agreement = entailment probability
    entailment_idx = _deberta_model.config.label2id.get("entailment", 2)
    return probs[0][entailment_idx].item()


def _chunk_text(text: str, max_tokens: int = 500) -> list[str]:
    """Split text into chunks by exact token count, preserving sentence boundaries.

    Uses the DeBERTa tokenizer for precise token counting when available,
    falling back to a character-based approximation otherwise.

    Args:
        text: The text to chunk.
        max_tokens: Maximum tokens per chunk (default 500, leaving room
            for special tokens within DeBERTa's 512-token limit).

    Returns:
        A list of text chunks, each within the token limit.
    """
    # Fallback to character-based chunking if tokenizer is not loaded
    if _deberta_tokenizer is None:
        return _chunk_text_by_chars(text)

    # Split text into logical sentences first
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks: list[str] = []
    current_chunk = ""
    current_tokens = 0

    for sentence in sentences:
        # Tokenize just the sentence to get its exact length
        # (add_special_tokens=False because the final model call will add them)
        sentence_tokens = len(_deberta_tokenizer.encode(sentence, add_special_tokens=False))

        # Edge case: a single sentence exceeds the limit (rare for standard text)
        if sentence_tokens > max_tokens:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                current_tokens = 0
            logger.warning(
                "Encountered a single sentence exceeding %d tokens; "
                "it will be truncated by the model.",
                max_tokens,
            )
            chunks.append(sentence)
            continue

        # Group sentences until the token limit is reached
        if current_tokens + sentence_tokens > max_tokens and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_tokens = sentence_tokens
        else:
            current_chunk = f"{current_chunk} {sentence}".strip()
            current_tokens += sentence_tokens

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks if chunks else [text]


def _chunk_text_by_chars(text: str, max_chars: int = 1800) -> list[str]:
    """Character-based fallback for chunking when tokenizer is unavailable.

    Uses 1800 chars as a reasonable approximation of ~500 tokens for
    English text (average ~3.6 chars/token).
    """
    if len(text) <= max_chars:
        return [text]

    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 > max_chars and current:
            chunks.append(current.strip())
            current = sentence
        else:
            current = current + " " + sentence if current else sentence

    if current.strip():
        chunks.append(current.strip())

    return chunks if chunks else [text[:max_chars]]


def _heuristic_agreement(text_a: str, text_b: str) -> float:
    """Simple heuristic agreement estimate for when DeBERTa is unavailable.

    Returns a score in the uncertain zone (0.4-0.5) to trigger Tier 2.
    """
    # Very basic: check for shared vocabulary
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.5

    overlap = len(words_a & words_b)
    union = len(words_a | words_b)
    jaccard = overlap / union if union > 0 else 0.0

    # Map Jaccard to uncertain zone to trigger Tier 2
    return 0.35 + jaccard * 0.25  # Range: 0.35-0.60


# ── Tier 2: LLM ───────────────────────────────────────────────────────


async def tier2_agreement(
    position_a: Position,
    position_b: Position,
    client: LLMClient,
    config: NLIConfig | None = None,
) -> AgreementAnalysis:
    """LLM-based accurate agreement analysis for long arguments.

    Takes full arguments and produces a structured agreement analysis.
    This is only invoked when Tier 1 falls in the uncertain zone.
    """
    import json

    cfg = config or NLIConfig()

    prompt = f"""Analyze whether these two debate positions substantively agree or disagree.

Position A ({position_a.role_name}):
{position_a.argument}

Position B ({position_b.role_name}):
{position_b.argument}

Respond with a JSON object with these fields:
- "substantively_agree": true if they agree on the core conclusion, false otherwise
- "agreement_score": a float from 0.0 (complete disagreement) to 1.0 (complete agreement)
- "points_of_agreement": list of specific points they agree on
- "points_of_disagreement": list of specific points they disagree on
- "is_fundamental_disagreement": true if the disagreement is about core conclusions vs. details
- "summary": one-sentence summary of the agreement/disagreement

Respond ONLY with the JSON object."""

    try:
        content, tokens = await client.complete(
            model_id=cfg.tier2_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.2,
        )

        # Parse JSON response
        cleaned = content.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned

        data = json.loads(cleaned)
        return AgreementAnalysis(**data)

    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Failed to parse Tier 2 agreement analysis: {e}")
        # Return a default uncertain analysis
        return AgreementAnalysis(
            substantively_agree=False,
            agreement_score=0.5,
            summary=f"Analysis failed: {str(e)[:100]}",
        )


# ── Combined Agreement ─────────────────────────────────────────────────


async def compute_agreement(
    position_a: Position,
    position_b: Position,
    client: LLMClient,
    config: NLIConfig | None = None,
) -> tuple[float, bool]:
    """Compute agreement between two positions using the two-tier system.

    Returns (agreement_score, tier2_was_invoked).
    """
    cfg = config or NLIConfig()

    # Always run Tier 1 first
    tier1_score = tier1_agreement(position_a.argument, position_b.argument, cfg)

    # Check if Tier 1 is confident
    if tier1_score <= cfg.tier1_uncertain_low:
        # Confident disagreement
        return tier1_score, False
    if tier1_score >= cfg.tier1_uncertain_high:
        # Confident agreement
        return tier1_score, False

    # Tier 1 is uncertain — escalate to Tier 2
    tier2_result = await tier2_agreement(position_a, position_b, client, cfg)
    return tier2_result.agreement_score, True


# ── Convergence Detection ──────────────────────────────────────────────


def check_convergence(
    debate_state: DebateState,
    config: NLIConfig | None = None,
    budget_remaining: int | None = None,
) -> ConvergenceResult:
    """Hybrid convergence check: round limit + NLI agreement + budget.

    Priority:
    1. Budget exhaustion (hard ceiling)
    2. Round limit reached (soft ceiling, but continue if disagreement is high)
    3. NLI convergence (adaptive)
    """
    cfg = config or NLIConfig()
    rounds = debate_state.rounds
    brief = debate_state.mission_brief

    # No rounds yet
    if not rounds:
        return ConvergenceResult(
            converged=False,
            reason=ConvergenceReason.DEBATE_CONTINUING,
        )

    # 1. Check budget exhaustion (hard ceiling)
    if budget_remaining is not None and budget_remaining <= 0:
        return ConvergenceResult(
            converged=True,
            reason=ConvergenceReason.BUDGET_EXHAUSTED,
            details=f"Budget exhausted after round {len(rounds)}",
        )

    # 2. Check round limit (soft ceiling)
    if len(rounds) >= brief.debate_rounds:
        # Check if there's still high disagreement — continue if so
        if rounds:
            avg_agreement = _average_agreement(rounds[-1])
            if avg_agreement < 0.5:
                return ConvergenceResult(
                    converged=False,
                    reason=ConvergenceReason.ROUND_LIMIT_BUT_DISAGREEMENT,
                    details=f"Round limit reached but avg agreement only {avg_agreement:.2f}",
                )
        return ConvergenceResult(
            converged=True,
            reason=ConvergenceReason.ROUND_LIMIT_REACHED,
            details=f"Round limit ({brief.debate_rounds}) reached",
        )

    # 3. Check NLI convergence (adaptive)
    if len(rounds) >= cfg.convergence_rounds:
        last_n_rounds = rounds[-cfg.convergence_rounds:]
        avg_scores = [_average_agreement(r) for r in last_n_rounds]

        if all(s >= cfg.convergence_threshold for s in avg_scores):
            return ConvergenceResult(
                converged=True,
                reason=ConvergenceReason.NLI_TIER1_CONVERGENCE,
                details=f"NLI agreement > {cfg.convergence_threshold} for {cfg.convergence_rounds} consecutive rounds",
            )

    return ConvergenceResult(
        converged=False,
        reason=ConvergenceReason.DEBATE_CONTINUING,
    )


def _average_agreement(round_result: Any) -> float:
    """Compute average agreement score from a round's agreement matrix."""
    matrix = round_result.agreement_matrix
    if not matrix:
        return 0.5

    scores = []
    for agent_id, others in matrix.items():
        for other_id, score in others.items():
            if agent_id < other_id:  # Avoid double-counting
                scores.append(score)

    return sum(scores) / len(scores) if scores else 0.5


# ── Position Stability ─────────────────────────────────────────────────


def compute_position_stability(
    current_position: Position,
    previous_position: Position | None,
    config: NLIConfig | None = None,
) -> float | None:
    """Compute position stability (NLI agreement between current and previous position).

    Returns None for the first round (no previous position).
    Returns a score in [0, 1] where higher = more stable (less change).
    """
    if previous_position is None:
        return None

    cfg = config or NLIConfig()
    return tier1_agreement(
        current_position.argument,
        previous_position.argument,
        cfg,
    )


def should_inject_novelty(
    position_stability_history: list[float],
    config: NLIConfig | None = None,
) -> bool:
    """Determine if novelty injection should be triggered.

    Triggers when position_stability > threshold for N consecutive rounds,
    indicating the agent is stuck on the same point.
    """
    cfg = config or NLIConfig()

    if len(position_stability_history) < cfg.position_stability_rounds:
        return False

    last_n = position_stability_history[-cfg.position_stability_rounds:]
    return all(s >= cfg.position_stability_threshold for s in last_n)
