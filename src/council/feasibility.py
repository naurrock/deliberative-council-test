"""
Futility detection for Deliberative Council.

Two-layer approach:
1. Heuristic layer (no LLM cost) — checks for obvious futility signals
2. LLM escalation layer — when heuristics are ambiguous, asks an LLM

Futility is when the debate is making no progress despite continued rounds,
indicating that further rounds would be wasted.
"""

from __future__ import annotations

import logging

from council.models import LLMClient
from council.types import DebateState, FutilityCheck

logger = logging.getLogger(__name__)


def check_futility_heuristic(debate_state: DebateState) -> FutilityCheck:
    """Cheap heuristic futility check. No LLM call.

    Checks for:
    1. All agents have position_stability > 0.85 for 3+ consecutive rounds
       (everyone is stuck on the same point)
    2. Agreement matrix shows no change across rounds (frozen debate)
    3. Round limit exceeded with no convergence improvement

    Returns a FutilityCheck with is_futile=True if futility is detected
    with high confidence, or is_futile=False with a confidence estimate
    for how likely futility is (for LLM escalation decisions).
    """
    rounds = debate_state.rounds
    if len(rounds) < 2:
        return FutilityCheck(is_futile=False, reason="Not enough rounds for futility check", confidence=0.0)

    # Check 1: All agents stuck (high position stability across the board)
    last_round = rounds[-1]
    if last_round.behavioral_signals:
        all_stuck = True
        for agent_id, signals in last_round.behavioral_signals.items():
            stability = getattr(signals, "position_stability", None)
            if stability is not None and stability < 0.85:
                all_stuck = False
                break
        if all_stuck and len(rounds) >= 3:
            return FutilityCheck(
                is_futile=True,
                reason="All agents have position stability > 0.85 for the latest round after 3+ rounds",
                confidence=0.85,
            )

    # Check 2: Agreement matrix frozen (no change between last 2 rounds)
    if len(rounds) >= 2:
        prev_avg = _average_agreement(rounds[-2])
        curr_avg = _average_agreement(rounds[-1])
        if abs(curr_avg - prev_avg) < 0.02 and len(rounds) >= 3:
            return FutilityCheck(
                is_futile=True,
                reason=f"Agreement scores frozen: {prev_avg:.3f} -> {curr_avg:.3f} over {len(rounds)} rounds",
                confidence=0.75,
            )

    # Check 3: Position stability high for 2+ consecutive rounds for majority of agents
    if len(rounds) >= 2 and last_round.behavioral_signals:
        stuck_count = 0
        total_agents = len(last_round.behavioral_signals)
        for agent_id, signals in last_round.behavioral_signals.items():
            stability = getattr(signals, "position_stability", None)
            if stability is not None and stability >= 0.80:
                stuck_count += 1
        if total_agents > 0 and stuck_count / total_agents >= 0.75 and len(rounds) >= 2:
            # High but not certain — suggest LLM escalation
            return FutilityCheck(
                is_futile=False,
                reason=f"Majority of agents ({stuck_count}/{total_agents}) show high position stability",
                confidence=0.5,
            )

    return FutilityCheck(is_futile=False, reason="No futility signals detected", confidence=0.0)


async def check_futility_llm(
    debate_state: DebateState,
    client: LLMClient,
    model_id: str | None = None,
) -> FutilityCheck:
    """Escalate futility check to an LLM when heuristics are ambiguous.

    This is called when the heuristic check returns is_futile=False but
    with a moderate confidence (0.3-0.7), suggesting possible futility
    that needs human-like judgment.
    """
    import json

    rounds = debate_state.rounds
    if not rounds:
        return FutilityCheck(is_futile=False, reason="No rounds to check")

    # Build a summary of the debate for the LLM
    debate_summary = _build_debate_summary(debate_state)

    prompt = f"""Analyze this debate and determine if it has reached futility — meaning further rounds are unlikely to produce progress.

{debate_summary}

Consider:
1. Are agents making the same arguments repeatedly without engaging with counterpoints?
2. Has the agreement level plateaued?
3. Are there genuinely irreconcilable differences that more rounds won't resolve?

Respond with a JSON object:
- "is_futile": true if further debate rounds are unlikely to help
- "reason": one-sentence explanation
- "confidence": float 0.0-1.0 how certain you are

Respond ONLY with the JSON object."""

    try:
        content, tokens = await client.complete(
            model_id=model_id or "openai/gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.2,
        )

        cleaned = content.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned

        data = json.loads(cleaned)
        return FutilityCheck(
            is_futile=bool(data.get("is_futile", False)),
            reason=str(data.get("reason", "")),
            confidence=float(data.get("confidence", 0.5)),
        )

    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Failed to parse LLM futility check: {e}")
        return FutilityCheck(
            is_futile=False,
            reason=f"LLM futility check failed: {str(e)[:100]}",
            confidence=0.0,
        )


def _average_agreement(round_result) -> float:
    """Compute average agreement from a round's agreement matrix."""
    matrix = round_result.agreement_matrix
    if not matrix:
        return 0.5

    scores = []
    for agent_id, others in matrix.items():
        for other_id, score in others.items():
            if agent_id < other_id:
                scores.append(score)

    return sum(scores) / len(scores) if scores else 0.5


def _build_debate_summary(debate_state: DebateState) -> str:
    """Build a concise summary of the debate for LLM futility check."""
    lines = []
    lines.append(f"Question: {debate_state.mission_brief.question}")
    lines.append(f"Rounds completed: {len(debate_state.rounds)}")
    lines.append(f"Max rounds: {debate_state.mission_brief.debate_rounds}")
    lines.append("")

    for round_result in debate_state.rounds[-3:]:  # Last 3 rounds
        lines.append(f"Round {round_result.round_number}:")
        for pos in round_result.positions:
            stability = ""
            if pos.position_stability is not None:
                stability = f" [stability: {pos.position_stability:.2f}]"
            lines.append(
                f"  {pos.role_name} (confidence: {pos.self_confidence:.2f}){stability}: "
                f"{pos.argument[:200]}..."
            )
        avg_agree = _average_agreement(round_result)
        lines.append(f"  Average agreement: {avg_agree:.3f}")
        lines.append("")

    return "\n".join(lines)
