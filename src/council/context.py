"""
Context window management for Deliberative Council.

Core: Full context (all prior rounds visible to each agent).
Extension stub: Progressive context (gradually reveal information).
"""

from __future__ import annotations

import logging

from council.types import EvidenceReport

logger = logging.getLogger(__name__)


def build_agent_context(
    agent_id: str,
    round_num: int,
    graph: dict[str, list[str]],
    all_positions: dict[str, list[dict]],  # agent_id -> [{argument, role_name, round}]
    research_evidence: list[EvidenceReport],
    strategy: str = "full",
    max_context_chars: int = 80_000,
) -> str:
    """Build the context window for an agent's next debate turn.

    Args:
        agent_id: The agent receiving this context.
        round_num: Current round number (0-indexed).
        graph: Communication graph mapping agent -> visible agents.
        all_positions: All positions from prior rounds, keyed by agent_id.
        research_evidence: Available research evidence reports.
        strategy: Context strategy.
            'full' — core: all prior rounds visible
            'progressive' — extension: gradually reveal
        max_context_chars: Maximum context size in characters (approximate).

    Returns:
        Formatted context string for the agent.

    Raises:
        NotImplementedError: If strategy is 'progressive' (extension not yet built).
    """
    if strategy == "full":
        return _build_full_context(
            agent_id, round_num, graph, all_positions, research_evidence, max_context_chars
        )
    elif strategy == "progressive":
        raise NotImplementedError(
            "Progressive context is an extension not yet implemented. "
            "Use strategy='full' for the core implementation."
        )
    else:
        raise ValueError(
            f"Unknown context strategy: {strategy}. Use 'full' or 'progressive'."
        )


def _build_full_context(
    agent_id: str,
    round_num: int,
    graph: dict[str, list[str]],
    all_positions: dict[str, list[dict]],
    research_evidence: list[EvidenceReport],
    max_context_chars: int,
) -> str:
    """Build a full context with all visible prior rounds and research evidence."""
    parts: list[str] = []

    # Research evidence section
    if research_evidence:
        parts.append("## Research Evidence\n")
        for report in research_evidence:
            parts.append(f"### Research: {report.sub_question}\n")
            for finding in report.key_findings:
                tag = finding.epistemic_tag.value
                sources_str = ", ".join(s.url for s in finding.sources)
                parts.append(
                    f"- [{tag}] {finding.claim} (relevance: {finding.relevance:.2f})"
                )
                if sources_str:
                    parts.append(f"  Sources: {sources_str}")
            if report.gaps:
                parts.append(f"\nGaps: {report.gaps}")
            parts.append("")
        parts.append("")

    # Prior rounds section
    visible_agents = graph.get(agent_id, [])
    all_visible = visible_agents + [agent_id]  # Agent can see its own prior positions

    if all_positions and any(all_positions.get(a) for a in all_visible):
        parts.append("## Prior Debate Rounds\n")
        for round_idx in range(round_num):
            parts.append(f"### Round {round_idx + 1}\n")
            for other_id in all_visible:
                positions = all_positions.get(other_id, [])
                if round_idx < len(positions):
                    pos = positions[round_idx]
                    role_name = pos.get("role_name", other_id)
                    argument = pos.get("argument", "")
                    confidence = pos.get("self_confidence", "")
                    parts.append(f"**{role_name}** (confidence: {confidence}):")
                    parts.append(argument)
                    parts.append("")

    context = "\n".join(parts)

    # Truncate if over limit (simple character-based)
    if len(context) > max_context_chars:
        context = _compress_context(context, max_context_chars)

    return context


def _compress_context(context: str, max_chars: int) -> str:
    """Compress context to fit within character limit.

    Strategy: Keep the most recent rounds in full, summarize older rounds.
    For now, we use simple truncation with a note.
    """
    if len(context) <= max_chars:
        return context

    # Keep the tail (most recent content)
    truncation_note = "\n\n[Note: Earlier context was truncated to fit context window.]\n\n"
    available = max_chars - len(truncation_note)
    return truncation_note + context[-available:]
