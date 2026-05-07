"""
Devil's advocate assignment for Deliberative Council.

Core: No devil's advocate (strategy='none').
Extension stubs: Rotation strategies (rotate, weakest, random).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DevilAdvocateAssignment:
    """Result of devil's advocate assignment for a round."""

    devil_agent_id: str | None = None
    original_role: str | None = None
    devil_prompt_addition: str | None = None


def assign_devils_advocate(
    agents: list[str],
    round_num: int,
    strategy: str = "none",
    **kwargs,
) -> dict[str, DevilAdvocateAssignment]:
    """Assign devil's advocate roles for a debate round.

    Args:
        agents: List of agent IDs participating in the debate.
        round_num: Current round number (0-indexed).
        strategy: Assignment strategy.
            'none' — core: no devil's advocate assigned
            'rotate' — extension: rotate through agents each round
            'weakest' — extension: assign to agent with lowest confidence
            'random' — extension: randomly assign
        **kwargs: Additional parameters (e.g., confidence_scores for 'weakest').

    Returns:
        Dictionary mapping agent IDs to their DevilAdvocateAssignment.

    Raises:
        NotImplementedError: If strategy is not 'none' (extensions not yet built).
    """
    if strategy == "none":
        return {agent_id: DevilAdvocateAssignment() for agent_id in agents}
    elif strategy == "rotate":
        raise NotImplementedError(
            "Devil's advocate rotation is an extension not yet implemented. "
            "Use strategy='none' for the core implementation."
        )
    elif strategy == "weakest":
        raise NotImplementedError(
            "Devil's advocate weakest-link strategy is an extension not yet implemented. "
            "Use strategy='none' for the core implementation."
        )
    elif strategy == "random":
        raise NotImplementedError(
            "Devil's advocate random assignment is an extension not yet implemented. "
            "Use strategy='none' for the core implementation."
        )
    else:
        raise ValueError(
            f"Unknown devil's advocate strategy: {strategy}. "
            "Use 'none', 'rotate', 'weakest', or 'random'."
        )
