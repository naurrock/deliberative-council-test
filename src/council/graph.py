"""
Communication graph construction for Deliberative Council.

Core: Full graph (all agents see all others).
Extension stub: Sparse graph (CortexDebate-inspired partial connectivity).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def build_communication_graph(
    agents: list[str],
    strategy: str = "full",
    **kwargs,
) -> dict[str, list[str]]:
    """Build a communication graph determining which agents see which others.

    Args:
        agents: List of agent IDs.
        strategy: Graph strategy. 'full' is the core; 'sparse' is extension.
        **kwargs: Additional parameters for sparse graph (ignored for full).

    Returns:
        Dictionary mapping each agent ID to a list of agent IDs it can see.

    Raises:
        NotImplementedError: If strategy is 'sparse' (extension not yet built).
    """
    if strategy == "full":
        return _build_full_graph(agents)
    elif strategy == "sparse":
        raise NotImplementedError(
            "Sparse communication graph is an extension not yet implemented. "
            "Use strategy='full' for the core implementation."
        )
    else:
        raise ValueError(f"Unknown graph strategy: {strategy}. Use 'full' or 'sparse'.")


def _build_full_graph(agents: list[str]) -> dict[str, list[str]]:
    """Build a full communication graph where every agent sees every other.

    With 2-4 debaters, the token savings from sparse communication are
    negligible, while the implementation complexity and debugging difficulty
    are substantial. Full graph is simpler, more predictable, and more debuggable.
    """
    graph = {}
    for agent in agents:
        graph[agent] = [a for a in agents if a != agent]
    return graph


def validate_graph(graph: dict[str, list[str]], agents: list[str]) -> bool:
    """Validate that a communication graph is well-formed.

    Checks:
    - Every agent appears as a key
    - No agent appears in its own visibility list
    - All referenced agents exist in the agent list
    """
    agent_set = set(agents)

    for agent_id in agents:
        if agent_id not in graph:
            logger.warning(f"Agent {agent_id} missing from communication graph")
            return False

        visible = graph[agent_id]
        if agent_id in visible:
            logger.warning(f"Agent {agent_id} can see itself in communication graph")
            return False

        for v in visible:
            if v not in agent_set:
                logger.warning(
                    f"Agent {agent_id} can see unknown agent {v} in communication graph"
                )
                return False

    return True
