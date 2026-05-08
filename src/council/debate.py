"""
Debate phase for Deliberative Council.

Orchestrates debate rounds with:
- Full communication graph (core)
- NLI agreement tracking (two-tier)
- Position stability tracking (for DoT detection)
- Novelty injection (when agents get stuck)
- Futility detection (heuristic + LLM escalation)
- Convergence detection (round limit + NLI + budget)
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Literal

from council.config import CouncilConfig, NLIConfig
from council.context import build_agent_context
from council.devil_advocate import assign_devils_advocate
from council.feasibility import check_futility_heuristic, check_futility_llm
from council.graph import build_communication_graph
from council.models import LLMClient, ModelRegistry, TokenBudget
from council.nli import compute_agreement, compute_position_stability, should_inject_novelty, check_convergence
from council.types import (
    BehavioralSignals,
    ConvergenceReason,
    Critique,
    DebateState,
    EvidenceReport,
    MissionBrief,
    Position,
    RoundResult,
)

logger = logging.getLogger(__name__)


# ── Debate Prompts ───────────────────────────────────────────────────────

DEBATE_POSITION_PROMPT = """You are {role_name}. Your perspective: {perspective}
Your expertise: {expertise}

{system_prompt}

## Your Task
You are participating in a structured debate on the following question:
**{question}**

{context}

## Instructions
Present your position on this question. You must:
1. State your argument clearly
2. Cite specific evidence (from research or your knowledge)
3. Rate your confidence (0.0-1.0)
4. Answer: What would change your position?

{novelty_instruction}

Respond with a JSON object:
{{
  "argument": "Your argument",
  "supporting_evidence": ["evidence item 1", "evidence item 2"],
  "self_confidence": 0.0-1.0,
  "metacognitive_notes": "What would change your position"
}}"""

DEBATE_CRITIQUE_PROMPT = """You are {role_name} in a debate. You've just presented your position, and now you need to critique another debater's argument.

## Your Position (for reference)
{your_argument}

## Their Position
{their_argument}

## Instructions
Critique their argument. You must:
1. Identify points of agreement
2. Identify points of disagreement
3. Provide counter-arguments
4. Present any new evidence they haven't considered

Respond with a JSON object:
{{
  "points_of_agreement": ["point 1", "point 2"],
  "points_of_disagreement": ["point 1", "point 2"],
  "counter_arguments": ["counter 1", "counter 2"],
  "new_evidence": ["evidence 1"]
}}"""

NOVELTY_INJECTION_PROMPT = """

## ⚠️ Novelty Injection
You have been making very similar arguments for multiple rounds. You must now:
- Approach the question from a completely different angle
- Introduce at least one new argument or piece of evidence you haven't mentioned before
- Challenge your own previous assumptions
- Consider what a critic of YOUR position would say
Do NOT simply repeat your previous arguments."""


# ── Debate Phase ─────────────────────────────────────────────────────────


async def run_debate(
    brief: MissionBrief,
    evidence: list[EvidenceReport],
    client: LLMClient,
    registry: ModelRegistry,
    budget: TokenBudget,
    prior_state: DebateState | None = None,
    config: CouncilConfig | None = None,
    graph_strategy: str = "full",
    debate_strategy: str = "none",
    context_strategy: str = "full",
) -> DebateState:
    """Run debate rounds until convergence, futility, or budget exhaustion.

    Args:
        brief: The Mission Brief from the Scout phase.
        evidence: Research evidence reports.
        client: LLM client for making completion calls.
        registry: Model registry for model selection.
        budget: Token budget for debate phase.
        prior_state: Previous debate state (for resume).
        config: Council configuration.
        graph_strategy: Communication graph strategy.
        debate_strategy: Devil's advocate strategy.
        context_strategy: Context window strategy.

    Returns:
        Updated DebateState after all rounds complete.
    """
    cfg = config or CouncilConfig()
    nli_config = cfg.nli

    # Initialize or resume debate state
    state = prior_state or DebateState(
        mission_brief=brief,
        evidence_reports=evidence,
    )

    # Set up communication graph — only non-research roles debate
    non_research_roles = [r for r in brief.suggested_roles if not r.is_research]
    agent_ids = [f"debater_{i}" for i in range(len(non_research_roles))]

    if not state.communication_graph:
        state.communication_graph = build_communication_graph(agent_ids, strategy=graph_strategy)

    # Select models for each debater
    agent_models = _assign_models_to_agents(non_research_roles, registry, cfg)

    # Track position stability history per agent
    stability_history: dict[str, list[float]] = {
        aid: [] for aid in agent_ids
    }

    # Run debate rounds
    max_rounds = brief.debate_rounds
    starting_round = len(state.rounds)

    for round_num in range(starting_round, max_rounds):
        # Check budget before starting a new round
        if budget.is_exhausted:
            logger.info("Budget exhausted, ending debate")
            state.is_resolved = True
            state.resolution_reason = ConvergenceReason.BUDGET_EXHAUSTED.value
            break

        logger.info(f"Starting debate round {round_num + 1}/{max_rounds}")

        # Assign devil's advocate roles
        devil_assignments = assign_devils_advocate(
            agent_ids, round_num, strategy=debate_strategy
        )

        # Run one debate round
        round_result = await _run_debate_round(
            round_num=round_num,
            brief=brief,
            evidence=evidence,
            state=state,
            client=client,
            registry=registry,
            budget=budget,
            agent_models=agent_models,
            non_research_roles=non_research_roles,
            nli_config=nli_config,
            graph_strategy=graph_strategy,
            context_strategy=context_strategy,
            stability_history=stability_history,
            devil_assignments=devil_assignments,
        )

        state.rounds.append(round_result)

        # Update stability history
        for agent_id, signals in round_result.behavioral_signals.items():
            if signals.position_stability is not None:
                stability_history[agent_id].append(signals.position_stability)

        # Check convergence
        convergence = check_convergence(state, nli_config, budget.remaining)
        if convergence.converged and convergence.reason != ConvergenceReason.ROUND_LIMIT_BUT_DISAGREEMENT:
            logger.info(f"Debate converged: {convergence.reason.value} — {convergence.details}")
            state.is_resolved = True
            state.resolution_reason = convergence.reason.value
            break

        # Check futility (heuristic first, then LLM escalation)
        futility = check_futility_heuristic(state)
        if futility.is_futile:
            logger.info(f"Futility detected: {futility.reason}")
            state.futility_flags.append(f"Futility round {round_num + 1}: {futility.reason}")
            state.is_resolved = True
            state.resolution_reason = ConvergenceReason.FUTILITY_DETECTED.value
            break
        elif futility.confidence >= 0.3:
            # Heuristic is uncertain — escalate to LLM
            # Use first available mid-tier model for futility check
            from council.types import ModelTier
            futility_model = None
            mid_models = registry.models_by_tier(ModelTier.MID)
            if mid_models:
                futility_model = mid_models[0].model_id
            else:
                available = registry.available_models()
                futility_model = available[0].model_id if available else None
            if futility_model is None:
                logger.warning("No model available for LLM futility check, skipping")
                continue
            futility_llm = await check_futility_llm(state, client, model_id=futility_model)
            if futility_llm.is_futile:
                logger.info(f"LLM futility check: {futility_llm.reason}")
                state.futility_flags.append(
                    f"LLM futility round {round_num + 1}: {futility_llm.reason}"
                )
                state.is_resolved = True
                state.resolution_reason = ConvergenceReason.FUTILITY_DETECTED.value
                break

    # If we exhausted rounds without resolution
    if not state.is_resolved:
        state.is_resolved = True
        state.resolution_reason = ConvergenceReason.ROUND_LIMIT_REACHED.value

    return state


async def _run_debate_round(
    round_num: int,
    brief: MissionBrief,
    evidence: list[EvidenceReport],
    state: DebateState,
    client: LLMClient,
    registry: ModelRegistry,
    budget: TokenBudget,
    agent_models: dict[str, str],
    non_research_roles: list[RoleSpec],
    nli_config: NLIConfig,
    graph_strategy: str,
    context_strategy: str,
    stability_history: dict[str, list[float]],
    devil_assignments: dict[str, Any],
) -> RoundResult:
    """Execute a single debate round."""
    agent_ids = list(agent_models.keys())

    # Build context for each agent
    all_positions = _collect_prior_positions(state)

    # Step 1: Generate positions for all agents concurrently
    positions: list[Position] = []
    position_map: dict[str, Position] = {}

    async def _generate_one_position(
        agent_id: str, role: RoleSpec
    ) -> Position:
        """Generate a single agent's position (for concurrent execution)."""
        # Check if novelty injection is needed
        novelty_instruction = ""
        agent_stability = stability_history.get(agent_id, [])
        if should_inject_novelty(agent_stability, nli_config):
            novelty_instruction = NOVELTY_INJECTION_PROMPT

        # Build context
        context = build_agent_context(
            agent_id=agent_id,
            round_num=round_num,
            graph=state.communication_graph,
            all_positions=all_positions,
            research_evidence=evidence,
            strategy=context_strategy,
        )

        # Generate position
        prompt = DEBATE_POSITION_PROMPT.format(
            role_name=role.name,
            perspective=role.perspective,
            expertise=role.expertise,
            system_prompt=role.system_prompt,
            question=brief.question,
            context=context,
            novelty_instruction=novelty_instruction,
        )

        try:
            response_text, tokens = await client.complete(
                model_id=agent_models[agent_id],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.7,
            )
            budget.consume(tokens)
            return _parse_position(response_text, agent_id, role.name)

        except Exception as e:
            logger.error(f"Failed to generate position for {agent_id}: {e}")
            return Position(
                agent_id=agent_id,
                role_name=role.name,
                argument=f"[Position generation failed: {str(e)[:100]}]",
                self_confidence=0.5,
            )

    # Run all position generations concurrently
    position_results = await asyncio.gather(
        *[
            _generate_one_position(aid, role)
            for aid, role in zip(agent_ids, non_research_roles)
        ]
    )
    for pos in position_results:
        positions.append(pos)
        position_map[pos.agent_id] = pos

    # Step 2: Compute position stability
    if round_num > 0 and state.rounds:
        prev_round = state.rounds[-1]
        for pos in positions:
            prev_pos = _find_previous_position(pos.agent_id, prev_round)
            stability = compute_position_stability(pos, prev_pos, nli_config)
            pos.position_stability = stability

    # Step 3: Generate critiques (each agent critiques visible others)
    critiques: list[Critique] = []
    for agent_id in agent_ids:
        if agent_id not in position_map:
            continue
        visible = state.communication_graph.get(agent_id, [])
        for other_id in visible:
            if other_id not in position_map:
                continue

            try:
                role = _get_role_for_agent(agent_id, non_research_roles, agent_ids)
                critique = await _generate_critique(
                    agent_id=agent_id,
                    other_id=other_id,
                    role=role,
                    own_position=position_map[agent_id],
                    other_position=position_map[other_id],
                    client=client,
                    model_id=agent_models[agent_id],
                    budget=budget,
                )
                critiques.append(critique)
            except Exception as e:
                logger.warning(f"Critique generation failed {agent_id}->{other_id}: {e}")

    # Step 4: Compute agreement matrix (NLI two-tier)
    # Initialize all top-level keys first to prevent overwrite when
    # iterating through subsequent agents.
    agreement_matrix: dict[str, dict[str, float]] = {aid: {} for aid in agent_ids}
    nli_tier2_invoked: dict[str, dict[str, bool]] = {aid: {} for aid in agent_ids}

    for i, aid in enumerate(agent_ids):
        for j, bid in enumerate(agent_ids):
            if i >= j:
                continue
            if aid not in position_map or bid not in position_map:
                continue

            try:
                score, tier2_used = await compute_agreement(
                    position_map[aid], position_map[bid], client, nli_config
                )
                agreement_matrix[aid][bid] = score
                agreement_matrix[bid][aid] = score
                nli_tier2_invoked[aid][bid] = tier2_used
                nli_tier2_invoked[bid][aid] = tier2_used
            except Exception as e:
                logger.warning(f"NLI agreement failed {aid}<->{bid}: {e}")
                agreement_matrix[aid][bid] = 0.5
                agreement_matrix[bid][aid] = 0.5
                nli_tier2_invoked[aid][bid] = False
                nli_tier2_invoked[bid][aid] = False

    # Step 5: Compute behavioral signals
    behavioral_signals: dict[str, BehavioralSignals] = {}
    for agent_id in agent_ids:
        pos = position_map.get(agent_id)
        if pos:
            stability = pos.position_stability
            novelty_triggered = should_inject_novelty(
                stability_history.get(agent_id, []), nli_config
            )
            behavioral_signals[agent_id] = BehavioralSignals(
                position_stability=stability,
                novelty_injection_triggered=novelty_triggered,
            )

    return RoundResult(
        round_number=round_num + 1,
        positions=positions,
        critiques=critiques,
        agreement_matrix=agreement_matrix,
        nli_tier2_invoked=nli_tier2_invoked,
        behavioral_signals=behavioral_signals,
    )


def _assign_models_to_agents(
    roles: list[RoleSpec], registry: ModelRegistry, config: CouncilConfig
) -> dict[str, str]:
    """Assign models to each debater, maximizing family diversity."""
    agent_models = {}
    assigned_families: list[str] = []

    for i, role in enumerate(roles):
        agent_id = f"debater_{i}"
        model_info = registry.select_model_for_role(
            role,
            family=config.family or None,
            exclude_families=config.exclude_families or None,
            local_only=config.local_only,
            api_only=config.api_only,
            model_override=config.model_overrides.get(agent_id),
            already_assigned_families=assigned_families,
        )
        if model_info:
            agent_models[agent_id] = model_info.model_id
            assigned_families.append(model_info.family)
        else:
            # No model found for this debater — raise error
            raise RuntimeError(
                f"No model available for debater {agent_id}. "
                f"Configure providers.yaml with at least one model."
            )

    return agent_models


def _collect_prior_positions(state: DebateState) -> dict[str, list[dict]]:
    """Collect all prior positions for context building."""
    all_positions: dict[str, list[dict]] = {}
    for round_result in state.rounds:
        for pos in round_result.positions:
            if pos.agent_id not in all_positions:
                all_positions[pos.agent_id] = []
            all_positions[pos.agent_id].append({
                "argument": pos.argument,
                "role_name": pos.role_name,
                "self_confidence": pos.self_confidence,
            })
    return all_positions


def _find_previous_position(agent_id: str, round_result: RoundResult) -> Position | None:
    """Find an agent's position from a previous round."""
    for pos in round_result.positions:
        if pos.agent_id == agent_id:
            return pos
    return None


def _get_role_for_agent(agent_id: str, roles: list[RoleSpec], agent_ids: list[str]) -> RoleSpec | None:
    """Get the RoleSpec for a given agent ID."""
    try:
        idx = agent_ids.index(agent_id)
        return roles[idx] if idx < len(roles) else None
    except ValueError:
        return None


def _parse_position(response_text: str, agent_id: str, role_name: str) -> Position:
    """Parse an LLM position response into a Position object."""
    try:
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned

        data = json.loads(cleaned)

        confidence = data.get("self_confidence", 0.5)
        try:
            confidence = float(confidence)
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.5

        return Position(
            agent_id=agent_id,
            role_name=role_name,
            argument=data.get("argument", response_text[:500]),
            supporting_evidence=data.get("supporting_evidence", []),
            self_confidence=confidence,
            metacognitive_notes=data.get("metacognitive_notes", ""),
        )

    except json.JSONDecodeError:
        # If JSON parsing fails, use the raw text as the argument
        return Position(
            agent_id=agent_id,
            role_name=role_name,
            argument=response_text[:1000],
            self_confidence=0.5,
        )


async def _generate_critique(
    agent_id: str,
    other_id: str,
    role: RoleSpec | None,
    own_position: Position,
    other_position: Position,
    client: LLMClient,
    model_id: str,
    budget: TokenBudget,
) -> Critique:
    """Generate a critique from one agent to another."""
    prompt = DEBATE_CRITIQUE_PROMPT.format(
        role_name=own_position.role_name,
        your_argument=own_position.argument[:1000],
        their_argument=other_position.argument[:1000],
    )

    response_text, tokens = await client.complete(
        model_id=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.5,
    )
    budget.consume(tokens)

    try:
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned

        data = json.loads(cleaned)

        return Critique(
            from_agent=agent_id,
            to_agent=other_id,
            points_of_agreement=data.get("points_of_agreement", []),
            points_of_disagreement=data.get("points_of_disagreement", []),
            counter_arguments=data.get("counter_arguments", []),
            new_evidence=data.get("new_evidence", []),
        )

    except json.JSONDecodeError:
        return Critique(
            from_agent=agent_id,
            to_agent=other_id,
            counter_arguments=[response_text[:500]],
        )
