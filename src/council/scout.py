"""
Scout phase for Deliberative Council.

Two-agent architecture:
1. Cheap tool-augmented agent — investigates question, produces draft Mission Brief
2. Mid-tier verification agent — reviews draft, challenges assumptions, finalizes

Scout's token consumption is tracked separately from the pipeline budget.
"""

from __future__ import annotations

import json
import logging

from council.config import CouncilConfig, NLIConfig
from council.models import LLMClient, ModelRegistry, TokenBudget
from council.tools import ToolRegistry
from council.types import Complexity, MissionBrief, RoleSpec

logger = logging.getLogger(__name__)

# ── Scout Agent Prompts ──────────────────────────────────────────────────

SCOUT_SYSTEM_PROMPT = """You are a Scout Agent for a multi-AI debate system. Your job is to investigate a question and determine how hard it is, what expertise is needed, and whether research is required.

You must respond with a JSON object containing:
{
  "question": "The question, possibly rephrased for clarity",
  "complexity": "trivial" | "moderate" | "complex" | "deep",
  "domain_tags": ["list", "of", "domains"],
  "is_likely_solvable": true/false,
  "why_might_be_hard": "Explanation of difficulty",
  "suggested_roles": [
    {
      "name": "Role Name",
      "perspective": "One-line perspective",
      "expertise": "Domain of expertise",
      "suggested_model": "Model family hint (e.g. 'deepseek', 'anthropic', 'openai')",
      "system_prompt": "Full system prompt for this role",
      "is_research": false,
      "research_subquestion": null
    }
  ],
  "research_needed": true/false,
  "research_subquestions": ["sub-question 1", "sub-question 2"],
  "debate_rounds": 0-3,
  "token_budget": 50000-500000,
  "human_checkpoints": ["any points where human input would help"],
  "reasoning": "Your full reasoning trace"
}

Complexity guidelines:
- trivial: Simple factual question, single model sufficient (0 debate rounds)
- moderate: Requires debate but not research (1 round)
- complex: Needs debate + research (2 rounds)
- deep: Full pipeline with extensive research (3 rounds)

Role guidelines:
- Generate 2-4 debate roles with DIFFERENT perspectives and expertise
- Each role should approach the question from a unique angle
- Assign different model families for diversity
- Include research roles if research_needed is true
- Write detailed system prompts that guide the role's behavior

Token budget guidelines:
- trivial: 10,000-50,000
- moderate: 50,000-150,000
- complex: 150,000-350,000
- deep: 350,000-500,000

Respond ONLY with the JSON object."""

VERIFIER_SYSTEM_PROMPT = """You are a Verification Agent for a multi-AI debate system. Your job is to review a Scout's draft Mission Brief and catch any misclassifications, missing perspectives, or budget miscalibrations.

You will receive a question and a draft Mission Brief in JSON format. Review it carefully and produce the final, corrected version.

Common Scout errors to watch for:
1. Classifying trick questions as trivial (questions with misleading premises need at least 'moderate')
2. Under-budgeting for genuinely hard questions
3. Assigning too few debate roles for complex questions
4. Missing important perspectives or expertise domains
5. Not flagging potentially unsolvable questions

You must respond with a JSON object in the same format as the Scout's output, but with your corrections applied. Add any notes about your changes in the "verification_notes" field.

Respond ONLY with the JSON object."""


# ── Scout Phase ──────────────────────────────────────────────────────────


async def run_scout(
    question: str,
    client: LLMClient,
    registry: ModelRegistry,
    tools: ToolRegistry | None = None,
    config: CouncilConfig | None = None,
) -> MissionBrief:
    """Run the two-agent Scout phase.

    Args:
        question: The user's question.
        client: LLM client for making completion calls.
        registry: Model registry for model selection.
        tools: Tool registry (for search capability during scouting).
        config: Council configuration.

    Returns:
        A MissionBrief configuring the rest of the pipeline.
    """
    cfg = config or CouncilConfig()
    scout_budget = TokenBudget(total=50_000)  # Scout gets its own budget

    # Step 1: Select models for Scout agents
    scout_model = _select_scout_model(registry, cfg)
    verifier_model = _select_verifier_model(registry, cfg)

    logger.info(f"Scout using models: scout={scout_model}, verifier={verifier_model}")

    # Step 2: Optionally pre-search the question for context
    search_context = ""
    if tools:
        search_context = await _preliminary_search(question, tools)

    # Step 3: Run cheap scout agent
    scout_prompt = f"Question: {question}\n"
    if search_context:
        scout_prompt += f"\nPreliminary search results:\n{search_context}\n"

    scout_response = await client.complete(
        model_id=scout_model,
        messages=[
            {"role": "system", "content": SCOUT_SYSTEM_PROMPT},
            {"role": "user", "content": scout_prompt},
        ],
        max_tokens=4096,
        temperature=0.4,
    )
    draft_brief_text, scout_tokens = scout_response
    scout_budget.consume(scout_tokens)

    # Step 4: Run mid-tier verification agent
    verifier_prompt = (
        f"Question: {question}\n\n"
        f"Draft Mission Brief from Scout:\n{draft_brief_text}\n\n"
        f"Review and correct this Mission Brief. Pay special attention to "
        f"complexity classification and role diversity."
    )

    verifier_response = await client.complete(
        model_id=verifier_model,
        messages=[
            {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": verifier_prompt},
        ],
        max_tokens=4096,
        temperature=0.3,
    )
    final_brief_text, verifier_tokens = verifier_response
    scout_budget.consume(verifier_tokens)

    # Step 5: Parse the final Mission Brief
    brief = _parse_mission_brief(final_brief_text, question)

    # Apply complexity override if configured
    if cfg.complexity_override:
        brief.complexity = cfg.complexity_override
        # Update debate rounds for overridden complexity
        rounds_map = cfg.debate.default_rounds
        brief.debate_rounds = rounds_map.get(cfg.complexity_override.value, brief.debate_rounds)

    # Apply budget override if configured
    if cfg.budget_override:
        brief.token_budget = cfg.budget_override

    logger.info(
        f"Scout complete: complexity={brief.complexity.value}, "
        f"roles={len(brief.suggested_roles)}, "
        f"research={brief.research_needed}, "
        f"scout_tokens={scout_budget.used}"
    )

    return brief


def _select_scout_model(registry: ModelRegistry, config: CouncilConfig) -> str:
    """Select the model for the cheap scout agent."""
    # Prefer local/cheap models for scouting
    constraints = {
        "local_only": config.local_only,
        "api_only": config.api_only,
        "exclude_families": config.exclude_families,
    }

    # Try to find a cheap model
    cheap_models = registry.models_by_tier(
        __import__("council.types", fromlist=["ModelTier"]).ModelTier.CHEAP
    )
    if cheap_models:
        if config.family:
            family_match = [m for m in cheap_models if m.family == config.family]
            if family_match:
                return family_match[0].model_id
        return cheap_models[0].model_id

    # Fallback: use the first available model
    available = registry.available_models()
    if available:
        return available[0].model_id

    # Last resort
    return "openai/gpt-4.1-mini"


def _select_verifier_model(registry: ModelRegistry, config: CouncilConfig) -> str:
    """Select the model for the mid-tier verification agent."""
    from council.types import ModelTier

    mid_models = registry.models_by_tier(ModelTier.MID)
    if mid_models:
        if config.family:
            family_match = [m for m in mid_models if m.family == config.family]
            if family_match:
                return family_match[0].model_id
        return mid_models[0].model_id

    # Fallback to premium if no mid available
    premium_models = registry.models_by_tier(ModelTier.PREMIUM)
    if premium_models:
        return premium_models[0].model_id

    # Last resort
    return "openai/gpt-4.1-mini"


async def _preliminary_search(question: str, tools: ToolRegistry) -> str:
    """Run a quick search to give the scout context."""
    try:
        results = await tools.execute("web_search", query=question, max_results=3)
        if results:
            lines = []
            for r in results:
                lines.append(f"- {r.title}: {r.snippet[:200]}")
            return "\n".join(lines)
    except Exception as e:
        logger.warning(f"Preliminary search failed: {e}")
    return ""


def _parse_mission_brief(brief_text: str, original_question: str) -> MissionBrief:
    """Parse the LLM's Mission Brief JSON output into a MissionBrief model.

    Handles common formatting issues and provides robust defaults.
    """
    try:
        # Clean up the response
        cleaned = brief_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned

        data = json.loads(cleaned)

        # Map complexity string to enum
        complexity_str = data.get("complexity", "moderate").lower()
        try:
            complexity = Complexity(complexity_str)
        except ValueError:
            logger.warning(f"Unknown complexity '{complexity_str}', defaulting to moderate")
            complexity = Complexity.MODERATE

        # Parse roles
        roles = []
        for role_data in data.get("suggested_roles", []):
            try:
                roles.append(
                    RoleSpec(
                        name=role_data.get("name", "Analyst"),
                        perspective=role_data.get("perspective", ""),
                        expertise=role_data.get("expertise", "General"),
                        suggested_model=role_data.get("suggested_model", ""),
                        system_prompt=role_data.get("system_prompt", ""),
                        is_research=role_data.get("is_research", False),
                        research_subquestion=role_data.get("research_subquestion"),
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to parse role: {e}")
                continue

        # Ensure we have at least 2 debate roles for non-trivial questions
        if complexity != Complexity.TRIVIAL and len(roles) < 2:
            roles = _add_default_roles(roles, complexity)

        return MissionBrief(
            question=data.get("question", original_question),
            complexity=complexity,
            domain_tags=data.get("domain_tags", []),
            is_likely_solvable=data.get("is_likely_solvable", True),
            why_might_be_hard=data.get("why_might_be_hard", ""),
            suggested_roles=roles,
            research_needed=data.get("research_needed", False),
            research_subquestions=data.get("research_subquestions", []),
            debate_rounds=data.get("debate_rounds", 1),
            token_budget=data.get("token_budget", 100_000),
            human_checkpoints=data.get("human_checkpoints", []),
            scout_reasoning=data.get("reasoning", ""),
            verification_notes=data.get("verification_notes", ""),
        )

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse Mission Brief JSON: {e}")
        # Return a safe default
        return MissionBrief(
            question=original_question,
            complexity=Complexity.MODERATE,
            is_likely_solvable=True,
            why_might_be_hard="Scout parsing failed; defaulting to moderate",
            suggested_roles=_add_default_roles([], Complexity.MODERATE),
            research_needed=False,
            debate_rounds=1,
            token_budget=100_000,
            scout_reasoning="Failed to parse scout output; using defaults",
            verification_notes="Parse error; auto-generated defaults",
        )


def _add_default_roles(existing: list[RoleSpec], complexity: Complexity) -> list[RoleSpec]:
    """Add default debate roles if the scout didn't generate enough."""
    roles = list(existing)
    default_debate_roles = [
        RoleSpec(
            name="Analytical Expert",
            perspective="Evidence-based analytical approach",
            expertise="General analysis",
            suggested_model="openai",
            system_prompt="You are an analytical expert. Focus on evidence, logical reasoning, and clear argumentation. Address counterarguments directly. Be precise and cite specific points.",
            is_research=False,
        ),
        RoleSpec(
            name="Skeptical Reviewer",
            perspective="Critical examination of claims and assumptions",
            expertise="Critical thinking",
            suggested_model="anthropic",
            system_prompt="You are a skeptical reviewer. Your role is to challenge assumptions, identify logical fallacies, and demand evidence for claims. Do not accept arguments at face value. Propose alternative interpretations.",
            is_research=False,
        ),
        RoleSpec(
            name="Pragmatic Realist",
            perspective="Practical implications and real-world applicability",
            expertise="Applied knowledge",
            suggested_model="deepseek",
            system_prompt="You are a pragmatic realist. Focus on practical implications, real-world constraints, and actionable conclusions. Challenge theoretical arguments that don't survive contact with reality.",
            is_research=False,
        ),
    ]

    # Add enough roles for the complexity level
    target = 2 if complexity == Complexity.MODERATE else 3
    for role in default_debate_roles:
        if len(roles) >= target:
            break
        # Don't add duplicates
        if not any(r.name == role.name for r in roles):
            roles.append(role)

    return roles
