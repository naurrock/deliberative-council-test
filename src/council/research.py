"""
Research phase for Deliberative Council.

Deploys parallel search-and-ingest agents using Jina.ai for web content.
Each agent investigates a sub-question, produces an EvidenceReport with
epistemic tags and multi-source URL support.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Literal

from council.config import CouncilConfig, ResearchConfig
from council.models import LLMClient, ModelRegistry, TokenBudget
from council.tools import ToolRegistry
from council.types import (
    EpistemicTag,
    EvidenceReport,
    EvidenceSource,
    ResearchFinding,
    RoleSpec,
)

logger = logging.getLogger(__name__)

# ── Research Agent Prompt ────────────────────────────────────────────────

RESEARCH_AGENT_PROMPT = """You are a research agent for a multi-AI debate system. Your job is to investigate a specific sub-question thoroughly and report findings with proper epistemic tagging.

## Your Sub-Question
{sub_question}

## Research Instructions
1. Search for information relevant to your sub-question
2. Extract key findings from the search results
3. Tag each finding with its epistemic status:
   - "sourced": Directly traceable to a specific source URL. You MUST cite the URL.
   - "inferred": Synthesized from multiple sources. List all source URLs.
   - "judgment": Your own assessment with no direct external source. Use sparingly.
4. When in doubt about a tag, DOWNGRADE (e.g., if unsure between sourced and inferred, use inferred)
5. Identify gaps in your findings

## Response Format
You must respond with a JSON object:
{{
  "key_findings": [
    {{
      "claim": "The specific finding",
      "sources": [{{"url": "https://...", "snippet": "Relevant quote", "title": "Page title"}}],
      "epistemic_tag": "sourced" | "inferred" | "judgment",
      "relevance": 0.0-1.0
    }}
  ],
  "gaps": "What you could not find or verify",
  "recommended_investigation": "What deeper search might reveal"
}}

## Important Rules
- Only tag as "sourced" if you can cite a specific URL from your search results
- Do NOT fabricate URLs or sources
- Prefer "sourced" findings over "inferred" or "judgment"
- If you found nothing relevant, report that honestly in "gaps"

Respond ONLY with the JSON object."""


# ── Research Phase ───────────────────────────────────────────────────────


async def run_research(
    brief: "MissionBrief",
    client: LLMClient,
    registry: ModelRegistry,
    tools: ToolRegistry,
    budget: TokenBudget,
    config: CouncilConfig | None = None,
    research_mode: Literal["strict", "augmented"] = "strict",
) -> list[EvidenceReport]:
    """Run parallel research agents with epistemic tagging.

    Args:
        brief: The Mission Brief from the Scout phase.
        client: LLM client for making completion calls.
        registry: Model registry for model selection.
        tools: Tool registry for Jina.ai access.
        budget: Token budget for research phase.
        config: Council configuration.
        research_mode: 'strict' (only sourced findings) or 'augmented' (all tags).

    Returns:
        List of EvidenceReports, one per research sub-question.
    """
    from council.types import MissionBrief

    cfg = config or CouncilConfig()
    research_cfg = cfg.research

    if not brief.research_needed or not brief.research_subquestions:
        logger.info("No research needed for this question")
        return []

    # Calculate per-agent budget
    # NOTE: The budget passed in is already the research sub-budget
    # (engine.py applies research_share before calling us).
    # Do NOT apply research_share again here — that would double-reduce.
    num_subquestions = len(brief.research_subquestions)
    per_agent_budget = budget.total // num_subquestions

    logger.info(
        f"Starting research: {num_subquestions} sub-questions, "
        f"budget={budget.total} tokens, mode={research_mode}"
    )

    # Select research models
    research_roles = _create_research_roles(brief.research_subquestions)

    # Run agents concurrently with a semaphore
    semaphore = asyncio.Semaphore(research_cfg.max_concurrent_agents)
    reports: list[EvidenceReport] = []

    async def research_one(
        role: RoleSpec, agent_budget: TokenBudget
    ) -> EvidenceReport:
        async with semaphore:
            return await _run_research_agent(
                role=role,
                client=client,
                registry=registry,
                tools=tools,
                budget=agent_budget,
                config=cfg,
                research_mode=research_mode,
            )

    # Create per-agent budgets
    tasks = []
    for i, role in enumerate(research_roles):
        agent_budget = TokenBudget(total=per_agent_budget)
        tasks.append(research_one(role, agent_budget))

    # Run all research agents concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Research agent {i} failed: {result}")
            reports.append(
                EvidenceReport(
                    agent_id=f"research_{i}",
                    sub_question=brief.research_subquestions[i] if i < num_subquestions else "",
                    gaps=f"Research agent failed: {str(result)[:200]}",
                )
            )
        else:
            reports.append(result)

    total_tokens = sum(r.tokens_used for r in reports)
    budget.consume(total_tokens)

    logger.info(f"Research complete: {len(reports)} reports, {total_tokens} tokens used")
    return reports


def _create_research_roles(subquestions: list[str]) -> list[RoleSpec]:
    """Create research agent roles from sub-questions."""
    roles = []
    for i, sub_q in enumerate(subquestions):
        roles.append(
            RoleSpec(
                name=f"Researcher_{i}",
                perspective="Factual investigation",
                expertise="Research",
                suggested_model="",  # Let registry pick from cheap tier
                system_prompt=f"You are a research agent investigating: {sub_q}",
                is_research=True,
                research_subquestion=sub_q,
            )
        )
    return roles


async def _run_research_agent(
    role: RoleSpec,
    client: LLMClient,
    registry: ModelRegistry,
    tools: ToolRegistry,
    budget: TokenBudget,
    config: CouncilConfig,
    research_mode: str,
) -> EvidenceReport:
    """Run a single research agent.

    1. Search for relevant information using Jina.ai
    2. Extract content from top results
    3. Use LLM to synthesize findings with epistemic tags
    """
    sub_question = role.research_subquestion or "General research"
    searches_performed = []

    # Step 1: Search for relevant information
    search_results = []
    try:
        results = await tools.execute("web_search", query=sub_question)
        search_results = results or []
        searches_performed.append({
            "query": sub_question,
            "num_results": len(search_results),
        })
    except Exception as e:
        logger.warning(f"Search failed for '{sub_question}': {e}")

    # Step 2: Extract content from top results
    extracted_content = []
    for sr in search_results[:3]:  # Top 3 results
        try:
            extract_result = await tools.execute("extract_content", url=sr.url)
            if extract_result and extract_result.success:
                extracted_content.append({
                    "url": sr.url,
                    "title": sr.title or extract_result.title,
                    "content": extract_result.content[:5000],
                })
        except Exception as e:
            logger.warning(f"Extraction failed for {sr.url}: {e}")

    # Step 3: Synthesize findings using LLM
    search_context = ""
    for ec in extracted_content:
        search_context += f"\nSource: {ec['url']}\nTitle: {ec['title']}\n{ec['content'][:2000]}\n"

    # Also include raw search snippets
    if search_results:
        snippets = "\n".join(
            f"- {sr.title}: {sr.snippet[:200]} ({sr.url})"
            for sr in search_results
        )
        search_context += f"\nSearch snippets:\n{snippets}"

    if not search_context.strip():
        return EvidenceReport(
            agent_id=role.name.lower().replace(" ", "_"),
            sub_question=sub_question,
            searches_performed=searches_performed,
            gaps="No search results found for this sub-question",
            tokens_used=0,
        )

    # Select research model (cheap tier)
    from council.types import ModelTier

    research_model = None
    cheap_models = registry.models_by_tier(ModelTier.CHEAP)
    if cheap_models:
        research_model = cheap_models[0].model_id
    else:
        available = registry.available_models()
        if available:
            research_model = available[0].model_id
        else:
            raise RuntimeError("No models available for research. Configure providers.yaml with at least one model.")

    prompt = RESEARCH_AGENT_PROMPT.format(sub_question=sub_question)
    user_msg = f"{prompt}\n\n## Search Results\n{search_context}"

    try:
        response_text, tokens = await client.complete(
            model_id=research_model,
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=2048,
            temperature=0.2,
        )
        budget.consume(tokens)

        # Parse the research findings
        findings, gaps, recommended = _parse_research_findings(
            response_text, search_results, research_mode
        )

        return EvidenceReport(
            agent_id=role.name.lower().replace(" ", "_"),
            sub_question=sub_question,
            searches_performed=searches_performed,
            key_findings=findings,
            gaps=gaps,
            recommended_investigation=recommended,
            tokens_used=tokens,
        )

    except Exception as e:
        logger.error(f"Research synthesis failed: {e}")
        return EvidenceReport(
            agent_id=role.name.lower().replace(" ", "_"),
            sub_question=sub_question,
            searches_performed=searches_performed,
            gaps=f"Research synthesis failed: {str(e)[:200]}",
            tokens_used=0,
        )


def _parse_research_findings(
    response_text: str,
    search_results: list,
    research_mode: str,
) -> tuple[list[ResearchFinding], str, str]:
    """Parse the LLM's research findings JSON into structured types.

    In strict mode, sourced tags without a URL are downgraded to inferred.
    """
    try:
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned

        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse research findings JSON: {e}")
        return [], "Failed to parse research findings", ""

    findings = []
    for finding_data in data.get("key_findings", []):
        try:
            tag_str = finding_data.get("epistemic_tag", "judgment").lower()
            try:
                tag = EpistemicTag(tag_str)
            except ValueError:
                tag = EpistemicTag.JUDGMENT

            sources = []
            for src_data in finding_data.get("sources", []):
                url = src_data.get("url", "")
                if url:
                    sources.append(
                        EvidenceSource(
                            url=url,
                            snippet=src_data.get("snippet", ""),
                            title=src_data.get("title", ""),
                        )
                    )

            # Strict mode enforcement: downgrade sourced without URL
            if research_mode == "strict" and tag == EpistemicTag.SOURCED and not sources:
                tag = EpistemicTag.INFERRED
                logger.debug("Downgraded sourced finding to inferred (no URL in strict mode)")

            relevance = finding_data.get("relevance", 0.5)
            try:
                relevance = float(relevance)
            except (TypeError, ValueError):
                relevance = 0.5

            findings.append(
                ResearchFinding(
                    claim=finding_data.get("claim", ""),
                    sources=sources,
                    epistemic_tag=tag,
                    relevance=max(0.0, min(1.0, relevance)),
                )
            )
        except Exception as e:
            logger.warning(f"Failed to parse finding: {e}")
            continue

    gaps = data.get("gaps", "")
    recommended = data.get("recommended_investigation", "")

    # In strict mode, filter out judgment findings
    if research_mode == "strict":
        findings = [f for f in findings if f.epistemic_tag != EpistemicTag.JUDGMENT]

    return findings, gaps, recommended
