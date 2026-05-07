"""Comprehensive tests for council.research — Research phase."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from council.config import CouncilConfig, ResearchConfig
from council.models import LLMClient, ModelRegistry, TokenBudget
from council.research import (
    _create_research_roles,
    _parse_research_findings,
    run_research,
)
from council.tools import ToolRegistry, SearchResult, ExtractResult
from council.types import (
    Complexity,
    EpistemicTag,
    EvidenceReport,
    EvidenceSource,
    MissionBrief,
    ModelInfo,
    ModelTier,
    ResearchFinding,
    RoleSpec,
)


# ── Test Fixtures ──────────────────────────────────────────────────────


def make_brief(
    research_needed: bool = True,
    subquestions: list[str] | None = None,
) -> MissionBrief:
    return MissionBrief(
        question="What is quantum computing?",
        complexity=Complexity.COMPLEX,
        is_likely_solvable=True,
        why_might_be_hard="Technical topic",
        suggested_roles=[],
        research_needed=research_needed,
        research_subquestions=subquestions or ["What is quantum computing?", "How does it differ from classical?"],
        debate_rounds=2,
        token_budget=200000,
        scout_reasoning="Technical question needs research",
        verification_notes="",
    )


def make_registry() -> ModelRegistry:
    models = [
        ModelInfo(model_id="ollama_chat/qwen3:8b", family="alibaba", tier=ModelTier.CHEAP, context_window=128_000, supports_local=True),
        ModelInfo(model_id="openai/gpt-4.1-mini", family="openai", tier=ModelTier.MID, context_window=128_000),
    ]
    return ModelRegistry(models)


# ── _create_research_roles Tests ────────────────────────────────────────


class TestCreateResearchRoles:
    def test_creates_one_role_per_subquestion(self):
        subquestions = ["What is X?", "How does Y work?", "Why is Z important?"]
        roles = _create_research_roles(subquestions)
        assert len(roles) == 3

    def test_roles_are_marked_as_research(self):
        roles = _create_research_roles(["Test question"])
        assert all(r.is_research for r in roles)

    def test_roles_have_subquestions(self):
        subquestions = ["What is X?", "How does Y work?"]
        roles = _create_research_roles(subquestions)
        for role, sq in zip(roles, subquestions):
            assert role.research_subquestion == sq

    def test_role_names_are_unique(self):
        roles = _create_research_roles(["Q1", "Q2", "Q3"])
        names = [r.name for r in roles]
        assert len(set(names)) == len(names)

    def test_empty_subquestions(self):
        roles = _create_research_roles([])
        assert len(roles) == 0


# ── _parse_research_findings Tests ─────────────────────────────────────


class TestParseResearchFindings:
    def test_valid_json_findings(self):
        """Parse a well-formed research findings JSON."""
        findings_json = json.dumps({
            "key_findings": [
                {
                    "claim": "Quantum computers use qubits",
                    "sources": [{"url": "https://example.com/quantum", "snippet": "Qubits are...", "title": "Quantum Basics"}],
                    "epistemic_tag": "sourced",
                    "relevance": 0.9,
                },
                {
                    "claim": "Quantum advantage is demonstrated",
                    "sources": [],
                    "epistemic_tag": "inferred",
                    "relevance": 0.7,
                },
            ],
            "gaps": "Limited practical implementations",
            "recommended_investigation": "Look into error correction",
        })

        findings, gaps, recommended = _parse_research_findings(
            findings_json, [], "strict"
        )
        assert len(findings) == 2
        assert findings[0].epistemic_tag == EpistemicTag.SOURCED
        assert findings[0].relevance == 0.9
        assert findings[1].epistemic_tag == EpistemicTag.INFERRED
        assert "Limited practical" in gaps

    def test_strict_mode_downgrades_sourced_without_url(self):
        """In strict mode, sourced findings without URLs should be downgraded."""
        findings_json = json.dumps({
            "key_findings": [
                {
                    "claim": "Some claim",
                    "sources": [],  # No URL!
                    "epistemic_tag": "sourced",
                    "relevance": 0.8,
                },
            ],
            "gaps": "",
            "recommended_investigation": "",
        })

        findings, _, _ = _parse_research_findings(findings_json, [], "strict")
        assert len(findings) == 1
        assert findings[0].epistemic_tag == EpistemicTag.INFERRED  # Downgraded

    def test_strict_mode_filters_judgment(self):
        """In strict mode, judgment-tagged findings should be filtered out."""
        findings_json = json.dumps({
            "key_findings": [
                {
                    "claim": "I think quantum is cool",
                    "sources": [],
                    "epistemic_tag": "judgment",
                    "relevance": 0.3,
                },
                {
                    "claim": "Qubits can be 0 and 1",
                    "sources": [{"url": "https://example.com", "snippet": "test", "title": "Test"}],
                    "epistemic_tag": "sourced",
                    "relevance": 0.9,
                },
            ],
            "gaps": "",
            "recommended_investigation": "",
        })

        findings, _, _ = _parse_research_findings(findings_json, [], "strict")
        assert len(findings) == 1  # Judgment filtered out
        assert findings[0].epistemic_tag == EpistemicTag.SOURCED

    def test_augmented_mode_keeps_judgment(self):
        """In augmented mode, judgment-tagged findings should be kept."""
        findings_json = json.dumps({
            "key_findings": [
                {
                    "claim": "I think quantum is cool",
                    "sources": [],
                    "epistemic_tag": "judgment",
                    "relevance": 0.3,
                },
            ],
            "gaps": "",
            "recommended_investigation": "",
        })

        findings, _, _ = _parse_research_findings(findings_json, [], "augmented")
        assert len(findings) == 1  # Judgment kept
        assert findings[0].epistemic_tag == EpistemicTag.JUDGMENT

    def test_invalid_json_returns_empty(self):
        """Invalid JSON should return empty findings with error message."""
        findings, gaps, _ = _parse_research_findings("Not JSON", [], "strict")
        assert findings == []
        assert "Failed to parse" in gaps

    def test_json_with_markdown_fences(self):
        """JSON wrapped in code fences should still parse correctly."""
        inner = json.dumps({
            "key_findings": [
                {
                    "claim": "Test claim",
                    "sources": [],
                    "epistemic_tag": "judgment",
                    "relevance": 0.5,
                },
            ],
            "gaps": "",
            "recommended_investigation": "",
        })
        wrapped = f"```json\n{inner}\n```"
        findings, _, _ = _parse_research_findings(wrapped, [], "augmented")
        assert len(findings) == 1

    def test_invalid_epistemic_tag_defaults_to_judgment(self):
        """Invalid epistemic tags should default to judgment."""
        findings_json = json.dumps({
            "key_findings": [
                {
                    "claim": "Test",
                    "sources": [],
                    "epistemic_tag": "invalid_tag",
                    "relevance": 0.5,
                },
            ],
            "gaps": "",
            "recommended_investigation": "",
        })
        findings, _, _ = _parse_research_findings(findings_json, [], "augmented")
        assert findings[0].epistemic_tag == EpistemicTag.JUDGMENT

    def test_relevance_out_of_bounds_clamped(self):
        """Relevance values outside [0,1] should be clamped."""
        findings_json = json.dumps({
            "key_findings": [
                {
                    "claim": "Test",
                    "sources": [],
                    "epistemic_tag": "judgment",
                    "relevance": 5.0,  # Way too high
                },
            ],
            "gaps": "",
            "recommended_investigation": "",
        })
        findings, _, _ = _parse_research_findings(findings_json, [], "augmented")
        assert findings[0].relevance == 1.0  # Clamped

    def test_empty_key_findings(self):
        """JSON with empty key_findings should return empty list."""
        findings_json = json.dumps({
            "key_findings": [],
            "gaps": "Nothing found",
            "recommended_investigation": "Try different keywords",
        })
        findings, gaps, recommended = _parse_research_findings(findings_json, [], "strict")
        assert findings == []
        assert "Nothing found" in gaps
        assert "different keywords" in recommended


# ── run_research Integration Tests ──────────────────────────────────────


class TestRunResearch:
    @pytest.mark.asyncio
    async def test_no_research_needed(self):
        """When research is not needed, should return empty list."""
        brief = make_brief(research_needed=False, subquestions=[])
        reg = make_registry()
        budget = TokenBudget(total=100000)
        client = LLMClient(reg, budget)
        tools = ToolRegistry()
        config = CouncilConfig()

        reports = await run_research(brief, client, reg, tools, budget, config)
        assert reports == []

    @pytest.mark.asyncio
    async def test_research_with_mocked_llm(self):
        """Test research phase with mocked search and LLM calls."""
        brief = make_brief(subquestions=["What is quantum computing?"])
        reg = make_registry()
        budget = TokenBudget(total=100000)
        client = LLMClient(reg, budget)
        config = CouncilConfig()

        # Create a mock JinaClient that returns search results
        mock_jina = MagicMock()
        search_result = SearchResult(
            url="https://example.com/quantum",
            title="Quantum Computing 101",
            snippet="Quantum computers use qubits",
            rank=1,
        )
        extract_result = ExtractResult(
            url="https://example.com/quantum",
            title="Quantum Computing 101",
            content="Quantum computing uses quantum bits (qubits)...",
            success=True,
        )
        mock_jina.search = AsyncMock(return_value=[search_result])
        mock_jina.extract = AsyncMock(return_value=extract_result)
        mock_jina.close = AsyncMock()

        tools = ToolRegistry(jina_client=mock_jina)

        research_json = json.dumps({
            "key_findings": [
                {
                    "claim": "Quantum computers use qubits",
                    "sources": [{"url": "https://example.com/quantum", "snippet": "Qubits are fundamental", "title": "Quantum Basics"}],
                    "epistemic_tag": "sourced",
                    "relevance": 0.95,
                },
            ],
            "gaps": "Limited practical examples",
            "recommended_investigation": "Look into quantum error correction",
        })

        async def mock_complete(model_id, messages, **kwargs):
            return (research_json, 300)

        with patch.object(client, 'complete', side_effect=mock_complete):
            reports = await run_research(brief, client, reg, tools, budget, config)

        assert len(reports) == 1
        assert isinstance(reports[0], EvidenceReport)
        assert reports[0].sub_question == "What is quantum computing?"
        assert len(reports[0].key_findings) == 1
        assert reports[0].key_findings[0].epistemic_tag == EpistemicTag.SOURCED

    @pytest.mark.asyncio
    async def test_research_handles_search_failure(self):
        """Research should handle search failures gracefully."""
        brief = make_brief(subquestions=["What is X?"])
        reg = make_registry()
        budget = TokenBudget(total=100000)
        client = LLMClient(reg, budget)
        config = CouncilConfig()

        # Create a mock JinaClient that fails
        mock_jina = MagicMock()
        mock_jina.search = AsyncMock(side_effect=Exception("Search service down"))
        mock_jina.extract = AsyncMock()
        mock_jina.close = AsyncMock()

        tools = ToolRegistry(jina_client=mock_jina)

        reports = await run_research(brief, client, reg, tools, budget, config)

        assert len(reports) == 1
        assert "No search results" in reports[0].gaps
