"""
Integration test that uses a live LLM model via z-ai-web-dev-sdk
to verify the Deliberative Council's Scout phase works end-to-end.

Marked with @pytest.mark.integration so it's excluded from normal runs.
Run with: pytest tests/test_live_model.py -m integration -v
"""

import json
import os
import subprocess
import tempfile
import pytest

from council.scout import _parse_mission_brief
from council.types import Complexity


def _call_live_llm(prompt: str) -> str:
    """Call the live LLM via z-ai CLI and return the content string."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        output_path = f.name

    try:
        result = subprocess.run(
            ["z-ai", "chat", "--prompt", prompt, "-o", output_path],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, f"z-ai CLI failed: {result.stderr}"

        with open(output_path) as f:
            data = json.load(f)

        return data["choices"][0]["message"]["content"]
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


class TestLiveModelScout:
    """Test Scout phase with a real LLM model."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_scout_json_parsing_with_real_llm(self):
        """Use the live LLM to produce a Scout-compatible JSON response, then parse it."""
        prompt = (
            'You are a Scout Agent for a multi-AI debate system. '
            'Classify the question "Is democracy the best form of government?" '
            'and respond with ONLY a JSON object (no markdown, no extra text) with these fields: '
            '{"question": "...", "complexity": "complex", "domain_tags": [...], '
            '"is_likely_solvable": false, "why_might_be_hard": "...", '
            '"suggested_roles": [{"name": "...", "perspective": "...", '
            '"expertise": "...", "suggested_model": "...", '
            '"system_prompt": "...", "is_research": false}], '
            '"research_needed": true, "research_subquestions": [...], '
            '"debate_rounds": 2, "token_budget": 250000, "reasoning": "..."}'
        )

        content = _call_live_llm(prompt)
        brief = _parse_mission_brief(content, "Is democracy the best form of government?")

        assert brief.question == "Is democracy the best form of government?"
        # The LLM might classify it as complex or deep; both are reasonable
        assert brief.complexity in [Complexity.COMPLEX, Complexity.DEEP, Complexity.MODERATE]
        assert len(brief.suggested_roles) >= 1

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_trivial_question_classification(self):
        """Test that a trivial question gets classified correctly."""
        prompt = (
            'You are a Scout Agent. Classify this question and respond with ONLY a JSON object: '
            '{"question": "What is 2+2?", "complexity": "trivial", '
            '"domain_tags": ["mathematics"], "is_likely_solvable": true, '
            '"why_might_be_hard": "", "suggested_roles": [], '
            '"research_needed": false, "research_subquestions": [], '
            '"debate_rounds": 0, "token_budget": 10000, "reasoning": "Simple arithmetic"}'
        )

        content = _call_live_llm(prompt)
        brief = _parse_mission_brief(content, "What is 2+2?")
        assert brief.complexity == Complexity.TRIVIAL
        assert brief.research_needed is False

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complex_question_gets_research(self):
        """Test that a complex question triggers research."""
        prompt = (
            'You are a Scout Agent. Classify this question and respond with ONLY a JSON object: '
            '{"question": "What are the latest advances in quantum error correction?", '
            '"complexity": "complex", "domain_tags": ["quantum computing", "physics"], '
            '"is_likely_solvable": true, "why_might_be_hard": "Rapidly evolving field", '
            '"suggested_roles": [{"name": "Quantum Researcher", "perspective": "Technical analysis", '
            '"expertise": "Quantum computing", "suggested_model": "deepseek", '
            '"system_prompt": "Focus on recent advances.", "is_research": false}], '
            '"research_needed": true, "research_subquestions": ["Recent QEC papers"], '
            '"debate_rounds": 2, "token_budget": 250000, "reasoning": "Requires current research"}'
        )

        content = _call_live_llm(prompt)
        brief = _parse_mission_brief(content, "What are the latest advances in quantum error correction?")
        assert brief.complexity in [Complexity.COMPLEX, Complexity.DEEP]
        assert brief.research_needed is True
