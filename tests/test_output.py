"""Tests for council.output — output formatting."""

import json
import pytest

from council.types import (
    Complexity,
    ConsensusLevel,
    FinalReport,
    KeyPoint,
    ModelUsage,
    PipelineTrace,
)
from council.output import format_report, _to_markdown, _to_json, _to_text


def make_report() -> FinalReport:
    return FinalReport(
        question="What is 2+2?",
        complexity=Complexity.TRIVIAL,
        rounds_completed=0,
        convergence_score=1.0,
        answer="4",
        key_points=[
            KeyPoint(
                point="2+2 equals 4",
                consensus=ConsensusLevel.STRONG,
            )
        ],
        raw_markdown="# Answer\n\n4",
        pipeline_trace=PipelineTrace(
            scout_tokens=1000,
            research_tokens=0,
            debate_tokens=0,
            synthesis_tokens=500,
        ),
    )


class TestMarkdownOutput:
    def test_basic_markdown(self):
        report = make_report()
        md = _to_markdown(report)
        assert "2+2" in md
        assert "trivial" in md
        assert "STRONG" in md

    def test_uses_raw_markdown_if_available(self):
        report = make_report()
        result = format_report(report, "markdown")
        assert result == report.raw_markdown


class TestJsonOutput:
    def test_valid_json(self):
        report = make_report()
        result = _to_json(report)
        data = json.loads(result)
        assert data["question"] == "What is 2+2?"
        assert data["complexity"] == "trivial"


class TestTextOutput:
    def test_plain_text(self):
        report = make_report()
        result = _to_text(report)
        assert "2+2" in result
        # No markdown formatting
        assert "**" not in result


class TestFormatSelection:
    def test_unknown_format_raises(self):
        report = make_report()
        with pytest.raises(ValueError, match="Unknown output format"):
            format_report(report, "xml")
