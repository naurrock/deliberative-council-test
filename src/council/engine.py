"""
Main orchestrator for Deliberative Council.

Orchestrates the full pipeline: Scout -> Research -> Debate -> Synthesis.
Handles error recovery, state management, and checkpoint/resume.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from council.config import CouncilConfig, load_config
from council.debate import run_debate
from council.models import LLMClient, ModelRegistry, TokenBudget
from council.research import run_research
from council.scout import run_scout
from council.synthesis import run_synthesis
from council.tools import ToolRegistry
from council.types import (
    Complexity,
    DebateState,
    EvidenceReport,
    FinalReport,
    MissionBrief,
    PipelineTrace,
)

logger = logging.getLogger(__name__)


# ── Checkpoint Manager ───────────────────────────────────────────────────


class CheckpointManager:
    """Manages pipeline checkpoints for crash recovery and resume.

    Saves intermediate state after each phase to both SQLite and
    file-based checkpoints.
    """

    def __init__(self, run_id: str, db_path: str | None = None):
        self.run_id = run_id
        self.db_path = db_path or str(Path.cwd() / ".council" / "checkpoints.db")
        self._ensure_db()

    def _ensure_db(self) -> None:
        """Ensure the checkpoint database exists."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT,
                    brief_json TEXT,
                    evidence_json TEXT,
                    debate_state_json TEXT,
                    report_json TEXT,
                    phase TEXT
                )
            """)

    def save_brief(self, brief: MissionBrief) -> None:
        """Save the Mission Brief checkpoint."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO checkpoints (run_id, created_at, brief_json, phase) VALUES (?, ?, ?, ?)",
                (self.run_id, datetime.now().isoformat(), brief.model_dump_json(), "scout"),
            )

    def save_evidence(self, evidence: list[EvidenceReport]) -> None:
        """Save research evidence checkpoint."""
        evidence_json = json.dumps([e.model_dump() for e in evidence])
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE checkpoints SET evidence_json = ?, phase = ? WHERE run_id = ?",
                (evidence_json, "research", self.run_id),
            )

    def save_debate_state(self, state: DebateState) -> None:
        """Save debate state checkpoint."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE checkpoints SET debate_state_json = ?, phase = ? WHERE run_id = ?",
                (state.model_dump_json(), "debate", self.run_id),
            )

    def load_checkpoint(self) -> dict[str, Any] | None:
        """Load the latest checkpoint for this run."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT brief_json, evidence_json, debate_state_json, report_json, phase FROM checkpoints WHERE run_id = ?",
                (self.run_id,),
            ).fetchone()

        if not row:
            return None

        return {
            "brief_json": row[0],
            "evidence_json": row[1],
            "debate_state_json": row[2],
            "report_json": row[3],
            "phase": row[4],
        }

    def get_phase(self) -> str | None:
        """Get the last completed phase."""
        cp = self.load_checkpoint()
        return cp["phase"] if cp else None


# ── Engine ───────────────────────────────────────────────────────────────


async def run_council(
    question: str,
    config: CouncilConfig | None = None,
    run_id: str | None = None,
) -> FinalReport:
    """Main entry point. Orchestrates the full pipeline with checkpoint/resume.

    Pipeline flow:
    1. Scout → MissionBrief
    2. Research → list[EvidenceReport] (if needed)
    3. Debate → DebateState
    4. Synthesis → FinalReport

    Args:
        question: The user's question.
        config: Council configuration. If None, uses defaults.
        run_id: Optional run ID for checkpoint tracking.

    Returns:
        A FinalReport with the synthesized answer and full pipeline trace.
    """
    import uuid

    cfg = config or load_config()
    rid = run_id or str(uuid.uuid4())[:8]

    logger.info(f"Starting Deliberative Council run {rid}")
    logger.info(f"Question: {question}")

    # Initialize components
    registry = ModelRegistry.from_config(cfg)
    tools = ToolRegistry()
    budget = TokenBudget(total=cfg.budget_override or cfg.budget.default_budget)

    # Detach global budget to prevent double-counting across phases.
    # Each phase (Scout, Research, Debate) tracks its own sub-budget
    # and reports consumption back to the main budget manually.
    client = LLMClient(registry, budget=None)

    checkpoint = CheckpointManager(rid)

    # Check for existing checkpoint (resume support)
    existing_phase = checkpoint.get_phase() if cfg.resume else None

    # ── Phase 1: Scout ──────────────────────────────────────────────────
    brief: MissionBrief | None = None
    if existing_phase and existing_phase in ("scout", "research", "debate", "synthesis"):
        cp = checkpoint.load_checkpoint()
        if cp and cp["brief_json"]:
            brief = MissionBrief.model_validate_json(cp["brief_json"])
            logger.info(f"Resumed from checkpoint: scout phase already complete")

    if brief is None:
        logger.info("Phase 1: Scout")
        brief = await run_scout(question, client, registry, tools, cfg)
        checkpoint.save_brief(brief)
        logger.info(f"Scout complete: complexity={brief.complexity.value}")

    # For trivial questions, skip to single-model synthesis
    if brief.complexity == Complexity.TRIVIAL:
        logger.info("Trivial question — skipping to synthesis")
        debate_state = DebateState(mission_brief=brief)
        report = await run_synthesis(brief, [], debate_state, client, registry, cfg)
        report.pipeline_trace.scout_tokens = budget.used
        await tools.close()
        return report

    # ── Phase 2: Research ───────────────────────────────────────────────
    evidence: list[EvidenceReport] = []
    if existing_phase and existing_phase in ("research", "debate", "synthesis"):
        cp = checkpoint.load_checkpoint()
        if cp and cp["evidence_json"]:
            evidence_data = json.loads(cp["evidence_json"])
            evidence = [EvidenceReport(**e) for e in evidence_data]
            logger.info(f"Resumed from checkpoint: research already complete ({len(evidence)} reports)")

    if not evidence and brief.research_needed:
        logger.info("Phase 2: Research")
        # Create a separate research budget
        research_budget = TokenBudget(
            total=int(budget.total * cfg.budget.research_share)
        )
        evidence = await run_research(brief, client, registry, tools, research_budget, cfg)
        budget.consume(research_budget.used)
        checkpoint.save_evidence(evidence)
        logger.info(f"Research complete: {len(evidence)} reports")
    elif not brief.research_needed:
        logger.info("Research not needed — skipping")

    # ── Phase 3: Debate ─────────────────────────────────────────────────
    debate_state: DebateState | None = None
    if existing_phase and existing_phase in ("debate", "synthesis"):
        cp = checkpoint.load_checkpoint()
        if cp and cp["debate_state_json"]:
            debate_state = DebateState.model_validate_json(cp["debate_state_json"])
            logger.info(f"Resumed from checkpoint: debate already complete")

    if debate_state is None or not debate_state.is_resolved:
        logger.info("Phase 3: Debate")
        debate_budget = TokenBudget(
            total=int(budget.total * cfg.budget.debate_share)
        )
        debate_state = await run_debate(
            brief=brief,
            evidence=evidence,
            client=client,
            registry=registry,
            budget=debate_budget,
            config=cfg,
            prior_state=debate_state,
            graph_strategy=cfg.debate.graph_strategy,
            debate_strategy=cfg.debate.debate_strategy,
            context_strategy=cfg.debate.context_strategy,
        )
        budget.consume(debate_budget.used)
        checkpoint.save_debate_state(debate_state)
        logger.info(f"Debate complete: {len(debate_state.rounds)} rounds")
    else:
        logger.info("Debate already resolved from checkpoint")

    # ── Phase 4: Synthesis ──────────────────────────────────────────────
    logger.info("Phase 4: Synthesis")
    report = await run_synthesis(brief, evidence, debate_state, client, registry, cfg)

    # Update pipeline trace with scout tokens
    report.pipeline_trace.scout_tokens = budget.used - (
        report.pipeline_trace.research_tokens
        + report.pipeline_trace.debate_tokens
        + report.pipeline_trace.synthesis_tokens
    )

    # Save final report
    await tools.close()

    logger.info(
        f"Council run {rid} complete. "
        f"Complexity={brief.complexity.value}, "
        f"Rounds={len(debate_state.rounds)}, "
        f"Total tokens={report.pipeline_trace.total_tokens}"
    )

    return report
