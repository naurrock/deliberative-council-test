"""
CLI interface for Deliberative Council.

Built with Typer for a clean, typed command-line interface.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="deliberative-council",
    help="Multi-Agent LLM Debate System for robust, nuanced answers.",
    no_args_is_help=True,
)


def _setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("liteLLM").setLevel(logging.WARNING)


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to debate"),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to YAML config file"
    ),
    complexity: Optional[str] = typer.Option(
        None, "--complexity",
        help="Override Scout's complexity classification (trivial/moderate/complex/deep)",
    ),
    budget: Optional[int] = typer.Option(
        None, "--budget", "-b", help="Override token budget"
    ),
    format: str = typer.Option(
        "markdown", "--format", "-f",
        help="Output format: markdown, json, text, pdf, docx",
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file path"
    ),
    family: Optional[str] = typer.Option(
        None, "--family", help="Restrict to a specific model family"
    ),
    exclude: Optional[list[str]] = typer.Option(
        None, "--exclude", help="Families to exclude"
    ),
    local_only: bool = typer.Option(
        False, "--local-only", help="Only use Ollama/local models"
    ),
    api_only: bool = typer.Option(
        False, "--api-only", help="Only use cloud API models"
    ),
    research_mode: str = typer.Option(
        "strict", "--research-mode",
        help="Research strictness: strict or augmented",
    ),
    graph_strategy: str = typer.Option(
        "full", "--graph-strategy",
        help="Communication graph: full (sparse is extension)",
    ),
    debate_strategy: str = typer.Option(
        "none", "--debate-strategy",
        help="Devil's advocate: none (others are extensions)",
    ),
    context_strategy: str = typer.Option(
        "full", "--context-strategy",
        help="Context window: full (progressive is extension)",
    ),
    model_override: Optional[list[str]] = typer.Option(
        None, "--model",
        help="Override model for a role: role=model_id (e.g. debater_0=openrouter/meta-llama/llama-3.3-70b-instruct:free)",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show configuration without running"
    ),
    resume: bool = typer.Option(
        False, "--resume", help="Resume from last checkpoint"
    ),
) -> None:
    """Ask the Deliberative Council a question."""
    _setup_logging(verbose)

    from council.config import load_config
    from council.types import Complexity

    # Build overrides dict
    overrides = {}
    if budget is not None:
        overrides["budget_override"] = budget
    if complexity is not None:
        try:
            overrides["complexity_override"] = Complexity(complexity)
        except ValueError:
            typer.echo(f"Invalid complexity: {complexity}. Use: trivial, moderate, complex, deep", err=True)
            raise typer.Exit(1)
    if family is not None:
        overrides["family"] = family
    if exclude:
        overrides["exclude_families"] = exclude
    if local_only:
        overrides["local_only"] = True
    if api_only:
        overrides["api_only"] = True
    overrides["format"] = format
    if output:
        overrides["output_path"] = str(output)
    overrides["verbose"] = verbose
    overrides["dry_run"] = dry_run
    overrides["resume"] = resume

    # Parse model overrides
    model_overrides = {}
    if model_override:
        for mo in model_override:
            if "=" in mo:
                role, model_id = mo.split("=", 1)
                model_overrides[role] = model_id
            else:
                typer.echo(f"Invalid model override: {mo}. Use format: role=model_id", err=True)
                raise typer.Exit(1)
    overrides["model_overrides"] = model_overrides

    # Load configuration
    cfg = load_config(config, **overrides)

    # Override debate strategies
    if graph_strategy != "full":
        cfg.debate.graph_strategy = graph_strategy
    if debate_strategy != "none":
        cfg.debate.debate_strategy = debate_strategy
    if context_strategy != "full":
        cfg.debate.context_strategy = context_strategy

    # Override research mode
    if research_mode == "augmented":
        from council.types import ResearchMode
        cfg.research.mode = ResearchMode.AUGMENTED

    if dry_run:
        _show_dry_run(question, cfg)
        return

    # Run the council
    report = asyncio.run(_run(question, cfg))

    # Format and output the report
    from council.output import format_report, save_report

    if output:
        path = save_report(report, output, format)
        typer.echo(f"Report saved to: {path}")
    else:
        content = format_report(report, format)
        typer.echo(content)


async def _run(question: str, config):
    """Async wrapper for running the council."""
    from council.engine import run_council
    return await run_council(question, config)


def _show_dry_run(question: str, config) -> None:
    """Show what would be run without actually running it."""
    typer.echo("=== Dry Run ===")
    typer.echo(f"Question: {question}")
    typer.echo(f"Complexity override: {config.complexity_override or 'auto'}")
    typer.echo(f"Budget: {config.budget_override or config.budget.default_budget:,} tokens")
    typer.echo(f"Family: {config.family or 'auto'}")
    typer.echo(f"Exclude families: {config.exclude_families or 'none'}")
    typer.echo(f"Local only: {config.local_only}")
    typer.echo(f"API only: {config.api_only}")
    typer.echo(f"Research mode: {config.research.mode.value}")
    typer.echo(f"Graph strategy: {config.debate.graph_strategy}")
    typer.echo(f"Debate strategy: {config.debate.debate_strategy}")
    typer.echo(f"Context strategy: {config.debate.context_strategy}")
    typer.echo(f"Output format: {config.format}")
    typer.echo(f"Model overrides: {config.model_overrides or 'none'}")
    typer.echo("")
    typer.echo(f"Models registered: {len(config.models)}")
    for m in config.models[:5]:
        typer.echo(f"  {m.model_id} ({m.family}, {m.tier.value})")
    if len(config.models) > 5:
        typer.echo(f"  ... and {len(config.models) - 5} more")


@app.command()
def models(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to YAML config file"
    ),
    local_only: bool = typer.Option(
        False, "--local-only", help="Only show local models"
    ),
) -> None:
    """List available models and their configurations."""
    from council.config import load_config

    cfg = load_config(config)
    models_list = cfg.models

    if local_only:
        models_list = [m for m in models_list if m.supports_local]

    typer.echo(f"{'Model ID':<40} {'Family':<12} {'Tier':<10} {'Context':<10} {'Cost In/M':<10}")
    typer.echo("-" * 82)
    for m in models_list:
        local_tag = " [L]" if m.supports_local else ""
        typer.echo(
            f"{m.model_id + local_tag:<40} {m.family:<12} {m.tier.value:<10} "
            f"{m.context_window // 1000:>7}K   ${m.input_cost_per_m:<8.2f}"
        )


@app.command()
def check(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to YAML config file"
    ),
) -> None:
    """Run health checks on configured model providers."""
    _setup_logging(verbose=False)

    import asyncio
    from council.config import load_config
    from council.models import ModelRegistry

    cfg = load_config(config)
    registry = ModelRegistry.from_config(cfg)

    typer.echo("Running health checks...")
    results = asyncio.run(registry.health_check())

    typer.echo(f"\n{'Model':<40} {'Status':<10} {'Latency':<10} {'Error'}")
    typer.echo("-" * 80)
    for r in results:
        status = "OK" if r.is_healthy else "FAIL"
        latency = f"{r.latency_ms:.0f}ms" if r.latency_ms else "N/A"
        error = (r.error or "")[:30]
        typer.echo(f"{r.model_id:<40} {status:<10} {latency:<10} {error}")

    healthy = sum(1 for r in results if r.is_healthy)
    typer.echo(f"\n{healthy}/{len(results)} models healthy")


if __name__ == "__main__":
    app()
