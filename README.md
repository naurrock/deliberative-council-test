# Deliberative Council

Multi-agent LLM debate system for robust, nuanced answers. Orchestrates a
**Scout → Research → Debate → Synthesis** pipeline that dispatches each role
to the cheapest available free-tier model, maximises family diversity among
debaters, and falls back across providers when one route is exhausted.

## Quick Start

```bash
# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
cd deliberative-council-test
uv sync

# Copy and fill in API keys
cp .env.template .env
# Edit .env with your keys (Cloudflare + OpenRouter is enough to start)

# Ask a question
uv run council ask "What are the implications of quantum computing for cryptography?"

# Dry run (show config without calling models)
uv run council ask "What is 2+2?" --dry-run

# List registered models
uv run council models

# Run health checks on model providers
uv run council check

# Override complexity
uv run council ask "Simple question" --complexity trivial --budget 10000
```

## Architecture

Four-phase pipeline: **Scout → Research → Debate → Synthesis**

| Phase    | Purpose                                           | Model tier |
|----------|---------------------------------------------------|------------|
| Scout    | Classify question, generate debate roles           | CHEAP → MID |
| Research | Tool-augmented agents search for evidence (Jina.ai) | CHEAP      |
| Debate   | Multi-agent debate with NLI agreement tracking     | MID        |
| Synthesis| Impartial model synthesizes final report            | PREMIUM    |

### Key Design Decisions

- **Two-strikes escalation**: First failure → TRANSIENT (60 s cooldown). Second
  failure while still cooling → DAILY (until midnight UTC). No error-message
  parsing needed — the pattern distinguishes RPM throttling from RPD exhaustion.
- **Cross-provider fallback**: When a model fails, the registry first looks for
  the same underlying model (same `canonical_id`) on a different provider, then
  falls back to the explicit chain, then to same-tier round-robin.
- **Family diversity**: Debaters are assigned different model families when
  possible; the Synthesizer is always from a family not represented in the debate.
- **Geo-blocking is config**: Models with `geo_blocked: true` are filtered at
  selection time — they never reach FailureTracker.
- **Two-tier NLI**: Tier 1 (DeBERTa, local, free) → Tier 2 (cheap LLM) for
  agreement detection. Falls back to heuristics if DeBERTa is unavailable.
- **Core-plus-extension**: v1 core (full graph, no devil's advocate, full context)
  → v2 extensions (sparse graph, rotating devil's advocate, progressive context).

## Configuration

Model configuration lives in **`config/providers.yaml`**, which contains three
sections:

| Section            | Purpose                                                |
|--------------------|--------------------------------------------------------|
| `providers:`       | Per-provider metadata (priority, quotas, geo-blocking) |
| `models:`          | Model registry (38 models across 6 providers)          |
| `fallback_chains:` | Ordered fallback lists per role type                   |

Subsystem defaults (NLI, budget, debate, research) live in
`config/default.yaml` and can be overridden via CLI flags or a custom YAML
file. If `config/providers.yaml` is absent, built-in free-tier defaults are
used.

### Priority Order

| Priority | Provider   | Quota                | Notes                         |
|----------|------------|----------------------|-------------------------------|
| 0        | Cloudflare | 10K neurons/day      | 14 LLM models, daily regen    |
| 1        | OpenRouter | 50 RPD (free bucket) | 8 free models, good diversity |
| 2        | Gemini     | 1,500 RPD            | **GEO-BLOCKED from HK**       |
| 3        | Groq       | ~14,400 RPD          | **GEO-BLOCKED from HK**       |
| 4        | SambaNova  | $5 credits           | Non-regenerating, conserve    |
| 5        | Cerebras   | Limited              | **GEO-BLOCKED from HK**       |

## Environment Variables

Set API keys in a `.env` file (copy from `.env.template`):

| Variable               | Provider     | Required? |
|------------------------|-------------|-----------|
| `CLOUDFLARE_API_KEY`   | Cloudflare  | **Yes**   |
| `CLOUDFLARE_ACCOUNT_ID`| Cloudflare  | **Yes**   |
| `OPENROUTER_API_KEY`   | OpenRouter  | **Yes**   |
| `GEMINI_API_KEY`       | Gemini      | Optional  |
| `GROQ_API_KEY`         | Groq        | Optional  |
| `SAMBANOVA_API_KEY`    | SambaNova   | Optional  |
| `CEREBRAS_API_KEY`     | Cerebras    | Optional  |

Cloudflare + OpenRouter together provide 22 non-geo-blocked models — enough
for the full pipeline with family diversity.

## Testing

```bash
# Run all unit tests (no API calls needed)
uv run pytest tests/ -v -m "not integration"

# Run specific test file
uv run pytest tests/test_nli.py -v

# Run with coverage
uv run pytest tests/ --cov=council --cov-report=term-missing -m "not integration"

# Run integration tests (requires API keys)
uv run pytest tests/test_live_model.py -v -m integration
```

## Project Structure

```
deliberative-council-test/
├── config/
│   ├── providers.yaml          # Model registry + provider metadata
│   └── default.yaml            # Subsystem defaults (NLI, budget, etc.)
├── src/council/
│   ├── types.py                # Core Pydantic type definitions
│   ├── config.py               # Configuration loading & validation
│   ├── models.py               # Model registry, FailureTracker, LLMClient
│   ├── nli.py                  # Two-tier NLI agreement system
│   ├── scout.py                # Scout phase (question classification)
│   ├── research.py             # Research phase (web search via Jina.ai)
│   ├── debate.py               # Debate phase (multi-agent)
│   ├── synthesis.py            # Synthesis phase (final report)
│   ├── feasibility.py          # Futility detection
│   ├── engine.py               # Pipeline orchestrator + checkpoints
│   ├── cli.py                  # Typer CLI interface
│   ├── output.py               # Report formatting & export
│   ├── tools.py                # Jina.ai search/extract tools
│   ├── context.py              # Agent context building
│   ├── graph.py                # Communication graph
│   └── devil_advocate.py       # Devil's advocate assignment
├── tests/                      # pytest test suite
├── .env.template               # API key template
└── pyproject.toml
```

## Documentation

| Document | Description |
|----------|-------------|
| [DESIGN_DOC.md](DESIGN_DOC.md) | Architecture, design decisions, and data flow |
| [ENGINEERING_DOC.md](ENGINEERING_DOC.md) | Implementation details and module reference |
| [PROJECT_STATE.md](PROJECT_STATE.md) | Current status, known issues, and roadmap |
| [TEST_REPORT.md](TEST_REPORT.md) | Testing infrastructure and results |
| [API_KEYS_GUIDE.md](API_KEYS_GUIDE.md) | Detailed provider setup and quota planning |
