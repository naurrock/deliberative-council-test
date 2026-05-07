# Deliberative Council

Multi-agent LLM debate system for robust, nuanced answers.

## Quick Start

```bash
# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
cd deliberative-council
uv sync

# Run tests
uv run pytest tests/ -v -m "not integration"

# Ask a question (requires API keys in environment)
uv run council ask "What are the implications of quantum computing for cryptography?"

# Dry run (show config without calling models)
uv run council ask "What is 2+2?" --dry-run

# List available models
uv run council models

# Run with local-only models (Ollama)
uv run council ask "Explain this concept" --local-only

# Override complexity
uv run council ask "Simple question" --complexity trivial --budget 10000

# Run health checks on model providers
uv run council check
```

## Architecture

Four-phase pipeline: **Scout → Research → Debate → Synthesis**

- **Scout**: Classifies question complexity, generates debate roles, determines if research is needed
- **Research**: Tool-augmented agents search for evidence using Jina.ai APIs
- **Debate**: Multi-agent debate with NLI agreement tracking, position stability, novelty injection, and futility detection
- **Synthesis**: Premium model synthesizes debate into a final report with key points and dissenting views

## Key Design Decisions

- **Two-tier NLI**: Tier 1 (DeBERTa, local, free) → Tier 2 (cheap LLM) for agreement detection
- **Position stability**: NLI-based float replacing SHA-256 hashing; triggers novelty injection when > threshold for N consecutive rounds
- **Family diversity**: Debaters are assigned different model families when possible; Synthesizer is always from a different family than debaters
- **Core-plus-extension**: v1 core (full graph, no devil's advocate, full context) → v2 extensions (sparse graph, rotating devil's advocate, progressive context)

## Configuration

Default config is in `config/default.yaml`. Override via CLI flags or a custom YAML file.

## Environment Variables

Set API keys as environment variables:
- `OPENAI_API_KEY` — OpenAI models
- `ANTHROPIC_API_KEY` — Anthropic models
- `GEMINI_API_KEY` — Google models
- `DEEPSEEK_API_KEY` — DeepSeek models
- `DASHSCOPE_API_KEY` — Alibaba/Qwen models
- `MISTRAL_API_KEY` — Mistral models
- `MOONSHOT_API_KEY` — Moonshot/Kimi models

Or use a `.env` file in the project root.

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
deliberative-council/
├── config/
│   └── default.yaml          # Default configuration
├── src/
│   └── council/
│       ├── __init__.py
│       ├── types.py           # Core Pydantic type definitions
│       ├── config.py          # Configuration loading & validation
│       ├── models.py          # Model registry & LiteLLM wrapper
│       ├── nli.py             # Two-tier NLI agreement system
│       ├── scout.py           # Scout phase (question classification)
│       ├── research.py        # Research phase (web search)
│       ├── debate.py          # Debate phase (multi-agent)
│       ├── synthesis.py       # Synthesis phase (final report)
│       ├── engine.py          # Pipeline orchestrator + checkpoints
│       ├── cli.py             # Typer CLI interface
│       ├── output.py          # Report formatting & export
│       ├── tools.py           # Jina.ai search/extract tools
│       ├── context.py         # Agent context building
│       ├── graph.py           # Communication graph
│       ├── devil_advocate.py  # Devil's advocate assignment
│       └── feasibility.py     # Futility detection
├── tests/
│   ├── test_types.py
│   ├── test_config.py
│   ├── test_models.py
│   ├── test_nli.py
│   ├── test_scout.py
│   ├── test_research.py
│   ├── test_debate.py
│   ├── test_synthesis.py
│   ├── test_engine.py
│   ├── test_output.py
│   ├── test_tools.py
│   ├── test_context.py
│   ├── test_graph.py
│   ├── test_devil_advocate.py
│   ├── test_feasibility.py
│   ├── test_live_model.py
│   └── test_config.py
└── pyproject.toml
```
