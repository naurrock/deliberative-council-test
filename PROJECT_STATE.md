# Deliberative Council — Project State

> **Last updated**: 2026-03-05  
> **Version**: v1.0 (feature-complete core, stubs for v2 extensions)  
> **Status**: Functional end-to-end pipeline; 25 of 38 models reachable from HK server

---

## 1. Current Status

The Deliberative Council is a Python multi-agent LLM debate system that orchestrates a four-phase pipeline — **Scout → Research → Debate → Synthesis** — to produce robust, nuanced answers from free-tier LLM APIs. The core pipeline is fully functional: questions flow from initial classification through optional web research, structured multi-round debate with NLI agreement tracking, and impartial cross-family synthesis into a final report.

### Model Fleet

| Metric | Count |
|--------|-------|
| Total models registered | 38 |
| Available from HK server | 25 |
| Geo-blocked from HK server | 7 |
| Conserved (non-regenerating credits) | 3 |
| Model families (cross-provider dedup) | 14 |
| Providers | 6 |

**Available models by provider (from HK):**

| Provider | Available | Geo-blocked | Conserved |
|----------|-----------|-------------|-----------|
| Cloudflare Workers AI | 14 | 0 | 0 |
| OpenRouter | 8 | 0 | 0 |
| SambaNova | 2 | 0 | 2 |
| Cerebras | 1 | 1 | 1 |
| Gemini | 0 | 3 | 0 |
| Groq | 0 | 3 | 0 |

The 14 model families — llama, qwen, deepseek, gpt-oss, llama4, gemma, mistral, nemotron, glm, moonshot, qwq, granite, hermes, minimax — provide genuine reasoning diversity for debate. The `canonical_id` system deduplicates models across providers (e.g., `llama-3.3-70b-instruct` appears on Cloudflare, OpenRouter, Groq, SambaNova, and Cerebras), enabling intelligent cross-provider fallback.

### Core Pipeline

- [x] **Scout phase** — Two-agent architecture (cheap classifier + mid-tier verifier), produces `MissionBrief` with complexity classification, role assignment, research sub-questions, and token budget
- [x] **Research phase** — Parallel Jina.ai-powered search agents with epistemic tagging (`SOURCED`/`INFERRED`/`JUDGMENT`), strict mode filters unsupported claims
- [x] **Debate phase** — Multi-round structured debate with NLI agreement tracking (two-tier: DeBERTa Tier 1 + LLM Tier 2), position stability, novelty injection, and futility detection
- [x] **Synthesis phase** — Impartial model from a different family produces `FinalReport` with consensus levels, dissenting views, and source URLs

### Infrastructure

- [x] **Two-strikes escalation FailureTracker** — Strike 1 = TRANSIENT (60s cooldown), Strike 2 = DAILY (until midnight UTC); no error message parsing required
- [x] **Cross-provider fallback** — Three-layer strategy: (1) same `canonical_id` on a different provider, (2) explicit fallback chain from YAML, (3) same-tier round-robin
- [x] **Provider-aware model selection** — `providers:` section parsed from YAML into `ProviderInfo` objects with priority ordering, quota awareness, and geo-blocking flags
- [x] **Checkpoint/resume** — SQLite-based `CheckpointManager` saves intermediate state after each phase; `--resume` flag recovers from crashes
- [x] **Token budget management** — Hierarchical budgets with per-phase sub-budgets; `research_share` applied once (no double-counting); scout has separate 50K budget
- [x] **CLI** — Three commands: `ask` (full pipeline), `models` (list registered models), `check` (health check with two-strikes tracking)
- [x] **Output formats** — markdown, json, text, pdf (via weasyprint), docx (via python-docx); format auto-detected from file extension

---

## 2. Server Environment

The system runs on a Hong Kong-based server, which introduces significant constraints on which free-tier LLM providers are reachable. These constraints are handled as **configuration concerns** — not code — so the system adapts automatically if the server is relocated or if a provider changes its geo-restriction policy.

### Provider Reachability from HK

| Provider | Priority | Status from HK | Daily Quota | Regenerates? |
|----------|----------|----------------|-------------|--------------|
| **Cloudflare Workers AI** | 0 | Works | 10K neurons/day | Yes (daily) |
| **OpenRouter** | 1 | Works | 50 RPD (free bucket) | Yes (daily) |
| **Gemini** | 2 | **GEO-BLOCKED** (403) | 1,500 RPD | Yes (daily) |
| **Groq** | 3 | **GEO-BLOCKED** (403) | ~14,400 RPD | Yes (daily) |
| **SambaNova** | 4 | Works | $5 signup credits | **No** (one-time) |
| **Cerebras** | 5 | **GEO-BLOCKED** (403) | Very limited | **No** (one-time) |

### Implications

- **Cloudflare is the workhorse.** With 14 models across 11 families and a daily-regenerating neuron budget, it carries the majority of the workload. All Cloudflare models use the OpenAI-compatible `/ai/v1` endpoint via litellm's `openai/` prefix, working around the buggy native cloudflare handler.
- **OpenRouter is the secondary.** Its 28 free models (8 registered here) provide family diversity and fallback capacity, but the 50 RPD limit is shared across ALL `:free` models — a single busy session can exhaust the daily bucket.
- **Gemini and Groq are dormant assets.** If the server moves to an allowed region (US, EU, etc.), these providers unlock immediately — no code changes required, just flip `geo_blocked: false` in `providers.yaml`. Their generous quotas (1,500 RPD and ~14,400 RPD respectively) would dramatically increase capacity.
- **SambaNova and Cerebras are last-resort conserved providers.** SambaNova's $5 credits and Cerebras's limited free tier are non-regenerating; once exhausted, they're gone. The `conserve: true` flag ensures the round-robin selector only uses them when no regenerating option is available.
- **No OpenAI, Anthropic, or paid tiers.** The system is designed to operate entirely on free-tier APIs. No API keys for paid providers are required or expected.

---

## 3. Recently Completed

The following items were completed in recent development sessions, addressing a comprehensive 18-item technical debt audit and adding key architectural improvements:

### Architecture Improvements

- [x] **Two-strikes escalation** — Replaced the previous error-classification-based failure tracking with the simpler and more robust pattern-based approach. No more brittle HTTP status code parsing; the pattern ("failed once" vs "failed twice") is the signal.
- [x] **Geo-blocking as config concern** — Models with `geo_blocked: true` in `providers.yaml` are filtered at selection time and never reach the FailureTracker. Runtime-discovered geo-blocking naturally escalates through the two-strikes system.
- [x] **`canonical_id` cross-provider dedup** — Added canonical identifiers that link the same underlying model across providers (e.g., `llama-3.3-70b-instruct` on Cloudflare, OpenRouter, Groq, SambaNova, and Cerebras). Enables cross-provider fallback to prefer the same reasoning capability on a different route.
- [x] **`ProviderInfo` type and providers section wiring** — The `providers:` section of `providers.yaml` is now parsed into validated `ProviderInfo` objects with priority, quota, regenerates, and geo_blocked fields. The `ModelRegistry` uses this metadata for priority-aware fallback and conserve awareness.
- [x] **Cross-provider fallback in `resolve_fallback()`** — The three-layer fallback strategy is fully implemented: (1) same canonical_id on a different provider, (2) explicit fallback chain, (3) same-tier round-robin. Provider priority ordering ensures regenerating providers are preferred over conserved ones.

### Bug Fixes

- [x] **Budget double-counting fixed** — `research_share` is applied once in `engine.py` when creating the research sub-budget, NOT again inside `run_research()`. The same pattern holds for `debate_share` and `synthesis_share`. This was a subtle bug that would have caused research agents to receive only 9% of the total budget instead of 30%.
- [x] **Hardcoded fallbacks removed** — All instances of `openai/gpt-4.1-mini` as a hardcoded fallback model have been replaced with proper error handling via `resolve_fallback()` or `RuntimeError` when no model is available. The system no longer silently switches to a paid/OpenAI model.
- [x] **`.env.template` updated** — All Cloudflare variables (`CLOUDFLARE_API_KEY`, `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_BASE`) are documented in the template, making first-time setup straightforward.

### Code Quality & Tooling

- [x] **`.gitignore` added** — Excludes `__pycache__`, `.env`, checkpoints database, output files, and 11 root-level test scripts.
- [x] **`hasattr` guards** — Replaced fragile attribute access with proper `hasattr`/`getattr` patterns for optional model fields.
- [x] **`tempfile.mktemp` → `tempfile.mkstemp`** — Eliminated the insecure deprecated tempfile usage in output module.
- [x] **`consensus_emoji` removed** — Stripped non-deterministic emoji from consensus level display; now uses plain text labels (`STRONG`, `MODERATE`, `CONTESTED`).
- [x] **`suggested_model="alibaba"` removed** — Replaced default model suggestion with proper registry-based selection.
- [x] **`FailureSeverity` as Enum** — Converted from string constants to proper `str, Enum` class for type safety.
- [x] **`float("inf")` dead check removed** — Eliminated unreachable code path in budget calculation.
- [x] **Documentation updated** — README, API_KEYS_GUIDE, DESIGN_DOC, and ENGINEERING_DOC all reflect the current architecture, model registry, and configuration system.

---

## 4. Known Issues & Limitations

These are the current constraints and edge cases that users and developers should be aware of. Items are categorized by severity and whether they have a workaround.

### Heavy Optional Dependencies

| Dependency | Module | Size | Impact if Missing | Workaround |
|------------|--------|------|-------------------|------------|
| `transformers` + `torch` | `nli.py` (DeBERTa Tier 1) | ~2 GB | NLI Tier 1 falls back to Jaccard heuristic (0.35–0.60 range); Tier 2 LLM invoked for all pairs | Accept slightly higher LLM token costs for agreement computation |
| `weasyprint` | `output.py` (PDF format) | ~50 MB | PDF output degrades to raw markdown | Use `--format docx` or `--format markdown` instead |
| `JINA_API_KEY` | `research.py` (web search) | N/A | Research phase degrades gracefully — returns empty `EvidenceReport` with gaps noted | Debate proceeds without web-sourced evidence; uses only LLM knowledge |

### Free-Tier Quota Constraints

| Provider | Constraint | Impact | Mitigation |
|----------|-----------|--------|------------|
| **OpenRouter** | 50 RPD shared across ALL `:free` models | A single complex question with 4 debaters × 3 rounds can consume 20+ requests; two such questions exhaust the daily bucket | Cloudflare carries the primary load; OpenRouter is secondary |
| **Cloudflare** | 10K neurons/day, varies by model size | Larger models (120B) consume more neurons per request; no precise request count possible | Monitor FailureTracker; two-strikes escalation marks exhausted models DAILY |
| **SambaNova** | $5 signup credits, non-regenerating | Once credits are depleted, SambaNova models are permanently unusable | `conserve: true` flag ensures they're only used as last resort |
| **Cerebras** | $0.00 credit remaining, geo-blocked | Currently provides zero usable capacity | Listed in config for completeness; would work if server relocates and credits are added |

### Stub Implementations

The following features are architecturally scaffolded but raise `NotImplementedError` or return defaults when invoked:

| Feature | Module | Current Behavior | Planned Behavior |
|---------|--------|-----------------|------------------|
| **Sparse communication graph** | `graph.py` | Only `"full"` strategy works (all agents see all others); `"sparse"` raises `NotImplementedError` | CortexDebate-inspired partial connectivity — agents only see a subset of others, reducing token costs and creating information silos that drive genuine disagreement |
| **Devil's advocate strategies** | `devil_advocate.py` | Only `"none"` works (no devil's advocate assigned); `"rotate"`, `"weakest"`, `"random"` raise `NotImplementedError` | Rotating assignment cycles through agents; weakest assigns to the least confident; random provides unpredictability |
| **Progressive context strategy** | `context.py` | Only `"full"` works (complete prior rounds in context); `"progressive"` is a stub | Gradually reveal information — early rounds see limited context, later rounds see more, creating a natural escalation of complexity |

### Other Limitations

- **`validate_graph()` is never called.** The function exists as a utility for verifying communication graph invariants (every agent appears as a key, no self-visibility, all references valid) but is not invoked anywhere in the pipeline. This is a minor gap — not a bug, since the full graph strategy trivially satisfies all invariants — but it should be called when sparse graphs are implemented.
- **Test fixtures use old model IDs.** Some test files still reference `openai/gpt-4.1-mini` model IDs from before the Cloudflare migration. Tests pass because they mock LLM calls, but fixture updates are needed for clarity.
- **11 root-level test scripts.** These are gitignored but still present in the working directory. They were created during early development for manual testing and should eventually be migrated into the `tests/` directory or removed.
- **DeBERTa chunking loses cross-paragraph coherence.** When arguments exceed the 512-token limit, the system chunks at sentence boundaries and averages pairwise chunk scores. This works for most cases but can miss logical connections that span paragraph boundaries.
- **Critiques are generated sequentially, not concurrently.** This is by design (to avoid overwhelming rate-limited APIs), but it makes the critique step the slowest part of each debate round. Concurrent critiques could be enabled when rate limits are confirmed to be sufficient.

---

## 5. Technical Debt Status

The original 18-item technical debt audit has been largely resolved. Below is the complete status of every item:

### Resolved (15 items)

| # | Item | Resolution |
|---|------|-----------|
| 1 | No `.gitignore` | Added comprehensive `.gitignore` excluding `__pycache__`, `.env`, checkpoints, outputs, and root test scripts |
| 2 | Budget double-counting | `research_share` applied once in `engine.py`; all phases receive pre-computed sub-budgets |
| 3 | Hardcoded `openai/gpt-4.1-mini` fallbacks | All replaced with `resolve_fallback()` or explicit `RuntimeError` |
| 4 | Missing `hasattr` guards | Replaced with `hasattr`/`getattr` patterns for optional model fields |
| 5 | `tempfile.mktemp` usage | Replaced with `tempfile.mkstemp` in output module |
| 6 | `consensus_emoji` non-determinism | Removed; consensus levels use plain text labels |
| 7 | `suggested_model="alibaba"` default | Replaced with registry-based model selection |
| 8 | `FailureSeverity` as bare strings | Converted to `str, Enum` class |
| 9 | `float("inf")` dead code | Removed unreachable code path |
| 10 | `providers:` section not parsed | Fully parsed into `ProviderInfo` objects with validation, env var expansion, and priority ordering |
| 11 | No cross-provider awareness | `canonical_id` index enables same-model fallback across providers |
| 12 | `.env.template` incomplete | All Cloudflare variables documented (`CLOUDFLARE_API_KEY`, `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_BASE`) |
| 13 | README outdated | Updated with current architecture, model fleet, and CLI reference |
| 14 | API_KEYS_GUIDE outdated | Updated with Cloudflare setup instructions and provider priority table |
| 15 | DESIGN_DOC and ENGINEERING_DOC outdated | Both documents rewritten to reflect current architecture and module structure |

### Remaining Low Priority (3 items)

| # | Item | Status | Risk | Notes |
|---|------|--------|------|-------|
| 16 | `validate_graph()` never called | Open | Very low | Utility function, not a bug; full graph trivially satisfies invariants; should be wired in when sparse graphs are implemented |
| 17 | 11 root-level test scripts | Gitignored | Minimal | These are manual test scripts from early development; should be migrated to `tests/` or removed |
| 18 | Test fixtures reference `openai/gpt-4.1-mini` | Open | Low | Tests pass because LLM calls are mocked, but fixture model IDs should be updated to match current `providers.yaml` entries |

---

## 6. Roadmap — v2 and Beyond

The v1 core pipeline is feature-complete. The following items are planned for v2, organized by priority and estimated complexity.

### High Priority — Core Extensions

| Feature | Module | Description | Complexity | Notes |
|---------|--------|-------------|------------|-------|
| **Sparse communication graph** | `graph.py` | Implement CortexDebate-inspired partial connectivity where agents only see a subset of others. Creates information silos that drive genuine disagreement rather than rapid consensus. Must call `validate_graph()` after construction. | Medium | Key research insight: full connectivity can lead to groupthink as agents converge too quickly |
| **Devil's advocate strategies** | `devil_advocate.py` | Implement `rotate` (cycle through agents), `weakest` (assign to least confident), and `random` strategies. Each forces an agent to argue against their own position, preventing premature consensus. | Low | Architecture already scaffolded; just needs strategy logic and prompt engineering |
| **Progressive context window** | `context.py` | Gradually reveal information across rounds — early rounds see limited context, later rounds see more. Creates a natural escalation of complexity and prevents agents from anchoring on early evidence. | Medium | Must handle truncation gracefully; progressive reveal may conflict with budget constraints in short debates |

### Medium Priority — Usability

| Feature | Description | Complexity | Notes |
|---------|-------------|------------|-------|
| **Streaming output** | Stream LLM responses as they're generated, providing real-time feedback during long debates. Requires modifying `LLMClient.complete()` to support async generators and updating the CLI to display partial output. | High | litellm supports streaming; need to handle partial JSON parsing for structured outputs |
| **Web UI / API server** | FastAPI-based HTTP server with a web UI for submitting questions, watching debate progress in real-time, and browsing past reports. Would replace CLI-only interaction for non-technical users. | High | Consider Server-Sent Events (SSE) for real-time debate updates; WebSocket for bidirectional control |
| **Real-time provider quota tracking** | Parse rate-limit headers from API responses (`X-RateLimit-Remaining`, `X-RateLimit-Reset`, etc.) to track actual quota consumption rather than relying on failure-based detection. Enables proactive load balancing before quota exhaustion. | Medium | Each provider uses different header names; need per-provider header parsing; fall back to two-strikes if headers are missing |

### Lower Priority — Ecosystem

| Feature | Description | Complexity | Notes |
|---------|-------------|------------|-------|
| **Automatic `providers.yaml` updates** | Script or built-in command that queries each provider's model catalog API and updates `providers.yaml` with new models, removing deprecated ones. Currently, adding a model requires a manual YAML edit. | Medium | Cloudflare and OpenRouter have model list APIs; others may not |
| **Additional provider integrations** | Together AI (free-tier models), Mistral (free-tier API), Cohere (trial key), and other emerging free providers. Each requires testing from HK, quota characterization, and YAML entries. | Low per provider | Config-driven architecture means zero code changes per provider; only YAML and testing |
| **Debate visualization** | Generate visual representations of the communication graph, agreement matrix heatmap, and position drift over rounds. Would help users understand how the debate evolved. | Medium | Could use matplotlib/seaborn for static images or D3.js for interactive web visualizations |
| **Multi-question batching** | Accept multiple questions and intelligently schedule them across the daily quota window, prioritizing high-complexity questions for early execution when quotas are fresh. | Medium | Requires persistent quota state across CLI invocations; SQLite checkpoint DB could be extended |

### Experimental / Research

| Feature | Description | Complexity | Notes |
|---------|-------------|------------|-------|
| **Agent memory across sessions** | Allow debaters to recall and reference conclusions from previous questions on related topics, building cumulative expertise over time. | High | Requires persistent agent identity, embedding-based retrieval of past positions, and careful handling of stale information |
| **Human-in-the-loop debate** | Allow a human participant to join the debate as a special agent, injecting their own arguments and critiques. The system would adapt its strategy based on human input. | High | Requires the API server mode as a prerequisite; real-time human input doesn't fit the batch CLI model |
| **Self-improving prompts** | Use debate outcomes (convergence speed, report quality ratings) to iteratively improve system prompts via automated A/B testing. | Very high | Needs a quality metric (human ratings? LLM-as-judge?) and careful experimental design to avoid prompt overfitting |

---

## Quick Reference

### File Layout

```
deliberative-council-test/
├── config/
│   ├── default.yaml          # Pipeline config (NLI, budget, debate, research)
│   └── providers.yaml        # Model registry, provider metadata, fallback chains
├── src/council/
│   ├── __init__.py
│   ├── cli.py                # Typer CLI (ask, models, check)
│   ├── config.py             # YAML loading, validation, env var expansion
│   ├── context.py            # Agent context building (full + progressive stub)
│   ├── debate.py             # Multi-round debate orchestration
│   ├── devil_advocate.py     # Devil's advocate assignment (none + stubs)
│   ├── engine.py             # Pipeline orchestrator with checkpoint/resume
│   ├── feasibility.py        # Two-layer futility detection
│   ├── graph.py              # Communication graph (full + sparse stub)
│   ├── models.py             # Model registry, failure tracker, LLM client
│   ├── nli.py                # Two-tier NLI agreement system
│   ├── output.py             # 5-format output (md, json, text, pdf, docx)
│   ├── research.py           # Parallel Jina.ai research agents
│   ├── scout.py              # Two-agent question classifier
│   ├── synthesis.py          # Cross-family impartial synthesis
│   ├── tools.py              # Tool registry (Jina search/extract)
│   └── types.py              # Pydantic data models
├── tests/                    # Unit tests (mocked LLM calls)
├── .env.template             # Environment variable template
├── API_KEYS_GUIDE.md         # Provider setup instructions
├── DESIGN_DOC.md             # Architecture and design decisions
├── ENGINEERING_DOC.md        # Module reference and internals
├── PROJECT_STATE.md          # This document
├── README.md                 # Project overview and quickstart
└── pyproject.toml            # Python project metadata
```

### Key Commands

```bash
# Run the full pipeline on a question
deliberative-council ask "Is quantum computing a threat to Bitcoin?"

# List all registered models
deliberative-council models

# Health check (ping each model)
deliberative-council check

# Run with overrides
deliberative-council ask "What is consciousness?" --complexity deep --budget 300000 --format pdf --output report.pdf

# Resume from last checkpoint
deliberative-council ask "..." --resume

# Dry run (show config without making LLM calls)
deliberative-council ask "..." --dry-run
```

### Environment Variables

| Variable | Required | Provider | Notes |
|----------|----------|----------|-------|
| `CLOUDFLARE_API_KEY` | Yes | Cloudflare | Workers AI API token |
| `CLOUDFLARE_ACCOUNT_ID` | Yes | Cloudflare | Account ID for API base URL |
| `OPENROUTER_API_KEY` | Yes | OpenRouter | Free-tier API key |
| `GEMINI_API_KEY` | No | Gemini | Only needed if server is in allowed region |
| `GROQ_API_KEY` | No | Groq | Only needed if server is in allowed region |
| `SAMBANOVA_API_KEY` | No | SambaNova | Conserve credits; only for last-resort fallback |
| `CEREBRAS_API_KEY` | No | Cerebras | Currently geo-blocked + no credits |
| `JINA_API_KEY` | No | Jina.ai | Optional; research degrades gracefully without it |
