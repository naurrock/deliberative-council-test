# Deliberative Council — Testing Infrastructure & Results

This document describes the testing infrastructure, coverage, execution strategies, and known gaps for the Deliberative Council project — a Python multi-agent LLM debate system that routes questions through a **Scout → Research → Debate → Synthesis** pipeline.

---

## 1. Testing Infrastructure

### 1.1 Overview

The project maintains two distinct tiers of tests, each serving a different purpose in the development and validation workflow:

| Tier | Location | Framework | Purpose |
|------|----------|-----------|---------|
| Unit tests | `tests/` | pytest | Fast, isolated, mocked — validates individual modules |
| Ad-hoc integration scripts | Project root | Standalone `asyncio` scripts | End-to-end validation with real LLM providers |

The unit tests are the primary safety net for refactoring and feature development. The root-level scripts are exploratory integration tests that exercise the full pipeline against live model providers (Cloudflare Workers AI, OpenRouter, z-ai SDK, Gemini, Groq, etc.). They evolved incrementally during development and are **not** pytest-discoverable — they each contain their own `async def main()` entry point and a copy-pasted `_zai_acompletion` helper that monkey-patches `litellm.acompletion`.

### 1.2 Unit Test Directory (`tests/`)

The `tests/` directory contains 16 pytest-discoverable test modules, one per source module (plus a live integration test file). All tests use standard pytest conventions: classes for grouping, `pytest.mark.asyncio` for async tests, and `unittest.mock` for isolating external dependencies.

**Test files and their corresponding source modules:**

| Test File | Source Module | Lines |
|-----------|---------------|-------|
| `test_types.py` | `council/types.py` | ~266 |
| `test_config.py` | `council/config.py` | ~215 |
| `test_models.py` | `council/models.py` | ~391 |
| `test_nli.py` | `council/nli.py` | ~418 |
| `test_scout.py` | `council/scout.py` | ~481 |
| `test_research.py` | `council/research.py` | ~347 |
| `test_debate.py` | `council/debate.py` | ~367 |
| `test_synthesis.py` | `council/synthesis.py` | ~453 |
| `test_engine.py` | `council/engine.py` | ~229 |
| `test_output.py` | `council/output.py` | ~77 |
| `test_tools.py` | `council/tools.py` | ~281 |
| `test_context.py` | `council/context.py` | ~241 |
| `test_graph.py` | `council/graph.py` | ~54 |
| `test_devil_advocate.py` | `council/devil_advocate.py` | ~37 |
| `test_feasibility.py` | `council/feasibility.py` | ~79 |
| `test_live_model.py` | (integration) | ~108 |

### 1.3 Root-Level Ad-Hoc Integration Scripts

The project root contains 11 standalone integration test scripts that exercise the pipeline against real LLM providers. These scripts are **not** pytest-discoverable — they each define their own `async def main()` and are run directly with `uv run python <script>`.

| Script | Description | LLM Backend |
|--------|-------------|-------------|
| `test_cloudflare.py` | Tests all Cloudflare Workers AI models with connectivity checks; simulates full Scout→Debate→Synthesis pipeline | Cloudflare (litellm with custom `api_base`/`api_key`) |
| `test_trivial.py` | Trivial-complexity pipeline test (2–3 LLM calls) | z-ai SDK CLI (monkey-patches `litellm.acompletion`) |
| `test_moderate.py` | Moderate-complexity pipeline test (1 debate round, no research) | z-ai SDK CLI |
| `test_moderate_v2.py` | Moderate test with async subprocess calls and retry/backoff for 429 errors | z-ai SDK CLI (async) |
| `test_moderate_v3.py` | Moderate test with direct HTTP calls to z-ai API, pre-flight rate-limit check, and 15-second call interval | z-ai HTTP API (aiohttp) |
| `test_moderate_direct.py` | Moderate test bypassing litellm entirely — direct HTTP to OpenRouter/SambaNova with family-diversity-aware model selection | OpenRouter, SambaNova (aiohttp) |
| `test_moderate_async.py` | Moderate test with async z-ai SDK calls (subprocess-based) | z-ai SDK CLI (async) |
| `test_moderate_provider.py` | Moderate test with `providers.yaml`-driven multi-provider routing and priority-aware selection | Any litellm-compatible provider |
| `test_moderate_final.py` | Moderate test with quota-aware multi-provider support (z-ai, Gemini, Groq) and pre-flight rate-limit probing | z-ai HTTP, Gemini, Groq |
| `test_round_robin.py` | Validates round-robin model selection, failure tracking/cooldown, Cloudflare inference, and conserve-flag behavior | Cloudflare via `providers.yaml` |
| `test_live_integration.py` | Three-phase test: Scout only, full trivial pipeline, full moderate pipeline | z-ai SDK CLI |

All root-level scripts (except `test_cloudflare.py` and `test_round_robin.py`) share a common pattern: they define a `_zai_acompletion` helper function that replaces `litellm.acompletion` at runtime, routing calls through the z-ai SDK CLI or HTTP API. This helper is copy-pasted across scripts rather than shared via import, which is a known maintainability issue (see Section 5).

### 1.4 Pytest Configuration

The project uses the following pytest configuration (defined in `pyproject.toml`):

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
markers = [
    "integration: marks tests that call external APIs (deselect with '-m \"not integration\"')",
]
```

Key points:
- **`asyncio_mode = "auto"`**: All async test functions are automatically handled by `pytest-asyncio` without requiring explicit `@pytest.mark.asyncio` decorators (though some tests still include them for clarity).
- **`integration` marker**: Tests that make real API calls (e.g., `test_live_model.py`) are marked with `@pytest.mark.integration` so they can be easily excluded from CI runs that lack API keys.
- **Dev dependencies**: `pytest>=8.0`, `pytest-asyncio>=0.23`, `pytest-cov>=5.0` are declared as dev dependencies and managed via `uv`.

---

## 2. Unit Test Coverage

### 2.1 test_types.py — Pydantic Model Validation

This module validates all Pydantic models defined in `council/types.py`, which form the data backbone of the entire pipeline. The tests are organized into six test classes covering the major type categories:

- **`TestComplexity`**: Validates the four complexity enum values (`TRIVIAL`, `MODERATE`, `COMPLEX`, `DEEP`) and string-based construction.
- **`TestPosition`**: Tests `Position` model construction with optional fields (`position_stability`, `supporting_evidence`, `metacognitive_notes`), and validates that `self_confidence` and `position_stability` are bounded to `[0, 1]` via Pydantic `ValidationError`.
- **`TestMissionBrief`**: Tests both minimal (trivial question) and complex (deep question with roles, sub-questions, and domain tags) construction. Verifies default empty lists for `domain_tags` and `research_subquestions`.
- **`TestResearchTypes`**: Tests `EvidenceSource`, `ResearchFinding` (with epistemic tags and relevance bounds), and `EvidenceReport` construction.
- **`TestDebateTypes`**: Tests `RoundResult` (including agreement matrix structure), `DebateState` (initial unresolved state), `ConvergenceResult`, and `FutilityCheck`.
- **`TestFinalReportTypes`**: Tests `KeyPoint` with consensus levels, `ModelUsage` tracking, `PipelineTrace.total_tokens` computed property, and `FinalReport` with all optional fields.

### 2.2 test_config.py — YAML Loading and Configuration

This module tests the configuration system in `council/config.py`, which loads settings from YAML files with environment variable expansion and default fallback.

- **`TestModelConfig`**: Validates `ModelConfig` defaults (`enabled=True`, `context_window=128_000`, zero costs for local models).
- **`TestNLIConfig`**: Tests the critical invariant that `position_stability_threshold` (0.80) > `convergence_threshold` (0.75), ensuring novelty injection triggers before convergence detection. Also validates custom threshold construction.
- **`TestBudgetConfig`**: Validates default budget (500K tokens) and the constraint that research/debate/synthesis shares must sum to approximately 1.0.
- **`TestDebateConfig`**: Tests default round counts per complexity level (`trivial=0`, `deep=3`), graph strategy, debate strategy, and context strategy defaults.
- **`TestResearchConfig`**: Validates `STRICT` mode as default, `max_concurrent_agents=3`, and Jina.ai URL configuration.
- **`TestCouncilConfig`**: Tests the top-level config object including complexity override, family constraints, exclude families, and model overrides.
- **`TestLoadConfig`**: Tests `load_config()` with actual YAML files (tempfile-based), missing file errors, parameter overrides, and default model/chain loading.
- **`TestDefaultModels`**: Validates that `DEFAULT_MODELS` covers all expected families (openai, anthropic, google, deepseek, alibaba, meta, mistral, microsoft, moonshot), all tiers, and includes local models.

### 2.3 test_models.py — Model Registry and LLM Client

This is one of the largest test files, covering the `ModelRegistry` class, `TokenBudget`, and `LLMClient` — the three components that manage model selection, budget tracking, and LLM communication.

- **`TestTokenBudget`**: Tests initial state, consumption, exhaustion detection, and allocation checks (`can_allocate`).
- **`TestModelRegistry`**: Tests creation, model lookup, availability filtering, family/tier grouping, cheapest-in-tier selection (with and without family constraint), and usage recording.
- **`TestModelSelection`**: Tests the `select_model_for_role` method across research roles (should get CHEAP tier), debater roles (MID tier), family constraints, family exclusion, local-only constraints, model overrides, family diversity preference, and fallback chain resolution when the primary model is unavailable.
- **`TestLLMClient`**: Tests `LLMClient.complete()` with mocked `litellm.acompletion`, including successful calls with budget tracking, budget exhaustion errors, per-call `budget_override` precedence over client-level budget, and the case where neither budget is set (unlimited mode).

### 2.4 test_nli.py — Two-Tier NLI Agreement System

This module tests the NLI (Natural Language Inference) agreement detection system, which is the core mechanism for determining when debate agents have converged.

- **`TestChunkingByChars`**: Tests the character-based fallback chunker with short text (single chunk), long text (multiple chunks), empty text, and max-chars enforcement.
- **`TestChunkingByTokens`**: Tests token-aware chunking with mocked tokenizer, including fallback to char-based chunking when no tokenizer is loaded, and handling of oversized single sentences.
- **`TestHeuristicAgreement`**: Tests the fallback heuristic scorer with identical text (high score), different text, and empty text (returns 0.5 neutral).
- **`TestTier1Agreement`**: Tests DeBERTa-based agreement scoring, including fallback to heuristic when DeBERTa is unavailable, and mocked DeBERTa model with controlled logits.
- **`TestTier2Agreement`**: Tests LLM-based deep agreement analysis with mocked JSON responses, and graceful degradation on unparseable responses (returns default `agreement_score=0.5`).
- **`TestConvergenceDetection`**: Tests `check_convergence()` with no rounds (continue), budget exhaustion (converged), round limit reached with high agreement (converged), round limit with low agreement (not converged), NLI convergence across consecutive rounds, and mixed rounds that break the convergence streak.
- **`TestPositionStability`**: Tests stability computation with no previous position (returns `None`) and with a previous position (uses NLI/heuristic).
- **`TestNoveltyInjection`**: Tests `should_inject_novelty()` with insufficient history, high stability triggering injection, low stability not triggering, and the critical invariant that `position_stability_threshold > convergence_threshold`.

### 2.5 test_scout.py — Scout Phase

Tests the Scout phase, which classifies questions and generates mission briefs.

- **`TestParseMissionBrief`**: Tests JSON parsing of scout responses, including valid JSON, invalid JSON (falls back to moderate complexity with default roles), markdown-fenced JSON, unknown complexity strings (default to MODERATE), missing fields (safe defaults), and the rule that non-trivial questions get at least 2 default roles while trivial questions keep zero.
- **`TestAddDefaultRoles`**: Tests default role generation for moderate (≥2), complex (≥3), and deep (≥3) complexities, duplicate name avoidance, and preservation of existing roles.
- **`TestModelSelection`**: Tests `_select_scout_model` (prefers CHEAP tier, respects local_only and family constraints) and `_select_verifier_model` (prefers MID tier, falls back to PREMIUM).
- **`TestRunScout`**: Integration tests with mocked LLM calls, verifying that the full scout run produces a valid `MissionBrief`, complexity override takes effect, and budget override works.
- **`TestPrompts`**: Validates that system prompts mention key elements (complexity levels, JSON output format, verification notes).

### 2.6 test_research.py — Research Phase

Tests the Research phase, which dispatches research agents to investigate sub-questions.

- **`TestCreateResearchRoles`**: Tests that one role is created per sub-question, roles are marked as research, sub-questions are attached, names are unique, and empty sub-questions produce no roles.
- **`TestParseResearchFindings`**: Extensive testing of the research findings parser, including valid JSON with sourced/inferred tags, strict mode downgrading of `SOURCED` tags without URLs to `INFERRED`, strict mode filtering of `JUDGMENT` tags, augmented mode keeping `JUDGMENT` tags, invalid JSON handling, markdown-fenced JSON, invalid epistemic tags (default to `JUDGMENT`), relevance clamping, and empty findings with gaps/recommendations.
- **`TestRunResearch`**: Integration tests with mocked search and LLM calls, including the no-research-needed path, successful research with Jina search results, and graceful handling of search failures.

### 2.7 test_debate.py — Debate Phase

Tests the Debate phase, where agents argue positions and critique each other.

- **`TestParsePosition`**: Tests JSON position parsing, raw text fallback, markdown-fenced JSON, out-of-bounds confidence clamping, missing confidence default (0.5), missing evidence default (empty list), and non-numeric confidence handling.
- **`TestAssignModelsToAgents`**: Tests model assignment with family diversity preference and local-only constraints.
- **`TestCollectPriorPositions`**: Tests collecting positions from no rounds, single rounds, and multiple rounds in chronological order.
- **`TestFindPreviousPosition`**: Tests finding an agent's position in a round and returning `None` for missing agents.
- **`TestRunDebate`**: Integration tests with mocked NLI and LLM calls, verifying that debate runs to the configured round limit and that prior state is respected for resume scenarios.

### 2.8 test_synthesis.py — Synthesis Phase

Tests the Synthesis phase, which produces the final report from debate results.

- **`TestBuildDebateSummary`**: Tests that summaries include round count, position details, stability scores, resolution reason, and futility flags.
- **`TestBuildEvidenceSummary`**: Tests that evidence summaries include findings, epistemic tags, gaps, and the "No research evidence" message for empty evidence.
- **`TestParseSynthesis`**: Tests valid JSON synthesis parsing (answer, key points with consensus levels, dissenting views), invalid JSON fallback, markdown-fenced JSON, unknown consensus level (defaults to MODERATE), and convergence score computation from the last round's agreement matrix.
- **`TestFallbackReport`**: Tests that fallback reports include positions and collect source URLs from evidence.
- **`TestSelectSynthesizer`**: Tests premium model preference, model override, mid-tier fallback when no premium models available, and family constraint.
- **`TestGenerateMarkdown`**: Tests that generated markdown includes the question, key points with dissent, and pipeline trace.
- **`TestRunSynthesis`**: Integration tests with mocked LLM calls, verifying successful report generation and fallback on LLM failure.

### 2.9 test_engine.py — Pipeline Orchestration

Tests the top-level pipeline orchestrator and checkpoint management.

- **`TestCheckpointManager`**: Tests saving/loading mission briefs, evidence, and debate state as SQLite checkpoints; phase tracking; loading nonexistent checkpoints (returns `None`); and automatic directory creation for the database file.
- **`TestRunCouncil`**: Integration tests with fully mocked pipeline phases, verifying that trivial questions skip research and debate (going directly from scout to synthesis), and that complex questions run all four phases (scout, research, debate, synthesis).

### 2.10 test_output.py — Format Conversion

Tests the output formatting module, which converts `FinalReport` objects to various formats.

- **`TestMarkdownOutput`**: Tests basic markdown generation and the rule that `raw_markdown` is used directly when available.
- **`TestJsonOutput`**: Tests JSON output produces valid, parseable JSON with correct fields.
- **`TestTextOutput`**: Tests plain text output with no markdown formatting (no `**` bold markers).
- **`TestFormatSelection`**: Tests that unknown format strings raise `ValueError`.

Note: PDF and DOCX output paths are implemented in `council/output.py` but are not covered by unit tests (see Section 5).

### 2.11 test_tools.py — Jina.ai Search and Extract

Tests the tool registry and Jina.ai client for web search and content extraction.

- **`TestSearchResult` / `TestExtractResult`**: Tests data model construction for successful and failed results.
- **`TestJinaClient`**: Tests default/custom configuration, mocked search (URL extraction from response text), search failure (empty list), mocked content extraction, extraction failure (error result), session cleanup, and cleanup with no session.
- **`TestJinaClientParsing`**: Tests search result URL extraction, URL deduplication, title extraction from URLs, and title extraction from HTML `<title>` tags.
- **`TestToolRegistry`**: Tests built-in tool registration (`web_search`, `extract_content`), tool descriptions, tool lookup, custom tool registration, unknown tool execution error, sync tool execution, and cleanup.

### 2.12 test_context.py — Context Window Management

Tests the context building system that determines what information each agent sees during debate.

- **`TestBuildAgentContext`**: Tests full strategy with no prior positions (research only), full strategy with prior positions from visible agents, agents seeing their own positions even without graph visibility, empty context handling, and error raising for unimplemented (`progressive`) and unknown strategies.
- **`TestContextCompression`**: Tests that short context is not compressed, long context is truncated with a "truncated" note, and compression preserves recent content (tail).
- **`TestResearchEvidenceInContext`**: Tests that research gaps, source URLs, and epistemic tags appear in agent context.

### 2.13 test_graph.py — Communication Graph

Tests the communication graph that determines agent visibility during debate.

- **`TestFullGraph`**: Tests full-mesh graph construction with 2 agents (bidirectional), 3 agents (each sees 2 others), single agent (empty list), and empty input (empty dict).
- **`TestSparseGraph`**: Validates that sparse graph raises `NotImplementedError`.
- **`TestValidateGraph`**: Tests graph validation for valid full graphs, missing agents, self-loops (invalid), and unknown agent references.

### 2.14 test_devil_advocate.py — Devil's Advocate Assignment

Tests the devil's advocate assignment strategy.

- **`TestNoneStrategy`**: Tests that the `none` strategy returns empty assignments for all agents, regardless of round number.
- **`TestExtensionStrategies`**: Validates that `rotate`, `weakest`, and `random` strategies raise `NotImplementedError`, and unknown strategies raise `ValueError`.

### 2.15 test_feasibility.py — Futility Detection

Tests the heuristic futility detection system that identifies when debate is making no progress.

- **`TestHeuristicFutility`**: Tests detection with insufficient rounds (not futile), all agents stuck (high position stability + frozen agreement → futile), agents making progress (not futile), and frozen agreement patterns across multiple rounds (futile).

### 2.16 test_live_model.py — Integration Tests with Real API Calls

This file contains tests marked with `@pytest.mark.integration` that make real LLM calls via the z-ai SDK CLI. These tests verify that the Scout phase can produce valid JSON that parses correctly when generated by an actual language model.

- **`TestLiveModelScout`**: Tests Scout JSON parsing with a real LLM (complex question classification, trivial question classification, and research-needed detection for complex questions).

---

## 3. Running Tests

### 3.1 Unit Tests Only (No API Keys Required)

Run all unit tests, excluding integration tests that require API keys:

```bash
uv run pytest tests/ -v -m "not integration"
```

This is the standard command for CI pipelines and local development. It runs all 16 test modules and completes in seconds since all external calls are mocked.

### 3.2 Integration Tests (Requires API Keys)

Run only the integration tests that make real LLM calls:

```bash
uv run pytest tests/ -v -m integration
```

This requires the z-ai SDK CLI (`z-ai`) to be installed and configured. These tests are slower (each LLM call takes 5–30 seconds) and are subject to rate limiting.

### 3.3 Specific Test Module

Run a single test module for focused development:

```bash
uv run pytest tests/test_nli.py -v
```

Or a single test class:

```bash
uv run pytest tests/test_nli.py::TestConvergenceDetection -v
```

### 3.4 Coverage Report

Generate a coverage report with line-level detail:

```bash
uv run pytest tests/ --cov=council --cov-report=term-missing
```

This produces a table showing coverage percentage per module and lists specific uncovered lines. The coverage tool measures against the `council` package in `src/council/`.

### 3.5 Root-Level Integration Scripts

Each root-level script is run independently as a standalone Python program:

```bash
# Cloudflare Workers AI connectivity and pipeline test
uv run python test_cloudflare.py
uv run python test_cloudflare.py --quick
uv run python test_cloudflare.py --full-pipeline

# Trivial complexity pipeline test (2-3 calls)
uv run python test_trivial.py

# Moderate complexity pipeline tests (various backends)
uv run python test_moderate.py
uv run python test_moderate_v2.py          # async + retry
uv run python test_moderate_v3.py          # direct HTTP + rate limit check
uv run python test_moderate_direct.py      # multi-provider direct HTTP
uv run python test_moderate_async.py       # async z-ai SDK
uv run python test_moderate_provider.py    # providers.yaml driven
uv run python test_moderate_final.py       # quota-aware multi-provider

# Round-robin model selection test
uv run python test_round_robin.py
uv run python test_round_robin.py --quick

# Full integration test (scout + trivial + moderate)
uv run python test_live_integration.py
```

Most moderate-complexity scripts accept a `--question` flag to customize the test question and various provider-specific flags (e.g., `--allow-conserve`, `--provider gemini`).

---

## 4. Integration Testing Strategy

### 4.1 Pre-Flight Health Check

Before running any integration test, verify that model providers are reachable:

```bash
uv run council check
```

This CLI command probes configured providers and reports which models are available, which are rate-limited, and which API keys are missing. It's the recommended first step before any live testing session.

### 4.2 Tiered Integration Testing

Integration tests are organized by complexity level, which directly controls the number of LLM calls and therefore the cost and time:

| Test Level | Complexity | Pipeline Phases | Approx. LLM Calls | Token Usage | When to Use |
|------------|-----------|-----------------|-------------------|-------------|-------------|
| Trivial | `TRIVIAL` | Scout + Synthesis only | 2–3 | ~2,000–5,000 | Quick smoke test, API key verification |
| Moderate | `MODERATE` | Scout + 2 Debaters (1 round) + Synthesis | 8–12 | ~15,000–30,000 | Feature validation, prompt engineering |
| Complex | `COMPLEX` | Scout + Research + 2–3 Debaters (2 rounds) + Synthesis | 15–25 | ~50,000–100,000 | End-to-end validation, demo |
| Deep | `DEEP` | Full pipeline, 3 rounds, full research | 25–40 | ~100,000–250,000 | Final acceptance, rare |

For daily development, the **trivial** test is sufficient to verify API connectivity. The **moderate** test is the workhorse for validating pipeline changes. Complex and deep tests should be reserved for pre-release validation.

### 4.3 Provider Budget Planning

The Deliberative Council routes requests across multiple LLM providers with different rate limits and pricing:

| Provider | Daily Call Budget | Cost | Notes |
|----------|-------------------|------|-------|
| Cloudflare Workers AI | ~300–500 calls | Free (neuron-limited) | Primary workhorse; bulk of scout/research/debate calls |
| OpenRouter (free tier) | ~50 calls/day (RPD) | Free | Bottleneck provider; use sparingly for diversity |
| Gemini | ~1,500 calls/day | Free tier | Generous quota; good for synthesis |
| Groq | ~30 RPM | Free tier | Fast inference; good for debate rounds |
| z-ai SDK | Varies (user daily quota) | Free | Sandbox provider; rate-limited but functional |

**Budget strategy**: Use Cloudflare for the bulk of calls (scout, research agents, most debaters). Reserve OpenRouter free-tier calls for family diversity — one debater per question from OpenRouter ensures cross-provider robustness. Use Gemini or Groq for synthesis, where response quality matters most.

### 4.4 Rate Limit Handling

All root-level integration scripts implement rate-limit handling with varying levels of sophistication:

1. **Minimum call interval**: Most scripts enforce a 3–15 second delay between API calls.
2. **Exponential backoff**: On 429 (rate limit) responses, scripts wait with exponential backoff (10s → 20s → 40s → 80s → 160s) before retrying.
3. **Pre-flight checks**: `test_moderate_v3.py` and `test_moderate_final.py` probe the API before starting to check if the daily quota is exhausted.
4. **Serial execution**: All scripts make API calls sequentially to stay well within free-tier rate limits.

When running integration tests, expect significant wall-clock time due to rate-limit delays. A moderate-complexity test typically takes 2–5 minutes including sleep time.

---

## 5. Known Test Gaps

### 5.1 Cross-Provider Fallback (canonical_id-based)

The `ModelRegistry.resolve_fallback()` method supports fallback chains based on canonical model IDs, allowing the system to transparently switch to an equivalent model on a different provider when the primary model is unavailable. This critical resilience feature has **no unit test** yet. A proper test would:

1. Create a registry with cross-provider fallback chains (e.g., `openai/gpt-4.1-mini` → `openrouter/meta-llama/llama-3.3-70b-instruct:free`)
2. Mark the primary model as unavailable
3. Verify that `resolve_fallback()` returns the secondary model
4. Verify that the fallback model has a different provider but equivalent tier

### 5.2 ProviderInfo Parsing from providers.yaml

The `providers.yaml` file defines provider-level configuration (API base URLs, environment variable names for keys, priority rankings, regeneration status). The code that parses this YAML into `ProviderInfo` objects has no dedicated unit test. A test should verify:

- Correct parsing of provider entries with all fields
- Missing optional fields use safe defaults
- Unknown providers are handled gracefully
- Priority ordering is preserved

### 5.3 Two-Strikes Escalation Edge Cases

The failure tracking system uses a "two-strikes" escalation model where models accumulate failure counts and enter cooldown periods. Edge cases that lack test coverage include:

- **Midnight rollover**: Failure counts that span UTC midnight, when daily rate limits reset
- **Concurrent failures**: Two simultaneous failures on the same model from different agents
- **Cooldown expiry**: Verifying that a model is re-selected after its cooldown period ends
- **Cascading failures**: What happens when all models in a fallback chain fail

### 5.4 Checkpoint Resume with Partially Completed Phases

The `CheckpointManager` can save and load pipeline state, enabling resume after interruption. However, there's no test for resuming from a partially completed phase — for example, when the debate phase completes 1 of 2 rounds before crashing. The existing `TestRunDebate.test_debate_with_existing_state` test covers the happy path but not the edge case where the checkpoint contains an incomplete round with missing positions or agreement matrices.

### 5.5 PDF/DOCX Output Formatting

The `council/output.py` module supports PDF (via WeasyPrint) and DOCX (via python-docx) output formats, but these are not tested by `test_output.py`. The current tests only cover markdown, JSON, and plain text. PDF and DOCX output is difficult to test without visual inspection or complex snapshot testing. Possible approaches:

- Verify that PDF generation produces a non-empty byte stream
- Verify that DOCX generation creates a valid ZIP archive (DOCX is ZIP-based)
- Use simple string matching on the DOCX XML content
- Reserve visual inspection for manual QA

### 5.6 Root-Level Scripts Should Be Migrated

The 11 root-level integration test scripts share a significant amount of duplicated code (particularly the `_zai_acompletion` helper and the report-printing boilerplate). They should be migrated into `tests/` with proper pytest fixtures, shared helper modules, and the `@pytest.mark.integration` marker. Benefits of migration:

- **Deduplication**: The `_zai_acompletion` helper would be defined once in a `conftest.py` or shared fixture module.
- **Discoverability**: `pytest --collect-only` would show all tests, including integration ones.
- **Reporting**: pytest's built-in reporting (junitxml, coverage) would capture integration test results.
- **Parameterization**: Provider-specific tests could use `@pytest.mark.parametrize` instead of separate scripts.

Suggested migration path:
1. Create `tests/integration/` directory with a `conftest.py` containing shared fixtures
2. Convert each root script to a pytest test class with `@pytest.mark.integration`
3. Replace `argparse`-based configuration with pytest fixtures and `@pytest.mark.parametrize`
4. Delete root scripts after migration is validated

---

## 6. Test Data Considerations

### 6.1 Current Model ID Fixtures

The current test fixtures use `"openai/gpt-4.1-mini"` as a generic model ID throughout the test suite. This ID appears in `SAMPLE_MODELS` (test_models.py), `make_registry()` helpers (test_scout.py, test_debate.py, test_synthesis.py, test_research.py), and various inline constructions. While these tests don't make real API calls (everything is mocked), the fixture names should still be consistent with the actual provider configuration to avoid confusion.

The problem: `"openai/gpt-4.1-mini"` is **not** a valid model ID in `providers.yaml`. It's a litellm-format identifier that assumes direct OpenAI API access, which the project doesn't use as a primary provider. The project routes calls through Cloudflare Workers AI, OpenRouter, and other providers, so the model IDs in `providers.yaml` look like `"openai/@cf/meta/llama-3.3-70b-instruct-fp8-fast"` or `"openrouter/meta-llama/llama-3.3-70b-instruct:free"`.

### 6.2 Recommended Fixture Updates

Test fixtures should use model IDs that actually exist in the project's `providers.yaml`, prioritizing free-tier models that are available for integration testing:

| Current Fixture | Suggested Replacement | Rationale |
|----------------|----------------------|-----------|
| `"openai/gpt-4.1-mini"` (MID tier) | `"openrouter/meta-llama/llama-3.3-70b-instruct:free"` | Free-tier MID model, actually in providers.yaml |
| N/A (no CHEAP tier fixture) | `"openai/@cf/meta/llama-3.2-3b-instruct"` | Free Cloudflare CHEAP model, in providers.yaml |
| `"openai/gpt-4.1"` (PREMIUM tier) | Keep as-is or use `"anthropic/claude-sonnet-4"` | Premium models are rarely called in free-tier testing |

This change would make the test data more realistic and reduce confusion when debugging test failures alongside integration test results. It would also make it possible to write integration tests that use the same model IDs as the unit test fixtures, enabling a smoother transition when migrating root-level scripts into `tests/`.

### 6.3 Test Isolation Notes

Several test classes mutate shared state (e.g., `ModelRegistry.available_models()` modifies `is_available` flags on `ModelInfo` objects). While this doesn't cause test failures due to fresh object creation in most `make_registry()` helpers, developers should be aware that:

- Tests that modify `ModelInfo.is_available` should create fresh registries, not reuse module-level singletons.
- The `reset_deberta()` function is called in `setup_method` for NLI tests to ensure the global DeBERTa model state doesn't leak between tests.
- The `litellm.acompletion` monkey-patch in root-level scripts is global and not safe for concurrent execution — run only one integration script at a time.

---

## Appendix: Quick Reference

```bash
# Run all unit tests
uv run pytest tests/ -v -m "not integration"

# Run integration tests (needs API keys)
uv run pytest tests/ -v -m integration

# Run a specific module
uv run pytest tests/test_nli.py -v

# Run with coverage
uv run pytest tests/ --cov=council --cov-report=term-missing

# Run root-level integration scripts
uv run python test_trivial.py
uv run python test_moderate_final.py --provider gemini
uv run python test_round_robin.py

# Health check before integration testing
uv run council check
```
