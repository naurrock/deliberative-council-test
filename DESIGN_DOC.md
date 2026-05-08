# Deliberative Council — Architecture & Design Decisions

## 1. System Overview

Deliberative Council is a Python multi-agent LLM debate system that produces robust, nuanced answers to questions by orchestrating a structured pipeline of specialized AI agents. The system's core premise is that a single LLM call—even to a powerful model—is insufficient for questions requiring genuine reasoning, trade-off analysis, or factual verification. Instead, the Council deploys multiple agents with distinct perspectives and model families through a four-phase pipeline—Scout, Research, Debate, Synthesis—each designed to add rigor and catch the failures that any individual model would miss.

The central problem Deliberative Council solves is **getting high-quality answers from free-tier LLMs**. The landscape of free LLM APIs is fragmented: Cloudflare Workers AI offers 10K neurons/day across 55 models, OpenRouter provides 28 free models at 50 RPD, Gemini gives 1,500 RPD but geo-blocks certain regions, Groq offers blazing speed but is also geo-blocked, and SambaNova provides non-regenerating signup credits. No single provider has the quota to handle sustained multi-agent workloads, and each provider's free tier resets on different schedules. Deliberative Council treats this fragmentation as a feature rather than a bug: by spreading load across providers, it maximizes total daily API capacity while using the diversity of model families (Llama, Qwen, DeepSeek, Gemma, Mistral, Nemotron, GLM, Moonshot, QwQ, and more) to generate genuinely different reasoning perspectives for debate.

The system is designed around three constraints: **cost** (everything must be free-tier), **reliability** (providers fail constantly—429s, geo-blocks, key revocations), and **quality** (the output must be better than any single free model could produce alone). Every design decision—from the two-strikes failure tracker to the canonical_id cross-provider deduplication system—exists to serve these constraints.

---

## 2. Pipeline Architecture

The pipeline processes a question through four sequential phases, each building on the output of the previous one. The engine (`council/engine.py`) orchestrates the full flow with checkpoint/resume support via SQLite, enabling crash recovery at any phase boundary.

```
Question
  │
  ▼
┌──────────────────────────────────────────────────┐
│  Phase 1: SCOUT                                 │
│  Cheap classifier → Mid-tier verifier            │
│  Produces: MissionBrief                          │
└──────────────────┬───────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────┐
│  Phase 2: RESEARCH (if needed)                   │
│  Parallel search agents via Jina.ai              │
│  Produces: list[EvidenceReport]                  │
└──────────────────┬───────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────┐
│  Phase 3: DEBATE                                 │
│  Multi-agent with NLI agreement tracking         │
│  Produces: DebateState                           │
└──────────────────┬───────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────┐
│  Phase 4: SYNTHESIS                              │
│  Impartial model from different family            │
│  Produces: FinalReport                           │
└──────────────────────────────────────────────────┘
```

### 2.1 Scout Phase

The Scout phase uses a **two-agent architecture** to classify a question and plan the pipeline. A cheap-tier model (e.g., Llama-3.2-3B) acts as the primary investigator, analyzing the question's complexity, domain, and research requirements. It optionally performs a preliminary web search for context, then produces a draft `MissionBrief` containing the complexity classification (`TRIVIAL`, `MODERATE`, `COMPLEX`, or `DEEP`), suggested debate roles with full system prompts, research sub-questions, debate round count, and token budget allocation.

A mid-tier verification agent then reviews this draft, catching common scout errors: trick questions misclassified as trivial, under-budgeting for genuinely hard questions, too few debate roles, missing perspectives, and unsolvable questions that should be flagged. This two-agent pattern—cheap drafter, mid-tier verifier—recurring throughout the system, provides quality assurance at minimal cost since the verifier only sees the draft rather than doing independent analysis.

For `TRIVIAL` questions, the pipeline short-circuits directly to synthesis, bypassing research and debate entirely. Scout has its own 50K token budget tracked separately from the main pipeline budget.

### 2.2 Research Phase

When the Scout determines research is needed, the Research phase deploys parallel search-and-ingest agents. Each agent investigates one sub-question from the MissionBrief using the Jina.ai suite: `s.jina.ai` for web search and `r.jina.ai` for content extraction. This provides free, high-quality web access without requiring API keys.

The critical design element is **epistemic tagging**. Every research finding is tagged with one of three levels:

| Tag | Meaning | Trust Level |
|-----|---------|-------------|
| `SOURCED` | Directly traceable to a specific source URL | Highest |
| `INFERRED` | Synthesized from multiple sources | Medium |
| `JUDGMENT` | Agent's own assessment, no external source | Lowest |

In `strict` research mode (the default), judgment-tagged findings are filtered out of the debate context entirely, and sourced findings without a URL are downgraded to inferred. This prevents the debate from building on unchecked LLM hallucinations. In `augmented` mode, all findings are included but clearly labeled, allowing for more speculative exploration while maintaining epistemic transparency.

Research agents run concurrently (up to `max_concurrent_agents`, default 3) with per-agent sub-budgets. The total research budget is the main budget multiplied by `research_share` (0.3), and each sub-question gets an equal share of that.

### 2.3 Debate Phase

The Debate phase is the core of the system. Multiple agents with distinct perspectives and model families engage in structured multi-round debate, tracked by a two-tier NLI agreement system. Each round consists of:

1. **Position generation** — All agents concurrently state their positions, citing evidence and rating their confidence. Novelty injection is triggered for agents whose position stability has been above the threshold for N consecutive rounds, forcing them to approach the question from a new angle.

2. **Critique generation** — Each agent critiques the positions of other agents visible in their communication graph, identifying points of agreement, disagreement, and counter-arguments.

3. **Agreement computation** — The two-tier NLI system computes pairwise agreement scores (see Section 6 for details), producing an agreement matrix and tracking when Tier 2 (LLM-based analysis) was invoked.

4. **Behavioral signal computation** — Position stability, evidence novelty, and novelty injection status are tracked per agent.

The debate terminates when one of several conditions is met: **convergence** (NLI agreement above threshold for consecutive rounds), **futility** (agents are stuck in unproductive cycles), **budget exhaustion** (hard ceiling), or **round limit** (soft ceiling, though the system can extend if disagreement is still high).

**Futility detection** uses a two-layer approach: a free heuristic layer checks for obvious signals (all agents stuck with high position stability, frozen agreement scores), and an LLM escalation layer provides human-like judgment when heuristics are ambiguous (confidence between 0.3 and 0.7). This mirrors the two-tier NLI pattern: use the cheap/free method first, escalate only when needed.

### 2.4 Synthesis Phase

The Synthesis phase produces the final `FinalReport` using a premium-tier model from a **different model family** than any debater. This is a deliberate design choice: if the synthesizer shares a model family with a debater, it may have an inherent affinity for arguments generated by that family's reasoning patterns. By selecting a synthesizer from an unrepresented family, the system ensures impartial weighing of all positions.

The synthesizer receives a structured debate summary (positions, agreement matrices, critiques, futility flags) and evidence summary (with epistemic tags). It produces a markdown answer, key points with consensus levels (`STRONG`, `MODERATE`, `CONTESTED`), dissenting views, and source URLs. The synthesis prompt explicitly instructs the model not to simply pick the majority view—a well-argued minority position may be correct.

If synthesis fails entirely, a fallback report is generated from the raw positions of the final debate round, ensuring the system always produces some output.

---

## 3. Model Selection & Fallback

Model selection is one of the most complex subsystems, designed to maximize free-tier utilization while maintaining family diversity for genuine debate. The `ModelRegistry` manages model availability, health, round-robin distribution, and a three-tier fallback strategy.

### 3.1 Selection Algorithm

When selecting a model for a role, the registry:

1. **Filters out geo-blocked models** — Models with `geo_blocked: true` in config are never selected, never tracked by the failure system.
2. **Applies user constraints** — Family preference, exclusion list, local-only or API-only mode.
3. **Maximizes family diversity** — For debaters, prefer unassigned families first; if all families are taken, pick from the least-used family.
4. **Selects by tier** — Scout/research use CHEAP, debaters use MID, synthesizer uses PREMIUM. Falls back to adjacent tier if no matches.
5. **Round-robins within the pool** — Distributes load across models at the same tier, preventing a single model from being hammered.

### 3.2 Three-Layer Fallback Strategy

When a model fails, `resolve_fallback` walks through three layers in order:

| Layer | Strategy | Example |
|-------|----------|---------|
| **1. Cross-provider same-model** | Same `canonical_id`, different provider | Cloudflare's llama-3.3-70b fails → try OpenRouter's llama-3.3-70b |
| **2. Explicit fallback chain** | Configured per role type in YAML | Debater chain: CF Llama → CF Qwen → CF DeepSeek → OR Llama → ... |
| **3. Same-tier round-robin** | Any available model at the same tier | Any MID-tier model not in cooldown |

Cross-provider fallback (Layer 1) is the highest-value strategy: you get the same reasoning capability through a different route. This is why the `canonical_id` system exists (see Section 5).

### 3.3 Conserve Flag

Models backed by non-regenerating credits (e.g., SambaNova's $5 signup credits) are flagged with `conserve: true`. The round-robin selector prefers non-conserved models and only uses conserved ones as a last resort. This ensures one-time credits are preserved for situations where no regenerating option is available.

### 3.4 Weighted Round-Robin with Family Diversity

Within a tier, models are selected via round-robin with health-aware sorting. The `_rr_select` method sorts candidates by health status (prefer non-cooling, non-daily-exhausted, non-conserved), then applies round-robin within the healthy pool. This distributes load evenly while respecting failure states.

---

## 4. Failure Tracking — Two-Strikes Escalation

The failure tracking system is the core reliability mechanism, built on a key design insight: **you don't need to classify errors by message content—the pattern speaks for itself**.

### 4.1 The Two-Strikes Rule

| Strike | Severity | Cooldown | Meaning |
|--------|----------|----------|---------|
| 1st failure | `TRANSIENT` | 60 seconds | Could be anything: RPM throttle, network blip, timeout, even 403/401 |
| 2nd failure (while TRANSIENT active) | `DAILY` | Until midnight UTC | Confirmed exhaustion: RPD quota depletion, persistent outage, revoked key |

A success resets the strike count to zero. The second failure is only registered if it occurs while the first strike's cooldown is still active—if a model fails, recovers after 60s, and then works fine, the strike count resets. But if it fails, waits 60s, and fails again, that's strong evidence of daily quota exhaustion rather than a transient hiccup.

### 4.2 Why Not Parse Error Messages?

Many failure tracking systems try to classify errors by parsing HTTP status codes and error messages: "429 means rate limit, 401 means bad key, 403 means forbidden." This approach has several problems in the free-tier LLM ecosystem:

- **Error messages are inconsistent** across providers. OpenRouter returns one format, Cloudflare another, Groq yet another. A 429 from one provider might mean RPM throttle (try again in 60s) while a 429 from another might mean RPD exhaustion (try again tomorrow).
- **Error messages are unreliable**. Some providers return 500 for quota exhaustion. Some return 429 for geo-blocking. Some return cryptic internal error messages.
- **Error messages change**. Provider APIs evolve; a classification system based on string matching becomes a maintenance burden.

The two-strikes system sidesteps all of this. It doesn't care *why* a model failed—it only cares about the *pattern*: "failed once" vs "failed twice in quick succession." This naturally distinguishes RPM throttling (one 429, then it works after 60s) from RPD exhaustion (429, wait 60s, still 429 → DAILY). No provider-specific error classification needed. No brittle message parsing. The pattern is the signal.

### 4.3 Geo-Blocking is a Config Concern

Models known to be geo-blocked (e.g., Gemini and Groq from Hong Kong) are flagged with `geo_blocked: true` in `providers.yaml`. They are **filtered at selection time** and never reach the FailureTracker at all. This is a deliberate separation of concerns: geo-blocking is a known, static property of the deployment environment, not a runtime failure condition. Runtime-discovered geo-blocking (a model that should work but doesn't due to an unexpected IP restriction) will naturally escalate through the two-strikes system: fail once (TRANSIENT), fail again (DAILY), effectively blocking it for the day at a cost of only two failed requests.

There is no "permanent" severity level. Models with permanently revoked keys will simply re-escalate to DAILY each day, costing at most one attempt per day—negligible waste compared to the complexity of a permanent-block system.

---

## 5. Cross-Provider Deduplication

The free LLM ecosystem has significant model overlap: the same underlying model weights are available through multiple providers, each with different quotas, rate limits, and latency characteristics. The `canonical_id` system captures this relationship.

### 5.1 Canonical ID

A `canonical_id` identifies the underlying model independently of the provider. For example:

| Model ID | Provider | canonical_id |
|----------|----------|-------------|
| `openai/@cf/meta/llama-3.3-70b-instruct-fp8-fast` | Cloudflare | `llama-3.3-70b-instruct` |
| `openrouter/meta-llama/llama-3.3-70b-instruct:free` | OpenRouter | `llama-3.3-70b-instruct` |
| `groq/llama-3.3-70b-versatile` | Groq | `llama-3.3-70b-instruct` |
| `sambanova/Meta-Llama-3.3-70B-Instruct` | SambaNova | `llama-3.3-70b-instruct` |
| `cerebras/llama-3.3-70b` | Cerebras | `llama-3.3-70b-instruct` |

These are all the same Llama 3.3 70B model, just served through different routes. The canonical_id tells the system they're interchangeable for fallback purposes.

### 5.2 Provider Priority Ordering

Providers are ordered by priority (lower number = preferred):

| Priority | Provider | Daily Quota | Notes |
|----------|----------|-------------|-------|
| 0 | Cloudflare | 10K neurons/day | Daily regen, 55 models, best ongoing option |
| 1 | OpenRouter | 50 RPD | 28 free models, good family diversity |
| 2 | Gemini | 1,500 RPD | Geo-blocked from HK |
| 3 | Groq | ~14,400 RPD | Fastest inference, geo-blocked from HK |
| 4 | SambaNova | $5 credits (non-regenerating) | Conserve for critical needs |
| 5 | Cerebras | Very limited | Geo-blocked, non-regenerating |

### 5.3 Cross-Provider Fallback in Practice

When `resolve_fallback` is called for a failed model, it first checks if the model has a `canonical_id`. If it does, it looks up all other models sharing that canonical_id, filters out the failed model itself, geo-blocked models, daily-exhausted models, and models in cooldown, then sorts the remaining alternatives by provider priority. This ensures that when Cloudflare's llama-3.3-70b exhausts its daily neurons, the system seamlessly switches to OpenRouter's llama-3.3-70b, preserving the same reasoning capability without disrupting the debate.

The cross-provider fallback also prefers non-conserved providers over conserved ones. If both SambaNova and OpenRouter offer the same model and both are available, OpenRouter (regenerating) is preferred over SambaNova (non-regenerating credits).

---

## 6. NLI Agreement System

The agreement tracking system determines when debate agents have converged on shared conclusions. It uses a two-tier architecture that balances cost, accuracy, and speed.

### 6.1 Tier 1: DeBERTa-v3-large (Local, Free, Deterministic)

The first tier uses `cross-encoder/nli-deberta-v3-large`, a locally-running Natural Language Inference model. It takes two texts as premise and hypothesis, classifies their relationship (entailment, neutral, contradiction), and returns the entailment probability as an agreement score in [0, 1].

**Advantages**: Completely free (runs on CPU), deterministic (same input → same output), and fast for short-to-medium text. The model runs lazily—loaded only on first use—and is forced to CPU to avoid GPU contention with other processes.

**Limitation**: DeBERTa has a 512-token input limit. For longer texts, the system chunks both inputs at sentence boundaries (max 500 tokens per chunk to leave room for special tokens) and averages the pairwise chunk scores. This chunking approach works reasonably well but loses some cross-paragraph coherence.

### 6.2 Tier 2: Cheap LLM (Accurate for Long Arguments)

When Tier 1 is uncertain, the system escalates to a cheap LLM (configured as `tier2_model`, default: `openrouter/meta-llama/llama-3.3-70b-instruct:free`). The LLM receives both full arguments and produces a structured `AgreementAnalysis` with:

- Whether positions substantively agree
- Agreement score (0.0–1.0)
- Specific points of agreement and disagreement
- Whether the disagreement is fundamental or superficial
- A one-sentence summary

### 6.3 Uncertainty Zone

Tier 2 is only invoked when Tier 1's score falls in the **uncertainty zone** between `tier1_uncertain_low` (0.25) and `tier1_uncertain_high` (0.65):

```
  0.0                    0.25     0.65                    1.0
  ├───────────────────────┤█████████├───────────────────────┤
  Confident disagreement   Uncertain  Confident agreement
  (use Tier 1 score)       (escalate   (use Tier 1 score)
                           to Tier 2)
```

This gating ensures that Tier 2 LLM calls—which cost tokens—are only made when Tier 1 genuinely can't decide. In practice, most agreement checks are resolved by Tier 1 alone, keeping the system fast and free.

### 6.4 Graceful Degradation

If DeBERTa fails to load (missing `transformers` dependency, insufficient memory, etc.), the system falls back to a simple Jaccard word-overlap heuristic that returns scores in the 0.35–0.60 range— squarely in the uncertain zone. This ensures Tier 2 is always invoked when Tier 1 is unavailable, maintaining agreement tracking at the cost of more LLM calls. The system never crashes due to a missing DeBERTa model; it just costs a bit more.

---

## 7. Budget Management

Token budget management prevents any single phase from monopolizing the pipeline's capacity. The system uses a hierarchical budget structure with clear ownership.

### 7.1 Budget Flow

```
Total Budget (e.g., 500K tokens)
  │
  ├── Scout Budget (50K, tracked separately)
  │
  └── Pipeline Budget (remainder)
       ├── Research Sub-Budget (30% = 150K)
       ├── Debate Sub-Budget  (50% = 250K)
       └── Synthesis Sub-Budget (20% = 100K)
```

The key invariant is that **phases do NOT double-apply shares**. The engine creates sub-budgets by multiplying the total budget by the configured share (`research_share`, `debate_share`, `synthesis_share`), then passes each sub-budget directly to the phase. The phase uses its sub-budget as-is without further share calculation. This prevents the subtle bug where a phase might inadvertently reduce its own budget by re-applying a share factor.

### 7.2 Scout's Separate Budget

Scout gets its own fixed 50K token budget, tracked in `MissionBrief.scout_tokens_used`. This is separate from the main pipeline budget because scout runs before the pipeline budget is allocated—the total budget for research/debate/synthesis is determined by the MissionBrief, which is produced by the scout itself. Mixing scout consumption into the pipeline budget would create a circular dependency.

### 7.3 Budget as Hard Ceiling

Budget exhaustion is a hard convergence ceiling. When the debate phase's sub-budget reaches zero, the debate terminates immediately with `ConvergenceReason.BUDGET_EXHAUSTED`, regardless of agreement levels or remaining rounds. This prevents runaway token consumption from a single question. The `TokenBudget` class tracks total and used tokens, and the `is_exhausted` property is checked before every LLM call.

### 7.4 Default Configuration

```yaml
budget:
  default_budget: 500000    # 500K tokens total
  max_per_agent: 100000     # 100K per agent
  research_share: 0.3       # 30% for research
  debate_share: 0.5         # 50% for debate (largest share)
  synthesis_share: 0.2      # 20% for synthesis
```

Debate receives the largest share because it's the most token-intensive phase (each round generates positions, critiques, and agreement checks for every agent pair). Synthesis needs the smallest share since it's a single LLM call.

---

## 8. Data Flow

Data flows through the pipeline as structured Pydantic models, providing runtime validation, serialization, and living documentation. Each phase's output becomes the next phase's input.

### 8.1 Type Flow Diagram

```
str (question)
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│  Scout                                                       │
│  Input: question (str)                                       │
│  Output: MissionBrief                                        │
│    ├── question: str                                         │
│    ├── complexity: Complexity (TRIVIAL|MODERATE|COMPLEX|DEEP)│
│    ├── domain_tags: list[str]                                │
│    ├── is_likely_solvable: bool                              │
│    ├── suggested_roles: list[RoleSpec]                       │
│    │     ├── name, perspective, expertise                    │
│    │     ├── suggested_model (family hint)                   │
│    │     └── system_prompt (full prompt for the role)        │
│    ├── research_needed: bool                                 │
│    ├── research_subquestions: list[str]                      │
│    ├── debate_rounds: int (0–5)                              │
│    ├── token_budget: int                                     │
│    ├── scout_tokens_used: int (separate tracking)           │
│    ├── scout_reasoning: str (full reasoning trace)           │
│    └── verification_notes: str                               │
└─────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│  Research                                                    │
│  Input: MissionBrief                                         │
│  Output: list[EvidenceReport]                                │
│    └── EvidenceReport                                        │
│          ├── agent_id: str                                   │
│          ├── sub_question: str                               │
│          ├── searches_performed: list[dict]                  │
│          ├── key_findings: list[ResearchFinding]             │
│          │     ├── claim: str                                │
│          │     ├── sources: list[EvidenceSource]             │
│          │     │     └── url, snippet, title                 │
│          │     ├── epistemic_tag: SOURCED|INFERRED|JUDGMENT  │
│          │     └── relevance: float (0.0–1.0)               │
│          ├── gaps: str                                       │
│          ├── recommended_investigation: str                  │
│          └── tokens_used: int                                │
└─────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│  Debate                                                      │
│  Input: MissionBrief + list[EvidenceReport]                  │
│  Output: DebateState                                         │
│    ├── mission_brief: MissionBrief                           │
│    ├── evidence_reports: list[EvidenceReport]                │
│    ├── rounds: list[RoundResult]                             │
│    │     ├── round_number: int                               │
│    │     ├── positions: list[Position]                       │
│    │     │     ├── agent_id, role_name, argument             │
│    │     │     ├── supporting_evidence: list[str]            │
│    │     │     ├── self_confidence: float                    │
│    │     │     ├── metacognitive_notes: str                  │
│    │     │     └── position_stability: float|None            │
│    │     ├── critiques: list[Critique]                       │
│    │     │     └── from→to with agreement/disagreement lists│
│    │     ├── agreement_matrix: dict[str, dict[str, float]]   │
│    │     ├── nli_tier2_invoked: dict[str, dict[str, bool]]   │
│    │     └── behavioral_signals: dict[str, BehavioralSignals]│
│    ├── communication_graph: dict[str, list[str]]             │
│    ├── futility_flags: list[str]                             │
│    ├── is_resolved: bool                                     │
│    └── resolution_reason: str|None                           │
└─────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│  Synthesis                                                   │
│  Input: MissionBrief + list[EvidenceReport] + DebateState    │
│  Output: FinalReport                                         │
│    ├── question: str                                         │
│    ├── complexity: Complexity                                │
│    ├── rounds_completed: int                                 │
│    ├── convergence_score: float                              │
│    ├── answer: str (markdown)                                │
│    ├── key_points: list[KeyPoint]                            │
│    │     ├── point: str                                      │
│    │     ├── consensus: STRONG|MODERATE|CONTESTED            │
│    │     ├── evidence: list[str] (source URLs)               │
│    │     └── dissent: str|None                               │
│    ├── dissenting_views: list[str]                           │
│    ├── research_sources: list[str] (all URLs)                │
│    ├── pipeline_trace: PipelineTrace                         │
│    │     ├── models_used: dict[str, ModelUsage]              │
│    │     ├── scout_tokens, research_tokens                   │
│    │     ├── debate_tokens, synthesis_tokens                 │
│    │     └── total_tokens (computed property)                │
│    ├── futility_notes: str|None                              │
│    └── raw_markdown: str (complete export)                   │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 Checkpoint Serialization

After each phase, the engine saves intermediate state to a SQLite database (`checkpoints.db`). The `CheckpointManager` stores each phase's output as JSON, enabling crash recovery and resume. If a run is interrupted during the debate phase, for example, the engine can reload the MissionBrief and EvidenceReport from the database and resume from the last completed phase.

---

## 9. Design Principles

### 9.1 Free-Tier First

Every design decision optimizes for maximizing daily API capacity at zero cost. The three-layer fallback strategy exists because no single free provider has sufficient quota for sustained multi-agent workloads. The two-tier NLI system exists because DeBERTa is free and local, with LLM escalation as a cost gate. The conserve flag exists to protect non-regenerating credits. When a choice must be made between quality and free-tier sustainability, the system chooses sustainability—because a system that exhausts its credits by noon produces no answers at all.

### 9.2 Family Diversity

Different model families represent genuinely different reasoning paradigms. Llama, Qwen, DeepSeek, Gemma, Mistral, Nemotron, GLM, Moonshot, and QwQ each have different training data, architectures, and default behaviors. When these models disagree, the disagreement is more likely to reflect genuine epistemic uncertainty than shared training artifacts. The system enforces family diversity at multiple levels: the Scout assigns different family hints to each role, the model registry's selection algorithm prefers unassigned families, and the Synthesizer is explicitly chosen from a family not represented in the debate.

### 9.3 Graceful Degradation

The system never crashes. If DeBERTa fails to load, it falls back to heuristics. If a research agent fails, it produces an empty EvidenceReport with a gap description. If synthesis fails, it generates a fallback report from raw debate positions. If all models for a tier are exhausted, it falls back to adjacent tiers. If JSON parsing fails, it uses raw text as the argument. Every component has a degradation path, ensuring the pipeline always produces some output—even if that output is less polished than ideal.

### 9.4 Config Over Code

Geo-blocking, provider priorities, quota limits, model lists, and fallback chains are all defined in YAML (`config/providers.yaml`, `config/default.yaml`), not in Python code. This means adding a new provider or model requires zero code changes—just a YAML edit. The `geo_blocked` flag, `conserve` flag, `canonical_id`, and provider priority are all configuration concerns that the code reads but never hardcodes. This separation makes the system adaptable to new providers, changing geo-restrictions, and evolving free-tier limits without developer intervention.

### 9.5 Pattern Over Parsing

The two-strikes failure escalation system embodies the principle that behavioral patterns are more reliable than error message content. Rather than trying to classify 429 vs 500 vs 403 vs timeout vs connection refused, the system simply asks: "did it fail once, or did it fail twice?" The pattern of two consecutive failures is a more reliable signal of quota exhaustion than any error message parsing, and it works identically across all providers regardless of their error format. This principle extends throughout the system: convergence detection uses agreement score patterns rather than semantic analysis, futility detection uses position stability patterns rather than argument classification, and budget management uses consumption patterns rather than cost estimation.
