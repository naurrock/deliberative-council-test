# Deliberative Council — API Key & Provider Guide

## Provider Overview

| Priority | Provider   | Free Quota           | Regen?  | Geo-blocked (HK)? | Models |
|----------|------------|----------------------|---------|-------------------|--------|
| 0        | Cloudflare | 10K neurons/day      | Daily   | No                | 14     |
| 1        | OpenRouter | 50 RPD shared bucket | Daily   | No                | 8      |
| 2        | Gemini     | 1,500 RPD            | Daily   | **Yes (403)**     | 3      |
| 3        | Groq       | ~14,400 RPD          | Daily   | **Yes (403)**     | 3      |
| 4        | SambaNova  | $5 signup credit     | **No**  | No                | 2      |
| 5        | Cerebras   | Very limited         | **No**  | **Yes (403)**     | 1      |

**Minimum viable setup**: Cloudflare + OpenRouter (22 non-geo-blocked models).

---

## Setting Up API Keys

### Step 1: Copy the template

```bash
cp .env.template .env
```

### Step 2: Fill in keys

#### Cloudflare Workers AI (Priority 0 — START HERE)

1. Go to https://dash.cloudflare.com/profile/api-tokens
2. Click "Create Token" → "Workers AI" template
3. Copy the token → `CLOUDFLARE_API_KEY`
4. Get your Account ID from the URL bar after logging in → `CLOUDFLARE_ACCOUNT_ID`
5. No credit card required

```ini
CLOUDFLARE_API_KEY=your-token-here
CLOUDFLARE_ACCOUNT_ID=your-account-id-here
# CLOUDFLARE_API_BASE is auto-generated from the account ID
```

**Quota**: 10,000 neurons/day (regenerates daily). A 70B model uses ~20-30
neurons per request, so you get roughly 300-500 requests/day on large models,
or thousands on small models.

**Why Cloudflare first?**: 14 models across 11 families (Llama, Qwen, DeepSeek,
GPT-OSS, Llama4, Gemma, Mistral, Nemotron, GLM, Moonshot, QwQ) — maximum
family diversity for debate. Daily regenerating quota means it never runs out
permanently.

#### OpenRouter (Priority 1)

1. Go to https://openrouter.ai/settings/keys
2. Create a key → `OPENROUTER_API_KEY`
3. No credit card required for free models

```ini
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

**Quota**: 20 RPM, 50 RPD on the free tier (shared across ALL `:free` models).
If you spend $10+ on paid models, the limit increases to 1,000 RPD.

**Important**: All 8 free models share the same 50 RPD bucket. Each request to
any `:free` model counts against the same limit.

#### Gemini (Priority 2 — GEO-BLOCKED from HK)

1. Go to https://aistudio.google.com/apikey
2. Create a key → `GEMINI_API_KEY`
3. No credit card required

```ini
GEMINI_API_KEY=your-key-here
```

**Quota**: 15 RPM, 1,500 RPD, 1M TPM — very generous. However, the server is
in Hong Kong, which gets 403 Forbidden from Gemini's API. These models are
registered with `geo_blocked: true` and are never tried. They would work if
the server moved to an allowed region.

#### Groq (Priority 3 — GEO-BLOCKED from HK)

1. Go to https://console.groq.com/keys
2. Create a key → `GROQ_API_KEY`
3. No credit card required

```ini
GROQ_API_KEY=your-key-here
```

**Quota**: ~14,400 RPD. Fast inference (LPU hardware). Same 403 geo-blocking
issue as Gemini from Hong Kong.

#### SambaNova (Priority 4 — CONSERVE)

1. Go to https://cloud.sambanova.ai/
2. Sign up → `SAMBANOVA_API_KEY`
3. $5 free credits on signup — **does not regenerate**

```ini
SAMBANOVA_API_KEY=your-key-here
```

**Important**: These credits are non-regenerating. SambaNova models are flagged
`conserve: true` in providers.yaml — the selection algorithm avoids them unless
all non-conserved options are exhausted. Save them for testing or emergencies.

#### Cerebras (Priority 5 — GEO-BLOCKED + CONSERVE)

1. Go to https://cloud.cerebras.ai/
2. Sign up → `CEREBRAS_API_KEY`

```ini
CEREBRAS_API_KEY=your-key-here
```

Dual disadvantage: geo-blocked from HK AND non-regenerating credits. Listed for
completeness but effectively unusable from the current server.

---

## Quota Management

### How the System Handles Quota Exhaustion

The two-strikes escalation system handles quota automatically:

1. **Strike 1** (any error): Model is marked TRANSIENT with a 60-second
   cooldown. This handles RPM throttling — wait 60s and try again.

2. **Strike 2** (fails again while still cooling): Model is marked DAILY,
   meaning it won't be retried until midnight UTC. This handles RPD exhaustion
   — no point retrying until the daily quota resets.

3. **Fallback**: When a model is DAILY, `resolve_fallback()` tries:
   - Same model on a different provider (cross-provider fallback)
   - Next model in the explicit fallback chain
   - Any available model at the same tier (round-robin)

### Daily Budget Planning

With Cloudflare (14 models) + OpenRouter (8 free models), the effective daily
budget for a typical run:

| Complexity  | API Calls | Models Used              | Daily Runs |
|-------------|-----------|--------------------------|------------|
| Trivial     | 2-3       | Scout + Verifier         | ~100+      |
| Moderate    | 8-12      | + 2-3 Debaters × 1 round | ~25-40     |
| Complex     | 20-30     | + Research agents        | ~10-15     |
| Deep        | 40-60     | + 3 rounds × 3 debaters | ~5-8       |

The bottleneck is typically OpenRouter's 50 RPD shared bucket, not Cloudflare's
neuron budget. Cloudflare handles the bulk of the load; OpenRouter provides
fallback diversity.

---

## Security Notes

### Risk Assessment

| Provider                | Risk if Key Exposed | Why                                    |
|-------------------------|--------------------|-----------------------------------------|
| Cloudflare (free)       | VERY LOW           | No credit card, hard neuron caps        |
| OpenRouter (free)       | VERY LOW           | Free models have no billing             |
| Gemini (free)           | VERY LOW           | No credit card, hard rate caps          |
| Groq (free)             | VERY LOW           | No credit card, rate-limited            |
| SambaNova ($5 credit)   | LOW                | $5 cap, no auto-billing                 |
| Cerebras                | LOW                | Limited free tier, no auto-billing      |

None of the configured providers have credit cards on file or auto-billing
enabled. The maximum financial exposure is $5 (SambaNova signup credit).

### Key Safety

- **NEVER** commit `.env` to version control (it's in `.gitignore`)
- All keys are free-tier with hard caps — no billing risk
- If a key is accidentally exposed, revoke it at the provider's dashboard
