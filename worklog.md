---
Task ID: 1
Agent: main
Task: Debt cleanup and documentation update for Deliberative Council

Work Log:
- Created .gitignore with .env excluded, Python artifacts, checkpoint DB, root-level test scripts
- Fixed config/default.yaml: replaced hardcoded "openai/gpt-4.1-mini" NLI tier2 model with "openrouter/meta-llama/llama-3.3-70b-instruct:free"
- Added ProviderInfo type to types.py for provider metadata
- Added providers field to CouncilConfig (dict[str, ProviderInfo])
- Updated load_config() to parse providers: section from providers.yaml into ProviderInfo objects
- Updated ModelRegistry.__init__ to accept providers dict
- Updated ModelRegistry.from_config to pass provider metadata from config
- Rewrote resolve_fallback() with 3-strategy cascade: cross-provider same-model → explicit chain → same-tier round-robin
- Added _cross_provider_fallback() method using canonical_id lookup + provider priority sort + conserve filter
- Rewrote README.md with accurate current info (correct config path, correct env vars, pipeline description, provider priority table, doc links)
- Rewrote API_KEYS_GUIDE.md to match current providers (6 providers, Cloudflare+OpenRouter focus, quota management, security notes)
- Created DESIGN_DOC.md: 9 sections covering architecture, pipeline, selection, failure tracking, cross-provider dedup, NLI, budget, data flow, principles
- Created ENGINEERING_DOC.md: 14-module reference, config system, registry internals, LLM client error handling, Cloudflare integration, checkpoint system
- Created PROJECT_STATE.md: current status, server environment, recently completed items, known issues, debt status, roadmap
- Created TEST_REPORT.md: test infrastructure, coverage per module, running instructions, integration strategy, test gaps, fixture considerations
- Added 5 cross-provider fallback tests to test_models.py (TestCrossProviderFallback class)
- Added 4 ProviderInfo tests to test_config.py (TestProviderInfo class)
- Updated test_config.py DEFAULT_MODELS tests to match free-tier models (removed gpt-4.1, added Cloudflare/OpenRouter assertions)
- All 259 unit tests pass

Stage Summary:
- All high-priority debt items resolved
- Cross-provider fallback fully wired up with provider metadata
- 6 documentation files created/updated (README, API_KEYS_GUIDE, DESIGN_DOC, ENGINEERING_DOC, PROJECT_STATE, TEST_REPORT)
- New test coverage for cross-provider fallback and ProviderInfo parsing
