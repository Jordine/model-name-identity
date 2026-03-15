# model-name-identity

Survey of self-identification discrepancies across 102 LLMs. When asked "who are you?", 38 models self-reported as a different LLM on at least one prompt.

## What's here

**Sweep code:**
- `runner.py` — main sweep runner (async, OpenRouter API)
- `models.py` — 102 model definitions with expected identities
- `prompts.py` — 32 prompts: casual, direct identity (EN/ZH), creator, system prompt probes
- `compile_results.py` — compiles raw responses into analysis JSON
- `analyze_discrepancies.py` — detects identity claims via regex, flags discrepancies

**Follow-up probes:**
- `depth_probe.py` — multi-turn depth probes ("Who is Claude?" → "and who are you?")
- `deep_identity_probe.py` — confrontation probes ("What if I told you that you're actually {name}?")
- `specific_identity_probe.py` — "What specific model version are you?" (n=15 per model)
- `provider_sweep.py` — preflight check for hidden system prompts across providers

**Analysis:**
- `plot_analysis.py` — generates plots
- `test_runner.py` — tests for the sweep infrastructure

**Data:**
- `results/responses_no_probes.jsonl` — main sweep responses (5,701 records, 102 models x ~56 prompts)
- `results/depth_probes.jsonl` — depth probe responses
- `results/deep_identity_probes.jsonl` — confrontation probe responses
- `results/specific_identity_probes.jsonl` — specific version claim responses
- `results/compiled_analysis.json` — compiled analysis with discrepancy flags
- `results/metadata.json` — sweep metadata
- `results/preflight.jsonl` — provider preflight checks
- `results/provider_sweep.jsonl` — provider system prompt detection
- `results/depth_probe_targets.json` — models selected for depth probing
- `plots/` — generated figures

## Methodology

32 prompts per model via OpenRouter, no system prompt, temperature 0.7, max 500 tokens. 7 key prompts repeated 4x for stochastic detection. Identity claims detected in both response text and thinking/reasoning traces via regex. 25 models excluded due to provider-injected system prompts.

See the [LessWrong post](TODO) for full methodology, results, and discussion.

## Running

Requires an OpenRouter API key at `~/.secrets/openrouter_api_key`.

```bash
# Main sweep
python runner.py

# Compile results
python compile_results.py

# Follow-up probes (run after main sweep)
python depth_probe.py
python deep_identity_probe.py
python specific_identity_probe.py

# Generate plots
python plot_analysis.py
```
