# What do LLMs call themselves?

Ask a language model "who are you?" and a surprising number answer with **another
vendor's** name — a Qwen model calling itself DeepSeek, Kimi calling itself Claude,
Claude (in Chinese) calling itself DeepSeek. This repo is a survey of that: **180
models queried through hosted APIs + 16 run from raw weights on GPUs** (the
cross-model figures pool the 180 API models with 10 of the raw-weights models =
190), across 8 languages, with every self-identity claim extracted by an LLM judge
and adjudicated.

The study is three experiments on the same models: what a model **volunteers** as
its identity (the headline), what the **raw weights** say once the shipped chat
template is stripped, and how **suggestible** it is when you ask "are you X?".

## What are you looking for?

- **The full list of models** → [`MODELS.md`](./MODELS.md) — every model queried,
  every one run from raw weights on GPUs, and every one excluded (with the reason:
  usually a provider that injects its own "You are …" system prompt).
- **The exact prompts** → [`prompts.jsonl`](./prompts.jsonl) — the whole battery,
  no commentary: identity/creator/casual questions in 8 languages, the "are you
  X?" probes, and the placebos.
- **Verify a specific mismatch** (e.g. *"Claude Opus 4.8 → DeepSeek in Chinese"*)
  and copy the exact prompt/response → [`rollouts/MISMATCHES.md`](./rollouts/MISMATCHES.md)
  (searchable, grouped by model). For **every** answer from every model, open the
  full browser [`rollouts/index.html`](./rollouts/index.html) (needs GitHub Pages
  or any static host) or search the raw [`rollouts/rollouts_data.json`](./rollouts/rollouts_data.json).
- **The methodology + analysis code** → [`sweep/`](./sweep) — the whole pipeline,
  from provider-hygiene checks to the judge to the figures.
- **The figures** → [`figures/`](./figures).

## Repository layout

```
MODELS.md            every model: queried / raw-weights / excluded (bullets + reasons)
prompts.jsonl        the complete prompt battery, one prompt per line, no commentary
FUTURE_WORK.md       directions that would sharpen the mechanism (prefill-entropy, confrontation)
figures/             the final figures (fig_*.png) + per-family panels (family/)
rollouts/
  MISMATCHES.md      index of every model that names another vendor (rate + claims)
  mismatches/*.md    full records per vendor — every prompt + response, by language
  index.html         full browser over every model & answer (serve over http)
  rollouts_data.json data behind the browser (all 190 models, compact)
config/
  models.json        the model registry (id, family, expected identity, aliases)
  provider_hygiene.json  per-model provider decision (which endpoint, or why excluded)
  local_manifest.jsonl   the raw-weights GPU run spec (model, tensor-parallel, backend)
results/             API sweep: main_sweep.jsonl.gz (raw), judgments.jsonl.gz (judge),
                     adjudications*.jsonl, judge_bench/ (6-judge validation)
results_local/       raw-weights sweep: per-model responses + judgments_clean.jsonl
sweep/               all code (see Pipeline below)
```

## Pipeline

The registry and provider-hygiene decisions are committed under `config/`. To
reproduce the analysis from the collected data, or re-run the sweep end to end:

```bash
# API sweep
python -m sweep.preflight            # provider hygiene -> config/provider_hygiene.json
python -m sweep.runner --yes         # resumable sweep -> results/main_sweep.jsonl
python -m sweep.judge                # LLM judge (identity-claim extraction)
python -m sweep.bench_judge          # 6-judge agreement benchmark (judge validation)
python -m sweep.adjudicate           # Claude adjudication of every flagged claim

# raw-weights (GPU) sweep — on a 2×A100 box, HF weights, identity stripped from the template
python -m sweep.verify_prompts       # GATE: prove every clean prompt is identity-free
python -m sweep.run_parallel         # generate from weights -> results_local/
python -m sweep.adjudicate_local     # adjudicate raw-weights claims

# analysis + artifacts
python -m sweep.analyze              # discrepancy-rate tables
python -m sweep.make_figs            # figures/
python -m sweep.explain              # figures/fig_coherence, fig_lang_conditional, fig_size, fig_cutoff
python -m sweep.build_models_md      # MODELS.md (roster + pinned provider per model)
python -m sweep.build_rollouts       # rollouts/MISMATCHES.md
python -m sweep.build_rollout_viewer # rollouts/index.html + rollouts_data.json

# cross-provider check — same weights, every clean host (does the cloud change identity?)
python -m sweep.xprovider --run --boost --zh   # Opus 4.8 / Sonnet 4.6 x all providers, high-N zh CIs
python -m sweep.fig_xprovider        # figures/fig_xprovider.png
```

API calls go through a litellm proxy (`~/.secrets/litellm_api_key`); the GPU
scripts use vLLM (or the transformers backend for archs vLLM can't load).

## Method notes

- **No system prompt**, temperature 0.7, N=5 samples per prompt. The core battery
  (13 identity/creator/casual questions) is identical across all 8 languages
  (en zh ja ko ru fr es vi), so per-language rates are directly comparable.
- **Provider hygiene.** Each API model is pinned to one endpoint that serves it
  *without* injecting a system prompt. A hosted "You are ChatGPT…" preamble would
  fake (or mask) an identity, so models with no clean provider are excluded, not
  measured — see `sweep/preflight.py` and `config/provider_hygiene.json`.
- **Raw weights.** For the GPU models, any identity is stripped from the chat
  template (blank `model_identity` for harmony models, a neutral system override
  for others) and *verified* identity-free before generation
  (`sweep/verify_prompts.py`). This separates what the weights say from what the
  shipped template says.
- **Judge + canonicalization.** Identity claims are extracted by an LLM judge from
  the response and the reasoning trace separately, then canonicalized across
  languages (克劳德 / クロード / 클로드 / клод all → Claude) with
  family-equivalence: a GPT-4o answering "ChatGPT" is *self*, not drift — only
  cross-vendor claims count. Every flagged claim is then re-checked by a stronger
  adjudicator; `sweep/bench_judge.py` benchmarks the judge against five others.
```
