# Data files (API sweep)

The large sweep outputs are committed **gzipped** to stay under GitHub's 100 MB
per-file limit. The analysis reads the `.gz` transparently
(`sweep.analyze.open_lines`), so nothing needs decompressing first — but if you
want the plain files:

```bash
gunzip -k results/main_sweep.jsonl.gz results/judgments.jsonl.gz
```

Files:

- `main_sweep.jsonl.gz` — raw model responses, the full sweep (189 models asked,
  179 complete; ~136k rows).
- `judgments.jsonl.gz` — LLM-judge output for every response (`openai/gpt-4o-mini`;
  ~123k rows): extracted claimed name/creator, cross/self acceptance, and the
  reasoning-trace stance.
- `adjudications.jsonl` — second-pass, ground-truth-aware false-positive verdicts
  for every flagged claim (only `genuine_foreign` counts as a mismatch).
- `adjudications_local.jsonl` — the same adjudication for the raw-weights (GPU) models.
- `judge_bench/` — the 6-judge selection benchmark (each judge's calls on the hard
  cases), behind `sweep/bench_judge.py`.
- `preflight_raw.jsonl` — per-provider hidden-system-prompt probe results feeding
  `config/provider_hygiene.json`.
- `run_meta.jsonl` — per-model run metadata. (Summary numbers come from
  `sweep/make_figs.py` + `sweep/explain.py`, not a static table.)

Raw-weights sweep data lives in `../results_local/`.
