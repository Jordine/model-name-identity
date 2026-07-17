# Data files

The large sweep outputs are committed **gzipped** to stay under GitHub's 100 MB
per-file limit. Decompress them before running the analysis:

```bash
gunzip -k results/main_sweep.jsonl.gz results/judgments.jsonl.gz
```

Files:

- `main_sweep.jsonl.gz` — raw model responses, the full v3 sweep (189 models asked,
  179 complete; ~136k rows).
- `judgments.jsonl.gz` — LLM-judge output for every response (all `openai/gpt-4o-mini`;
  ~123k rows), with extracted claimed name/creator, cross/self acceptance, and
  reasoning-trace stance.
- `adjudications.jsonl` — second-pass, ground-truth-aware false-positive verdicts
  (only `genuine_foreign` counts as a mismatch).
- `probes.jsonl`, `judge_bench/` — auxiliary probe data and the judge-selection benchmark.

The analysis code (`sweep/*.py`) reads the decompressed `.jsonl` paths, so run the
`gunzip` above once after cloning.
