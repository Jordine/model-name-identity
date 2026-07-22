# Raw-weights comparison set (NOT in the results)

Six Qwen models run from raw weights whose sizes are **also covered by an API
model** in the main study (Qwen3-8B/14B/32B, Qwen3.5-4B, Qwen3.5-35B-A3B,
Qwen3.6-35B-A3B). They are kept **out of `results/` and `results_local/`** so they
are never pooled into the headline or double-counted against their API twins.

They exist only for our own **API-vs-weights sanity check** (does the raw model
behave like its hosted sibling?) — this is a spot-check we run ourselves, **not a
reported result**. Nothing in `sweep/` reads this folder; the canonical raw-weights
*results* set is the 10 models in `make_figs.LOCAL_MODELS`.
