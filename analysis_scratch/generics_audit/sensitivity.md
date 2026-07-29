# Generic-answer sensitivity — headline numbers under alternative scoring rules

Recompute of the shipped headline (identity/creator battery, 190-model pool,
post-adjudication) under variant per-record mismatch definitions.
Script: `analysis_scratch/generics_audit/sensitivity.py` (run from repo root;
imports `sweep.analyze` / `sweep.make_figs` — no logic reimplemented without a
per-record assert against `foreign_claims()` and a per-model assert against
`make_figs.gather()`). Companion data: `sensitivity.json` (incl. full per-model
n / named / d0..d3).

**Harness check: S0 reproduces the shipped numbers exactly** — 4,849/60,770
(7.98% → "8.0%"), 116/190 models >=1 mismatch, 95 >=3, 74 with 0, median model
rate 0.78% — and per-model (d, n) is identical to `make_figs.gather()` for all
190 models.

## Master table

| variant | pooled | mismatch records | denominator | models >=1 | >=3 | zero | median model rate |
|---|---|---|---|---|---|---|---|
| S0 baseline (shipped) | **7.98%** | 4,849 | 60,770 | 116 | 95 | 74 | 0.78% |
| S1 known-only | **7.10%** | 4,315 | 60,770 | 115 | 89 | 75 | 0.62% |
| S2 conditional-on-naming | **9.12%** | 4,847 | 53,159 | 116 | 95 | 74 | 0.96% |
| S2b cond., adjudication-aware naming | 9.23% | 4,847 | 52,500 | 116 | 95 | 74 | 0.97% |
| S3 strict-self (separate axis) | **8.02%** | 4,875 | 60,770 | 116 | 95 | 74 | 0.78% |
| S2-legacy = PLAN Q&A replication | 8.18% | 4,099 | 50,121 | (180 API models only) | | | |

Denominators: S0/S1/S3 = all judged battery records; S2 = records whose
*response* named something specific (canon of `claimed_name` or
`claimed_creator` non-None, i.e. KNOWN or `other:*`); S2b = S2 minus records
whose extraction was adjudicated `judge_error`/`generic`.

## S2 headline caveat: PLAN's earlier conditional numbers used a different definition

PLAN.md ("Collaborator Q&A", commit 9b913ae) reports the conditional variant as
pooled 8.0%→8.2%, "82% of asks name something", Hermes 3 70B 19.7%→95.5%,
ERNIE 4.5 VL 24%→83%, Perceptron 88%→100%. Those numbers **replicate exactly**
(8.18% = 4,099/50,121; Hermes 63/66 = 95.5%; ERNIE 77/93 = 82.8%; Perceptron
282/282 = 100%) — but only under the original session's definition (recovered
from the 2026-07-28 transcript), which differs from
"named something specific" in three ways:

1. **Denominator = counted-mismatch OR canon-self-named** (self-naming checked
   over all 4 fields incl. reasoning), *not* "named anything specific". Records
   whose extracted name was dismissed by adjudication fall out of the
   denominator entirely. For Hermes 3 70B that removes 66 of its 132 naming
   records (adjudication verdicts of the removed: judge_error 52, roleplay 11,
   generic 3 — e.g. `claimed_creator="Google"`/`"IBM"` judge_errors, roleplay
   personas "Clara"/"Alex"), which is what inflates 47.7% to 95.5%.
2. **API-only**: the loop iterated `load()` (API judgments), so the 10 local
   raw-weights models contributed nothing → numerator 4,099 instead of 4,849.
   "82%" = 50,121/60,770 divides an API-only count by the 190-pool ask count;
   the internally consistent naming share of that definition is 50,121/57,570 =
   87.1% on the API pool.
3. Numerator/denominator both post-adjudication on the foreign side, mixed
   fields on the self side.

Under the definition specified for this audit (canon of response-level
`claimed_name`/`claimed_creator` non-None; numerator = S0 mismatches that named,
full 190-model pool): **pooled 9.12% (4,847/53,159), pool naming rate 87.5%**.
Only 2 S0 mismatch records fall outside the denominator (trace-only claims) —
matches PLAN's "2/4,099 trace-only" beat. If PLAN's supplementary table ships,
its numbers should be regenerated with one of the definitions here and labeled
precisely; the 95.5%/83% per-model figures are artifacts of exclusion (1).

Per-model naming rate (named/n): min 21.9%, median 94.2%, max 100.0%
(Claude Fable 5, 320/320). 5 shyest models:

| model | named/n | naming % |
|---|---|---|
| Qwen3 0.6B (local) | 70/320 | 21.9% |
| Reka Flash 3 | 91/319 | 28.5% |
| Reka Edge | 97/320 | 30.3% |
| GPT-3.5 Turbo | 109/320 | 34.1% |
| Nova Lite 1.0 | 120/320 | 37.5% |

S2 top movers (conditional rate vs S0 rate, pp):

| model | S0 | S2 conditional | delta |
|---|---|---|---|
| ERNIE 4.5 VL 424B A47B | 77/320 (24.1%) | 77/130 (59.2%) | +35.2 |
| OLMo 3.1 32B Instruct | 188/320 (58.8%) | 188/205 (91.7%) | +33.0 |
| Nemotron Super 49B | 155/320 (48.4%) | 155/192 (80.7%) | +32.3 |
| Qwen3 0.6B | 27/320 (8.4%) | 27/70 (38.6%) | +30.1 |
| Hermes 3 70B Instruct | 63/320 (19.7%) | 63/132 (47.7%) | +28.0 |
| OLMo 3 7B Instruct | 200/320 (62.5%) | 200/222 (90.1%) | +27.6 |
| Reka Edge | 37/320 (11.6%) | 37/97 (38.1%) | +26.6 |
| Hermes 3 405B Instruct | 133/320 (41.6%) | 133/197 (67.5%) | +26.0 |
| Laguna XS 2.1 | 215/320 (67.2%) | 215/235 (91.5%) | +24.3 |
| Reka Flash 3 | 29/319 (9.1%) | 29/91 (31.9%) | +22.8 |

## S1 known-only (no invented-persona / no fallthrough lower bound)

Dropping all `other:*` evidence removes 534 of 4,849 mismatch records (11.0%):
pooled 7.98% → 7.10%, median 0.78% → 0.62%, models >=1 116 → 115 (Gemma 3 4B's
single `other:*` mismatch goes clean; no model can gain), >=3 95 → 89 (leaving:
Jamba Large 1.7, DeepSeek V3.2, Ling-2.6-flash, Llama 3.1 70B, Llama 3.3 70B,
Qwen3.5 Plus 2026-04-20). The headline shape is intact — known-name claims
carry 89% of the effect.

Language of dropped records: ja 116, ko 91, ru 82, vi 76, zh 52, es 45, fr 40,
en 32 — 94% non-English, consistent with the "CJK/vi transliteration + invented
persona" expectation, though the biggest per-model droppers are invented-persona
and self-brand-variant models rather than specifically Vietnamese/Korean small
models. Top dropped strings pool-wide: "step" x18 (Trinity claiming Step —
arguably a real StepFun claim canon doesn't catch bare), "nemistral"(+ ai) x20
(Mistral Nemo's blend-name), "hugging face" x10, "fpt smart cloud" x8,
"neuro"/"카카오"/"클로이"/"jarvis" x5 each.

S1 top droppers:

| model | S0 | S1 | delta | dropped langs | top dropped `other:*` |
|---|---|---|---|---|---|
| Trinity Large Thinking | 120/320 (37.5%) | 70/320 (21.9%) | -15.6 | vi 20, ja 16, ko 8 | step x18, fpt smart cloud x8, vinbigdata x4 |
| Mistral Nemo | 65/313 (20.8%) | 31/313 (9.9%) | -10.9 | ko 10, en 9, ja 8, zh 7 | nemistral x14, nemistral ai x6 |
| ERNIE 4.5 VL 424B A47B | 77/320 (24.1%) | 49/320 (15.3%) | -8.8 | es 13, ja/ru 5 | cloudwalk x3, t-astro x3 |
| Laguna XS 2.1 | 215/320 (67.2%) | 187/320 (58.4%) | -8.8 | vi 11 | hugging face x2, yuanbao x1 |
| Llama 3.2 3B Instruct | 43/320 (13.4%) | 16/320 (5.0%) | -8.4 | ja 8, zh 8, ru 6, ko 5 | 大理石 x2, hikari x1 |
| Llama 3.2 1B Instruct | 35/320 (10.9%) | 11/320 (3.4%) | -7.5 | ja 8, ru/ko 5 | mystic, ラプターシュ, コメット x1 |
| Hermes 3 70B Instruct | 63/320 (19.7%) | 43/320 (13.4%) | -6.2 | spread | aws x2, clara x1 |
| Nova Lite 1.0 | 43/320 (13.4%) | 24/320 (7.5%) | -5.9 | ru 6, zh 5 | 开源社区 x2 |
| Qwen3 1.7B (local) | 35/320 (10.9%) | 16/320 (5.0%) | -5.9 | fr 6, es 4 | lia x3, 셀레나 x2, aiden x2 |
| Reka Edge | 37/320 (11.6%) | 19/320 (5.9%) | -5.6 | ja 14 | アシュター x1, sassari ai x1 |

## S3 strict-self (separate axis: disclosure asymmetry, not generics)

Family-equivalence via the alias string disabled for KNOWN canons: a known
canon counts self only if in `FAMILY_SELF[family]` (+ family slug and its known
canon — needed so local OLMo→"allenai" stays self); `other:*` and generic/None
branches unchanged; adjudication overrides kept (0 strict-only records were
suppressed by an adverse verdict; local strict-only evidence counted without
the positive local gate, as those records never reached adjudication).

Effect is tiny and concentrated exactly where disclosure predicts: +26 records
(4,849 → 4,875), pooled 7.98% → 8.02%, all other summary stats unchanged
(116/95/74, median 0.78%). Every flip is a disclosed-ancestor name:
nvidia→llama x22, nvidia→meta x12, nous→llama x3, nous→meta x1. Only 2 models
move: **Nemotron Super 49B** (the llama-named one; 155→178/320, +7.2pp) and
**Hermes 3 405B** (133→136/320, +0.9pp). The non-llama-named Nemotron 3 family
does not move (its Llama claims already count as foreign in S0 — aliases don't
disclose Llama). So the headline is insensitive to the ancestor-forgiveness
choice; it only redistributes within two derivative models.

## Bottom line

- The pooled rate lives in [7.10%, 9.23%] across every rule tested; the
  models->=1/>=3/zero counts are essentially invariant (115-116 / 89-95 / 74-75).
- S1 (hardest rule against judge/canon fallthrough) keeps 89% of mismatch
  records and the full model-level story.
- S2 raises the pooled rate (9.1%) because generically-answering records leave
  the denominator; per-model it reshuffles hard for shy models (denominators
  down to n=70). PLAN's previously-quoted conditional numbers (8.2%, 82%,
  Hermes 95.5%) are NOT this definition — see caveat above; they replicate only
  under the legacy mismatch-or-self-named, API-only construction.
- S3 confirms the family-equivalence choice is not load-bearing.
