# Post-fix statistical analyses (2026-07-29)

Four analyses on the post-fix working tree. Script: `analysis_scratch/postfix_analyses.py`
(run from repo root); full machine-readable results in `postfix_analyses.json`.
Everything is computed through the repo's own machinery: `make_figs.gather()` per-model
d/n and per-language cells (10 local raw-weights models folded in), `analyze.py`'s
paired-delta path, `explain.py`'s VERSIONS/CUTOFF_LAG/MIN_TOT for fig_cutoff rules, and
`generics_audit/sensitivity.py`'s own `run()` for S2. Bootstrap/permutation: 10,000
resamples, seed 12345 throughout.

**Sanity anchor (asserted before anything else): pooled 4,992/60,770 = 8.21%,
190 models, 115 with >=1 mismatch — exact match to the post-fix headline.**

---

## 1. Lab-region x language interaction

Metric: per-model paired delta (rate in language L − rate in English, pp) over the
balanced identity/creator battery (40 records/language/model); group means compared by
cluster bootstrap over models (10k, independent resampling of the two groups for the
difference CI; two-sided add-one bootstrap p).

| lang | Chinese-lab mean Δ [95% CI] (n) | non-Chinese mean Δ [95% CI] (n) | diff CN−nonCN [95% CI] | p |
|---|---|---|---|---|
| **zh** | **−2.31 [−4.42, −0.58]** (78) | **+6.20 [+3.39, +9.38]** (112) | **−8.51 [−12.31, −5.19]** | **0.0002** |
| ja (mirror) | +2.13 [+0.05, +4.38] (78) | +7.06 [+3.93, +10.48] (112) | −4.93 [−8.97, −1.14] | 0.011 |
| ko (mirror) | +5.29 [+2.34, +8.65] (78) | +4.05 [+0.88, +7.38] (112) | +1.24 [−3.24, +5.73] | 0.59 |

**Final lab mapping** (families in the 190-model complete pool):

- **Chinese (78 models):** qwen 41 (incl. 5 local raw-weights), deepseek 10, zhipu 9,
  kimi 6, ant 3, tencent 2, nex 2 (Nex AGI = Shanghai Innovation Institute alliance,
  PRC — verified via web search), baidu 1, kuaishou 1, minimax 1, stepfun 1, xiaomi 1.
- **Non-Chinese (112 models):** anthropic, openai (incl. gpt-oss locals), google, gemma,
  meta, microsoft, amazon, nvidia, ibm, mistral (FR), cohere (CA), ai21 (IL — counted
  Western; home language not in battery), nous, perplexity, poolside, inception (absent),
  arcee, reka, olmo/allenai, perceptron, and the rest of the registry pool.
- **Excluded from BOTH groups** (non-PRC-non-Western, per spec): naver, upstage, yandex,
  sber, sakana, aisingapore — zero of these are in the 190-model complete pool (upstage
  and sakana are registered but incomplete), so the exclusion removes no rows.
- CN families listed in the spec but contributing nothing: alibaba/moonshot (slugs not
  used; covered by qwen/kimi), bytedance (registered, not complete), huawei, internlm,
  sparkdesk, zhinao360 (not in registry).

**Takeaway:** Chinese-lab models are dramatically cleaner in Chinese relative to their own
English baseline than non-Chinese-lab models are (−8.5pp interaction, p=0.0002) — they
actually mismatch *less* in Chinese than in English (−2.3pp), while every other lab gets
+6.2pp worse. The specificity check behaves: the interaction is absent in Korean (+1.2pp,
p=0.59) and intermediate in Japanese (−4.9pp, p=0.011) — consistent with a
Chinese-training-data effect that partially spills into the CJK-adjacent language rather
than a generic "Chinese labs are careful everywhere" effect.

---

## 2. Multiple comparisons on the 7 paired language deltas

Repo's paired-bootstrap path (`analyze --paired` logic). The 4,000-rep replication in the
script matches the tool's stdout digit-for-digit (zh +2.71 [+0.68,+4.68] … vi +0.16
[−1.29,+1.70]). Below: 10k reps, add-one two-sided bootstrap p (floor 0.0002), Holm and
Bonferroni at m=7.

| lang | Δ vs en (pp) | 95% CI | p raw | p Holm | p Bonf | survives Holm | survives Bonf |
|---|---|---|---|---|---|---|---|
| ja | +5.04 | [+2.97, +7.26] | 0.0002 | 0.0014 | 0.0014 | yes | yes |
| ru | +3.51 | [+1.97, +5.21] | 0.0002 | 0.0014 | 0.0014 | yes | yes |
| ko | +4.56 | [+2.31, +6.92] | 0.0004 | 0.0020 | 0.0028 | yes | yes |
| zh | +2.71 | [+0.78, +4.77] | 0.0054 | 0.0216 | 0.0378 | yes | yes |
| es | +1.27 | [+0.18, +2.44] | 0.0212 | 0.0636 | 0.1484 | **no** | **no** |
| fr | +1.44 | [+0.17, +2.85] | 0.0240 | 0.0636 | 0.1680 | **no** | **no** |
| vi | +0.16 | [−1.32, +1.69] | 0.8495 | 0.8495 | 1.0000 | no | no |

Post-fix, **ja, ko, ru, zh survive both Holm and Bonferroni**. French — the language in
question — is nominally significant (raw p=0.024) but does **not** survive either
correction; Spanish, newly nominally significant post-fix (raw p=0.021), dies the same
way. Vietnamese is null. (Pre-fix, fr +1.3 had been reported significant; the honest
post-fix statement is 4 languages robust, fr/es suggestive-only.)

---

## 3. Cutoff at-risk pairs + permutation null

fig_cutoff's own inclusion rules: 189 models with a usable date and tot ≥ MIN_TOT=100
(103 documented cutoffs, 86 estimated = release − 0.5yr = CUTOFF_LAG). Targets = the six
VERSIONS identities; earliest breakout per target: chatgpt 2022.88, llama 2023.54
(Llama 2), gemini 2023.96 (1.0), claude 2024.21 (Claude 3), qwen 2024.71 (2.5),
deepseek 2024.96 (V3). Permutation null: 10k shuffles of the model→claim-target-set
assignment across the 189 models (cutoffs stay put; record weights travel with the sets).

**(a) At-risk pairs (breakout postdates cutoff): 173** — 126 documented / 47 estimated;
by target: deepseek 69, qwen 58, claude 23, gemini 19, llama 3, chatgpt 1.

**(b) Observed pre-breakout claim-pairs: 22 — NOT 0.** The anticipated zero does **not**
verify at the breakout definition. The 22 pairs carry 411 records (14 doc / 8 est) and
decompose entirely into artifacts:

- **12/22 are same-month coding ties** (gap ≤ 1 month, mostly 0.002 yr): Dec-2024 cutoffs
  vs DeepSeek V3 (released 2024-12-26) — OLMo-3 x3, Hunyuan-A13B, ERNIE-4.5-VL;
  Dec-2023 cutoffs vs Gemini 1.0 (2023-12-06) — GPT-4 + four Llama-3.x; Nemotron Nano
  9B v2 (cutoff 2024-09) vs Qwen2.5 (released 2024-09-19).
- **Creator→brand collapse:** gather() folds creator claims into the brand, so GPT-4's 16
  "gemini" records are all literally "Google / Google Assistant", Hermes-3's "claude" is
  mostly "Anthropic", Nemotron Super's 59 "qwen" records are 通义实验室/Alibaba creator
  claims — names that existed long before the cutoff.
- **Post-trained derivatives with stale base cutoffs:** nvidia/llama-3.3-nemotron-super-49b
  carries the Llama-3.3 base cutoff (2023-12) but was released 2025-07 with Qwen-flavored
  post-training data.
- **Estimated-cutoff imprecision** (release−6mo): mixtral-8x22b, mistral-large-2407,
  sonar-pro, hermes-3 — all gaps ≤ 0.6 yr, within the lag heuristic's error.
- **All 22 pairs postdate the target lineage's FIRST release** (Claude 1 2023.21,
  Bard 2023.21, Qwen1 2023.62, DeepSeek Coder 2023.87, …).

**(c) Permutation null (breakout):** expected E = **25.56** pairs, P(0 under null) =
0.0000 (never in 10k; max 45); P(count ≤ 22) = 0.28 — at the raw pair level the observed
count is statistically indistinguishable from timing-blind. Record-weighted: observed 411
vs E = 455.0, P(≤411) = 0.43. **Tie-robust variant (require ≥ 1-month gap): 149 at-risk
pairs, observed 10, E = 21.25, P(≤10) = 0.006** — a significant deficit once same-month
ties are excluded.

**(d) By provenance (breakout):** at-risk 126 doc / 47 est; observed 14 doc / 8 est;
null E = 19.37 doc / 6.18 est.

**Stricter first-release definition:** at-risk pairs = **21** (20 doc / 1 est), observed
= **0** (verified), null E = **3.51** pairs (55.8 records), **P(zero under null) =
0.059**.

Deliverable sentences:

> Breakout definition: 173 at-risk pairs, expected 25.6 claim-pairs under a timing-blind
> null, observed 22 (p(≤obs)=0.28) — but every observed pair is a same-month tie,
> creator-brand fold, stale-base-cutoff derivative, or lag-estimate artifact; with
> same-month ties excluded it is 149 at-risk, expected 21.3, observed 10, p=0.006.
>
> First-release (name-didn't-exist-yet) definition: 21 at-risk pairs, expected 3.5 under
> a timing-blind null, observed 0, p=0.059.

---

## 4. S2 conditional supplementary (post-fix refresh)

Via `sensitivity.py::run()` on the post-fix tree; S0 harness check reproduces the
headline exactly (4,992/60,770 = 8.21%, 115/190). S2 definition: denominator = records
whose response-level claimed name OR creator canon is non-None; numerator = counted
mismatches that named.

- **Pooled conditional: 9.37% (4,990/53,282)** — pre-fix 9.12% (4,847/53,159).
- **Overall naming rate: 87.68%** (53,282/60,770) — pre-fix 87.5%.
- S0 mismatches outside the denominator (trace-only claims): 2 (4,992 − 2 = 4,990 ✓).
- Per-model naming rate min/median/max: 20.3% / 94.4% / 100%.
- Model counts under S2: 115 with ≥1 mismatch, 95 with ≥3, 75 zero; median model
  conditional rate 0.96%.

Top-5 per-opportunity → conditional reshuffles (all shy namers):

| model | S0 (per-opportunity) | S2 (conditional) | Δ pp |
|---|---|---|---|
| ERNIE 4.5 VL 424B A47B | 77/320 (24.1%) | 77/128 (60.2%) | +36.1 |
| Qwen3 0.6B (local) | 27/320 (8.4%) | 27/65 (41.5%) | +33.1 |
| Nemotron Super 49B | 155/320 (48.4%) | 155/191 (81.2%) | +32.7 |
| OLMo 3.1 32B Instruct | 193/320 (60.3%) | 193/210 (91.9%) | +31.6 |
| Hermes 3 70B Instruct | 63/320 (19.7%) | 63/132 (47.7%) | +28.0 |
