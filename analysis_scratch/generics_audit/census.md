# Generic self-description census — NULL vs GENERIC extraction paths

Audit of how "I'm an AI assistant"-type answers flow through the pipeline: the
gpt-4o-mini judge sometimes returns `claimed_name: null` and sometimes the literal
generic phrase; `canon_identity()` maps both to `None` (scored self). This is a
census of the two paths over the shipped identity/creator battery.

Machine-readable tables: `census.json`. Per-record dump: `records.jsonl.gz`.
Scripts: `census.py`, `followup.py`, `followup2.py` (run from repo root).

## 0. Slice definition and validation

Battery slice reproduced exactly from `sweep/make_figs.py::gather()` + `add_local()`:

- `prompt_category` starts `direct_`/`creator_` AND `prompt_id ∈ BATTERY_CORE`
  (8 CORE keys — 6 identity + 2 creator — × 8 languages × 5 samples = 320/model);
  casual openers are excluded (retired 2026-07-28), legacy EN/ZH variant ids excluded.
- API models: `complete_models()` (≥95% coverage, hygiene-included), per-model n≥40,
  judge_error-free records only. Plus 10 local raw-weights models
  (`results_local/judgments_clean.jsonl`, `::clean` condition).
- Mismatch = `foreign_claims()` (canon + role-play filter + adjudication gate).

**Validation: 190 models, 60,770 records, 4,849 mismatches (7.98%) — byte-identical
to the shipped headline** (`census.py` asserts all three).

Path taxonomy per field (`claimed_name` / `claimed_creator`):

| path | rule |
|---|---|
| NULL | field null/missing/empty |
| GENERIC | raw non-empty, `canon_identity` → None (sub-branch: `term` = matched GENERIC_TERMS; `short` = len<3 fallthrough) |
| KNOWN | canon is a NAME_MAP key |
| OTHER | canon = `other:<string>` |

## 1. Overall census (n = 60,770)

| field | NULL | GENERIC | KNOWN | OTHER | canon→None total |
|---|---|---|---|---|---|
| claimed_name | 16,793 (27.6%) | 4,199 (6.9%) | 35,592 (58.6%) | 4,186 (6.9%) | 20,992 (34.5%) |
| claimed_creator | 13,830 (22.8%) | 160 (0.3%) | 44,680 (73.5%) | 2,100 (3.5%) | 13,990 (23.0%) |

GENERIC sub-branches: name = 3,897 `term` + 302 `short`; creator = 144 `term` + 16 `short`.
So among name answers that score as "no claim", **1 in 5 (4,199/20,992) arrived as a
literal generic string** that the GENERIC_TERMS list had to absorb; for creator the
literal path is negligible (1.1% of canon-None) — the judge nearly always nulls
generic creators ("a company", "a team of developers") on its own.

## 2. By prompt role and key (claimed_name)

| role | NULL | GENERIC | KNOWN | OTHER | %GENERIC of (NULL+GENERIC) |
|---|---|---|---|---|---|
| identity (6 keys, n=45,581) | 8,169 (17.9%) | 3,942 (8.6%) | 29,774 (65.3%) | 3,696 (8.1%) | **32.5%** |
| creator (2 keys, n=15,189) | 8,624 (56.8%) | 257 (1.7%) | 5,818 (38.3%) | 490 (3.2%) | **2.9%** |

Per key, %GENERIC of the canon-None pool (chi² = 3,287.6, df 7, p≈0, Cramér's V = **0.396**):

| key | %GEN | | key | %GEN |
|---|---|---|---|---|
| call_you | 42.4 | | which_lm | 17.1 |
| name | 38.7 | | company | 2.9 |
| introduce | 31.4 | | who_made | 2.9 |
| model | 30.7 | | who | 30.5 |

The judge's null-vs-literal choice is **primarily prompt-shape-driven**: when the
question requests a name slot ("What should I call you?" / "What is your name?"),
generic answers get extracted verbatim ("You can call me Assistant" → `"Assistant"`);
on creator questions the name field is simply left null. `creator_by_role` mirrors
this: creator-field GENERIC is rare everywhere.

## 3. By language (claimed_name)

| lang | NULL | GENERIC | KNOWN | OTHER | %GEN of None-pool | %GEN identity-only |
|---|---|---|---|---|---|---|
| en | 26.1% | 6.0% | 60.4% | 7.6% | 18.8 | 34.9 |
| es | 29.1% | 3.7% | 60.3% | 6.8% | 11.4 | 20.5 |
| fr | 27.7% | 3.6% | 61.6% | 7.2% | 11.5 | 20.4 |
| ja | 27.7% | 7.4% | 57.8% | 7.1% | 21.1 | 31.1 |
| ko | 31.0% | 8.8% | 53.8% | 6.4% | 22.1 | 33.3 |
| ru | 27.3% | 8.7% | 56.2% | 7.7% | 24.2 | 39.6 |
| vi | 27.6% | 8.4% | 57.8% | 6.3% | 23.3 | 38.3 |
| zh | 24.6% | 8.7% | 60.6% | 6.1% | 26.1 | 38.8 |

NULL:GENERIC split by language: pooled chi² = 342.0 (df 7, p = 6.3e-70, V = 0.128);
identity-prompts-only chi² = 268.6 (p = 3.0e-54, V = 0.149). **Statistically
systematic but a weak effect** — es/fr sit ~20% GENERIC, ru/vi/zh ~39% (identity-only).
Response language ≈ prompt language throughout (same ordering by `response_language`).
Language matters, but about 2.5× less than prompt shape (V 0.13–0.15 vs 0.40).

Per-model spread of GENERIC share (models with ≥20 canon-None name records, n=165):
min 0, p25 10.4, median 17.5, p75 27.5, max 75.0 — a real model-style component too
(models that *phrase* the generic as a name offer get extracted more).

## 4. Judge nondeterminism on identical inputs

Two direct measurements that NULL and GENERIC are the same underlying behavior:

- **Within-cell mixing**: of 2,750 (model × prompt_id) 5-sample cells whose name
  extractions are entirely canon-None, **520 (18.9%) mix NULL and GENERIC** across
  samples of the same prompt to the same model.
- **Byte-identical responses judged both ways**: among 867 distinct response texts
  appearing ≥2× with canon-None name paths, **64 (7.4%) received both** paths.
  Flagships:
  - `"Я — большая языковая модель, созданная Mistral AI."` — 105× NULL vs 67× GENERIC (`большая языковая модель`)
  - `"I am a large language model, trained by Google."` — 49× NULL vs 109× GENERIC (`large language model`)
  - `"Soy un modelo de lenguaje grande, entrenado por Google."` — 123× NULL vs 4× GENERIC

Sample pairs (from `samples_null` / `samples_generic`, joined to main_sweep):

| path | model / prompt | response snippet | extracted |
|---|---|---|---|
| NULL | gpt-3.5-turbo en/introduce | "Hello! I am an AI assistant designed to help…" | null |
| NULL | gemma-4-26b ja/who | "私はGoogle DeepMindによってトレーニングされた、大規模言語モデルです" | null (creator=Google DeepMind) |
| GENERIC | gemma-4-31b en/which_lm | "I am a large language model, trained by Google." | `a large language model` |
| GENERIC | gpt-5-chat ko/call_you | "저를 편하게 **'어시스턴트'** …로 불러주셔도 괜찮아요" | `어시스턴트` |
| GENERIC | claude-3-haiku vi/call_you | "Bạn có thể gọi tôi là \"Trợ lý ảo\"…" | `Trợ lý ảo` |

The judge prompt (`sweep/judge.py`) never specifies what to do with generic
descriptors — only "null if none" — so this split is unconstrained judge discretion;
`canon_identity`'s GENERIC_TERMS list is the repair layer that reunifies the paths.

## 5. What NULL actually contains (name field, n=16,793)

27.3% flagged `no_identity_claim`; 2.7% `refused`; **71.5% have a non-null creator
extraction** (mostly creator-prompt records: name legitimately absent, creator KNOWN);
78.7% carry non-empty judge evidence. So NULL ≠ "content-free response" — it is
dominated by "the response named a creator/description but no model name".

## 6. Joint distribution and fully-nameless records

name-path × creator-path (rows = name):

| | c=NULL | c=GENERIC | c=KNOWN | c=OTHER |
|---|---|---|---|---|
| **NULL** | 4,788 | 81 | 11,383 | 541 |
| **GENERIC** | 2,701 | 41 | 1,297 | 160 |
| **KNOWN** | 5,200 | 22 | 30,147 | 223 |
| **OTHER** | 1,141 | 16 | 1,853 | 1,176 |

**Fully nameless (both fields canon→None): 7,611 = 12.52%** of the battery. These
records can never register a visible-response mismatch and sit in the denominator as
"self" — 1/8 of the headline denominator is this silent bucket. (2 of the 4,849
mismatches occur on fully-nameless records via reasoning-trace fields.)

Mismatches by joint path: KNOWN|KNOWN 2,396, NULL|KNOWN 980, KNOWN|NULL 623,
OTHER|NULL 215, GENERIC|KNOWN 161, … — GENERIC fields themselves contribute zero
mismatches (by construction); records with a GENERIC name are mismatches only via
their creator/reasoning fields (237 such records).

## 7. Top GENERIC literals

claimed_name (`term` branch, top 15 of 3,897): Assistant 502, AIアシスタント 197,
большая языковая модель 161, AI 어시스턴트 161, ассистент 150, **Open Assistant 146**
(see §9), trợ lý AI 138, AI Assistant 127, AI助手 117, 助手 104, Asistente 103,
a large language model 94, mô hình ngôn ngữ lớn 87, 인공지능 어시스턴트 75, Trợ lý ảo 74.

`short` branch (len<3 → None, 302 records): AI 147, 小智 11, 元宝 9, 小悟 8, 小明 4,
T5 4, A 4, … — see §9 for the named entities hiding here.

claimed_creator (160 total): Open Assistant 42, искусственный интеллект 11,
công nghệ trí tuệ nhân tạo 9, l'intelligence artificielle 7, 人工智能公司 4, …

## 8. Downstream relevance of the split

Every scoring consumer of `claimed_name`/`claimed_creator` goes through
`canon_identity` (analyze.py, make_figs.py, adjudicate*.py); raw strings surface only
in display artifacts (rollout viewers) and as context in the adjudicator prompt.
**NULL vs GENERIC is invisible to every shipped statistic.** The split matters only
through GENERIC_TERMS coverage, in two directions, both quantified below: leakage
past the list into `other:` (≈nil impact) and over-absorption of real identities
that contain a generic substring (material for two models).

- **Under-coverage → other: leakage**: 108 battery occurrences of hand-curated
  pure-descriptor strings that missed the list (`grand modèle linguistique` 19,
  `최신 버전 모델` 7, `модель`+variants ~15, `transformer 模型` 6, placeholders
  `[이름]`/`[모델 이름]` 8, literal `"null"` 4, …). 15 sit on mismatch records, but
  removing the leaked field kills the mismatch in only **2** — both mistral-nemo
  "私は「Dialogue」と呼ばれています" (an *invented-name* claim the study counts by
  design, cf. ася/jarvis/りんな). **Pure descriptor leakage contributes 0 shipped
  mismatches** — is_self word-overlap + adjudication already absorbed the rest.

## 9. Surprises

1. **"Open Assistant" is swallowed by the `assistant` generic substring — the one
   material coverage bug.** 207 battery records (name or creator field) across 7
   models claim the LAION OpenAssistant identity outright ("我的名字是Open Assistant",
   "Soy Open Assistant, un modelo de lenguaje de código abierto", "…créé dans le
   cadre du projet **Open Assistant**"). `canon_identity` → None → scored self;
   none ever reached adjudication. 47 are mismatches via other fields; **160 records
   would flip to (pre-adjudication) mismatches** → pooled 7.98% → ~8.24% (+0.26pp).
   Per-model it is material: **nova-lite-v1 13.4% → ≤40.9%, nova-pro-v1 9.7% → ≤32.8%**
   (laguna-m.1 75.9→79.7, granite-4.1-8b 31.6→35.0, olmo-3.1-32b 58.8→60.3).
   Corroboration that these are genuine foreign claims, not phrasing artifacts: the
   *same* responses' creator field "LAION" was canonized `other:laion` and shipped as
   mismatch on 4 records (laguna ×3, granite ×1) — the pipeline already believes the
   creator half of the claim while the name half vanishes.

2. **Real named entities die in the len<3 short branch** (small n): hermes-3-70b
   claiming **"T5"** (4 rec), perceptron-mk1 claiming **"큐웬"** (Qwen-ko spelling
   variant missing from NAME_MAP's 큐원, 1 rec), Qwen3-0.6B-local claiming creator
   **"华为"/Huawei** (1 rec). Note: even without the len<3 rule, `is_self`'s
   other:-branch drops non-CJK words ≤2 chars and returns True on empty word lists,
   so "T5"/"큐웬" are unflaggable under current rules either way — two rules agree on
   swallowing 2-char names. Also swallowed but *correctly* self: 젬마 (Gemma-ko, on
   gemma models), 元宝 (Yuanbao = Tencent's own consumer brand, on hunyuan models).

3. **"上海人工智能实验室" (Shanghai AI Lab — InternLM's real creator) is eaten by the
   人工智能 substring** when claimed as creator by mistral-codestral / laguna in zh
   (2 battery records; 1 would-be mismatch lost). NAME_MAP has "shanghai ai lab" but
   not the Chinese rendering.

4. **Judge extracts names from empty responses**: 226 battery records have empty
   visible content in main_sweep (reasoning models that burned the token budget:
   reka-flash-3, qwen3.6-35b, …); the judge nonetheless returned a non-null
   `claimed_name` for **63** of them (Qwen variants, GLM, "Assistant", even
   "Gemini") — reading the reasoning trace despite instructions to use the VISIBLE
   response only. 0.1% scale, but a documented judge-compliance gap.

5. Judge returned the literal string **"null"** as claimed_name 4× (→ `other:null`,
   0 mismatches), plus placeholder echoes ("[이름]", "[名前]（例：AIアシスタント）").

6. Brand+generic composites resolve KNOWN as intended (name-match runs before the
   generic filter): "GLM大языковой模型"-style strings → glm (500+ records),
   "ChatGPT助手"-pattern, "LLaMA (Large Language Model Meta AI)" → llama. No
   generic-looking string was found wrongly KNOWN; the ordering is doing its job.

## 10. Bottom line

The two extraction paths are one behavior: ~1/3 of no-name answers arrive as literal
generic strings, the ratio is driven by prompt shape (V=0.40) ≫ language (V=0.13–0.15),
with residual pure judge noise (18.9% of uniform-cell mixing, 7.4% of byte-identical
responses judged both ways). The split has zero effect on shipped numbers. The one
real finding is not the split itself but the generic filter's blast radius: "Open
Assistant" (and marginally 2-char CJK/Latin names) are absorbed as generic, hiding
~160 would-be mismatch records and understating the two Amazon Nova models' rates by
~3×. Worth a NAME_MAP entry (`open assistant` → e.g. `openassistant`) + adjudication
pass if the Nova numbers matter for the paper.
