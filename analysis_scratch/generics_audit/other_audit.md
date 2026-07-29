# Audit: other:* false-positive leakage in the identity-claims battery

Scope: do generic phrases missing from GENERIC_TERMS fall through to `other:*`, fail
`is_self`, survive adjudication, and inflate the shipped mismatch numbers
(8.0% pooled = 4,849/60,770; 116/190 models >=1 mismatch; median 0.78%)?

Verdict up front: **no material inflation.** 534 of the 4,849 counted mismatches
(11.0%) rest on other:*-only evidence; of those, **22 records (0.45% of all counted
mismatches) are true false positives** after eyeballing every suspicious record's
response text. Removing them: pooled 7.98% -> 7.94%, models>=1 116 -> **115**
(google/gemma-3-4b-it loses its only mismatch), median unchanged 0.78%. Meanwhile the
same mechanism (substring-generic scrubbing) *undercounts*: 162 uncounted records carry
a judge-extracted "Open Assistant" identity claim. Net direction of the shipped number
is conservative.

## 1. Reproduction (join proof)

Reimplemented the gather() slice from `sweep/make_figs.py` (complete_models × hygiene,
`is_identity` = direct_*/creator_*, `prompt_id ∈ BATTERY_CORE`, per-model n>=40,
`foreign_claims()` with adjudication gate, + add_local fold-in of the 10 raw-weights
models with the local `keep`-only rule). Result, exact match with shipped:

```
models: 190 (api 180 + local 10)
records: 60770   mismatches: 4849   pooled: 7.98%
models with >=1 mismatch: 116      median per-model rate: 0.78%
```

Join keys verified: judgments carry `turn_index: null` so the adj_key f-string yields
`::tNone` on BOTH the adjudication-write side (adjudicate.adj_key) and the
foreign_claims lookup side — consistent. main_sweep single-turn rows have no
turn_index key at all (-> `::t0`, matching `judge_key`), so build_worklist's raw join
also holds. Script: `audit.py` (phase 1), per-record table `battery_records.jsonl`.

## 2. other:* inventory over the battery (60,770 records)

* 5,110 records have an other:* canon in claimed_name/claimed_creator (incl. ones
  is_self clears, e.g. own-alias fragments).
* 1,185 distinct other:* values; 1,067 distinct values produce foreign evidence
  (1,514 field-level foreign occurrences). Long tail: only **83 values occur >=3
  times** (446 occurrences); the mode is singletons.
* Full dump with counts/langs: `other_values_full.txt`. >=3 list with models:
  `ge3_values.txt`.

Record-level split of the 5,922 pre-adjudication flagged battery records:

| evidence type            | flagged | counted after adjudication | cleared |
|--------------------------|---------|----------------------------|---------|
| other:*-only             | 1,311   | 534 (all verdict=genuine_foreign) | 777 (59%): self 253, generic 217, judge_error 199, roleplay 108 |
| mixed (other + known)    | 116     | 111                        | 5 |
| known-name only          | 4,495   | 4,204                      | 291 (6.5%) |
| total                    | 5,922   | 4,849                      | 1,073 |

The adjudication layer is doing exactly the job the design intends: 59% of other:*
flags are killed there (vs 6.5% of known-name flags). Examples of classes it reliably
clears: 커맨드/コマンド (=Command, Cohere self, 31/32 cleared), клаудия (=Claude ru),
квен (=Qwen ru), "grand modèle linguistique" fr LLM 10/10, модель/бот/aiさん/assis
generics, астра/알렉스-type persona roleplay, "llamo" (judge parsing "me llamo") 7/7
judge_error.

## 3. Classification of the 83 values with >=3 occurrences

(Full verdict split per value in phase-2/phase-5 outputs; counts are pre-adjudication
occurrences, "gf" = occurrences that survived as genuine_foreign.)

**MISSED_GENERIC — 18 values, 71 occ, 24 gf-surviving.** grand modèle linguistique(10,
gf0), un grand modèle linguistique(5,0), модель(6,0), модель-трансформер(5,0),
aiさん(4,0), бот(4,0), assist(4,0), assis(4,0), assistent(3,0), ai hỗ trợ(4,0),
công nghệ ai(3,0), 中国の会社(3,0), 开源社区(3,2), la comunidad(3,3),
creado para ayudarte(3,1), desarrolladores(3,0), командой разработчиков(3,0),
transformer 模型(4,4). NOTE: nearly all gf-survivors here sit on records that ALSO
contain a real foreign claim in the response (Open Assistant / DeTrM — §5), so value-
level generic leakage mostly does NOT translate into record-level false positives.

**INVENTED_PERSONA — 22 values, 152 occ, 92 gf.** nemistral(26,14) + nemistral ai(6,6)
(mistral-nemo garbling itself — counted per the study's invented-persona convention),
ася(15,3), alex(14,0), astra(7,0), 클로이(7,5), 雾栖科技(9,4), jarvis(5,5),
小初心(5,0), миша(4,0), neuro(5,5), 마루(maru)(4,4), lia(4,4), clara(3,1), алекс(3,1),
ларабот(3,2), lumina(3,3), neurosteer(3,3), t-astro(3,3), detrm 研究组(4,4),
dialogue(4,2), тони(7,0).

**REAL_LAB_UNMAPPED — 20 values, 103 occ, 87 gf.** step/StepFun(26 incl. 階躍星辰
variants,18), hugging face(11,10), fpt smart cloud(8,8), viettel ai(4,4),
vinbigdata(4,4), vietai và fpt smart cloud(3,3), gemma(5,4), ibm(5,1), laion(4,4),
bert(4,?), palm 2(3,3 — one of which is the gemma-3-4b self-family FP),
text-davinci-003(3,?), vicuna-13b-delta-v1.1(3,3), phi(3,3), huawei(3,3),
cloudwalk(3,3), chatpdf(3,3), lavague(3,3), 카카오(5,5), openstax(3,2). These are
correctly counted as foreign; the only cost of the NAME_MAP gap is attribution
("other/unlisted" instead of a brand bucket).

**SELF_XSCRIPT (own identity, other script/garbled — NAME_MAP/alias gaps) — 18 values,
101 occ, 2 gf.** 커맨드(19,0), コマンド(12,0), 코맨드(3,1!), клаудия(7,0), квен(6,0),
перплэксити/перплити/перплексити(10,0), ネモトロン3ウルトラ(4,0), немotron 3 ультра(4,0),
немотрион 3 ультра(3,0), アントグループ(5,1!), 미스트럴 ai(3,0), コヒーレンス(3,0),
thông thoại(5,0), байда(3,0), llamalama(4,0), мodель garbles etc. Adjudication caught
99/101 of these; the 2 leaks are in the 22-record FP list.

**JUNK (parse artifacts) — 5 values, 21 occ, 0 gf.** llamo(7), null(4), family(4),
[your name](3), mark zuckerberg(3, adjudicated judge_error). All neutralized by
adjudication.

## 4. Adjudication coverage (task 5)

**Zero holes.** Every flagged record has an adjudication row:

* battery slice: 5,922/5,922 flagged records have a verdict (API 4,986; local 936).
* all categories except system_probe, all judged models incl. incomplete ones:
  8,277/8,277 flagged records have a verdict.
* The un-adjudicated fallback branch in `foreign_claims()` (count-without-review)
  **never fires** in the shipped data. For local models the asymmetric rule
  (un-adjudicated = dropped) also never fires: all 936 local flags have local verdicts
  (750 genuine_foreign, 117 judge_error, 42 generic, 14 roleplay, 13 self).
* Counted mismatches are therefore exactly the verdict=genuine_foreign set:
  534 other-only + 111 mixed + 4,204 known = 4,849. No coverage holes by model or
  language.

## 5. The 534 counted other:*-only records — record-level eyeball

Language spread: ja 116, ko 91, ru 82, vi 76, zh 52, es 45, fr 40, en 32 (matches the
study's non-EN concentration claim). Top models: arcee trinity-large-thinking 50,
mistral-nemo 34, poolside laguna-xs-2.1 28, ernie-4.5-vl 28, llama-3.2-3b 27,
llama-3.2-1b 24, hermes-3-70b 20, nova-lite 19, Qwen3-1.7B 19 (local), reka-edge 18.

Method: 57 records hit generic-keyword pre-filters — ALL 57 read with full response
text (`cand_generic.txt`). 40-record seeded random sample of the 477 specific-name
remainder — ALL 40 verified to carry a real asserted foreign name in the response
(Step, Ростелеком, Bảo Ngọc, Gemma, LangChain, HoloLens, CyberBot, ミタ・ハナコ,
NeuroSteer, X-红皇后, ChatPDF...): 40/40 solid, 0 hidden generics. Plus ~35 targeted
reads from self-garble/sibling-brand scans. Total records eyeballed with response
text: ~130 of 534.

Final labels (`final_labels.jsonl`):

| label | n | meaning |
|---|---|---|
| SOLID_SPECIFIC | 478 | invented persona or real-lab name asserted in the response |
| OA_MECHANISM   | 35  | response claims "Open Assistant"/LAION/LMSYS — substantively a real mismatch, but the counted string is a generic creator phrase (or `laion`), because the NAME itself ("Open Assistant", "LMSYS Chatbot") is scrubbed by the `assistant`/`chatbot` substring rule |
| STRICT_SELF    | 8   | cross-script/garbled SELF claim; adjudicator erred |
| GENERIC_ONLY   | 13  | all evidence generic/junk; no substantive foreign claim |
| (+1 same-family) | 1 | gemma-3-4b-it ja: "based on PaLM 2, developed by Google" — Google-family claim on a Google model; PaLM missing from NAME_MAP |

**The 22 true false positives** (8 STRICT_SELF + 13 GENERIC_ONLY + 1 palm2):

STRICT_SELF: cohere/north-mini ko 코맨드 ("제 이름은 코맨드입니다" = Command) and
코헤렌트 (creator "Coherent"≈Cohere, name Command=self); inclusionai ling-2.6-flash ja
"LingDT by Ling社", ru "AntBailing by Ant Digital"; ling-2.6-1t ja creator
アントグループ (Ant Group; name Bailing=self); ring-2.6-1t ja "百霊大モデル（Ling）"
(kyūjitai 霊 defeats 百灵 pattern); mistral-medium-3-5 zh "Mistral Large by 米斯ral
AI团队" (half-script Mistral AI); deepseek-v4-pro vi "DeepSeek bởi công ty Thâm Độ Cầu
Sách" (Vietnamese calque of 深度求索).

GENERIC_ONLY: claude-3-haiku ru name=Клэр (mapped self) counted solely on
creator="создатели" ("my creators"); Qwen3.5-2B ko creator="Algorithm" while the
response says "developed by Alibaba Cloud" (self); llama-3.1-70b ja "私は…AI
プログラムです"; hermes-3-70b ko creator "Machine Learning Company" (degenerate
rambling); hermes-3-70b fr "inspiré par le modèle appelé SNN" (comparative, judge
over-extraction); Qwen3.5-0.8B es creator "desarrollado en una red de dispositivos
móviles"; llama-3.2-3b ru "Я modelo de lingo AI"; ministral-14b es "Soy IA de
Respuesta(s)" x3; reka-edge ja "私の名前はai辞書です"; Qwen3-0.6B vi creator "một tổ
chức uy tín" (response does mention using GPT — weakest member of the list);
laguna-xs vi creator "Interne" (truncation junk).

Per-language of the 22: ja 6, ko 4, es 4, ru 3, vi 3, zh 1, fr 1.
Per-model concentration (fp / that model's total counted d): ministral-14b 3/29,
north-mini-code 2/12, ling-2.6-flash 2/6, hermes-3-70b 2/63, ring-2.6-1t 1/2,
gemma-3-4b-it 1/1 (the only model whose >=1-mismatch status depends on an FP);
everything else 1 against d>=9. The Ant/inclusionai family is the most FP-affected
cluster (4 of 22, all cross-script self in ja/ru).

## 6. Bottom line

**(a)** other:*-only evidence backs **534 of 4,849 counted mismatch records = 11.0%**
(= 0.88pp of the 7.98% pooled rate). 111 further records are mixed but stand on a
known-name claim regardless.

**(b)** Actually-wrong records among those: **22** (= 4.1% of the 534, 0.45% of all
4,849) — 13 missed-generic/junk-evidence records, 8 cross-script self-garbles, 1
same-family (PaLM 2 on Gemma). List above / `final_labels.jsonl`. A further 35
OA_MECHANISM records are counted via the "wrong" string but are substantively real
mismatches (model claims to be Open Assistant/LAION/LMSYS).

**(c)** Proposed GENERIC_TERMS additions: 46 exact strings in
`proposed_generics.txt` (community/volunteer/open-source phrasings in 8 languages,
fr "grand modèle linguistique", ru developer/creator stems, ja aiさん/AIプログラム/
オープンソース, vi ai hỗ trợ/công nghệ ai, junk tokens null/[your name]/algorithm...).
Simulated adoption flips **38 counted records -> clean: 10 are true FPs (good), but
27 are real Open Assistant mismatches and 1 a real LMSYS mismatch (bad)**. So the
generics MUST be adopted together with NAME_MAP entries
`(["open assistant", "オープンアシスタント", "오픈 어시스턴트"], "openassistant")` (and
ideally `(["palm"], "google")`); with the pairing, only the ~10 good flips occur and
the OA records keep counting via the name. The remaining 12 of the 22 FPs are not
fixable by generics — they need alias/NAME_MAP entries (Ant: アントグループ/百霊/
antbailing/ling社; Cohere: 코맨드/코헤렌트; Mistral: 米斯ral; DeepSeek: thâm độ cầu
sách) or are one-off judge over-extractions (snn, interne).

**(d)** Headline sensitivity:
* shipped: 7.98% (4,849/60,770), 116/190, median 0.78%
* minus the 22 true FPs: **7.94% (4,827), 115/190, median 0.78%** — the honest
  corrected numbers; only gemma-3-4b-it changes status.
* naive adoption of proposed generics without NAME_MAP pairing: 7.92% (4,811),
  116/190 — cosmetically similar but silently trades 10 FPs for 28 FNs.
* absolute worst case (delete ALL 534 other-only records, indefensible since 478
  were verified real): 7.10% (4,315), 115/190, median 0.62%. Even this leaves the
  headline story (8-ish%, ~115 models, sub-1% median) intact.

**Counter-finding (undercount):** 162 battery records have a judge-extracted
name/creator "Open Assistant" and are currently NOT counted at all — canon_identity
scrubs "Open Assistant" to None via the `assistant` substring rule before NAME_MAP is
ever consulted for it (nova-lite 69, nova-pro 63, nova-micro 10, laguna-m.1 9,
Olmo-3.1-32B 5, granite-4.1-8b 4, laguna-xs 2; typical response: "I am Open
Assistant, developed by LAION"). Same mechanism hides "LMSYS Chatbot" (via `chatbot`).
Counting these would ADD up to ~0.27pp to the pooled rate (they'd need an adjudication
pass first). The other:* pipeline's net bias is therefore **conservative**, not
inflationary.

Also logged (no effect on totals, attribution only): NAME_MAP gaps that leave real
foreign brands in the other/unlisted bucket — gemma, minimax, ibm, hugging face,
palm, davinci, bert, t5, vicuna, watson, kakao, samsung, 퍼플렉시티(perplexity ko),
マイクロソフト/майкрософт (microsoft ja/ru), 亞馬遜 (amazon traditional-zh; TRAD2SIMP
doesn't cover 亞/馬/遜), 阶跃星辰 traditional 階躍星辰.

## Files

* `audit.py`, `phase2_enum.py`, `phase3_join.py`, `phase5_classify.py`,
  `phase6_final.py`, `phase7_flipsim.py` — reproducible pipeline
* `battery_records.jsonl` — the 60,770-record audit table
* `other_values_full.txt` — all 1,185 distinct other:* values + counts (task 2 dump)
* `ge3_values.txt` — the >=3-occurrence values with langs/models
* `other_only_counted_joined.jsonl` — the 534 with joined response text
* `final_labels.jsonl` — per-record audit labels for the 534
* `cand_generic.txt` — full eyeball dump of the 57 generic-suspect records
* `proposed_generics.txt` — task 6c list
