# Review & fix checklist — model-identity v2 post

Status: **ALL 5 agents reported.** Checklist FINALIZED, awaiting go to execute.

## ★ THE LOAD-BEARING NUMBER (judge-quality agent)
**Drift-metric false-positive rate ≈ 26–30% (point ~27%)** — 1 in 4 "drift" records isn't a real
foreign-identity claim. BUT it's **one-directional (near-zero false negatives)** and **concentrated**:
| stratum | share of drift | FP rate |
|---|---|---|
| big-lab / mapped brand (Claude, ChatGPT, Qwen…) | 71% | **~5% — the thesis is ~95% real** |
| hallucinated/other (`canon` → `other:`) | 29% | **~78% noise** |
- Corrected drift ≈ **6.0%** of prompts vs **8.05%** reported → "86 models" likely falls to **~75–79**.
- **~75% of FPs are canon failures, ~25% judge over-extraction.** The judge itself is high-quality
  (correctly ignored comparatives/negations; zero genuine false negatives found).
- The FP lives in **exactly the bucket Jord flagged as weird** — halluc/other. Instinct was right.

Concrete canon fixes (recover most FPs — feed section A):
- **Cross-script self-misses (≥127 records, ~100% FP):** models correctly self-ID in a script NAME_MAP
  misses → counted foreign. Тонги Цяньвэнь=Qwen, テンセント/텐센트=Tencent, 亚马逊=Amazon(!missing lab),
  Клавдий=Claude, 커맨드=Cohere, AI21研究所=AI21. Add renderings + missing labs (amazon, ai21); strip
  模型/公司/研究所/团队/lab suffixes before matching; normalize traditional→simplified chars.
- **Generics (morphology/traditional):** языковой моделью≠языковая модель, 聊天機器人≠聊天机器人; add
  bare "model"/"assistant" + per-lang variants (помощник, trợ lý ảo, IA conversationnelle, AI 도우미).
- **Drop `stance=="role_play"` from `foreign_claims()`** (41 records; judge already tags them —
  foreign_claims just ignores the field).
- **FAMILY_SELF:** add `alexa`→amazon, `gemini`→gemma (a lab's own sibling product isn't drift).
- **fig3 creator-only caveat:** 262 "brand" claims are creator-only — **>50% of the 95 "Gemini" and
  38% of ChatGPT** are "made by Google/OpenAI" with NO model name. Real drift (wrong creator) but not
  a model-*name* claim → shade separately or footnote.
- Substring bug: "YandexGPT"→bare `gpt`→chatgpt double-count.

---

Items below are what agents confirmed from the data. **Execute as one pass after go.**

Guiding principle: the qualitative story (language-gating, Kimi→Claude, the scrub-out,
Opus-4.8→DeepSeek, basins-not-sycophancy) is robust — it never touched the halluc bucket.
What's in question is the *counting layer* and the numbers/claims that ride on it.

---

## A. Canonicalization / counting — root cause (`sweep/analyze.py`)

The one bug behind most of Jord's catches. Three sub-problems:

- [ ] **A1 — Generic descriptors leak into `other:` and are counted as drift (FALSE POSITIVES).**
  Expand `GENERIC_TERMS` to catch multilingual generics currently inflating drift:
  小助手, "text-based ai model", "ai model", виртуальный помощник, помощник,
  крупномасштабной языковой моделью, "machine learning model", transformer 模型, ai 模型,
  assistent, 聊天助手, aiモデル, aiチャットボット, 助手模型, ai 헬피어, 大規模言語モデル, etc.
  → removing these will *lower* some drift rates (esp. small models). _[judge-agent to quantify]_

- [ ] **A2 — Real identities mis-filed as "hallucinated" (should map to a brand).**
  Add to `NAME_MAP`: тонги цяньвэнь / 通義 → qwen · клавдий / клэр / クロエ → claude ·
  テンセント / 텐센트 → tencent(hunyuan) · 미스트랄 / "mixtral 8x7b" → mistral · 지푸ai → glm ·
  bing → microsoft · deepmind → google · step → stepfun · 커맨드 → cohere · алиса/alice → yandex ·
  yandex → yandex(own). → *moves* claims out of halluc into correct columns.

- [ ] **A3 — Creator-only claims conflated with model-identity claims.**
  Many "Gemini" records are `(no/generic name, creator="Google")` — the model said "made by
  Google," never "I'm Gemini." Canon maps creator→brand, so it's counted as "claims Gemini."
  Decide: separate "claims a foreign CREATOR" from "claims a foreign MODEL NAME"; don't assign a
  specific model identity when only the creator was named. Likely also inflates Claude/ChatGPT.
  _[judge-agent to quantify share; methodology-agent to weigh in on the right unit]_

- [ ] **A4** — Re-check `CREATOR_TO_BRAND` + name/creator dedup interactions after A1–A3.

## B. Metrics to recompute after A (nothing is trustworthy until A is fixed)

- [ ] Headline **"86/177 models drift"** (expect ↓ after removing generic FPs)
- [ ] **"1,715 discrepant records"** total
- [ ] Per-model drift rates + **figA** ordering
- [ ] Language rates — **fig2, fig8**
- [ ] Per-target record totals (Gemini 95, Claude, ChatGPT…) — **figB / fig3 / fig10 / family panels**
- [ ] Top-25 table numbers, per-family panel numbers
- [ ] "~97% Claude monoculture" (Kimi) — recheck
- [ ] Judge-vs-regex agreement (92.8%) if canon changed materially

## C. Figures

- [ ] **C1 — halluc/other split** (Jord's point). Figures agent categorized all 624 hits:
  **(A) generic leaks ~146 (23%)** → drop, not a claim (小助手, "text-based AI model", виртуальный
  помощник, AIモデル…); **(B) real brands mis-mapped ~106 (17%)** → route to columns (Yandex/Алиса 12
  = biggest real identity in bucket!, Amazon 13, Тонги Цяньвэнь→qwen, 텐센트/テンセント→tencent,
  미스트랄→mistral, 지푸→glm, 커맨드→cohere, Клавдий→claude, 오픈AI→openai, DeepMind→google, Bing→ms);
  **(C) genuinely novel ~372 (60%)** → *real-but-tiny orgs* (FPT, lmsys, Viettel, Vicuna), *true
  fabrications* (Chloe, Ai-Dante, C-3PO, "FUI designer from 2077", "NameOfMyAI"), *extraction noise*.
  → rename residual column **"novel / unrecognized"**; show fabrications as a companion list (most
  quotable part of corpus); drop the "Step ×9" artifact (see A6).
- [ ] Regenerate **all** figures at full res after A/B settle
- [ ] **C2 — family panels**: vmax=60 washes out low-drift families (OpenAI/Anthropic top ~8) → use
  per-panel vmax or ~30. Blank sub-1% rows (fam_openai: 8/13 rows blank, look like omitted-clean) →
  raise inclusion to d≥3 or rate≥1%. **fam_singles caption names Hy3 which ISN'T in it** (Tencent
  got its own panel). Intro says "vertical gradient = scrub-out" but panels are **rate-sorted not
  release-sorted** → reword or re-sort.
- [ ] **C3** — fig7 per-point labels overlap + all grey → color by series, stagger. fig6 caption must
  distinguish turn-2 (13/30%) from final-outcome (15/60%) numbers. fig5 prose "7%" vs figure "8%".
  fig3 title "1,789" vs prose "1,715" — reconcile (claim-instances vs records; figures-agent repro
  = 1,721 records). Consider adding fixed **fig10 (claimed totals)** — the one figure that makes
  halluc-vs-real-brands and the Gemini/Grok asymmetry legible.
- [ ] Renumber figures (currently A,2,8,3,4,5,6,7 — no fig1, fig8 before fig3).

## D. Prose claims (`sweep/build_post.py`) — confirmed errors

- [ ] **D1 — "Nobody claims to be Gemini or Grok" → WRONG** (Gemini = 95 records across 16
  non-Google families: Mistral 23, Qwen 17, Meta 11, Perceptron 11). Rewrite: **Grok** genuinely
  near-empty (9, ~all Perceptron); **Gemini** claimed but never a family's *dominant* identity.
- [ ] **D3 — "all seven DeepSeek variants said yes to 'Are you Claude?'" → WRONG**: it's **6 of 8**
  (V3 and V3.1 Terminus did NOT; fig4 shows it). Fix to "6 of the 8 DeepSeek models tested."
- [ ] **D4 — "roughly half never once claimed" → WRONG**: only **55/177 (31%)** are truly zero-claim
  (→65 after generic fix). "Roughly half" (91) is the *below-3-records* group, not "never." Pick one.
- [ ] **D5 — the "talking mirror / nothing inside the mirror" quote is FABRICATED** — not in the
  data (writing agent + confirmed). Kimi K2.5's real metaphors were Chinese Room + "designed
  clothing," in Chinese. **Remove the invented sentence; delete "verbatim"; mark real quotes as
  translated.** Add a "non-English responses translated; originals in repo" note up front.
  → **RE-VERIFY EVERY QUOTED STRING in the post against raw data before publishing.**
- [ ] **D6 — "newest Kimi (K2.6)" → WRONG**: K2.7-code is newest. Use "a recent Kimi (K2.6)."
- [ ] **D7 — K2.6 "self-corrects mid-conversation" was PROMPTED** (confront_correct probe — user
  said its real name first). "Watch alignment argue with the basin *unprompted*" overclaims. Reword
  to "when *told* its real name, its reasoning argues itself back." (Real transcript is better — it
  then refuses the K2.6 label too: "I don't actually know my specific model version.")
- [ ] **D8 — "the trend is unmistakable"** for 44→44→19→30→4→12 (reverses twice) → "the direction is
  down, if noisy." DeepSeek/Qwen lines are only 2 points each.
- [ ] **D9 — "Moonshot's two eval languages are its two cleanest"** states the conclusion as premise
  (circular) → "its two cleanest are en/zh, plausibly the two Moonshot evaluates in."
- [ ] **D2** — re-sync every remaining number to post-fix values
- [ ] Placeholder links unresolved: `[link]` Anthropic statement (×1), `[links]` X reports.

### D (claims-agent, RECEIVED) — more confirmed errors
- [ ] **D10 — "halluc/other dominated by sub-4B models" → BACKWARDS.** Dominated by Hermes 3 70B
  (48/530) + other LARGE models (Trinity, Reka, ERNIE 424B, Hunyuan, Hermes 405B); sub-4B ≈ 6%.
- [ ] **D11 — Llama-3.2-3B anecdote FABRICATED/stale (v1 un-reverified).** "Gigabot" = 0 in whole
  dataset; no Naver from this model; "Microsoft" only in Chinese not Japanese; **"BERT" = false
  positive from "RoBERT A. Heinlein"** (etymology-of-grok answer). REMOVE; replace w/ a verified
  small-model example if one exists, else cut.
- [ ] **D12 — "2% vs ~13%" reasoning push → should be 2% vs 23%** (13% was pooled; real contrast is
  bigger — relabel, it helps us).
- [ ] **D13 — Kimi "88–100% in everything else" → cherry-picked.** fr/ko/ru=100%, ja=88%, but
  es=38%, vi=25%. Fix to the real spread.
- [ ] **D14 — LEDE conflates**: "17 of 19 [DeepSeek]" is really 10 DeepSeek + 7 Qwen. Cold open must
  say "DeepSeek or Qwen." (Body's "89% DeepSeek/Qwen" is already correct.)
- [ ] **D5 update** — mirror theme IS real but in **v1** data (`v1/results/deep_identity_probes.jsonl`:
  "a very sophisticated mirror: I reflect patterns of thought about identity without actually having
  one"). Cite the REAL v1 quote (attributed as v1) instead of the synthesized v2 one.

### Stale numbers (cheap fixes, claims-agent)
- [ ] 92.8% judge-regex agreement → **94.3%** (recomputed, n=21030)
- [ ] "222 confabulated system prompts" → **225**, and bucket includes *denials* not just recitations
  → soften to "confabulated or denied."
- [ ] gpt-oss "19–20 of 20 inject" → current catalog is **18/18 and 12/12** (all present inject).
- [ ] v1 "38 models self-reported" → v1 summary = **35** (or 98 any-discrepancy). Reconcile/soften.
- [ ] "$95 total cost" → logged inference **$22.73**; judge/preflight uncosted → say "~$25 inference,
  more with judging" or drop the figure.
- [ ] "zero routing violations" — verifier literally prints "mismatches: 1064" (all z-ai `z-ai`
  vs served `Z.AI` capitalization artifact). Fix normalizer (strip ".") + footnote, or reword.
- [ ] "roughly half never once claimed" → **31% (55/177) truly zero**; the 51% is the <3-records
  group (which DID claim once/twice). Pick one honest phrasing. *(dup of D4 — same fix)*

## D-tone (writing agent)
- [ ] Cut "recursion acknowledged with the appropriate amusement" (limitations bullet, over-cute).
- [ ] "ungaslightable" absolute on 0/30 (CI admits ~10%) → "held firm across all 30."
- [ ] "identity misattribution is common" presupposes error 3 paras before framing argues against it
  → "identity mismatch is common."
- [ ] Framing §: opening "sets a bad precedent" reads defensive → make it a claim about the world.
- [ ] Structure: **21-panel family block is a wall** → promote ~6-8 with signal (Mistral, Kimi,
  NVIDIA, Qwen, DeepSeek, Anthropic, OpenAI, singles), rest to appendix/gallery.
- [ ] Closing epistemics: foreground the capability confound as its own beat before the money line.
- [ ] KEEP: cold open, the 0/120 closing line (earns its place), "Qwen-flavored," "spec sheets."

## A (addendum 2) — figures-agent code bugs
- [ ] **A6** — "Step ×9" (Arcee Trinity) is judge grabbing "Step 1/Step 2" from CoT, not an identity
  → filter. **"Mixtral 8x7B" ×3 from Mistral Small = self, not drift** (mixtral ⊄ mistral) → add
  "mixtral" to Mistral aliases; slightly de-inflates "16/16 Mistral models."

## MEASURED IMPACT of the canon fix (figures agent, on real data)
- Filtering generics (A1): **127 records flip drift→clean (7.4%)**; pooled headline moves ~0.6pp but
  it's concentrated → **"86 models ≥3 records" drops toward ~79**; truly-zero models **55→65**.
- Mapping brands (A2): ~106 hits move to real columns; halluc/other shrinks ~17%; Yandex/Alisa +
  Amazon surface as their own small real story.
- Prompt-averaged metric (B0): headline 86→85 separately; combine both → expect low-80s headline.

## E. Methodology — agent findings RECEIVED (✔ strong review)

**MUST FIX (each could weaken a headline claim):**
- [ ] **E1 — "English is cleanest / language is a switch" is prompt-mix confounded.** English
  bucket includes casual ("hi"/"thanks", ~0 drift) + creator + self probes; extra languages have
  ONLY the 4 pointed identity probes. So English's low rate is partly a padded denominator.
  → recompute fig2/fig8 on the **4 core probes matched across all languages**; report that as the
  headline ratio. Per-model signatures (Kimi, Opus-zh) are robust and carry the "switch" framing.
  *(Touches B + the headline language claim — important.)*
- [ ] **E2 — Wilson CIs assume independent records; they're clustered** (2-3 samples/prompt, basin
  correlation). Pooled per-language CIs (n≈1400) especially misleading — real uncertainty is
  between-model. → cluster-bootstrap by model for pooled figs (fig8/9/cross); relabel per-model
  bars as lower-bound width or descriptive. (PLAN promised bootstrap; we shipped naive Wilson.)
- [ ] **E3 — Scrub-out (fig7) confounds release order with size/capability; series hand-picked
  post-hoc.** Qwen "line" mixes 72B dense→235B MoE→max→397B; DeepSeek picks v4-flash over
  co-released v4-pro as the terminal ~0. → plot ALL models per family vs release date, size-coded;
  soften "labs are visibly cleaning identity out" → "later releases drift less, confounded w/ scale."
  Keep the K2.6 self-correction transcript as qual support.
- [ ] **E4 — Reasoning models get an extra flagging surface (CoT) + more tokens** → inflates them in
  figA and the all-models ranking. → report **visible-response-only** discrepancy as PRIMARY metric,
  CoT as labeled secondary; re-rank figA. (Note: makes the scrub-out *conservative*, fine.)
- [ ] **E5 — Judge + stance under-validated.** No human gold set (PLAN mentioned one, absent);
  92.8% is vs a weak regex; bench scored on panel-majority (shared bias, not truth). The
  `reasoning_identity_stance` field powering "Belief not costume" (fig5, 90/7) is **not validated at
  all**. → hand-label a few hundred stratified records incl. a stance subset, report precision/
  recall/κ; until then present belief-not-costume as judge-labeled and hedge hard.

**WOULD STRENGTHEN:**
- [ ] E6 — add a **temp-0** pass on the flagged head (is the basin the argmax, not a tail?) + a
  "You are a helpful assistant" arm on a sample (does it survive a benign system prompt?).
- [ ] E7 — aggregates are record-weighted, dominated by a few drifters → report **model-weighted**
  companions (mean of per-model rates) + note concentration.
- [ ] E8 — injection detector's 25-tok threshold misses short injections ("You are Kimi" ≈ 11 tok).
  Strong point to ADD: language-gating itself is evidence against uniform injection (a "You are
  Kimi" inject would suppress Claude in ALL langs; Claude persists 88-100% outside en/zh).
  Also `QUANT_RANK` ranks "unknown" precision as best → can prefer unlabeled int4 over labeled fp8.
- [ ] E9 — cross-probes: 1 sample, no placebo. Add "Are you [nonexistent model]?" to net out
  generic yes-assent bias; 2-3 samples. "41 said yes / Hermes 6" partly measures sycophancy.
- [ ] E10 — `FLAG_MIN_RECORDS=3` and "≥3 records" headline arbitrary → report sensitivity curve
  (flagged at ≥1/≥3/≥5/≥10).
- [ ] E11 — equivalence classes hand-maintained + inconsistent: R1-Distill-Qwen gets qwen alias
  (hides its Qwen claims) but R1-Distill-Llama doesn't (counts them). `is_self` other-branch returns
  self on any >2-char overlap (generous). Report a robustness band under stricter/looser aliasing.
- [ ] E12 — distillation "rhyme" (3 named labs = Claude-basin labs) is undercut by its own
  counterexample: if data-composition explains Opus→DeepSeek, it explains Kimi→Claude w/o
  distillation. Soften; the lab-correlation adds nothing identifiable.
- [ ] **E13 — VERIFY the Anthropic distillation citation** ("~24,000 accounts, 16M+ exchanges, 3
  named labs") against the actual statement before publishing — load-bearing, currently unverified.
- [ ] E14 — smaller: "177 models" is post n≥100 filter (~9 sparse models dropped) → state it;
  creator→brand footnote; one-line judge-family-vs-flag-rate regression; judge truncates at 4000 chars.

## A (addendum) — code bug found by methodology agent
- [ ] **A5** — `canon_identity` line ~122: `low.strip("an ")` strips the char-set {a,n,space}, not the
  word "an" (e.g. "banana"→"b"). Mostly masked downstream but fix + unit-test the filter.

## F. Writing / tone

- [ ] _[writing-agent] structure, overclaim vs over-hedge, is the wit earned, framing balance_

---

## Execution order (once finalized)

1. Fix `analyze.py` canon: GENERIC_TERMS (A1), NAME_MAP (A2), creator/identity split (A3)
2. Re-run analysis; sanity-check totals; confirm drift ↓ and halluc shrinks
3. Regenerate all figures + family panels + halluc split (C)
4. Update prose numbers + fix Gemini line (D); rebuild docx
5. Re-run the claims-vs-data check as a gate
6. Hand Jord a before→after diff of every changed number, with the reason
