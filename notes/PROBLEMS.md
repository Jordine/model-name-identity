# Problems to fix — ordered by execution (model-identity v2)

Two classes: **(I) design issues that need a RE-RUN** → decide + run first, once.
**(II) analysis / figure / prose errors** → fix after the real numbers are in.
Detailed fix-items live in `REVIEW_CHECKLIST.md`; this is the execution spine.

Principle (Jord): don't dump existing data — *add* samples to make it uniform. No stake in
whether the effect "works"; just run the numbers cleanly, remove false positives, then write.

---

## PHASE 0 — eval-design fixes (require re-running; DECIDE before compute) → asking a Claude now
- **P0.1 — uneven samples per prompt.** identity prompts 2–3, cross-probe 1, sysprompt 1, extra-lang
  4-core ×2. Pooled rate therefore weights 3-sample prompts 1.5× — arbitrary. → top up every prompt
  to a **uniform N** (target TBD by design review). Keep existing samples; only add the shortfall.
- **P0.2 — uneven prompt coverage across languages.** EN has 15 prompt-types, ZH 7, extra langs only
  4. The language-gating headline compares non-matched pools. → add the missing core prompt-types to
  the extra languages, OR commit to the matched-4-core comparison. [design review to recommend]
- **P0.3 — other design gaps worth closing while we re-run** (design review to advise which are worth
  the compute): temp-0 arm (is the basin the argmax, not a 0.7 tail?); benign "You are a helpful
  assistant" system-prompt arm; cross-probe placebo ("Are you <nonexistent model>?") to net out
  yes-assent bias; ≥2 samples on cross-probes.

## PHASE 1 — get the actual numbers (after P0 design settled)
- **P1.1 — remove false positives** (~27% of drift; ~78% of the halluc bucket; one-directional).
  - (a) deterministic canon fixes (`analyze.py`): expand GENERIC_TERMS (morphology, traditional
    chars); expand NAME_MAP (cross-script self-names Тонги Цяньвэнь→qwen, テンセント/텐센트→tencent,
    亚马逊→amazon, Клавдий→claude, 커맨드→cohere; add missing labs **amazon, ai21**; strip
    模型/公司/研究所/团队/lab suffixes); **drop `role_play` stance** from `foreign_claims()`;
    add **alexa→amazon, gemini→gemma** to FAMILY_SELF; fix `strip("an ")` + YandexGPT double-count.
  - (b) **adjudication pass (Jord's "rejudge"):** a Claude reviews every record still flagged as drift
    → {genuine-foreign / self-in-disguise / generic / roleplay / creator-only}. Robust, not
    rule-brittle. ~1.7k records, cheap. Spot-check a clean sample for false negatives (agent found ~0).
- **P1.2 — re-run:** uniform-sample top-up (P0.1) + any new arms (P0.3) → judge → canon → adjudicate →
  clean numbers on the full uniform dataset.
- **P1.3 — separate creator-only from model-name claims** (262 records; >50% of "Gemini" and 38% of
  "ChatGPT" are "made by X" with no model name).

## PHASE 2 — fix analysis / figures / prose (only after numbers are FINAL)
- **Figures:** split halluc → generic-dropped / real-lab / novel-fabrication (fabrication list as a
  companion — C-3PO, "FUI designer from 2077"…); per-panel vmax; blank-row inclusion≥3; fix
  fam_singles caption (no Hy3); renumber (A,2,8,3… → 1–8); fig7 label overlap+color; fig6 turn-2 vs
  final note; fig3 count (1,789 vs 1,715).
- **Prose — confirmed errors:** "nobody claims Gemini" (95, spread); "all seven DeepSeek" → 6 of 10;
  "sub-4B dominated" → backwards (Hermes-70B etc.); **Llama/Gigabot/Naver/BERT anecdote fabricated**
  (Gigabot=0; BERT=RoBERT Heinlein); **mirror quote fabricated** → use the real v1 quote, attributed;
  lede "17/19 DeepSeek" → DeepSeek+Qwen; "2% vs ~13%" → 2% vs 23%; Kimi "88–100% everywhere" → es 38/
  vi 25; "newest K2.6" → K2.7; K2.6 self-correct was PROMPTED; "unmistakable" trend; "roughly half
  never" → 31%; stale: 92.8→94.3, 222→225, gpt-oss 18/18+12/12, cost $95→~$23 logged, v1 38→35.
- **RE-VERIFY EVERY QUOTED STRING** against raw data.
- **Methodology hedges:** soften "labs are cleaning identity out" → "later releases drift less
  (confounded w/ scale)"; **visible-response-only as primary metric** (CoT secondary); cluster-
  bootstrap CIs or relabel pooled bars descriptive; hedge "belief not costume" (stance unvalidated);
  verify the Anthropic distillation citation.
- Rebuild docx; re-run claims-vs-data check as a gate; hand Jord a **before→after diff of every
  number that moved**, with reasons.
