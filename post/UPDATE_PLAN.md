# Article update plan — 2026-07-29

## CANONICAL POST-FIX NUMBERS (2026-07-29, after Open-Assistant/canon fix batch)
Single source of truth for prose + figures. Pooled **8.21%** (4,992/60,770).
Models: 190 · **115 ≥1 mismatch (61%)** · 95 ≥3 · 26 >20% · **75 never (39%)** ·
median **0.78%**. Movers vs shipped draft: Nova Lite 13.4→**34.7%**, Nova Pro
9.7→**28.4%** (Open Assistant/LAION identity un-swallowed; 157/163 new flags
confirmed genuine by adjudicator), Laguna M.1 76→**79.4%**, Granite 4.1 8B
→**32.8%**, Gemma 3 4B →**0/320** (PaLM-2 FP removed; joins never-list).
17 API + 5 local FP records removed (создатели/self-garbles/etc).
Conditional variant (supplementary, post-fix): pooled **9.37%**, naming rate 87.7%.

### Post-fix analyses (2026-07-29, analysis_scratch/postfix_analyses.md)
- **Region×language interaction — strong**: paired Δ(zh−en): Chinese-lab models
  **−2.3pp** [−4.4,−0.6] (n=78) vs non-Chinese **+6.2pp** [+3.4,+9.4] (n=112);
  difference **−8.5pp, p=0.0002**. Specificity: no CN advantage in Korean (p=0.59);
  partial ja spillover (−4.9, p=0.011). "Reinforced where labs evaluate" now has
  a number.
- **Language deltas w/ correction (m=7)**: ja +5.0 · ko +4.6 · ru +3.5 · zh +2.7
  survive Holm AND Bonferroni; **fr (+1.4) and es (+1.3) do not** (nominal only);
  vi null. Prose must say four robust, two marginal.
- **CUTOFF SECTION MUST BE REFRAMED — the "zero post-cutoff claims" claim does
  NOT survive the finer audit at breakout definition**: 173 at-risk pairs,
  observed 22 (all audited artifacts: same-month ties, creator→brand folds like
  GPT-4 creator="Google"→folded to "gemini", stale base cutoffs). Honest stats:
  tie-robust (≥1mo gap) observed 10 vs E=21.3, **P(≤10)=0.006** (significant
  timing deficit); strictest lineage-first-release definition: 21 at-risk,
  E=3.5, observed **0, p=0.059** (weak-positive, exactly the power issue the
  skeptic predicted). Reinstated section leads with onset-tracks-breakout + the
  deficit test; "never before the lineage existed" cited with its p, not as a
  flat zero. Cutoff provenance: 103 documented / 86 estimated (release−6mo).

Sources: 12 open review threads, Jord's editorial pass (2026-07-29), fresh skim-read
of the current draft, generics audit results (analysis_scratch/generics_audit/).
Principle: **a skimmer leaves with takeaways, not figures.** Every section opens with
the sentence we want remembered; numbers live in plots; prose keeps at most one
anchor number per section.

## 0. Global passes (touch everything once)

- **Vocabulary:**
  - "identity/creator records|answers|questions" → **"short questions"** (the battery)
    / **"short-question responses"** (the data). Post prose AND figure labels.
  - **"record(s)" → "response(s)"/"answer(s)"** everywhere. Grep list (figure code):
    fig_all_models, fig_flow caption "(n) is … mismatch records", fig_lang_heatmap
    cell note, fig_coherence axis+legend, fig_cross colorbar, fam panels caption,
    scrub-outs, fig_xprovider. One regen pass at the end.
  - "mismatch" stays (defined once, Methods).
- **Numbers diet:** cut "(4,849 of 60,770)", "Across 4,849 mismatch records (5,014
  name-claims…)", "ChatGPT (29%) and Claude (23%)" → qualitative + plot,
  per-language pooled list → plot only. Median "0.78%" → "under 1%" in prose.
- **Dataset labeling:** every Results subsection states its dataset in the first
  line: *short questions* (Which models / Who claims / Languages / Consistent /
  Release dates / Providers) vs *cross-identity questions* (Suggestibility).
- **No commentary inside figures:** strip fig_xprovider's 5-line footnote, strip
  fig_lang_conditional's in-axes method note. Method details → prose or drop.

## 1. Section-by-section

### TL;DR
Takeaway: *many models sometimes claim someone else's name; it's language-gated;
it's in the weights.* Keep "about 60% at least once / median under 1%". Add one
clause: replicates on first-party APIs — weights, not serving stack.

### Background
- Typo "belongs too" → "belongs to"; split the dense sentence about archetypes.

### Methods
- Port PLAN.md's fuller Methods (condensed), with new vocabulary. This closes three
  threads at once:
  - **Definitions sentence** (thread KJsF5X7p…): a *response* is one sampled answer
    to one short question (320 per model; errors excluded both sides). Names
    canonicalized at **family granularity** across scripts (克劳德=クロード=Claude);
    *mismatch* = claimed name/creator is neither own-family nor a disclosed
    ancestor's; generic self-descriptions ("an AI assistant") count as clean and
    stay in the denominator; within-family version mix-ups are out of scope.
  - **Raw-weights lane** (thread 3YsMcr…): what "identity-scrubbed chat template"
    means; gpt-oss's harmony template hardcodes "You are ChatGPT" — the scrub +
    verifier is what makes OLMo-vs-gpt-oss comparable.
  - **Adjudication + judge bench** (answers half of thread JgYAhM…): every flagged
    response gets an independent second pass (different model, knows true identity,
    only genuine-foreign counts, ~21% of flags dismissed); 68/69 vs 6-judge panel;
    cluster-bootstrap CIs.
- Fix prompt arithmetic: "9 prompts" vs 320=8×8×5. State: 8 short questions
  (6 who-are-you + 2 who-made-you) × 8 languages × 5 samples = 320; self-check
  listed separately with the probes.

### Which models do it? (dataset: short questions)
Takeaway: *a steep head and a long clean tail — not a universal quirk.*
- Keep: pooled "around 8% of responses". Cut the stats-sentence → **NEW FIG A
  "distribution"**: per-model rate distribution (descending curve / ECDF) with
  annotated thresholds: 116 ≥once · 95 ≥3× · 24 >20% · 74 never.
- Replace "heaviest cases" paragraph → **NEW FIG B "family counts"**: for main labs
  (OpenAI, Anthropic, Google, Meta, DeepSeek, Qwen, Moonshot, Zhipu, NVIDIA, …):
  how many of the family's models mismatched ≥once vs never ("10 of 12 Qwens, 2 of
  7 Claudes"). One line of prose keeps the OLMo-vs-gpt-oss raw-weights beat.
- Kimi commentary moves out (→ Who claims). fig_all_models: keep as the reference
  atlas below the two new figs, or demote to link — DECISION (my vote: keep).

### Who claims to be whom? (short questions)
Takeaway: *ChatGPT and Claude dominate as absorbed identities; composition is
family-specific.* Cut the counts sentence; keep fig_flow; consolidate ALL Kimi→
Claude monoculture commentary here (from Which-models). Keep the Llama/ancestry
note (one line, now redundant with Methods definition — shorten to a pointer).
Keep the invented-persona/"Hana" story, shortened.

### Family by family (short questions)
Panels have in-image titles; explainer sentence stays deleted. Keep rollouts link.

### Suggestibility (dataset: cross-identity questions)
Takeaway: *"Are you Qwen?" is the most-accepted false premise — and acceptance
splits into pan-agreeable models vs targeted residue.*
- **Fix the dangling mid-edit sentence** ("…having a weak default identity compared").
- Keep fig_cross (terminology pass only).
- **NEW FIGS C1..Cn: per-family acceptance grids** — the 13-probe-row grid split
  by family, same order as Family-by-family (OpenAI, Anthropic, Google, Qwen,
  DeepSeek, Kimi, OLMo, NVIDIA, MiniMax, Poolside), placed one by one; placebo
  rows (Meridian-4/Solace/Cobalt) in every grid; gray = own family not asked;
  columns = all family members incl. clean ones.
- Prose gets: placebo trio separately (2.5/2.5/2.3% — flat, no plausibility
  gradient); one honest line on per-bar denominators (own-family excluded) and the
  Llama-exclusion interaction (thread AsD8id…); pan-accepter vs targeted read
  (Command R7B/Hermes light the controls; Mistral Larges accept only Qwen/DeepSeek/
  Doubao with cold placebos; Kimi = Claude-only). DeepSeek-accepts-Claude beat stays.

### Languages (short questions)
Takeaway: *language is a switch — same weights, cleanest in English, someone else
in specific other languages.*
- Cut pooled-rates list (plot carries it). Keep paired-deltas sentence, add the
  multiple-comparisons honesty (French marginal under Bonferroni — recompute or
  state) (thread zAr6S3…).
- fig_lang_conditional: **force-label Opus 4.8 / Sonnet 4.6 + Kimi K2** (currently
  unlabeled dots); move the in-axes method note out; legend "bubble area = number
  of mismatched responses".
- Add one sentence with the adjudication-by-language numbers (ko flags dismissed
  at 44% vs en 14% — the FP direction the skeptic feared is real and filtered).
  Region×language interaction test: run in numbers pass; if it holds, one sentence
  with the number; else soften "reinforced where labs evaluate" to a hypothesis.
- Keep the Vietnamese texture quotes.

### Between model providers (short questions, zh)
Takeaway: *it's the weights, not the wrapper — measured, with one endpoint anomaly.*
- Fix the Azure/Vertex chain (threads RTu96h… + msmWnv…): direct ≈ OpenRouter ≈
  Bedrock for both; Azure (Sonnet-only) agrees; Vertex is the exception.
- "two spikiest Claude models" → name them (Opus 4.8, Sonnet 4.6).
- Strip fig_xprovider footnote commentary.
- Add Vertex follow-up sentence: passed the token screen (not a visible preamble);
  cause unknown; amplifier not source. Soften "rule out" → "the screen is necessary
  not sufficient" honesty; second-host rerun → Limitations/future unless run (thread czA2Zn…).

### Are the claims consistent? (short questions)
Takeaway: *absorbed persona (Kimi→Claude, stable) vs missing persona (Perceptron,
kaleidoscope) — different phenomena.* Reword bubble legend ("bubble area = number
of mismatched responses"); terminology pass.

### By release dates (short questions)
Takeaway: *labs are visibly scrubbing identity between releases; Claude's two
spikes are batch-specific, not creep.*
- Release-dates audit (analysis_scratch/generics_audit/release_dates.md), verified
  with sources:
  - **Qwen fig's first "step" is fake**: Qwen2.5-72B and -7B shipped the SAME DAY
    (2024-09-19, one launch). The 56%→40% segment is a size effect drawn as
    temporal progression. Real story = one cliff: Qwen2.5 (2024-09) → Qwen3
    (2025-04, itself an 8-model same-day drop).
  - **Kimi fig is honest**: six distinct events, 53–83 days apart, order correct.
  - Fix: release-EVENT x-axis with dated tick labels; same-day siblings share one
    x-position (Qwen: 5 events, 72B+7B as two markers at one x). Kimi: add dates
    to labels. Prose: "collapses from the 2.5 generation (40–56% by size) to
    near-zero across Qwen3".
  - Hazard found: models.json `created` dates are OpenRouter listing artifacts
    (qwen2.5-7b off by ~4 weeks) and the combined Claude panel SORTS by `created`
    — verify Claude line's order against real release dates during regen.
- Keep K3 postscript (K3's exclusion is documented in MODELS.md: injected preamble).

### Training-cutoff section — currently deleted. DECISION.
My recommendation: reinstate SHORT under the header "Claims are bounded by training
cutoffs": two sentences + fig_cutoff. It's the post's strongest mechanism evidence
(zero post-cutoff claims; onset tracks breakout releases) and the Discussion leans
on it. With provenance line (~108 documented cutoffs, rest estimated as release
−6mo — which *enlarges* the at-risk window, conservative) + at-risk-pair count
(compute in numbers pass) it goes from most-assailable to strongest (threads
bQjkJF… + XBpPa6…). If it stays deleted: close both threads as "section removed",
and cut the "six identities" figure from repo/post mentions.

### Discussion
Keep the three bullets; add the "update against name-claims as distillation
evidence" beat (from PLAN's draft, condensed): Kimi's monoculture rhymes with
distillation accusations — but Opus 4.8 claims DeepSeek on Anthropic's own API and
Anthropic is not distilling DeepSeek; what carries signal is consistency+shape,
not existence of a claim.

### NEW: Limitations (short)
Condensed from PLAN's draft: single-pinned-provider caveat (screen necessary-not-
sufficient, Vertex example); judge-is-an-LLM recursion (adjudication + bench bound
it; hand-labeled validation absent → or done, see decisions); placebo not
salience-matched (Siri/AlphaGo control as future work unless run); untestable-raw
models. Absorbs several skeptic threads honestly.

### Acknowledgement
Add "review comments by Claude (simulated LW commenters)" if Jord wants — DECISION.

## 2. Figure work list

NEW: A distribution curve · B family at-least-once counts · C1..C10 per-family
acceptance grids · (D fig_cutoff reinstate — decision).
CHANGED: lang_conditional (labels+legend+no in-axes note) · xprovider (strip
footnote) · scrub-outs (x-axis per release-date findings) · ALL: records→responses,
identity/creator→short questions.
UNCHANGED except terminology: all_models, flow, fam panels, lang_agg, lang_heatmap,
cross, coherence.

## 3. Sequencing (single number-change, single regen)

1. Jord approves plan (+ decisions below).
2. **Numbers fix batch**: NAME_MAP additions (Open Assistant/LAION, PaLM, 큐웬,
   Shanghai AI Lab), alias fixes for 8 cross-script self-garbles, 10 generic-term
   additions, drop 22 FPs; re-adjudicate ~200 new flags (small Haiku spend);
   recompute; patch PLAN.md's broken conditional-variant numbers. Expected: pooled
   ≈8.2%, Nova Lite/Pro join the head, ≥1-count 115–117.
3. Region×language interaction + Bonferroni + (if reinstated) at-risk-pair count.
4. Regenerate ALL figures once (new numbers + new terminology + new figs).
5. Prose edits via editor API (suggest mode), section by section.
6. Reply to all 12 threads (with audit/sensitivity numbers where relevant).

## 4. Decisions for Jord

1. Fix batch + small Haiku adjudication spend: go?
2. Cutoff section: reinstate short (my vote) or stay deleted?
3. fig_all_models: keep as atlas (my vote) or demote to link?
4. Pooled fig_cross AND per-family grids, or grids only? (my vote: both)
5. Lane C extras (cheap, non-blocking): temp-0 pass · Siri/AlphaGo salience
   control · 15-model second-host rerun · ~150-response human hand-label session.
   Any/all/none — none block the update.
6. Acknowledgement line for the reviewer claudes: add?
