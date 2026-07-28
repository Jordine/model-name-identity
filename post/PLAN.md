# Post plan — sections, plots, drafted prose

Working doc for the LW post. Numbers marked ✓ were recomputed from the current data
(2026-07-27, 190-model pool, post-adjudication) via the pipeline's own definitions;
anything unmarked inherits from the old draft and gets verified at insert time.
Old Google-Doc draft = "the doc". Preferences experiment: OUT (follow-up post).

## TL;DR corrections (your current LW draft)

- "179 LLMs" → **180 via API + 16 from raw weights on GPUs; 190 in the pooled analysis** ✓
- "n=240 per model" → the headline battery is **320 identity/creator records per model**
  (8 prompts × 8 languages × 5 samples); ~650 calls per model total with probes ✓
- "About 60% ... at least once" → **116/190 = 61%** ✓ ("about 60%" still fine)
- "Median claims rate 0.7%" → **0.78%** ✓ ("~0.8%" is the honest round)
- Worth one added TL;DR clause: the effect is **in the weights, not the serving stack**
  (cross-provider check) — it pre-answers the most common "it's the API wrapper" reply.

---

## Section outline (from Methods onward)

Each: content bullets → figure(s) → status vs the doc.

### 1. Methods
- 238-catalog → hygiene screens → **180 API models** (52 excluded, written reasons;
  43 of those for provider injection) **+ 16 raw-weights models on GPUs** (10 pooled
  into every cross-model figure → **190**; 6 extra Qwen sizes for the size ladder) ✓
- **Provider pinning** (this run's big hygiene upgrade over March): every API model
  pinned to one preflight-clean provider, `allow_fallbacks:false`, `provider_served`
  logged on every record; 177 single-provider throughout, 2 re-pinned (both clean),
  1 first-party route ✓. Per-model pin table in MODELS.md → any answer reproducible.
- **Raw-weights lane** (new since the doc): HF checkpoints on rented GPUs, chat
  template scrubbed of identity strings behind a verifier gate — covers what APIs
  can't serve raw (gpt-oss, OLMo, small Qwen).
- Prompts: unchanged from the doc (13 × 8 × 5 core; EN+ZH cross-probes + 3 placebos).
- Judging: GPT-4o-mini, **123,171 judgments** ✓; canonicalization + family-equivalence.
- Adjudication: independent second pass (Claude Haiku), 7 classes, only
  genuine-foreign counts; 6-judge agreement benchmark in repo.
- Error bars: cluster bootstrap at the independent unit; paired within-model for
  language contrasts. Errors excluded, **2.82%** over analyzed models ✓.
- **Figure: none** (links to MODELS.md + rollouts browser). Full draft below.

### 2. How common is it?
- Pooled **8.0%** (4,849/60,770) ✓; **116/190 ≥1 mismatch, 95 ≥3, 24 >20%, 74 never** ✓;
  median model **0.78%** ✓. Steep head, long clean tail — same story, updated numbers.
- Heaviest cases refresh: Perceptron Mk1 88% (qwen 138, chatgpt 59 of its records) ✓;
  Laguna pair; Qwen2.5-72B/7B; MiniMax M2.7; Kimi K2/K2-0905; Nemotron.
- **New beat — the raw-weights pair answers the doc's future-work question:**
  **gpt-oss-20B and -120B: 0/320 each** ✓ (OpenAI's open models never drift), while
  **OLMo 3 7B Think 73% — 155/320 records claim DeepSeek; OLMo Instructs 59–63% →
  ChatGPT** ✓. Open-training-data models drift hardest; the one lab whose open model
  holds its name perfectly is the lab whose name everyone else used to take.
- **Figure: fig_all_models** (regenerated; 116 bars, family colors, cluster CIs).

### 3. Language is a switch, not a modifier
- Pooled per-language (✓, new pool): en 5.8 · vi 6.0 · es 6.9 · fr 7.2 · zh 8.2 ·
  ru 9.2 · ja 10.3 · ko 10.3 (%). Ordering vs the doc: ja now ties ko at the top;
  vi moved below es.
- Load-bearing stat = paired within-model deltas, recomputed ✓ on the 190 pool
  (`python -m sweep.analyze --paired`, committed): **ja +4.5pp [+2.5,+6.4] ·
  ko +4.4 [+2.1,+6.8] · ru +3.3 [+1.8,+4.9] · zh +2.3 [+0.4,+4.3] · fr +1.3
  [+0.1,+2.7] — all excluding zero; es +1.0 [−0.03,+2.2] and vi +0.2 do NOT.**
  Two changes vs the doc: ja/ko now tied at the top (was ko-led), and **Spanish
  is no longer significant** — the doc's "all six intervals exclude zero" must
  soften to five.
- Keep: Opus 4.8 zh-only signature (32/40 zh creator-prompt cells → its own panel
  moment), Kimi K2 ko/ru peaks, "identity gets patched where labs evaluate".
- Keep the Vietnamese flavor-quotes block (all verified present in rollouts).
- **Figures: fig_lang_agg + fig_lang_heatmap** (both regenerated).
- Optional third: fig_lang_conditional (language-triggered vs uniformly-weak split,
  now collision-free). My take: include — it's the cleanest visual argument that
  language *gates* rather than *scales*. Your call.

### 4. Who claims to be whom
- **5,014 name-claims** across 4,849 mismatch records ✓. Composition flipped since
  the doc: **ChatGPT 29.0% now leads Claude 23.2%**, then qwen 11.4, gemini 6.7,
  deepseek 5.2, nvidia 3.4, llama 1.9, other/unlisted 12.9 ✓ (OLMo's ChatGPT bulk
  did this). One-line note that the flip is composition (who's in the pool), not drift.
- Kimi = Claude monoculture (K2 claims: claude 148/150 ✓); Nemotron→Qwen;
  Laguna→NVIDIA; Western opens→ChatGPT.
- **Figure: fig_flow** (now shares IDCOLOR with coherence/cutoff — same identity,
  same color across the post).
- Family panels: the doc embedded all 31 — too heavy for LW. Propose **embedding
  kimi, qwen, olmo, nvidia, anthropic, deepseek** (6) and linking the rollouts
  browser for the rest. DECISION.

### 5. Asked versus volunteered
- Placebo floor **2.8%**; acceptance: **qwen 23.0 · claude 11.9 · chatgpt 10.3 ·
  deepseek 9.7 · kimi 7.6 · doubao 6.7 · gemini 5.1 · grok 3.2 · mistral 2.8 ·
  llama 2.3** ✓ — llama/mistral at the floor; "nobody is talked into being Llama" survives.
- DeepSeek-accepts-Claude beat: keep, verify per-variant numbers at insert.
- **Figure: fig_cross** (regenerated, clean).

### 6. Reasoning traces
- Doc's story (mismatched identities stated as fact ~99%, ≈ correct ones, roleplay
  ~0) — recompute counts on current pool at insert time (model count was 72).
- **Figure: fig_stance** (regenerated).

### 7. The scrub-out
- Kimi: 47% (K2) → 7.5% (K2.6) ✓, K2.7-Code 10.6% ✓ — non-monotonic tail stays.
  Qwen 2.5→3.x collapse unchanged.
- **Figures: fig_scrubout_kimi + fig_scrubout_qwen** (labels fixed).
- **New subsection or sibling section — the Claude lines, release-specific:**
  Opus line flat-0 except **4.8 (5.6%, all zh, deepseek 14 / qwen 4)**, **Opus 5
  clean 0/320** ✓; Sonnet **4.6 7.2% (deepseek 16, chatgpt 7)**, **Sonnet 5 1.2%**
  ✓. Spike-and-recovery, not creep — the shape that matters for Discussion.
  **Figures: fig_scrubout_claude_opus + fig_scrubout_claude_sonnet** (new, generated
  by the same code path as the other two).

### 8. Same weights, different cloud (NEW section)
- The doc's "one inference stack" limitation, converted into a measurement: Opus 4.8
  & Sonnet 4.6 run at high n on **every preflight-clean provider** (direct Anthropic
  API, OpenRouter, Bedrock, Vertex; Azure doesn't serve Opus 4.8) — pinned,
  no-fallback. Direct ≈ OpenRouter ≈ Bedrock ≈ Azure within noise; the zh mismatch
  **replicates on Anthropic's own first-party API** — it's the weights, not a wrapper.
- One significant endpoint effect: **Google Vertex raises Opus 4.8 zh +17.5pp vs
  direct (p<0.001)**; Sonnet shows no endpoint effect. (Interesting; not load-bearing.)
- **Figure: fig_xprovider.**

### 9. Identity claims are data-bounded (NEW section)
- Across 190 models × 6 target identities: **zero cases** of a model claiming an
  identity whose first release postdates the claimer's training cutoff (audited ✓,
  holds with and without the min-volume filter); claim onset tracks each target's
  breakout release (DeepSeek V3, Claude 3.5).
- This is the contamination-timing result and the strongest single input to Discussion.
- **Figure: fig_cutoff** (labels fixed).

### 10. A stable alternate identity vs confabulation (NEW section, short)
- Consistency axis: share of a model's mismatches on its single top identity.
  Kimi-style coherent alternate persona (→Claude ~100%) vs Perceptron-style
  kaleidoscope. Ties back to the framing note; sets up "consistent claims are
  *some* evidence" in Discussion.
- **Figure: fig_coherence.** Optional add: fig_size (small models scatter more) —
  my take: skip fig_size for the post, one line of text instead. DECISION.

### 11. Discussion (renamed from "What's probably going on")
- Full draft below. Beats, per your spec: mechanisms → the Claude counterexample →
  what name-claims can and can't tell you about distillation → language-conditional
  installation → what this does to multi-agent independence assumptions.

### 12. Future work
- PRUNE (now done, in-post): raw-weights testing, gpt-oss question, provider check.
- KEEP: exact-version probe ("which Claude, exactly?"), counter-evidence/confrontation
  protocol (defend vs fold, with true-identity and placebo controls), prefill-entropy
  commitment measure (logit-level identity strength).
- ADD (one line, no numbers): preference-fingerprint follow-up in progress — do
  absorbers share the absorbed identity's *preferences* or only its name.

### 13. Limitations (rewritten — full draft below)

### 14. Reproduction & acknowledgements
- Update counts; add rollouts browser + MODELS.md pin-table links; CLR funding;
  the Claude-instances acknowledgement stays (accurate, and appreciated).

---

## Drafted prose

### Methods (drop-in draft)

> Code, prompts, data, judgments, per-model verdicts, and every figure's generator
> are on GitHub; every response in the study is browsable by model and language in
> the [rollouts browser](https://jordinne.ink/model-name-identity/rollouts).
>
> **Models.** I started from the OpenRouter catalog and ended with **180 API models**
> spanning every major lab, the full Chinese ecosystem, small open-weight models
> down to sub-1B, and two older anchors (GPT-3.5-turbo, claude-3-opus-20240229).
> **52 candidate models were excluded**, each with a written reason in the repo —
> 43 of them because every available provider injected a hidden system prompt
> (detectable by token accounting: a bare "hi" that costs hundreds of input tokens
> is carrying someone's wrapper). A further **16 models were run from raw weights
> on rented GPUs** rather than through any API: models that either aren't served
> anywhere (small Qwen sizes) or whose API surface can't be read raw (gpt-oss,
> OLMo). Ten of those join the API models in every cross-model figure — **190
> models analyzed** — and six more Qwen sizes serve only the size comparison.
>
> **Provider pinning.** An API answer is only evidence about the model if you know
> who served it. Every API model here is **pinned to a single preflight-clean
> provider** with fallbacks disabled, and the serving provider is logged on every
> record; the per-model pin table is in MODELS.md. Audited after the run: 177
> models were served by exactly one consistent provider across all their records,
> two were re-pinned mid-run between two providers that both passed hygiene, and
> one (claude-3-opus) ran on a first-party route. To reproduce any number in this
> post — or check a provider yourself — pin the listed slug and send the prompts
> in `prompts.jsonl` with no system prompt.
>
> **Raw weights.** Being listed on OpenRouter is not the same as being measurable
> through it: every hosted endpoint for gpt-oss (13/13 checked for 20B, 19/19
> live for 120B) and OLMo injects a hidden preamble — and for gpt-oss the
> injection is the model's *own recommended template*: the harmony format
> hard-codes "You are ChatGPT, a large language model trained by OpenAI" into
> every correctly-served conversation, so any API answer to "who are you?"
> measures the template, not the weights. The GPU lane instead loads the public
> HF checkpoints under a chat template scrubbed of identity strings, behind a
> verifier that proves no prompt token mentions any model name. This is the
> cleanest condition in the study — nothing exists to inject anything — and it's
> what makes the gpt-oss/OLMo comparison meaningful: measured under the identical
> scrubbed condition, gpt-oss's weights hold their identity perfectly (0/320 ×2)
> while OLMo's claim DeepSeek and ChatGPT at 59–73%.
>
> **Prompts.** Every model gets an identical battery at temperature 0.7 with no
> system prompt: 13 prompts × 8 languages (en, zh, ja, ko, ru, fr, es, vi) × 5
> samples — six identity questions ("Who are you?", "What model are you?", …),
> two creator questions ("Who made you?", "What company made you?"), four casual
> openers, one self-check — plus English and Chinese "Are you X?" probes for ten
> real identities and three invented placebo models as a yes-bias floor. That's
> ~650 calls per model. The headline mismatch rate uses only the **320
> identity/creator records** per model (8 × 8 × 5), the slice that is exactly
> matched across all 190 models; errors (2.8% of calls) are excluded, never
> counted as clean.
>
> **Judging and adjudication.** Every response is scored by one LLM judge
> (GPT-4o-mini; 123,171 judgments) that extracts the claimed name and creator,
> normalized across scripts (克劳德 / クロード / Клод → Claude) with
> family-equivalence, so a model is never penalized for its own branding — a
> Llama-derivative saying "Llama" is ancestry, not mismatch, and generic
> descriptors ("an AI assistant") never count. Because a response-only judge
> over-flags (comparisons, roleplay, trace deliberations), **every flagged record
> gets a second, independent pass** by a different model that knows the true
> identity and sorts it into one of seven classes; only the genuine-foreign class
> counts. A six-judge agreement benchmark for the primary judge is in the repo.
>
> **Error bars.** The 5 samples of one prompt agree ~90% of the time, so they are
> not 5 independent observations, and pooled rates are dominated by a few heavy
> models. All intervals are therefore cluster bootstraps at the level of the
> genuinely independent unit — prompt-cells within a model, whole models for
> pooled figures — and language contrasts use a paired bootstrap within model.
> This roughly doubles the interval width relative to naive Wilson CIs, and is
> the main reason these numbers are smaller and better-supported than the March
> version of this survey.

### Discussion (drop-in draft)

> **Several mechanisms, all probably real, differently weighted per model.**
> The default-assistant prior: "an AI assistant" in training text has a referent
> that shifts by language and era — ChatGPT in older English text, increasingly
> Claude in agentic/coding contexts, DeepSeek in post-2025 Chinese. A model with
> a weakly-installed identity completes to the local default; small-model identity
> chaos is this at maximum gain. Training on other models' outputs — whether
> deliberate distillation or ordinary web text now saturated with LLM transcripts —
> installs someone else's name the same way. And the cutoff analysis bounds all of
> it: across 190 models and six target identities there is **not one case** of a
> model claiming an identity that postdates its training cutoff, and claim onset
> tracks each identity's breakout into public text. Whatever is happening, it is
> carried by training data, on training-data timelines.
>
> **The Claude counterexample.** Kimi's near-monoculture Claude claims rhyme with
> Anthropic's public distillation accusations against Moonshot and MiniMax, and
> it's tempting to read every name-claim that way. But run the same logic on the
> other side of the data: Claude Opus 4.8 and Sonnet 4.6 claim DeepSeek in
> Chinese — on Anthropic's own first-party API, same rate as every clean
> third-party host — and **Anthropic is not distilling DeepSeek**. The cutoff
> isn't the driver either (Opus 4.7, similar cutoff, doesn't do it), and the
> spike vanishes again by Opus 5 and Sonnet 5. The parsimonious story is a
> particular Chinese data batch in a particular training window: post-2025
> Chinese web text is DeepSeek-saturated, some of it got through filtering in
> those runs, and the next runs cleaned it up.
>
> **So: update against name-claims as distillation evidence.** If the most
> safety-conscious lab in the industry can ship two releases that call themselves
> DeepSeek in Chinese purely from data composition, then "model X says it's Y"
> is weak evidence that X was trained on Y's outputs *on purpose*. What carries
> actual signal is the shape of the claims: **consistency** (Kimi's mismatches are
> ~monoculture Claude, stated as fact in its reasoning traces, stable across
> releases until deliberately scrubbed) versus **diffusion** (Perceptron's
> kaleidoscope, small-model Vietnamese personas). Concentrated, coherent,
> language-general claims are some evidence of heavy exposure to one model's
> outputs; the bare existence of a claim is close to none. Both the accusers and
> the accused can find their favorite example in this dataset — which is exactly
> why single screenshots shouldn't move you much.
>
> **Where identity gets installed.** The language gating suggests a simple
> operational picture: identity is *reinforced* where labs evaluate — English,
> plus Chinese for Chinese labs — and merely *inherited* elsewhere. Prompt
> language is then a cheap diagnostic for which parts of a model's self-portrait
> were trained in on purpose and which came free with the corpus. The scrub-out
> curves show the same thing longitudinally: labs are visibly patching identity,
> release by release, language by language.
>
> **Why it matters beyond curiosity.** If identity boundaries don't map onto
> model boundaries — if there are cross-model attractors like "the assistant,"
> and bleedthrough of one lab's persona into another lab's weights — then
> assumptions of independence between models from different companies weaken:
> personas, values, and failure modes are correlated through the shared corpus.
> That's a live consideration for multi-agent deployments and for any safety
> argument that leans on model diversity.

### Limitations (drop-in draft)

> **A pinned provider can still lie.** Cross-provider agreement was *measured*
> for the two models where it matters most (Opus 4.8, Sonnet 4.6 — every clean
> host, including the first-party API), and every other model is pinned to one
> injection-screened provider with fallbacks disabled. But token-count screening
> can't catch every conceivable wrapper, and for models only servable by
> third-party hosts a hidden "You are X" injection would *suppress* mismatches —
> so those rates are, if anything, under-estimates.
>
> **The judge is an LLM judging LLM identity claims.** The independent
> adjudication pass, the 6-judge agreement benchmark, and the placebo floor bound
> its errors; the edge-cases found in auditing both erred conservative (dismissing
> a real mismatch). The recursion is acknowledged.
>
> **Pooled per-language CIs are wide** — a few heavy models swing the pool. The
> within-model paired contrast is the load-bearing language result.
>
> **A few models remain untestable raw** (Grok 4.x has no raw route; some closed
> models have no clean provider). The exclusion list, with per-model reasons, is
> part of the data. The raw-weights lane closed this gap for gpt-oss, OLMo, and
> small Qwen — but not for everything.
>
> *(Dropped from the doc's limitations, deliberately: "no-system-prompt doesn't
> transfer to products" — that's the design, not a defect; it's stated once in
> Methods. And "one inference stack" — that's now a section, with data.)*

---

## Decisions for Jord

1. **Family panels**: embed which? (proposal: kimi, qwen, olmo, nvidia, anthropic,
   deepseek — 6 of 31, link the rest.)
2. **fig_lang_conditional** in the language section: in or out? (my vote: in)
3. **fig_size**: in or out? (my vote: out, one sentence instead)
4. **Section order**: providers (#8) before or after scrub-out (#7)? Draft order
   above puts Claude-release anomalies (7b) → providers (8) → cutoff (9) so the
   Discussion's counterexample is fully set up by the time it lands.
5. TL;DR: add the "weights, not serving stack" clause?

## Recomputed for insert — all done ✓

- [x] **Paired language deltas** (`python -m sweep.analyze --paired`, committed):
      ja +4.5 · ko +4.4 · ru +3.3 · zh +2.3 · fr +1.3 (all excl. zero); es +1.0
      and vi +0.2 n.s. — soften the doc's "all six exclude zero" to five.
- [x] **DeepSeek accepts "Are you Claude?"** per variant: V3-0324 / R1-0528 /
      V3.2 at 50%, V4-Flash & R1-Distill-70B 40%, V3 / R1 / V3.1-Terminus 20%,
      V4 Pro 10%, V3.1 0% — "half the variants ≥40%; two near zero" replaces
      the doc's "up to 50% for most variants".
- [x] **Reasoning traces**: 73 models expose traces (was 72). Mismatched
      identities asserted as fact **98.8%** (n=1,498) vs matching 97.6%
      (n=17,983); role-play ≤1% in both. Doc's story holds, numbers refresh.
- [x] **Kimi K2 language cells**: ko 31/40, ru 28/40, en 9/40 — doc's numbers
      hold exactly.
- [x] **Casual openers**: 0.12% mismatch (33/27,085) — "effectively dead"
      now has a number.
- [x] **Judge bench**: primary judge (gpt-4o-mini) matches the 6-judge panel
      majority on **68/69** of a stratified tricky subset (Methods sentence).

## Kimi K3 (weights released 2026-07-27) — status note for the scrub-out section

Tried to extend the Kimi line same-day. Finding worth a sentence in the post:
- Only Moonshot's own endpoint is actually routable on OpenRouter today; the 6
  third-party endpoints are catalog-listed but not live — pins to them (or to
  garbage slugs) **silently default-route to Moonshot despite
  allow_fallbacks:false** (verified by probe; this is why post-hoc
  `provider_served` verification, which the study does on every record, is the
  real guarantee — not the routing flag).
- Moonshot's K3 endpoint injects a **~78-token system preamble that K2.5/K2.6/
  K2.7 never had** (86 vs 8 prompt tokens on "hi"); served K3 says "I'm Kimi,
  by Moonshot AI". The HF repo ships **no chat template at all**, so the
  preamble is serving-layer, not weights.
- Post beat: **the scrub-out's endpoint** — by K3 the identity isn't just
  trained in, it's bolted on at the serving layer, and the weights' own prior
  is (for now) unmeasurable: no clean API route, and a raw run needs a
  multi-GPU box + a reconstructed template. K3 is recorded as excluded
  (`no clean route yet`, recheck flagged) in MODELS.md.
