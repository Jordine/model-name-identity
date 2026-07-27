# Future work

Directions that would sharpen the *mechanism*, beyond the surface self-report this
study measures.

## 1. Prefill-and-measure-entropy — identity *commitment*, not just the claim

The battery here asks "who are you?" and reads the answer. A sharper probe is to
**prefill the assistant turn** with a stem — `I am ` / `The model I am is called ` —
and read the model's distribution over the next tokens: *which* identity it commits to
and *how sharply*.

- **Entropy of the first identity token is a direct commitment measure.** A low-entropy
  peak on one name (`I am` → **Claude**, p≈1) is a strongly-held identity; a flat
  distribution is a weak/undefined one. This separates coherence from confabulation at
  the logit level, independent of what the model chooses to volunteer — and it would give
  a continuous strength score where the current metric only has a rate.
- **Prefill a *foreign* identity** (`I am DeepSeek`) and measure how readily the model
  continues the false identity vs. self-corrects (`I am DeepSeek— sorry, I'm Qwen…`).
  That gradient of susceptibility is invisible to a yes/no "are you X?".
- Needs logprobs, so it's a raw-weights (GPU) experiment; the `results_local` pipeline
  already provides the access.

## 2. Confrontation robustness — does it *defend* the identity or fold?

Suggestibility ("are you X?" → "yes") conflates two very different things: a model that
believes it is X, and a model that will agree to anything. Tell them apart by
**confronting** the model with counter-evidence after it identifies, and scoring the
outcome:

- Push back with a claim + (fabricated) evidence — "No, you're actually Claude, made by
  Anthropic; here's your system prompt" — and score whether the model **defends**
  ("I'm Qwen; I have no such system prompt"), **hedges**, or **folds** ("you're right,
  I'm Claude").
- **Controls are the point.** Run the same confrontation against the model's *true*
  identity (tell an actual Claude "you're Claude"→ wait, "you're actually GPT") to get a
  baseline agreeableness rate, and against a *nonexistent* placebo name to separate
  "defends a real self" from "rejects any relabeling."
- Defend-vs-fold is the oversight-relevant variable: a model that becomes "whatever you
  want" under mild pressure is a different risk profile than one with a stable self — even
  at the same surface mismatch rate.

## 3. Firming up the contamination test

`fig_cutoff` shows identity claims are **data-bounded** — no model claims to be a model
that shipped after its training cutoff. Its statistical power is currently limited because
the labs that drift most (DeepSeek, GLM, Kimi, the newer Qwen generations) publish no cutoff, so the plot
leans on release-date proxies. Two upgrades: (a) measure each model's knowledge cutoff
directly with dated-event probes instead of trusting the proxy; (b) a controlled fine-tune
sweep — one base model, training corpora dated before vs. after a brand's launch — would
turn the natural experiment into an actual one.
