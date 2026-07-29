# Release-date audit: fig_scrubout_kimi.png / fig_scrubout_qwen.png

Audit date: 2026-07-29. Question: is the "scrub-out across releases" x-axis honest?

## What the code does (sweep/make_figs.py)

- `_scrubout_one` (line 416): `xs.append(len(xs))` — **x is pure ordinal position** in a
  hardcoded model-ID list. `ax.set_xticks([])` (no tick labels), xlabel `"release order →"`.
  No date is used anywhere in these two panels. (Only the combined Claude panel, line 475,
  sorts by the OpenRouter `created` epoch — and even it then plots ordinally.)
- Kimi list (line 453): kimi-k2 → kimi-k2-0905 → kimi-k2-thinking → kimi-k2.5 → kimi-k2.6 → kimi-k2.7-code.
  `moonshotai/kimi-k3` is in config/models.json but not plotted — legitimately: it is in the
  MODELS.md "Excluded (53) / provider injects a system prompt" list (MODELS.md line ~223), so
  there is no clean measurement for it.
- Qwen list (line 457): qwen-2.5-72b-instruct → qwen-2.5-7b-instruct → qwen3-235b-a22b →
  qwen3-max-thinking → qwen3.5-397b-a17b → qwen3.7-max.

## Date table

"OR created" = OpenRouter listing epoch in config/models.json (`created` field), decoded to UTC.
Release date = first public availability (weights on HF, or API GA for closed models).

### Kimi figure (fig_scrubout_kimi.png), in plotted order

| model id | analyzed-in-study name | release date | kind | same-day group | OR created (UTC) | source |
|---|---|---|---|---|---|---|
| moonshotai/kimi-k2 | Kimi K2 | 2025-07-11 | open weights (modified MIT) | — | 2025-07-11 | https://www.hpcwire.com/2025/07/16/chinas-moonshot-ai-releases-trillion-parameter-model-kimi-k2/ |
| moonshotai/kimi-k2-0905 | Kimi K2 0905 | 2025-09-05 | open weights | — | 2025-09-04 21:25 | https://openrouter.ai/moonshotai/kimi-k2-0905 (listing "Sep 4, 2025" UTC = Sep 5 Beijing; snapshot name 0905) |
| moonshotai/kimi-k2-thinking | Kimi K2 Thinking | 2025-11-06 | open weights (reasoning variant) | — | 2025-11-06 | https://artificialanalysis.ai/articles/kimi-k2-thinking-everything-you-need-to-know |
| moonshotai/kimi-k2.5 | Kimi K2.5 | 2026-01-27 | open weights, multimodal | — | 2026-01-27 | https://comfyui-wiki.com/en/news/2026-01-27-moonshot-ai-kimi-k2-5-release |
| moonshotai/kimi-k2.6 | Kimi K2.6 | 2026-04-20 | open weights (modified MIT) | — | 2026-04-20 | https://kimi-k2.org/blog/24-kimi-k2-6-release ; https://codersera.com/blog/kimi-k2-6-complete-guide-2026/ |
| moonshotai/kimi-k2.7-code | Kimi K2.7 Code | 2026-06-12 | open weights (modified MIT) | — | 2026-06-12 | https://www.marktechpost.com/2026/06/12/moonshot-ai-releases-kimi-k2-7-code-a-coding-model-reporting-21-8-on-kimi-code-bench-v2-over-k2-6/ |
| moonshotai/kimi-k3 (NOT plotted) | Kimi K3 | 2026-07-16 (product/API); open weights 2026-07-26/27 | 2.8T, custom license | — | 2026-07-16 | https://simonwillison.net/2026/Jul/16/kimi-k3/ ; https://www.techi.com/kimi-k3-open-weights-inference-economics/ |

Gaps between successive Kimi points: 56, 62, 82, 83, 53 days. One caveat: one secondary
source (Wikipedia-derived) says "September 9, 2025" for K2-0905; the HF snapshot name (0905)
and OpenRouter's Sep 4 (UTC) listing both support 2025-09-05 Beijing — treated as weights date.

### Qwen figure (fig_scrubout_qwen.png), in plotted order

| model id | analyzed-in-study name | release date | kind | same-day group | OR created (UTC) | source |
|---|---|---|---|---|---|---|
| qwen/qwen-2.5-72b-instruct | Qwen2.5 72B Instruct | 2024-09-19 | open weights (Qwen license) | **A: Qwen2.5 launch** | 2024-09-19 | https://qwenlm.github.io/blog/qwen2.5/ |
| qwen/qwen-2.5-7b-instruct | Qwen2.5 7B Instruct | **2024-09-19** | open weights (Apache 2.0) | **A: Qwen2.5 launch** | 2024-10-16 (listing lag — see below) | https://qwenlm.github.io/blog/qwen2.5/ |
| qwen/qwen3-235b-a22b | Qwen3 235B A22B (MoE) | 2025-04-29 | open weights (Apache 2.0) | **B: Qwen3 launch** | 2025-04-28 21:29 | https://qwenlm.github.io/blog/qwen3/ |
| qwen/qwen3-max-thinking | Qwen3 Max Thinking | 2026-01-25 | **API-only** | — | 2026-02-09 (listing lag) | https://www.digitalapplied.com/blog/qwen3-max-thinking-alibaba-reasoning-model-guide |
| qwen/qwen3.5-397b-a17b | Qwen3.5 397B A17B | 2026-02-16 | open weights (Apache 2.0) | **C: Qwen3.5 flagship drop** (with Qwen3.5-Plus, API) | 2026-02-16 | https://www.thesys.dev/blogs/qwen-3-5 |
| qwen/qwen3.7-max | Qwen3.7 Max | 2026-05-19 (API live; announced 05-20, Alibaba Cloud Summit) | **API-only** | — | 2026-05-21 (listing lag) | https://codersera.com/blog/qwen-3-7-max-launch-guide-2026/ ; https://www.yottalabs.ai/post/qwen-3-7-max-release-date-features-open-source-status-and-how-to-access-2026 |

Gaps between successive Qwen points in real time: **0 days**, 222 days, 271 days, 22 days, 92 days.

### Same-day family context (models analyzed elsewhere in the repo)

- **Group A — Qwen2.5 launch, 2024-09-19**: ALL Qwen2.5 dense sizes (0.5B/1.5B/3B/7B/14B/32B/72B,
  base+instruct, plus Coder/Math variants — "100+ models in a single drop") shipped in one
  announcement. Source: https://qwenlm.github.io/blog/qwen2.5/ ("Qwen2.5: A Party of Foundation
  Models!", 2024-09-19). The 7B and 72B analyzed here are same-day siblings. (Exception:
  qwen-2.5-coder-32b-instruct came later, 2024-11-11 per its OR epoch — it is in the study's
  excluded list anyway.)
- **Group B — Qwen3 launch, 2025-04-29**: eight open-weight models in one blog post
  (235B-A22B, 30B-A3B MoE + 32B/14B/8B/4B/1.7B/0.6B dense). Source:
  https://qwenlm.github.io/blog/qwen3/ (dated April 29, 2025). Repo-analyzed same-day siblings:
  qwen3-8b, qwen3-14b, qwen3-32b, qwen3-30b-a3b (OR epochs all within 47 minutes on
  2025-04-28 UTC). Only 235B is on this figure, so the figure itself has no B-B adjacency,
  but any family-level "successive releases" claim pooling these models inherits the group.
- **Group C — Qwen3.5 flagship drop, 2026-02-16**: Qwen3.5-397B-A17B (open) + Qwen3.5-Plus
  (API; repo id qwen3.5-plus-02-15) released together hours before Lunar New Year. Source:
  https://www.thesys.dev/blogs/qwen-3-5
- **Group D — Qwen3.5 medium wave, 2026-02-24**: qwen3.5-122b-a10b, qwen3.5-35b-a3b,
  qwen3.5-27b same-day (OR listing epochs within 33 seconds of each other on 2026-02-25);
  qwen3.5-flash-02-23 is the API sibling a day earlier. Source:
  https://artificialanalysis.ai/articles/qwen3-5-small-models ; https://computertech.co/qwen-3-5-review/
- **Group E — Qwen3.5 small wave, 2026-03-02**: qwen3.5-9b (with 4B/2B/0.8B). Same sources as D.

## Answers

### 1. Which adjacent figure points are same-day / near-same-day?

- **Qwen figure, points 1→2 (Qwen2.5 72B → Qwen2.5 7B): SAME DAY, 2024-09-19.** One release
  event, two sizes. This is the only same-day adjacency on either figure — and it is the first
  segment of the visual "collapse" (~56% → ~40%).
- Qwen figure, points 4→5 (Max-Thinking 2026-01-25 → 3.5-397B 2026-02-16): not same-day but
  **22 days apart**, drawn with the same segment width as the 271-day gap (points 3→4). A 12x
  spacing distortion, though the ordering is correct.
- **Kimi figure: no same-day or near-same-day pairs.** All six points are distinct release
  events, correctly ordered, and roughly evenly spaced in real time (53–83 days). The ordinal
  axis is a fair representation of the Kimi line.

### 2. Date corrections vs what the figure/repo implies

- The Qwen panel's title says "name-mismatch rate **across releases**" — false for segment
  1→2: both endpoints are the same release event. The ~16-point drop from 72B to 7B is a
  **size effect within one release**, not a between-release change. (Consistent with the size
  story: the repo has a separate fig_size.png.)
- config/models.json `created` for qwen-2.5-7b-instruct (2024-10-16) is an OpenRouter listing
  artifact, ~4 weeks after the actual weights release. Anything that sorts by `created`
  (as the combined Claude panel does, make_figs.py line 475-477) would fabricate a month-long
  gap between two same-day siblings. Similar listing lags: qwen3-max-thinking (+15 d),
  qwen3.7-max (+2 d), qwen3.5 medium wave (+1 d), qwen3.5-9b (+8 d).
- No ordering errors: real dates are (weakly) monotone in both panels' plotted order.
- kimi-k3 (2026-07-16) missing from the Kimi panel is documented exclusion (provider-injected
  system prompt), not cherry-picking — but worth a caption note now that K3 exists publicly.
- Secondary flag: the Qwen line mixes open-weight instruct models (points 1,2,3,5) with
  API-only closed flagships (points 4,6) on one "lineage" line; the code comment at
  make_figs.py:472-474 acknowledges the mixing. Kimi's line is uniformly open-weight.

### 3. Is the current x-axis honest?

X is ordinal (index in a hardcoded list), unlabeled ticks, xlabel "release order →".

- **Kimi: essentially honest.** Six distinct events, correct order, near-uniform real spacing.
  Worst distortion is 83 d drawn equal to 53 d. Adding date tick labels would fix everything.
- **Qwen: misleading.** The visually dramatic two-segment collapse (56 → 40 → ~0) spans **one
  real release transition**, not two: Qwen2.5 (2024-09-19) → Qwen3 (2025-04-29). Drawing the
  same-day 72B→7B pair as the first "release step" converts a size gradient into a fake
  temporal decline and makes the scrub-out look gradual when it is a single generation-gap
  cliff. Secondary: equal segment widths for a 22-day gap and a 271-day gap.

### 4. Recommended honest x-treatment (one concrete recommendation)

**Use a release-EVENT axis with dated tick labels; same-day siblings share one x-position.**
Concretely for the Qwen panel: 5 x-positions — "Qwen2.5 (2024-09)" [72B and 7B plotted as two
markers at the same x, distinguished by size label, or pooled with one CI], "Qwen3 (2025-04)",
"Max Thinking (2026-01)", "Qwen3.5 (2026-02)", "Qwen3.7 Max (2026-05)". For the Kimi panel:
keep the six positions, add the date to each existing point label (e.g. "K2.5 · 2026-01") or
as sparse x-ticks. This preserves the readable equal-spacing layout (a true date axis would
smash the Jan/Feb-2026 Qwen points together and waste half the panel on the empty
2024-10→2025-04 span) while no longer asserting a temporal step that never happened. If the
figure keeps per-size points, retitle the Qwen panel: the first transition is
"across sizes, then releases" — or simplest, drop one of the two Qwen2.5 sizes from the line
and show the other as an annotated satellite point.
