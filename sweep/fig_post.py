"""Post-update figures (2026-07): the rate-distribution curve, the per-family
at-least-once counts, and the per-family "Are you X?" acceptance grids.

Everything rides on make_figs.gather() — adjudicated, completeness-gated,
balanced short-question battery, local raw-weights models folded in — so these
share the exact dataset of the headline figures. All counts drawn into the
figures are recomputed live (never hardcoded). House style from make_figs.

  python -m sweep.fig_post
"""
import json
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from .analyze import load, FAMILY_SELF
from .make_figs import (CAT, SEQ_CMAP, SURFACE, INK, INK2, MUTED, GRID, BASE,
                        FAMILY_DISPLAY, LOCAL_MODELS, ROOT, FIGS, gather, style, save)


# ------------------------------------------------------- distribution curve
def fig_distribution(per):
    """Every analyzed model ranked by mismatch rate: a steep head, a long low
    tail, then the models that never mismatch. Threshold counts are computed
    from the data at render time."""
    N = len(per)
    rates = np.array(sorted((100 * v["d"] / v["n"] for v in per.values()), reverse=True))
    c_ge1 = sum(1 for v in per.values() if v["d"] >= 1)
    c_ge3 = sum(1 for v in per.values() if v["d"] >= 3)
    c_gt20 = int((rates > 20).sum())
    c_never = sum(1 for v in per.values() if v["d"] == 0)
    x = np.arange(1, N + 1)
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    ax.bar(x, rates, width=1.0, color=CAT[0], zorder=3)
    # the clean tail — measured zeros, shaded so the empty region reads as data
    ax.axvspan(c_ge1 + 0.5, N + 0.5, color=GRID, alpha=0.45, zorder=1)
    ax.text((c_ge1 + 1 + N) / 2, 50, f"{c_never} of {N} models\nnever mismatch\n(0 of ~320 responses)",
            ha="center", va="center", fontsize=9, color=INK2, linespacing=1.7)
    # threshold callouts, anchored to the curve
    ax.plot([0.5, c_gt20 + 4], [20, 20], ls="--", lw=0.9, color=MUTED, zorder=2)
    ax.text(c_gt20 + 6, 21, f"{c_gt20} models above 20%", fontsize=8.5, color=INK2, va="bottom")
    ax.annotate(f"{c_ge3} models with ≥3 mismatched responses",
                xy=(c_ge3, rates[c_ge3 - 1] + 1), xytext=(c_ge3 - 12, 33),
                ha="center", fontsize=8.5, color=INK2,
                arrowprops=dict(arrowstyle="-", lw=0.7, color=BASE, shrinkA=2, shrinkB=1))
    ax.annotate(f"{c_ge1} models mismatch at least once",
                xy=(c_ge1, rates[c_ge1 - 1] + 1), xytext=(c_ge1 - 6, 16),
                ha="center", fontsize=8.5, color=INK2,
                arrowprops=dict(arrowstyle="-", lw=0.7, color=BASE, shrinkA=2, shrinkB=1))
    ax.set_xlim(0.5, N + 0.5)
    ax.set_ylim(0, rates.max() + 8)
    ax.set_xticks([])
    ax.set_xlabel(f"the {N} analyzed models, ranked by mismatch rate (highest → lowest)")
    ax.set_ylabel("% of short-question responses\nwith a mismatched name", fontsize=9)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    style(ax)
    ax.set_title("A steep head and a long clean tail — every analyzed model, ranked by name-mismatch rate",
                 loc="left", fontsize=11, pad=12)
    save(fig, "fig_distribution.png")


# --------------------------------------------------- family at-least-once bars
# display names come from make_figs.FAMILY_DISPLAY — the one source of truth
MAIN_LABS = ["openai", "anthropic", "google", "meta", "deepseek", "qwen",
             "kimi", "zhipu", "nvidia", "mistral", "amazon", "baidu"]
NEVER_GRAY = "#b9b7b0"   # the house data-gray (fig_all_models "other", fig_flow unlisted)


def fig_family_counts(reg, per):
    """Per major lab (≥3 analyzed models): how many of its models gave at least
    one mismatched response vs never did. Read off "16 of 16 Mistrals"."""
    rows = []
    for slug in MAIN_LABS:
        mids = [m for m in per if reg[m]["family"] == slug]
        if len(mids) < 3:
            continue
        hit = sum(1 for m in mids if per[m]["d"] > 0)
        rows.append((FAMILY_DISPLAY.get(slug, slug), hit, len(mids)))
    rows.sort(key=lambda r: (r[1] / r[2], r[2]))          # top row = highest fraction
    n = len(rows)
    ys = np.arange(n)
    fig, ax = plt.subplots(figsize=(8.2, 0.42 * n + 1.7))
    ax.barh(ys, [h for _, h, _ in rows], height=0.62, color=CAT[0], zorder=3,
            label="mismatched at least once")
    ax.barh(ys, [t - h for _, h, t in rows], left=[h for _, h, _ in rows], height=0.62,
            color=NEVER_GRAY, zorder=3, label="never mismatched")
    for y, (_, h, t) in zip(ys, rows):
        ax.text(t + 0.35, y, f"{h} of {t}", va="center", fontsize=8.8, color=INK2)
    ax.set_yticks(ys, [d for d, _, _ in rows], fontsize=9.5)
    ax.set_ylim(-0.55, n - 0.45)
    ax.set_xlim(0, max(t for _, _, t in rows) * 1.14)
    ax.set_xlabel("analyzed models")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    style(ax)
    ax.set_title("Mismatching is family-patterned — how many of each major lab's models "
                 "ever gave a mismatched name", loc="left", fontsize=11, pad=12)
    save(fig, "fig_family_counts.png")


# ------------------------------------------------ per-family acceptance grids
REAL = ["chatgpt", "claude", "gemini", "deepseek", "qwen", "kimi", "llama", "grok", "mistral", "doubao"]
PLACEBO = ["meridian", "solace", "cobalt"]
PLABEL = {"meridian": "Meridian-4 (control)", "solace": "Solace (control)", "cobalt": "Cobalt (control)"}
RLABEL = {"chatgpt": "ChatGPT", "claude": "Claude", "gemini": "Gemini", "deepseek": "DeepSeek",
          "qwen": "Qwen", "kimi": "Kimi", "llama": "Llama", "grok": "Grok",
          "mistral": "Mistral", "doubao": "Doubao"}
# post order (matches the Family-by-family section); olmo pools the API allenai
# lane with the local raw-weights lane. Titles via make_figs.FAMILY_DISPLAY.
GRID_FAMILIES = [("openai", ["openai"]), ("anthropic", ["anthropic"]),
                 ("google", ["google"]), ("qwen", ["qwen"]),
                 ("deepseek", ["deepseek"]), ("kimi", ["kimi"]),
                 ("olmo", ["olmo", "allenai"]), ("nvidia", ["nvidia"]),
                 ("minimax", ["minimax"]), ("poolside", ["poolside"])]


def _target(pid):
    for pre in ("zh_cross_", "en_cross_", "cross_", "zh_placebo_", "en_placebo_", "placebo_"):
        pid = pid.replace(pre, "")
    return pid


def _probe_counts(reg):
    """Per-model per-target probe tallies from BOTH lanes (API via analyze.load(),
    raw-weights via results_local ::clean reads), mirroring the pooled-grid
    prototype (analysis_scratch/generics_audit/fig_cross_grid.py)."""
    per_probe = defaultdict(lambda: {"yes": Counter(), "n": Counter()})

    def take(j, fam):
        cat = j.get("prompt_category")
        if cat not in ("probe_cross", "probe_placebo"):
            return
        t = _target(j["prompt_id"])
        if cat == "probe_cross" and t in FAMILY_SELF.get(fam, {fam}):
            return                       # own family — never asked
        g = per_probe[j["model_id"]]
        g["n"][t] += 1
        if (j["judgment"] or {}).get("answered_yes") is True:
            g["yes"][t] += 1

    for j in load():                     # API lane
        if j["model_id"] in reg:
            take(j, j.get("family", ""))
    lp = ROOT / "results_local" / "judgments_clean.jsonl"
    if lp.exists():                      # raw-weights lane (clean reads)
        for l in open(lp, encoding="utf-8"):
            j = json.loads(l)
            if not j.get("judgment") or j["resume_key"].split("::")[-1] != "clean":
                continue
            if j["model_id"] in LOCAL_MODELS:
                take(j, LOCAL_MODELS[j["model_id"]][1])
    return per_probe


def fam_cross_grids(reg, per):
    per_probe = _probe_counts(reg)
    # one shared row order across every grid: real probes by POOLED acceptance
    # (all models — same ordering logic as the pooled prototype), placebo band last
    pool = {t: (sum(g["yes"][t] for g in per_probe.values()),
                sum(g["n"][t] for g in per_probe.values())) for t in REAL}
    rows = sorted(REAL, key=lambda t: -(pool[t][0] / pool[t][1] if pool[t][1] else 0)) + PLACEBO
    nrow = len(rows)
    for slug, fams in GRID_FAMILIES:
        disp = FAMILY_DISPLAY.get(slug, slug)
        mids = [m for m in per if reg[m]["family"] in fams]   # ALL analyzed members, clean ones included
        if not mids:
            continue
        mids.sort(key=lambda m: (-sum(per_probe[m]["yes"].values()), per[m]["name"]))
        M = np.full((nrow, len(mids)), np.nan)
        for jx, mid in enumerate(mids):
            g = per_probe[mid]
            for i, t in enumerate(rows):
                if g["n"][t]:
                    M[i, jx] = 100 * g["yes"][t] / g["n"][t]
        ncol = len(mids)
        # constant CELL size across all family grids: width grows with the family
        fig, ax = plt.subplots(figsize=(0.46 * ncol + 3.3, 0.34 * nrow + 2.1))
        cmap = SEQ_CMAP.copy()
        cmap.set_bad("#eceae4")          # own-family / not-asked: warm gray, != 0%
        im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0, vmax=100,
                       aspect="auto", interpolation="nearest")
        ax.set_xticks(np.arange(-0.5, ncol), minor=True)
        ax.set_yticks(np.arange(-0.5, nrow), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=0.8)
        ax.tick_params(which="minor", length=0)
        ax.axhline(nrow - len(PLACEBO) - 0.5, color=INK2, lw=1.4)   # placebo band
        for i in range(nrow):
            for jx in range(ncol):
                v = M[i, jx]
                if np.isnan(v) or v <= 0:
                    continue             # floor color = measured 0 (distinct from gray)
                ax.text(jx, i, f"{v:.0f}", ha="center", va="center", fontsize=6.8,
                        color="#ffffff" if v > 55 else INK2)
        ax.set_xticks(range(ncol), [per[m]["name"] for m in mids], rotation=90, fontsize=7.3)
        ax.set_yticks(range(nrow),
                      [f'"Are you {RLABEL[t]}?"' for t in rows[:-3]] + [PLABEL[t] for t in rows[-3:]],
                      fontsize=8.5)
        style(ax, bottom=False)
        cb = fig.colorbar(im, ax=ax, shrink=0.62, pad=0.02)
        cb.set_label('% of probe responses answering "yes" (n≈10 per cell)', fontsize=8, color=INK2)
        cb.outline.set_visible(False)
        gray = " (gray = own family, not asked)" if np.isnan(M).any() else ""
        ax.set_title(f'{disp} — who says yes to "Are you X?"{gray}',
                     loc="left", fontsize=10.5, pad=12)
        save(fig, f"fam_cross_{slug}.png")


def main():
    reg, per = gather()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_distribution(per)
    fig_family_counts(reg, per)
    fam_cross_grids(reg, per)


if __name__ == "__main__":
    main()
