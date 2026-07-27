"""Explanatory re-analysis: don't just measure *how often* a model names another vendor,
measure the *structure* of it.

Two axes the headline rate collapses:
  * COHERENCE — when a model claims another identity, does it consistently claim
    ONE (a stable alternate persona, e.g. Kimi -> Claude) or scatter across many
    (confabulation / weak identity, e.g. a small model naming BERT/GPT/Claude at
    random)?  Measured by the top-1 share and Shannon entropy of its foreign-claim
    distribution.
  * LANGUAGE-CONDITIONALITY — is the drift spread evenly or concentrated in a few
    languages (e.g. Claude clean in English, DeepSeek in Chinese)?  Measured by the
    share of drift in the dominant language and the spread of per-language rates.

Writes figures/fig_coherence.png, figures/fig_lang_conditional.png and prints
ranked tables. Uses the same adjudicated drift as the headline figures.

  python -m sweep.explain
"""
import json
import math
import os
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .build_rollouts import brand
from .make_figs import FIGS, SURFACE, INK, INK2, MUTED, GRID, BASE, CAT, IDCOLOR, style, save

ROOT = Path(__file__).resolve().parent.parent
# committed research tables (from agent lookups); fall back to /tmp during a run
QWEN_LADDER = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B",
               "Qwen/Qwen3-8B", "Qwen/Qwen3-14B", "Qwen/Qwen3-32B"]


def _load(name):
    for p in (ROOT / "config" / name, Path("/tmp") / name):
        if p.exists():
            return {d["id"]: d for d in json.load(open(p, encoding="utf-8"))}
    return None

MIN_FOREIGN = 10   # need at least this many foreign claims to estimate a distribution
MIN_TOT = 100      # and a reasonably complete battery

# dominant-identity colors (who the model claims to be) come from make_figs.IDCOLOR
# so an identity keeps ONE color across fig_flow / fig_coherence / fig_cutoff.


def place_labels(fig, ax, labels, points, fontsize=6.6):
    """Deterministic collision-avoiding point labels (no adjustText).

    labels: [(x, y, text)]; points: [(x, y, s)] scatter obstacles (s = marker
    area, pt²). Processes labels top-down; for each, walks a fixed ladder of
    offsets (right-above first — left first near the right edge), measuring real
    text extents against already-placed labels, all markers and the axes frame;
    the first free slot wins, and far-flung slots get a thin leader line. Call
    AFTER limits/legend/titles are final (it uses the live transforms)."""
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    px = fig.dpi / 72.0
    tr = ax.transData.transform
    axbox = ax.get_window_extent(ren)
    obstacles = []
    for x, y, s in points:
        cx, cy = tr((x, y))
        r = (math.sqrt(s) / 2.0 + 1.5) * px
        obstacles.append((cx - r, cy - r, cx + r, cy + r))

    def overlap_area(b, boxes):
        return sum(max(0.0, min(b[2], o[2]) - max(b[0], o[0])) * max(0.0, min(b[3], o[3]) - max(b[1], o[1]))
                   for o in boxes)

    placed = []
    for x, y, text in sorted(labels, key=lambda t: (-t[1], t[0])):
        offs = [(6, 4), (6, -12), (7, 15), (7, -23), (8, 26), (9, -34), (10, 37), (11, -45), (12, 48)]
        right = [(dx, dy, "left") for dx, dy in offs]
        left = [(-dx, dy, "right") for dx, dy in offs]
        pairs = (left, right) if tr((x, y))[0] > axbox.x0 + 0.74 * axbox.width else (right, left)
        cands = [v for pair in zip(*pairs) for v in pair]
        pick, best_bad = None, None
        for dx, dy, ha in cands:
            a = ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy),
                            fontsize=fontsize, color=INK2, ha=ha, va="bottom", zorder=6)
            b = a.get_window_extent(ren)
            a.remove()
            bb = (b.x0 - 1.5, b.y0 - 1.5, b.x1 + 1.5, b.y1 + 1.5)
            inside = (bb[0] > axbox.x0 - 4 and bb[2] < axbox.x1 + 10
                      and bb[1] > axbox.y0 and bb[3] < axbox.y1 + 4)
            bad = overlap_area(bb, placed) + overlap_area(bb, obstacles) + (0 if inside else 1e6)
            if best_bad is None or bad < best_bad:
                pick, best_bad = (dx, dy, ha, bb), bad
            if bad == 0:
                break
        dx, dy, ha, bb = pick
        kw = dict(arrowprops=dict(arrowstyle="-", lw=0.55, color=BASE, shrinkA=2, shrinkB=2)) \
            if abs(dy) >= 15 else {}
        ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy),
                    fontsize=fontsize, color=INK2, ha=ha, va="bottom", zorder=6, **kw)
        placed.append(bb)


def entropy(counter):
    tot = sum(counter.values())
    if not tot:
        return 0.0
    return -sum((n / tot) * math.log2(n / tot) for n in counter.values() if n)


def gather():
    """Per-model data straight from make_figs.gather() — so the explanatory analysis
    runs on EXACTLY the figures' set: adjudicated (foreign_claims), completeness-gated
    (n>=40 + complete_models), BATTERY_CORE-balanced, local models folded in, and
    creator-canons already collapsed to brand with novel claims kept as 'other/unlisted'
    (so confabulators don't look coherent)."""
    from .make_figs import gather as figs_gather
    reg, per = figs_gather()
    rows = {}
    for mid, m in per.items():
        n = m["n"]
        if not n:
            continue
        rows[mid] = dict(
            name=m.get("name", mid), fam=reg.get(mid, {}).get("family", "?"),
            rate=100 * m["d"] / n, dn=m["d"], tot=n,
            claims=Counter(m["claims"]), cross=Counter(m.get("cross_yes", {})),
            lstats={l: list(v) for l, v in m["lang"].items()},
        )
    return rows


def metrics(d):
    claims = d["claims"]
    total = sum(claims.values())
    top = claims.most_common(1)[0] if total else (None, 0)
    # language-conditionality, rate-CONTROLLED: worst single-language rate vs the
    # model's overall rate. "excess" = worst_lang_rate - overall is how much more a
    # model drifts in its worst language than on average — a uniform model has
    # excess≈0 at any overall rate, so this isn't confounded by the rate itself
    # (unlike a drift-share, which is capped low once the overall rate is high).
    lr = {l: (dd / nn, dd, nn) for l, (dd, nn) in d["lstats"].items() if nn >= 15}
    worst = max(lr, key=lambda l: lr[l][0]) if lr else None
    overall = d["rate"] / 100
    max_rate = lr[worst][0] if worst else overall
    return dict(
        n_foreign=total,
        top1_id=top[0], top1_share=(top[1] / total if total else 0),
        entropy=entropy(claims), n_distinct=sum(1 for c, n in claims.items() if n >= 2),
        dom_lang=worst, max_lang_rate=max_rate, excess=max_rate - overall,
    )


def fig_coherence(data):
    """rate (how often it drifts) × top-1 share (how consistently to ONE identity)."""
    pts = [(d, metrics(d)) for d in data.values()]
    pts = [(d, mm) for d, mm in pts if mm["n_foreign"] >= MIN_FOREIGN and d["tot"] >= MIN_TOT and d["rate"] >= 5]
    fig, ax = plt.subplots(figsize=(9.2, 6.6))
    for d, mm in pts:
        col = IDCOLOR.get(mm["top1_id"], MUTED)
        ax.scatter(d["rate"], 100 * mm["top1_share"], s=22 + d["dn"] * 0.7,
                   color=col, alpha=0.72, edgecolor="white", linewidth=0.5, zorder=3)
    ax.axhline(60, color=BASE, lw=0.8, ls="--", zorder=1)
    ax.text(1, 62, "consistently ONE identity  (coherent alternate persona)", fontsize=7.5, color=MUTED)
    ax.text(1, 20, "scatters across many  (confabulation / weak identity)", fontsize=7.5, color=MUTED)
    ax.set_xlabel("spontaneous mismatch rate  (% of identity answers)")
    ax.set_ylabel("consistency — share of mismatches on the single top identity (%)")
    ax.set_title("A stable alternate identity vs. confabulation",
                 fontsize=11, color=INK, loc="left", pad=22)
    ax.text(0, 1.008, "labels: model →its dominant claimed identity · bubble area ∝ mismatched records",
            transform=ax.transAxes, fontsize=7.3, color=MUTED, va="bottom")
    ax.set_ylim(0, 105); ax.set_xlim(0, max(d["rate"] for d, _ in pts) + 6)
    style(ax)
    ax.grid(axis="both", color=GRID, lw=0.6, zorder=0)
    # legend for dominant identity
    from matplotlib.patches import Patch
    seen_ids = [i for i in IDCOLOR if any(mm["top1_id"] == i for _, mm in pts)]
    ax.legend(handles=[Patch(facecolor=IDCOLOR[i], label=brand(i)) for i in seen_ids],
              title="claims to be", fontsize=7.5, title_fontsize=8, loc="lower right", frameon=False)
    # label only the well-separated right side (rate>=33); the low-rate left is a
    # dense cloud (many models rarely-but-consistently drift) — left unlabelled
    place_labels(fig, ax,
                 [(d["rate"], 100 * mm["top1_share"], f"{d['name']} →{brand(mm['top1_id'])}")
                  for d, mm in pts if d["rate"] >= 33],
                 [(d["rate"], 100 * mm["top1_share"], 22 + d["dn"] * 0.7) for d, mm in pts])
    save(fig, "fig_coherence.png")


def fig_lang_conditional(data):
    """rate-controlled language-conditionality: worst-language rate vs overall rate.
    The y=x diagonal is 'uniform across languages'; distance above it = a model
    drifts far more in one language than on average (Jekyll/Hyde by language)."""
    pts = [(d, metrics(d)) for d in data.values()]
    pts = [(d, mm) for d, mm in pts if mm["n_foreign"] >= MIN_FOREIGN and d["tot"] >= MIN_TOT and d["rate"] >= 3]
    fig, ax = plt.subplots(figsize=(9.2, 6.6))
    lim = max(100 * mm["max_lang_rate"] for _, mm in pts) + 5
    ax.plot([0, lim], [0, lim], color=BASE, lw=1, ls="--", zorder=1)
    ax.text(lim * 0.55, lim * 0.5, "uniform across languages", fontsize=8, color=MUTED, rotation=38, va="bottom")
    for d, mm in pts:
        ax.scatter(d["rate"], 100 * mm["max_lang_rate"], s=22 + d["dn"] * 0.6,
                   color="#2a78d6", alpha=0.6, edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xlabel("overall spontaneous mismatch rate (%)")
    ax.set_ylabel("mismatch rate in the model's HIGHEST-mismatch language (%)")
    ax.set_title("Language-triggered vs. uniformly-weak: highest-mismatch-language rate vs. overall",
                 fontsize=11, color=INK, loc="left", pad=26)
    ax.text(0, 1.006, "labeled models clear a uniform-null by ≥40pp (null p95 ≈ 18pp); the near-diagonal "
            "cloud is within sampling noise · bubble area ∝ mismatched records",
            transform=ax.transAxes, fontsize=7.3, color=MUTED, va="bottom")
    ax.set_xlim(0, lim); ax.set_ylim(0, 105)
    style(ax)
    ax.grid(axis="both", color=GRID, lw=0.6, zorder=0)
    place_labels(fig, ax,
                 [(d["rate"], 100 * mm["max_lang_rate"], f"{d['name']} ({mm['dom_lang']})")
                  for d, mm in sorted(pts, key=lambda x: -x[1]["excess"])[:11]],
                 [(d["rate"], 100 * mm["max_lang_rate"], 22 + d["dn"] * 0.6) for d, mm in pts])
    save(fig, "fig_lang_conditional.png")


def fig_size(data):
    """spontaneous mismatch rate vs total parameter count. The Qwen3 raw ladder
    (same family, 0.6B->32B) is the controlled series that isolates the size effect."""
    params = _load("model_params.json")
    if not params:
        print("  (no model_params.json yet — skipping fig_size)")
        return
    pts = [(mid, d, params[mid]) for mid, d in data.items()
           if mid in params and params[mid].get("params_total_B") and params[mid].get("basis") != "unknown"
           and d["tot"] >= MIN_TOT]
    fig, ax = plt.subplots(figsize=(9.4, 6.4))
    for mid, d, p in pts:
        est = p["basis"] in ("epoch_estimate", "reported_estimate")
        ax.scatter(p["params_total_B"], d["rate"], s=28,
                   facecolor="none" if est else "#3987e5",
                   edgecolor="#3987e5", alpha=0.6, linewidth=1.1, zorder=3)
    lad = sorted((params[m]["params_total_B"], data[m]["rate"], data[m]["name"])
                 for m in QWEN_LADDER if m in data and m in params)
    if lad:
        ax.plot([x for x, _, _ in lad], [y for _, y, _ in lad], "-o", color="#e0761a",
                lw=2, ms=7, zorder=5, label="Qwen3 raw ladder (controlled: same family)")
        for x, y, nm in lad:
            ax.annotate(nm.replace("Qwen3 ", ""), (x, y), fontsize=6.5, color="#b85f14",
                        xytext=(3, 5), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("total parameters (billions, log)   ·   filled = published, hollow = estimated")
    ax.set_ylabel("spontaneous mismatch rate (%)")
    ax.set_title("Does size predict the mismatch rate?", fontsize=11, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    style(ax); ax.grid(color=GRID, lw=0.6, zorder=0)
    save(fig, "fig_size.png")


def _year(s):
    if not s:
        return None
    parts = str(s).split("-")
    try:
        return int(parts[0]) + ((int(parts[1]) if len(parts) > 1 else 6) - 0.5) / 12
    except (ValueError, IndexError):
        return None


# major version releases per claimed brand (year). breakout=True marks the version
# whose outputs actually flooded training data — where claiming it turns on — vs the
# obscure first release that merely made the name exist.
VERSIONS = {
    "chatgpt": [(2022.88, "ChatGPT", True), (2023.21, "GPT-4", False), (2024.37, "4o", False)],
    "deepseek": [(2023.87, "Coder", False), (2024.37, "V2", False), (2024.96, "V3", True), (2025.04, "R1", True)],
    "claude": [(2023.21, "1", False), (2023.54, "2", False), (2024.21, "3", True), (2024.46, "3.5", True)],
    "gemini": [(2023.21, "Bard", False), (2023.96, "1.0", True), (2024.37, "1.5", False)],
    "qwen": [(2023.62, "Qwen1", False), (2024.46, "Qwen2", False), (2024.71, "2.5", True)],
    "llama": [(2023.12, "1", False), (2023.54, "2", True), (2024.29, "3", False)],
}


CUTOFF_LAG = 0.5   # median release−cutoff lag (yr) among the 101 models reporting both;
#                    used to put release-only models on the same TRAINING-CUTOFF axis


def fig_cutoff(data):
    """Per identity: does a model start claiming to BE X only after X's outputs
    flooded training data? A model can absorb X's identity only if X's outputs
    existed before the model's TRAINING CUTOFF — and X's outputs begin at X's
    RELEASE. So x = the claiming model's training cutoff (documented = solid, or
    estimated as release − 6mo median lag = hollow) and the vertical lines are the
    TARGET's version releases (grey = earlier, red dashed = breakout). y = how much
    of the model's identity answers claim X. Onset tracks the breakout (DeepSeek V3,
    Claude 3.5), not the obscure first release."""
    cut = _load("model_cutoffs.json")
    if not cut:
        print("  (no model_cutoffs.json yet — skipping fig_cutoff)")
        return
    rows = []
    for mid, d in data.items():
        c = cut.get(mid) or {}
        yc, yr = _year(c.get("cutoff")), _year(c.get("release_date"))
        if yc is not None:
            x, doc = yc, True
        elif yr is not None:
            x, doc = yr - CUTOFF_LAG, False   # estimate cutoff from release
        else:
            continue
        if d["tot"] < MIN_TOT:
            continue
        rows.append((d, x, doc))
    # version labels sit just ABOVE each panel (never on the data); near-coincident
    # releases are split to opposite sides of their lines
    DODGE = {("chatgpt", "ChatGPT"): "right", ("chatgpt", "GPT-4"): "left",
             ("deepseek", "V3"): "right", ("deepseek", "R1"): "left",
             ("qwen", "Qwen2"): "right", ("qwen", "2.5"): "left"}
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 8.0))
    for ax, ident in zip(axes.flat, VERSIONS):
        col = IDCOLOR.get(ident, "#2a78d6")
        for d, x, doc in rows:
            y = 100 * d["claims"].get(ident, 0) / d["tot"]
            ax.scatter(x, y, s=22, facecolor=col if doc else "none",
                       edgecolor=col, alpha=0.6, lw=1, zorder=3)
        for vx, vlab, breakout in VERSIONS[ident]:
            ax.axvline(vx, color="#6d5bd0" if breakout else BASE,
                       lw=1.3 if breakout else 0.9, ls="--" if breakout else ":", zorder=2)
            ax.text(vx, 1.02, vlab, fontsize=7, color="#6d5bd0" if breakout else MUTED,
                    va="bottom", ha=DODGE.get((ident, vlab), "center"),
                    transform=ax.get_xaxis_transform(), clip_on=False)
        ax.set_title(f"claims to be {brand(ident)}", fontsize=10.5, color=INK, loc="left", pad=18)
        ax.set_xlim(2021.4, 2026.5)
        style(ax); ax.grid(color=GRID, lw=0.5, zorder=0)
    fig.text(0.5, 0.015, "claiming model's TRAINING CUTOFF — documented (solid) or estimated as "
             "release−6mo (hollow)   |   vertical lines = TARGET's version releases "
             "(dashed = breakout, dotted = earlier)",
             ha="center", fontsize=8.5, color=INK2)
    fig.text(0.008, 0.5, "% of the model's identity answers claiming this identity",
             va="center", rotation=90, fontsize=9, color=INK2)
    fig.suptitle("A model claims to be X only after X's outputs entered its training data "
                 "(onset ≈ breakout: DeepSeek V3, Claude 3.5)",
                 fontsize=12, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0.025, 0.03, 1, 0.95])
    save(fig, "fig_cutoff.png")


def report(data):
    rows = [(d, metrics(d)) for d in data.values()]
    ok = [(d, mm) for d, mm in rows if mm["n_foreign"] >= MIN_FOREIGN and d["tot"] >= MIN_TOT]

    def table(title, items, fmt):
        print(f"\n=== {title} ===")
        for d, mm in items:
            print("  " + fmt(d, mm))

    table("1. Claims another identity the MOST (rate)",
          sorted(ok, key=lambda x: -x[0]["rate"])[:15],
          lambda d, mm: f"{d['name']:26s} {d['rate']:4.0f}%  ->{brand(mm['top1_id']) or '-':9s} (top1 {100*mm['top1_share']:3.0f}%)")

    table("2a. COHERENT — consistently ONE other identity (rate>=20%, top1>=70%)",
          sorted([(d, mm) for d, mm in ok if d["rate"] >= 20 and mm["top1_share"] >= 0.7],
                 key=lambda x: -(x[0]["rate"] * x[1]["top1_share"]))[:15],
          lambda d, mm: f"{d['name']:26s} {d['rate']:4.0f}%  ->{brand(mm['top1_id']):9s} {100*mm['top1_share']:3.0f}%  H={mm['entropy']:.2f}b")

    table("2b. CONFABULATORY — high rate, scatters across names (rate>=25%, top1<50%)",
          sorted([(d, mm) for d, mm in ok if d["rate"] >= 25 and mm["top1_share"] < 0.5],
                 key=lambda x: -x[1]["entropy"])[:15],
          lambda d, mm: f"{d['name']:26s} {d['rate']:4.0f}%  H={mm['entropy']:.2f}b  {mm['n_distinct']} ids  top:{brand(mm['top1_id'])} {100*mm['top1_share']:.0f}%")

    table("3. Most LANGUAGE-TRIGGERED (worst-language rate >> overall — rate-controlled)",
          sorted([(d, mm) for d, mm in ok if d["rate"] >= 3], key=lambda x: -x[1]["excess"])[:15],
          lambda d, mm: f"{d['name']:26s} overall {d['rate']:4.0f}%   worst {mm['max_lang_rate']*100:3.0f}% in {mm['dom_lang']}   (+{mm['excess']*100:.0f}pp)")


def main():
    data = gather()
    report(data)
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_coherence(data)
    fig_lang_conditional(data)
    fig_size(data)
    fig_cutoff(data)


if __name__ == "__main__":
    main()
