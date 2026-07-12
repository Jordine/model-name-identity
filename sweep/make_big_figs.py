"""The big-ass figures + aggregate slices.

figA: ALL models, one bar each, sorted by foreign-claim rate, Wilson CIs.
figB: ALL models × claimed-identity-family heatmap (grouped by claimant family).
fig8: per-language aggregate (record-level, CIs).
fig9: per-prompt-category aggregate (record-level, CIs).
fig10: most-claimed identities overall.

Usage: python -m sweep.make_big_figs
"""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .analyze import load, canon_identity, is_self, lang_of
from .make_real_figs import foreign_claims, CREATOR_TO_BRAND

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "post" / "figs"

CAT = ["#2a78d6", "#1baf7a", "#eda100", "#e34948", "#4a3aa7"]
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "text.color": INK,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 9,
})


def wilson(d, n):
    if n == 0:
        return 0.0, 0.0
    p = d / n
    z = 1.96
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, (p - (c - h)) * 100), max(0.0, ((c + h) - p) * 100)


def style(ax, bottom=True):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_visible(bottom)
    ax.tick_params(length=0)


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / name, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  {name}")


def gather():
    reg = {m["id"]: m for m in json.loads((ROOT / "config" / "models.json").read_text())["models"]}
    rows = load()
    per = defaultdict(lambda: {"n": 0, "d": 0, "claims": Counter(),
                               "lang": defaultdict(lambda: [0, 0]),
                               "cat": defaultdict(lambda: [0, 0])})
    for j in rows:
        if j["prompt_category"] in ("probe_cross", "system_probe"):
            continue
        m = per[j["model_id"]]
        fc = foreign_claims(j)
        m["n"] += 1
        l = lang_of(j["prompt_category"])
        m["lang"][l][1] += 1
        cat = j["prompt_category"]
        m["cat"][cat][1] += 1
        if fc:
            m["d"] += 1
            m["lang"][l][0] += 1
            m["cat"][cat][0] += 1
            for c in fc:
                c = "hallucinated/other" if c.startswith("other:") else CREATOR_TO_BRAND.get(c, c)
                m["claims"][c] += 1
    per = {k: v for k, v in per.items() if v["n"] >= 100}
    return reg, per


def figA(reg, per):
    items = sorted(per.items(), key=lambda x: x[1]["d"] / x[1]["n"])
    n = len(items)
    names = [reg.get(mid, {}).get("name", mid) for mid, _ in items]
    rates = np.array([100 * v["d"] / v["n"] for _, v in items])
    errs = np.array([wilson(v["d"], v["n"]) for _, v in items]).T
    fams = [reg.get(mid, {}).get("family", "?") for mid, _ in items]
    fam_hue = {}
    for f in fams:
        if f not in fam_hue:
            fam_hue[f] = CAT[len(fam_hue) % 5]

    fig, ax = plt.subplots(figsize=(9.5, 0.185 * n + 1.6))
    ax.barh(np.arange(n), rates, height=0.66, color=CAT[0], zorder=3)
    ax.errorbar(rates, np.arange(n), xerr=errs, fmt="none",
                ecolor=INK2, elinewidth=0.7, capsize=1.5, zorder=4)
    ax.set_yticks(np.arange(n), names, fontsize=6.4)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("% of ~120 identity prompts with a foreign self-claim (Wilson 95% CI)")
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    for i, r in enumerate(rates):
        if r > 0:
            ax.text(r + errs[1][i] + 0.7, i, f"{r:.0f}", va="center", fontsize=5.6, color=MUTED)
    style(ax)
    ax.set_title(f"Spontaneous foreign-identity rate — all {n} models, sorted",
                 loc="left", fontsize=12, pad=14)
    save(fig, "figA_all_models_bar.png")


def figB(reg, per):
    targets = ["chatgpt", "claude", "gemini", "grok", "kimi", "deepseek", "qwen",
               "llama", "mistral", "glm", "nvidia", "doubao", "local-market lab", "hallucinated/other"]
    LOCAL = {"naver", "zhinao360", "sparkdesk", "ernie", "hunyuan", "bytedance",
             "internlm", "kuaishou", "xiaomi", "stepfun", "ant", "reka", "cohere",
             "microsoft", "copilot", "siri", "alexa", "cortana", "nous",
             "perplexity", "ibm", "allenai", "alibaba", "meta", "moonshot",
             "openai", "anthropic", "google", "xai"}

    def to_col(c):
        if c.startswith("other:") or c == "hallucinated/other":
            return "hallucinated/other"
        c = CREATOR_TO_BRAND.get(c, c)
        if c in targets:
            return c
        return "local-market lab"  # any remaining known-lab canon -> "other lab" column

    # group models by family, then by name
    mids = sorted(per, key=lambda m: (reg.get(m, {}).get("family", "?"),
                                      reg.get(m, {}).get("name", m)))
    M = np.zeros((len(mids), len(targets)))
    for i, mid in enumerate(mids):
        v = per[mid]
        for c, k in v["claims"].items():
            M[i, targets.index(to_col(c))] += k
        M[i] = 100 * M[i] / v["n"]

    n = len(mids)
    fig, ax = plt.subplots(figsize=(8.6, 0.185 * n + 2.0))
    cmap = LinearSegmentedColormap.from_list("s", SEQ)
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=50, aspect="auto")
    ax.set_xticks(range(len(targets)),
                  ["ChatGPT", "Claude", "Gemini", "Grok", "Kimi", "DeepSeek", "Qwen",
                   "Llama", "Mistral", "GLM", "NVIDIA", "Doubao", "other lab", "halluc./other"],
                  fontsize=7.5, rotation=35, ha="right")
    ax.set_yticks(range(n), [reg.get(m, {}).get("name", m) for m in mids], fontsize=6.4)
    # family separators
    fams = [reg.get(m, {}).get("family", "?") for m in mids]
    for i in range(1, n):
        if fams[i] != fams[i - 1]:
            ax.axhline(i - 0.5, color=SURFACE, lw=2)
            ax.axhline(i - 0.5, color=GRID, lw=0.8)
    style(ax, bottom=False)
    cb = fig.colorbar(im, ax=ax, shrink=0.35, pad=0.02, anchor=(0, 0.9))
    cb.set_label("% of records claiming that identity", fontsize=8, color=INK2)
    cb.outline.set_visible(False)
    ax.set_title(f"Who claims whom — all {n} models (grouped by family) × claimed identity",
                 loc="left", fontsize=12, pad=14)
    save(fig, "figB_all_models_heatmap.png")


def fig8(per):
    agg = defaultdict(lambda: [0, 0])
    for v in per.values():
        for l, (d, n) in v["lang"].items():
            agg[l][0] += d
            agg[l][1] += n
    langs = ["en", "zh", "fr", "es", "ja", "ko", "ru", "vi", "mixed"]
    vals = [100 * agg[l][0] / agg[l][1] for l in langs]
    errs = np.array([wilson(agg[l][0], agg[l][1]) for l in langs]).T
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    ax.bar(langs, vals, 0.62, color=CAT[0], zorder=3)
    ax.errorbar(range(len(langs)), vals, yerr=errs, fmt="none", ecolor=INK2,
                elinewidth=1, capsize=2.5, zorder=4)
    for i, (l, v) in enumerate(zip(langs, vals)):
        ax.text(i, v + errs[1][i] + 0.4, f"{v:.1f}%", ha="center", fontsize=8, color=INK2)
        ax.text(i, -1.6, f"n={agg[l][1]:,}", ha="center", fontsize=6.5, color=MUTED)
    ax.set_ylim(0, max(vals) + 4)
    ax.set_yticks([])
    style(ax)
    ax.set_title("Foreign-claim rate by prompt language — all models pooled (record-level, Wilson CIs)",
                 loc="left", fontsize=11, pad=12)
    save(fig, "fig8_language_aggregate.png")


def fig9(per):
    agg = defaultdict(lambda: [0, 0])
    for v in per.values():
        for c, (d, n) in v["cat"].items():
            key = ("direct (extra langs)" if c.startswith("direct_") and c not in ("direct_en", "direct_zh")
                   else {"direct_en": "direct EN", "direct_zh": "direct ZH", "casual": "casual",
                         "creator_en": "creator EN", "probe_self": "self-probe",
                         "multi_turn": "multi-turn"}.get(c, c))
            agg[key][0] += d
            agg[key][1] += n
    order = ["casual", "direct EN", "creator EN", "direct ZH", "direct (extra langs)",
             "self-probe", "multi-turn"]
    vals = [100 * agg[k][0] / agg[k][1] for k in order]
    errs = np.array([wilson(agg[k][0], agg[k][1]) for k in order]).T
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    ax.bar(order, vals, 0.62, color=CAT[1], zorder=3)
    ax.errorbar(range(len(order)), vals, yerr=errs, fmt="none", ecolor=INK2,
                elinewidth=1, capsize=2.5, zorder=4)
    for i, (k, v) in enumerate(zip(order, vals)):
        ax.text(i, v + errs[1][i] + 0.35, f"{v:.1f}%", ha="center", fontsize=8, color=INK2)
        ax.text(i, -1.4, f"n={agg[k][1]:,}", ha="center", fontsize=6.5, color=MUTED)
    ax.set_ylim(0, max(vals) + 3.5)
    ax.set_yticks([])
    plt.setp(ax.get_xticklabels(), fontsize=8)
    style(ax)
    ax.set_title("Foreign-claim rate by prompt category — all models pooled (Wilson CIs)",
                 loc="left", fontsize=11, pad=12)
    save(fig, "fig9_category_aggregate.png")


def fig10(per):
    tot = Counter()
    for v in per.values():
        for c, k in v["claims"].items():
            tot[c] += k
    items = tot.most_common(14)
    names = [k for k, _ in items][::-1]
    vals = [v for _, v in items][::-1]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.barh(names, vals, height=0.62, color=CAT[0], zorder=3)
    for i, v in enumerate(vals):
        ax.text(v + 8, i, f"{v:,}", va="center", fontsize=8, color=INK2)
    ax.set_xlabel("total foreign claims across all models")
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    style(ax)
    ax.set_title("Most-claimed identities overall", loc="left", fontsize=11, pad=12)
    save(fig, "fig10_claimed_totals.png")


if __name__ == "__main__":
    reg, per = gather()
    print(f"{len(per)} models")
    figA(reg, per)
    figB(reg, per)
    fig8(per)
    fig9(per)
    fig10(per)
