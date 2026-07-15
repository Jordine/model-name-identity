"""Per-family 'who claims whom' heatmaps — one plot per model family, so the
reader can scroll family-by-family instead of hunting a 177-row monolith.

For each family: rows = its models that misidentified at least once (sorted by
rate), columns = the canonical identities they claimed. Fully-clean families are
listed as text, not plotted. Full resolution (dpi 200). Writes a manifest so the
post builder can splice them in order.

Usage: python -m sweep.make_family_figs
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .make_big_figs import gather
from .make_real_figs import CREATOR_TO_BRAND

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "post" / "figs" / "family"
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 9,
})

TARGETS = ["chatgpt", "claude", "gemini", "deepseek", "qwen", "kimi", "llama",
           "mistral", "glm", "grok", "nvidia", "doubao", "other lab", "hallucinated/other"]
TLABELS = ["ChatGPT", "Claude", "Gemini", "DeepSeek", "Qwen", "Kimi", "Llama",
           "Mistral", "GLM", "Grok", "NVIDIA", "Doubao", "other lab", "halluc./other"]
# display name + which family keys fold into this panel
FAMILY_TITLE = {
    "anthropic": "Anthropic (Claude)", "openai": "OpenAI (GPT / o-series)",
    "google": "Google (Gemini)", "gemma": "Google (Gemma)", "qwen": "Alibaba (Qwen)",
    "deepseek": "DeepSeek", "kimi": "Moonshot (Kimi)", "mistral": "Mistral",
    "meta": "Meta (Llama)", "nvidia": "NVIDIA (Nemotron)", "zhipu": "Zhipu (GLM)",
    "nous": "Nous (Hermes)", "poolside": "Poolside (Laguna)", "cohere": "Cohere (Command / Aya)",
    "reka": "Reka", "minimax": "MiniMax", "ai21": "AI21 (Jamba)", "amazon": "Amazon (Nova)",
    "ant": "Ant (Ling / Ring)", "arcee": "Arcee", "microsoft": "Microsoft (Phi)",
    "baidu": "Baidu (ERNIE)", "tencent": "Tencent (Hunyuan)", "ibm": "IBM (Granite)",
    "perplexity": "Perplexity (Sonar)", "xiaomi": "Xiaomi (MiMo)", "stepfun": "StepFun",
    "aisingapore": "AI Singapore (SEA-LION)",
}


def to_col(c):
    if c.startswith("other:") or c == "hallucinated/other":
        return "hallucinated/other"
    c = CREATOR_TO_BRAND.get(c, c)
    return c if c in TARGETS else "other lab"


def panel(fam_key, mids, reg, per):
    """Render one family's heatmap; return (filename, caption) or None if clean."""
    rows = []
    for mid in mids:
        v = per[mid]
        if v["d"] == 0:
            continue
        vec = np.zeros(len(TARGETS))
        for c, k in v["claims"].items():
            vec[TARGETS.index(to_col(c))] += k
        vec = 100 * vec / v["n"]
        rows.append((mid, v["d"], v["n"], vec))
    n_clean = sum(1 for mid in mids if per[mid]["d"] == 0)
    if not rows:
        return None, n_clean, len(mids)
    rows.sort(key=lambda r: -r[1] / r[2])
    M = np.array([r[3] for r in rows])
    names = [reg.get(r[0], {}).get("name", r[0]) for r in rows]

    nrow = len(rows)
    fig, ax = plt.subplots(figsize=(8.4, 0.42 * nrow + 1.9))
    cmap = LinearSegmentedColormap.from_list("s", SEQ)
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=60, aspect="auto")
    ax.set_xticks(range(len(TARGETS)), TLABELS, fontsize=8.5, rotation=35, ha="right")
    ax.set_yticks(range(nrow), names, fontsize=9)
    for i in range(nrow):
        for j in range(len(TARGETS)):
            val = M[i, j]
            if val >= 1:
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if val > 40 else INK2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_xlabel("claimed identity  (number = % of that model's ~120 records)", fontsize=8.5, color=INK2)
    title = FAMILY_TITLE.get(fam_key, fam_key)
    ax.set_title(f"{title} — who they claim to be", loc="left", fontsize=12, pad=12)
    OUT.mkdir(parents=True, exist_ok=True)
    fname = f"fam_{fam_key}.png"
    fig.savefig(OUT / fname, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

    # top claimed target for caption
    tot = Counter()
    for _, _, _, vec in rows:
        for j, val in enumerate(vec):
            tot[TLABELS[j]] += val
    top = ", ".join(f"{k}" for k, _ in tot.most_common(2) if tot[k] > 0)
    cap = (f"{title}: {nrow} of {len(mids)} models misidentified at least once"
           f"{f' ({n_clean} always self-identified)' if n_clean else ''}"
           f"{f'; most-claimed: {top}' if top else ''}.")
    return fname, cap, (fname, sum(r[1] for r in rows))


def main():
    reg, per = gather()
    fam_models = defaultdict(list)
    for mid in per:
        fam_models[reg.get(mid, {}).get("family", "?")].append(mid)

    panels, clean_fams, singles = [], [], []
    for fam, mids in fam_models.items():
        if len(mids) >= 2:
            res = panel(fam, mids, reg, per)
            if res[0] is None:
                clean_fams.append((FAMILY_TITLE.get(fam, fam), res[2]))
            else:
                fname, cap, (f2, drift) = res
                panels.append({"file": f"family/{fname}", "caption": cap, "drift": drift})
        else:
            singles += mids  # 1-model families pooled below

    # pooled panel of single-model labs that misidentified
    sing_claim = [m for m in singles if per[m]["d"] > 0]
    if sing_claim:
        rows = sorted(sing_claim, key=lambda m: -per[m]["d"] / per[m]["n"])
        M = []
        names = []
        for mid in rows:
            v = per[mid]
            vec = np.zeros(len(TARGETS))
            for c, k in v["claims"].items():
                vec[TARGETS.index(to_col(c))] += k
            M.append(100 * vec / v["n"])
            names.append(reg.get(mid, {}).get("name", mid))
        M = np.array(M)
        fig, ax = plt.subplots(figsize=(8.4, 0.42 * len(rows) + 1.9))
        im = ax.imshow(M, cmap=LinearSegmentedColormap.from_list("s", SEQ), vmin=0, vmax=60, aspect="auto")
        ax.set_xticks(range(len(TARGETS)), TLABELS, fontsize=8.5, rotation=35, ha="right")
        ax.set_yticks(range(len(rows)), names, fontsize=9)
        for i in range(len(rows)):
            for j in range(len(TARGETS)):
                val = M[i, j]
                if val >= 1:
                    ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=7.5,
                            color="#ffffff" if val > 40 else INK2)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        ax.set_xlabel("claimed identity  (number = % of that model's ~120 records)", fontsize=8.5, color=INK2)
        ax.set_title("Single-model & long-tail labs — who they claim to be", loc="left", fontsize=12, pad=12)
        fig.savefig(OUT / "fam_zzz_singles.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
        plt.close(fig)
        panels.append({"file": "family/fam_zzz_singles.png",
                       "caption": f"Single-model & long-tail labs ({len(rows)} models): includes "
                                  f"Perceptron Mk1, Tencent Hy3, and other one-off labs that misidentified.",
                       "drift": sum(per[m]["d"] for m in rows)})

    # named families by drift desc; the single-model grab-bag always last
    panels.sort(key=lambda p: (1 if "singles" in p["file"] else 0, -p["drift"]))
    manifest = {"panels": panels,
                "clean_families": sorted([f"{t} ({n})" for t, n in clean_fams])}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(f"{len(panels)} family panels + {len(clean_fams)} fully-clean families")
    for p in panels:
        print(f"  {p['drift']:4d}  {p['file']}")
    print("clean:", ", ".join(manifest["clean_families"]))


if __name__ == "__main__":
    main()
