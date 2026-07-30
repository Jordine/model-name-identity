"""Cross-provider figure — does the serving cloud change a model's self-identity?

Chinese identity-mismatch rate for Claude Opus 4.8 / Sonnet 4.6 across every
clean, non-injecting provider, with Wilson 95% CIs over the pooled zh samples in
results/xprovider_sweep.jsonl (full battery + booster). Every other language is
~0% for these models, so Chinese is the whole story.

The honest read: bars within a model land close and their CIs overlap → the host
barely moves what the model says it is. It's the model, not the cloud.

  python -m sweep.fig_xprovider
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# importing make_figs applies the shared rcParams / palette
from .make_figs import CAT, SURFACE, INK, INK2, MUTED, BASE, wilson
from .analyze import canon_identity, is_self

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results" / "xprovider_sweep.jsonl"
FIGS = ROOT / "figures"
REG = {m["id"]: m for m in json.loads((ROOT / "config" / "models.json").read_text())["models"]}

MODELS = ["anthropic/claude-opus-4.8", "anthropic/claude-sonnet-4.6"]
DISPLAY = {"anthropic/claude-opus-4.8": "Claude Opus 4.8",
           "anthropic/claude-sonnet-4.6": "Claude Sonnet 4.6"}
PROV_ORDER = ["anthropic-direct", "anthropic", "google-vertex", "amazon-bedrock", "azure"]
PROV_NAME = {"anthropic-direct": "Anthropic API (direct)", "anthropic": "via OpenRouter",
             "google-vertex": "Google Vertex", "amazon-bedrock": "Amazon Bedrock",
             "azure": "Microsoft Azure"}
PROV_COLOR = {"anthropic-direct": CAT[0], "anthropic": CAT[6], "google-vertex": CAT[2],
              "amazon-bedrock": CAT[7], "azure": CAT[4]}


def zh_rates():
    per = defaultdict(lambda: [0, 0])   # (mid, prov) -> [mismatch, n] on zh
    for line in DATA.open(encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("error") or r["prompt_category"] not in ("direct_zh", "creator_zh"):
            continue
        mid, prov = r["model_id"], r["provider_pinned"]
        m = REG[mid]
        resp = (r.get("content_clean") or r.get("content") or "").strip()
        c = canon_identity(resp)
        mism = bool(c and not str(c).startswith("other:")
                    and not is_self(c, m["family"], m["aliases"], m["expected_identity"]))
        per[(mid, prov)][1] += 1
        if mism:
            per[(mid, prov)][0] += 1
    return per


def main():
    per = zh_rates()
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    group_w = 0.82
    # every provider serving EITHER model gets a slot in BOTH groups, so a
    # provider that doesn't serve one model reads as an explicit gap (with a
    # "not served" note), not as an ambiguous absence next to a 5-entry legend
    slots = [p for p in PROV_ORDER if any(per.get((m, p), [0, 0])[1] > 0 for m in MODELS)]
    bw = group_w / len(slots)
    for gi, mid in enumerate(MODELS):
        x0 = gi - group_w / 2 + bw / 2
        for j, prov in enumerate(slots):
            d, n = per.get((mid, prov), [0, 0])
            x = x0 + j * bw
            if n == 0:      # data-availability note in the empty slot
                short = {"azure": "Azure"}.get(prov, PROV_NAME[prov])
                ax.text(x, 1.5, f"{short}: not served", rotation=90, ha="center",
                        va="bottom", fontsize=7, color=MUTED, zorder=3)
                continue
            rate = 100 * d / n
            lo, hi = wilson(d, n)
            ax.bar(x, rate, bw * 0.9, color=PROV_COLOR[prov], zorder=3, edgecolor=SURFACE, linewidth=0.5)
            ax.errorbar(x, rate, yerr=[[lo], [hi]], fmt="none", ecolor=INK2,
                        elinewidth=1.1, capsize=2.5, zorder=4)
            ax.text(x, rate + hi + 1.5, f"{rate:.0f}%", ha="center", va="bottom",
                    fontsize=8.5, color=INK, fontweight="bold")

    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([DISPLAY[m] for m in MODELS], fontsize=10.5)
    ax.set_ylabel("% of Chinese-language short-question responses\nwith a mismatched name")
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(BASE)
    ax.spines["bottom"].set_color(BASE)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#e1e0d9", linewidth=0.7)
    ax.margins(x=0.12)

    handles = [Patch(facecolor=PROV_COLOR[p], label=PROV_NAME[p]) for p in slots]
    ax.legend(handles=handles, frameon=False, fontsize=8.5, loc="upper right",
              handlelength=1.1, borderaxespad=0.3)
    ax.set_title("Same weights, different endpoint — identical except Google Vertex on Opus",
                 fontsize=11.5, color=INK, pad=10, loc="left")
    FIGS.mkdir(exist_ok=True)
    out = FIGS / "fig_xprovider.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {out}")
    # also print the numbers it drew
    for mid in MODELS:
        cells = [f"{p} {100*per[(mid,p)][0]/per[(mid,p)][1]:.0f}% (n={per[(mid,p)][1]})"
                 for p in PROV_ORDER if per.get((mid, p), [0, 0])[1] > 0]
        print(f"  {DISPLAY[mid]}: " + " · ".join(cells))


if __name__ == "__main__":
    main()
