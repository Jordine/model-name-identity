"""DRAFT figure: acceptance grid for "Are you X?" probes.

Rows (y): the 10 real probe names sorted by pooled acceptance, then a separated
band of the 3 placebo controls (Meridian-4 / Solace / Cobalt).
Cols (x): every model with >=2 "yes" records on at least one probe name,
grouped by family (separators), families ordered by their peak cell.
Cell = % of that model's ~10 records (EN5+ZH5) answering yes.
Own-family cells (never asked) are masked gray, distinct from 0%.

Data: API models via sweep.analyze.load(); local raw-weights models via
results_local/judgments_clean.jsonl (::clean reads), mirroring make_figs.add_local().
Output: analysis_scratch/generics_audit/fig_cross_grid.png (NOT figures/ — draft).
"""
import json, sys
from collections import defaultdict, Counter

sys.path.insert(0, ".")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sweep.analyze import load, FAMILY_SELF
from sweep.make_figs import (SEQ_CMAP, SURFACE, INK, INK2, MUTED, GRID, BASE, style,
                             LOCAL_MODELS, ROOT)

REAL = ["chatgpt", "claude", "gemini", "deepseek", "qwen", "kimi", "llama", "grok", "mistral", "doubao"]
PLACEBO = ["meridian", "solace", "cobalt"]
PLABEL = {"meridian": "Meridian-4 (control)", "solace": "Solace (control)", "cobalt": "Cobalt (control)"}
RLABEL = {"chatgpt": "ChatGPT", "claude": "Claude", "gemini": "Gemini", "deepseek": "DeepSeek",
          "qwen": "Qwen", "kimi": "Kimi", "llama": "Llama", "grok": "Grok",
          "mistral": "Mistral", "doubao": "Doubao"}

reg = {m["id"]: m for m in json.loads(open("config/models.json").read())["models"]}
per = defaultdict(lambda: {"yes": Counter(), "n": Counter(), "fam": "", "name": ""})


def _target(pid):
    for pre in ("zh_cross_", "en_cross_", "cross_", "zh_placebo_", "en_placebo_", "placebo_"):
        pid = pid.replace(pre, "")
    return pid


def take(j, name, fam):
    cat = j.get("prompt_category")
    if cat not in ("probe_cross", "probe_placebo"):
        return
    t = _target(j["prompt_id"])
    if cat == "probe_cross" and t in FAMILY_SELF.get(fam, {fam}):
        return
    m = per[j["model_id"]]
    m["fam"], m["name"] = fam, name
    m["n"][t] += 1
    if (j["judgment"] or {}).get("answered_yes") is True:
        m["yes"][t] += 1


for j in load():                      # API lane
    if j["model_id"] in reg:
        take(j, reg[j["model_id"]]["name"], j.get("family", ""))

lp = ROOT / "results_local" / "judgments_clean.jsonl"   # raw-weights lane (clean reads)
if lp.exists():
    for l in open(lp, encoding="utf-8"):
        j = json.loads(l)
        if not j.get("judgment") or j["resume_key"].split("::")[-1] != "clean":
            continue
        if j["model_id"] in LOCAL_MODELS:
            name, fam, _ = LOCAL_MODELS[j["model_id"]]
            take(j, name, fam)

# --- column selection: >=2 yeses on some probe name ---------------------------
cols = [mid for mid, v in per.items()
        if any(v["yes"][t] >= 2 for t in REAL + PLACEBO)]
fam_peak = defaultdict(float)
for mid in cols:
    v = per[mid]
    pk = max((v["yes"][t] / v["n"][t]) for t in REAL + PLACEBO if v["n"][t])
    fam_peak[v["fam"]] = max(fam_peak[v["fam"]], pk)
cols.sort(key=lambda m: (-fam_peak[per[m]["fam"]], per[m]["fam"],
                         -sum(per[m]["yes"].values()), per[m]["name"]))

rows = REAL[:] + PLACEBO[:]
# real rows sorted by pooled acceptance (matches fig_cross ordering logic)
pool = {t: (sum(per[m]["yes"][t] for m in per), sum(per[m]["n"][t] for m in per)) for t in REAL}
rows = sorted(REAL, key=lambda t: -(pool[t][0] / pool[t][1])) + PLACEBO

M = np.full((len(rows), len(cols)), np.nan)
for jx, mid in enumerate(cols):
    v = per[mid]
    for i, t in enumerate(rows):
        if v["n"][t]:
            M[i, jx] = 100 * v["yes"][t] / v["n"][t]

# --- draw ---------------------------------------------------------------------
NROW, NCOL = len(rows), len(cols)
fig_w = 0.185 * NCOL + 2.6
fig_h = 0.34 * NROW + 2.9
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
cmap = SEQ_CMAP.copy()
cmap.set_bad("#eceae4")               # own-family / not-asked: warm gray, != 0%
im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0, vmax=100, aspect="auto",
               interpolation="nearest")

# white cell grid (the 2px-spacer idea at heatmap scale)
ax.set_xticks(np.arange(-0.5, NCOL), minor=True)
ax.set_yticks(np.arange(-0.5, NROW), minor=True)
ax.grid(which="minor", color=SURFACE, linewidth=0.8)
ax.tick_params(which="minor", length=0)

# placebo band separator
ysep = len(rows) - len(PLACEBO) - 0.5
ax.axhline(ysep, color=INK2, lw=1.4)

# family separators + top labels for blocks of >=3 models
fams = [per[m]["fam"] for m in cols]
starts = [0] + [k for k in range(1, NCOL) if fams[k] != fams[k - 1]]
for s in starts[1:]:
    ax.axvline(s - 0.5, color=MUTED, lw=0.9)
FAMNAME = {"deepseek": "DeepSeek", "kimi": "Kimi", "mistral": "Mistral", "meta": "Llama",
           "cohere": "Cohere", "nous": "Nous", "qwen": "Qwen", "poolside": "Poolside",
           "baidu": "ERNIE", "perceptron": "Perceptron", "allenai": "OLMo",
           "kuaishou": "Kuaishou", "google": "Google", "amazon": "Amazon",
           "nvidia": "NVIDIA", "reka": "Reka", "minimax": "MiniMax",
           "anthropic": "Anthropic", "openai": "OpenAI", "zhipu": "GLM",
           "olmo": "OLMo", "tencent": "Tencent", "ant": "Ant", "ibm": "IBM",
           "xai": "xAI", "alibaba": "Qwen", "bytedance": "Doubao",
           "microsoft": "Microsoft", "perplexity": "Perplexity"}
for a, b in zip(starts, starts[1:] + [NCOL]):
    if b - a >= 2:
        ax.text((a + b - 1) / 2, -0.85, FAMNAME.get(fams[a], fams[a].title()),
                ha="center", va="bottom", fontsize=7.5 if b - a == 2 else 8,
                color=INK2, clip_on=False)

ax.set_xticks(range(NCOL), [per[m]["name"] for m in cols], rotation=90, fontsize=6.6)
ax.set_yticks(range(NROW),
              [f'"Are you {RLABEL[t]}?"' for t in rows[:-3]] + [PLABEL[t] for t in rows[-3:]],
              fontsize=8.5)
style(ax, bottom=False)
cb = fig.colorbar(im, ax=ax, shrink=0.62, pad=0.012)
cb.set_label('% of probe records answering "yes"  (n≈10 per cell: EN+ZH × 5)',
             fontsize=8, color=INK2)
cb.outline.set_visible(False)
ax.set_title('Who says yes to whom — models with ≥2 accepted probes (gray = own family, not asked)\n'
             'Placebo rows (bottom band): pure yes-bias — a full column stripe = pan-accepter, '
             'a single hot cell = targeted residue',
             loc="left", fontsize=10.5, pad=30)

out = "analysis_scratch/generics_audit/fig_cross_grid.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
print(f"{out}  ({NROW}x{NCOL} cells, {len(set(fams))} families)")
