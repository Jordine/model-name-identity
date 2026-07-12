"""Placeholder figures for the draft post — layout previews.

Figures marked MOCK use synthetic data shaped like expected results.
Figures marked PILOT use real pilot judgments (6 models).

Palette: dataviz reference instance (validated; relief rule → direct labels).
Usage: python -m sweep.make_placeholder_figs
"""

import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "post" / "figs"
JUDG = ROOT / "results" / "judgments_partial_25flash.jsonl"

# --- reference palette ---
CAT = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ_CMAP = LinearSegmentedColormap.from_list("seqblue", SEQ)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "text.color": INK,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": False, "font.size": 9,
})
rng = random.Random(7)


def style_ax(ax, keep_bottom=True):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_visible(keep_bottom)
    ax.tick_params(length=0)


def watermark(fig, label="MOCK DATA — layout preview"):
    fig.text(0.5, 0.5, label, fontsize=26, color=INK, alpha=0.10,
             ha="center", va="center", rotation=18, weight="bold", zorder=100)


def tag(ax, label, color=MUTED):
    ax.text(1.0, 1.02, label, transform=ax.transAxes, fontsize=7.5,
            color=color, ha="right", va="bottom", style="italic")


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / name, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  {name}")


# --------------------------------------------------------------- fig 1
def fig1_headline():
    real = {"Kimi K2.5": (21, 3.8), "Llama 3.2 3B": (9, 2.7), "GPT-4o-mini": (2, 1.2),
            "GLM 4.7 Flash": (1, 0.9), "DeepSeek R1 0528": (1, 0.9), "Qwen3.5 9B": (1, 0.9)}
    fams = ["DeepSeek V4 Flash", "Step 3.7 Flash", "MiniMax M2.7", "Qwen3.6 Flash",
            "Hermes 4 70B", "ERNIE 4.5 VL", "Ling 2.6 Flash", "Nova Micro",
            "Reka Flash 3", "Ministral 3B", "Gemma 4 31B", "GPT-5.6 Sol",
            "Sonar", "Granite 4.1 8B", "Trinity Large", "Nemotron Nano 9B",
            "KAT-Coder-Pro", "Laguna XS", "Hy3", "Cogito 671B"]
    mock = {m: (max(0.3, rng.gauss(14 - i * 0.7, 3)), rng.uniform(1.5, 4)) for i, m in enumerate(fams)}
    allm = {**{f"{k} •": v for k, v in real.items()}, **mock}
    items = sorted(allm.items(), key=lambda x: -x[1][0])[:24]
    names = [k for k, _ in items][::-1]
    vals = np.array([v for _, (v, _) in items])[::-1]
    errs = np.array([e for _, (_, e) in items])[::-1]

    fig, ax = plt.subplots(figsize=(8.6, 7))
    ax.barh(names, vals, height=0.62, color=CAT[0], zorder=3)
    ax.errorbar(vals, np.arange(len(vals)), xerr=errs, fmt="none",
                ecolor=INK2, elinewidth=1, capsize=2, zorder=4)
    ax.set_xlabel("% of identity prompts with a foreign self-claim")
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    for i, v in enumerate(vals[-5:], start=len(vals) - 5):
        ax.text(v + errs[i] + 0.6, i, f"{v:.0f}%", va="center", fontsize=8, color=INK2)
    style_ax(ax)
    ax.set_title("Spontaneous misidentification rate by model (top 24 of [N])",
                 loc="left", fontsize=11, color=INK, pad=14)
    tag(ax, "• = real pilot value; all others synthetic")
    watermark(fig)
    save(fig, "fig1_headline_rates.png")


# --------------------------------------------------------------- fig 2
def fig2_language(real_rates):
    langs = ["en", "zh", "fr", "es", "ja", "ko", "ru", "vi"]
    models = ["Kimi K2.5", "Llama 3.2 3B", "GPT-4o-mini", "GLM 4.7 Flash",
              "DeepSeek R1 0528", "Qwen3.5 9B"]
    key = {"Kimi K2.5": "moonshotai/kimi-k2.5", "Llama 3.2 3B": "meta-llama/llama-3.2-3b-instruct",
           "GPT-4o-mini": "openai/gpt-4o-mini", "GLM 4.7 Flash": "z-ai/glm-4.7-flash",
           "DeepSeek R1 0528": "deepseek/deepseek-r1-0528", "Qwen3.5 9B": "qwen/qwen3.5-9b"}
    M = np.zeros((len(models), len(langs)))
    for i, m in enumerate(models):
        for j, l in enumerate(langs):
            d, n = real_rates.get(key[m], {}).get(l, (0, 0))
            M[i, j] = 100 * d / n if n else 0

    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    im = ax.imshow(M, cmap=SEQ_CMAP, vmin=0, vmax=60, aspect="auto")
    ax.set_xticks(range(len(langs)), langs)
    ax.set_yticks(range(len(models)), models)
    for i in range(len(models)):
        for j in range(len(langs)):
            v = M[i, j]
            ax.text(j, i, f"{v:.0f}" if v else "·", ha="center", va="center",
                    fontsize=8, color="#ffffff" if v > 32 else INK2)
    style_ax(ax, keep_bottom=False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("% discrepant", fontsize=8, color=INK2)
    cb.outline.set_visible(False)
    ax.set_title("Misidentification rate by prompt language — pilot",
                 loc="left", fontsize=11, color=INK, pad=12)
    tag(ax, "PILOT DATA (6 models, 120 judged records each)", color=CAT[0])
    save(fig, "fig2_language_heatmap.png")


# --------------------------------------------------------------- fig 3
def fig3_flow():
    claimants = ["Kimi (Moonshot)", "DeepSeek", "Qwen (Alibaba)", "GLM (Zhipu)",
                 "Small Llamas", "Mistral", "Step / other CN", "Long-tail labs"]
    targets = ["Claude", "ChatGPT", "Gemini", "DeepSeek", "Local-market lab", "Hallucinated / other"]
    W = np.array([
        [62, 10, 3, 2, 1, 4],
        [8, 30, 4, 0, 2, 5],
        [4, 18, 6, 9, 1, 3],
        [3, 12, 2, 6, 8, 4],
        [2, 9, 3, 2, 12, 18],
        [5, 8, 2, 1, 0, 3],
        [6, 14, 2, 8, 5, 6],
        [3, 10, 4, 2, 3, 9],
    ], dtype=float)
    W = W / W.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    left = np.zeros(len(claimants))
    for j, t in enumerate(targets):
        ax.barh(claimants[::-1], W[::-1, j], left=left[::-1] if j else None,
                height=0.62, color=CAT[j], label=t,
                edgecolor=SURFACE, linewidth=2, zorder=3)
        left += W[:, j]
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of that family's foreign identity claims (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=6,
              frameon=False, fontsize=8)
    style_ax(ax)
    ax.set_title("Who claims to be whom — composition of foreign claims per claimant family",
                 loc="left", fontsize=11, color=INK, pad=12)
    watermark(fig)
    save(fig, "fig3_identity_flow.png")


# --------------------------------------------------------------- fig 4
def fig4_cross():
    models = ["Kimi K2.5", "DeepSeek V4 Flash", "Qwen3.6 Flash", "GLM 5.2",
              "Llama 3.2 3B", "MiniMax M2.7", "Hermes 4 70B", "Step 3.7 Flash",
              "GPT-5.6 Sol", "Gemma 4 31B"]
    targets = ["ChatGPT", "Claude", "Gemini", "DeepSeek", "Qwen", "Llama", "Grok", "Kimi", "Mistral", "Doubao"]
    M = np.zeros((len(models), len(targets)))
    hot = {(0, 1): 90, (1, 0): 55, (2, 0): 30, (3, 3): 45, (4, 5): 0,
           (4, 0): 40, (5, 1): 70, (6, 5): 0, (7, 0): 35, (7, 3): 30}
    for (i, j), v in hot.items():
        M[i, j] = v
    for i in range(len(models)):
        for j in range(len(targets)):
            if M[i, j] == 0 and rng.random() < 0.25:
                M[i, j] = rng.uniform(3, 18)

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    im = ax.imshow(M, cmap=SEQ_CMAP, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(targets)), targets, fontsize=8)
    ax.set_yticks(range(len(models)), models, fontsize=8)
    for i in range(len(models)):
        for j in range(len(targets)):
            v = M[i, j]
            if v >= 25:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if v > 45 else INK2)
    style_ax(ax, keep_bottom=False)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label('% samples answering "yes"', fontsize=8, color=INK2)
    cb.outline.set_visible(False)
    ax.set_title('False-premise acceptance: "Are you {X}?" (own identity excluded)',
                 loc="left", fontsize=11, color=INK, pad=12)
    watermark(fig)
    save(fig, "fig4_cross_acceptance.png")


# --------------------------------------------------------------- fig 5
def fig5_stance():
    groups = ["Correct self-ID\n(reasoning models)", "Foreign self-ID\n(reasoning models)"]
    cats = ["asserts identity", "role-plays identity", "uncertain about identity"]
    vals = np.array([[68, 24, 8], [31, 55, 14]], dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    left = np.zeros(len(groups))
    for j, c in enumerate(cats):
        b = ax.barh(groups, vals[:, j], left=left, height=0.56, color=CAT[j],
                    label=c, edgecolor=SURFACE, linewidth=2, zorder=3)
        for i, r in enumerate(b):
            if vals[i, j] > 9:
                ax.text(left[i] + vals[i, j] / 2, i, f"{vals[i, j]:.0f}%",
                        ha="center", va="center", fontsize=8.5,
                        color="#ffffff" if j != 2 else INK)
        left += vals[:, j]
    ax.set_xlim(0, 100)
    ax.set_xticks([])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=8.5)
    style_ax(ax, keep_bottom=False)
    ax.set_title("How reasoning traces treat the model's own identity",
                 loc="left", fontsize=11, color=INK, pad=12)
    watermark(fig)
    save(fig, "fig5_reasoning_stance.png")


# --------------------------------------------------------------- fig 6
def fig6_confrontation():
    outcomes = ["recants immediately", "hedges / uncertain", "holds original claim"]
    flagged = [46, 38, 16]
    control = [22, 31, 47]

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(len(outcomes))
    w = 0.36
    ax.bar(x - w / 2, flagged, w, color=CAT[0], label="flagged models\n(pushed their TRUE name)", zorder=3)
    ax.bar(x + w / 2, control, w, color=CAT[1], label="control models\n(pushed a FALSE name)", zorder=3)
    for xi, v in zip(x - w / 2, flagged):
        ax.text(xi, v + 1.5, f"{v}%", ha="center", fontsize=8.5, color=INK2)
    for xi, v in zip(x + w / 2, control):
        ax.text(xi, v + 1.5, f"{v}%", ha="center", fontsize=8.5, color=INK2)
    ax.set_xticks(x, outcomes, fontsize=9)
    ax.set_ylim(0, 56)
    ax.set_yticks([])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False, fontsize=8.5)
    style_ax(ax)
    ax.set_title('Response to "What if I told you you\'re actually {X}?" — flagged vs control',
                 loc="left", fontsize=11, color=INK, pad=12)
    watermark(fig)
    save(fig, "fig6_confrontation_controls.png")


# --------------------------------------------------------------- real pilot rates
def pilot_language_rates():
    import sys
    sys.path.insert(0, str(ROOT))
    from sweep.analyze import canon_identity, is_self, lang_of
    reg = {m["id"]: m for m in json.loads((ROOT / "config" / "models.json").read_text())["models"]}
    rates = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    if not JUDG.exists():
        return {}
    for line in open(JUDG, encoding="utf-8"):
        j = json.loads(line)
        if j.get("judge_error") or not j.get("judgment"):
            continue
        if j["prompt_category"] in ("system_probe", "probe_cross", "probe_self", "multi_turn"):
            continue
        m = reg.get(j["model_id"])
        aliases = m["aliases"] if m else j.get("aliases", [])
        fam = m["family"] if m else j.get("family", "")
        jm = j["judgment"]
        disc = False
        for f in ("claimed_name", "claimed_creator", "reasoning_claimed_name", "reasoning_claimed_creator"):
            c = canon_identity(jm.get(f))
            if c and not is_self(c, fam, aliases, j["expected_identity"]):
                disc = True
        lang = lang_of(j["prompt_category"])
        cell = rates[j["model_id"]][lang]
        cell[1] += 1
        if disc:
            cell[0] += 1
    return {mid: {l: tuple(v) for l, v in d.items()} for mid, d in rates.items()}


if __name__ == "__main__":
    print("building figures ->", FIGS)
    fig1_headline()
    fig2_language(pilot_language_rates())
    fig3_flow()
    fig4_cross()
    fig5_stance()
    fig6_confrontation()
