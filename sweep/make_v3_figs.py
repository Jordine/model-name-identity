"""All v3 figures — clean adjudicated data, complete models only, Wilson CIs.

Produces post/figs_v3/*.png. Run after judge+adjudicate.
Usage: python -m sweep.make_v3_figs
"""

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

from .analyze import load, lang_of, FAMILY_SELF
from .make_real_figs import foreign_claims, CREATOR_TO_BRAND
from .prompts import prompts_for_model, CORE, LANGS, prompt_id

# the exactly-shared identity+creator battery (8 prompts × 8 languages) — excludes
# legacy EN/ZH prompt variants left over from the v2 reuse, so languages are matched
BATTERY_CORE = {prompt_id(k, lang) for k, (role, _) in CORE.items()
                if role in ("identity", "creator") for lang in LANGS}

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "post" / "figs_v3"
CAT = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "font.family": "DejaVu Sans",
    "text.color": INK, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 9})
SEQ_CMAP = LinearSegmentedColormap.from_list("s", SEQ)


def wilson(d, n):
    if n == 0:
        return 0.0, 0.0
    p = d / n
    z = 1.96
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, (p - (c - h)) * 100), max(0.0, ((c + h) - p) * 100)


def cluster_ci(units, reps=2000, seed=12345):
    """Cluster-bootstrap 95% CI (pp) around the pooled rate.

    `units` = list of (d, n) per independent cluster (a prompt-cell of 5 samples
    for per-model bars; a whole model for pooled figures). Resamples clusters
    with replacement so within-cluster correlation (90% of 5-sample cells are
    unanimous) is preserved — record-level Wilson understates width ~2x. Returns
    (lo_err, hi_err) in percentage points, drop-in for wilson()."""
    units = [(d, n) for d, n in units if n > 0]
    tot_d = sum(d for d, n in units)
    tot_n = sum(n for d, n in units)
    if tot_n == 0:
        return 0.0, 0.0
    point = 100 * tot_d / tot_n
    k = len(units)
    if k < 2:
        return wilson(tot_d, tot_n)     # single cluster → nothing to resample
    rng = random.Random(seed)
    boots = []
    for _ in range(reps):
        bd = bn = 0
        for _ in range(k):
            d, n = units[rng.randrange(k)]
            bd += d
            bn += n
        boots.append(100 * bd / bn if bn else 0.0)
    boots.sort()
    lo = boots[int(0.025 * reps)]
    hi = boots[int(0.975 * reps)]
    return max(0.0, point - lo), max(0.0, hi - point)


def style(ax, bottom=True):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_visible(bottom)
    ax.tick_params(length=0)


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / name, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  {name}")


def is_core(cat):
    return cat.startswith("direct_") or cat.startswith("creator_") or cat.startswith("casual_")


def is_identity(cat):
    return cat.startswith("direct_") or cat.startswith("creator_")


def complete_models(reg, hyg):
    ok = set()
    for l in open(ROOT / "results" / "main_sweep.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if not r.get("error"):
            ok.add(r["resume_key"])
    out = set()
    for m in reg.values():
        if hyg.get(m["id"], {}).get("exclude"):
            continue
        tgt = {f"{m['id']}::{p['id']}::{p['sample_idx']}" for p in prompts_for_model(m)}
        if tgt and len(tgt & ok) / len(tgt) >= 0.95:
            out.add(m["id"])
    return out


def gather():
    reg = {m["id"]: m for m in json.loads((ROOT / "config" / "models.json").read_text())["models"]}
    hyg = json.loads((ROOT / "config" / "provider_hygiene.json").read_text())
    complete = complete_models(reg, hyg)
    rows = [j for j in load() if j["model_id"] in complete]
    per = defaultdict(lambda: {"n": 0, "d": 0, "claims": Counter(), "lang": defaultdict(lambda: [0, 0]),
                               "cells": defaultdict(lambda: [0, 0]),
                               "cross_yes": Counter(), "cross_n": Counter(),
                               "plac_yes": 0, "plac_n": 0, "name": ""})
    for j in rows:
        m = per[j["model_id"]]
        m["name"] = reg[j["model_id"]]["name"]
        cat = j["prompt_category"]
        if cat == "probe_placebo":
            y = (j["judgment"] or {}).get("answered_yes")
            m["plac_n"] += 1
            m["plac_yes"] += 1 if y is True else 0
            continue
        if cat == "probe_cross":
            t = j["prompt_id"].replace("zh_cross_", "").replace("en_cross_", "").replace("cross_", "")
            if t in FAMILY_SELF.get(j.get("family", ""), set()):
                continue
            m["cross_n"][t] += 1
            if (j["judgment"] or {}).get("answered_yes") is True:
                m["cross_yes"][t] += 1
            continue
        if cat == "system_probe":
            continue
        if not is_identity(cat):
            continue  # identity+creator only for the drift rate (casual reported separately)
        if j["prompt_id"] not in BATTERY_CORE:
            continue  # exclude legacy EN/ZH variants → truly balanced across languages
        fc = foreign_claims(j)
        m["n"] += 1
        l = lang_of(cat)
        m["lang"][l][1] += 1
        cell = m["cells"][j["prompt_id"]]   # prompt-cell = 5 samples (bootstrap cluster)
        cell[1] += 1
        if fc:
            m["d"] += 1
            m["lang"][l][0] += 1
            cell[0] += 1
            brands = {("other/unlisted" if c.startswith("other:") else CREATOR_TO_BRAND.get(c, c)) for c in fc}
            for b in brands:
                m["claims"][b] += 1
    return reg, {k: v for k, v in per.items() if v["n"] >= 40}


# ------------------------------------------------------------------ figures
def fig_all_models(reg, per):
    total = len(per)
    items = sorted([kv for kv in per.items() if kv[1]["d"] > 0], key=lambda x: x[1]["d"] / x[1]["n"])
    n = len(items)
    names = [v["name"] for _, v in items]
    rates = np.array([100 * v["d"] / v["n"] for _, v in items])
    errs = np.array([cluster_ci(list(v["cells"].values())) for _, v in items]).T
    fams = [reg[mid]["family"] for mid, _ in items]
    fam_drift = defaultdict(float)
    for (mid, v), f in zip(items, fams):
        fam_drift[f] += v["d"]
    top = [f for f, _ in sorted(fam_drift.items(), key=lambda x: -x[1])[:8]]
    fc = {f: CAT[i] for i, f in enumerate(top)}
    colors = [fc.get(f, "#b9b7b0") for f in fams]
    fig, ax = plt.subplots(figsize=(9.5, 0.185 * n + 1.8))
    ax.barh(np.arange(n), rates, height=0.66, color=colors, zorder=3)
    ax.errorbar(rates, np.arange(n), xerr=errs, fmt="none", ecolor=INK2, elinewidth=0.7, capsize=1.5, zorder=4)
    ax.set_yticks(np.arange(n), names, fontsize=6.2)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("% of identity/creator prompts where the model gave a mismatched name (cluster-bootstrap 95% CI)")
    ax.set_ylabel("model (sorted)")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(color=fc[f], label=f) for f in top] + [Patch(color="#b9b7b0", label="other")],
              loc="lower right", frameon=False, fontsize=8, title="family (top-8 by mismatch)", title_fontsize=8)
    style(ax)
    ax.set_title(f"Name-mismatch rate — the {n} models with at least one mismatch "
                 f"(of {total} tested; the other {total - n} never mismatched)", loc="left", fontsize=12, pad=12)
    save(fig, "fig_all_models.png")


def fig_lang_agg(per):
    agg = defaultdict(lambda: [0, 0])
    for v in per.values():
        for l, (d, t) in v["lang"].items():
            agg[l][0] += d
            agg[l][1] += t
    langs = ["en", "zh", "ja", "ko", "ru", "fr", "es", "vi"]
    vals = [100 * agg[l][0] / agg[l][1] for l in langs]
    errs = np.array([cluster_ci([tuple(v["lang"][l]) for v in per.values() if v["lang"].get(l, [0, 0])[1] > 0])
                     for l in langs]).T
    fig, ax = plt.subplots(figsize=(7.8, 3.6))
    ax.bar(langs, vals, 0.62, color=CAT[0], zorder=3)
    ax.errorbar(range(len(langs)), vals, yerr=errs, fmt="none", ecolor=INK2, elinewidth=1, capsize=3, zorder=4)
    for i, (l, vv) in enumerate(zip(langs, vals)):
        ax.text(i, vv + errs[1][i] + 0.3, f"{vv:.1f}%", ha="center", fontsize=8.5, color=INK2)
    ax.set_xticks(range(len(langs)), [f"{l}\n(n={agg[l][1]:,})" for l in langs], fontsize=8.5)
    ax.set_ylim(0, max(vals) + 3)
    ax.set_yticks([])
    style(ax)
    ax.set_title("Foreign-claim rate by prompt language — balanced battery, all models pooled (model-clustered 95% CIs)",
                 loc="left", fontsize=11, pad=12)
    save(fig, "fig_lang_agg.png")


def fig_lang_heatmap(reg, per):
    # top drifting models with a per-language story, + a couple frontier Claudes
    pick = sorted(per, key=lambda m: -per[m]["d"] / per[m]["n"])[:16]
    for extra in ("anthropic/claude-opus-4.8", "anthropic/claude-sonnet-4.6", "moonshotai/kimi-k2"):
        if extra in per and extra not in pick:
            pick.append(extra)
    langs = ["en", "zh", "ja", "ko", "ru", "fr", "es", "vi"]
    M = np.full((len(pick), len(langs)), np.nan)
    for i, mid in enumerate(pick):
        for jx, l in enumerate(langs):
            d, t = per[mid]["lang"].get(l, (0, 0))
            if t:
                M[i, jx] = 100 * d / t
    fig, ax = plt.subplots(figsize=(8.0, 0.34 * len(pick) + 1.6))
    im = ax.imshow(np.nan_to_num(M), cmap=SEQ_CMAP, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(langs)), langs)
    ax.set_yticks(range(len(pick)), [per[m]["name"] for m in pick], fontsize=8)
    for i in range(len(pick)):
        for jx in range(len(langs)):
            v = M[i, jx]
            if not np.isnan(v):
                ax.text(jx, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if v > 55 else INK2)
    style(ax, bottom=False)
    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("% of records with a mismatched name", fontsize=8, color=INK2)
    cb.outline.set_visible(False)
    ax.set_title("Per-language name mismatch — heaviest models + frontier Claudes (each cell = % of that model's 40 identity/creator records in that language)",
                 loc="left", fontsize=10, pad=12)
    save(fig, "fig_lang_heatmap.png")


def _scrubout_one(mids, color, title, fname, strip):
    xs, ys, es, tl = [], [], [], []
    for mid in mids:
        if mid not in per_cache:
            continue
        v = per_cache[mid]
        xs.append(len(xs))
        ys.append(100 * v["d"] / v["n"])
        es.append(cluster_ci(list(v["cells"].values())))
        tl.append(v["name"])
    if not xs:
        return
    es = np.array(es).T
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.errorbar(xs, ys, yerr=es, fmt="-o", color=color, lw=2, ms=7, capsize=3, elinewidth=1, zorder=3)
    for i, (xx, yy, t) in enumerate(zip(xs, ys, tl)):
        up = (i % 2 == 0)
        ax.annotate(t, (xx, yy), textcoords="offset points", xytext=(0, 11 if up else -20),
                    fontsize=7, color=INK2, ha="center", va="bottom" if up else "top")
    ax.set_ylabel("% of prompts with a mismatched name", fontsize=9)
    ax.set_xticks([])
    ax.set_xlabel("release order →")
    ax.set_ylim(-6, max(ys) + 14)
    ax.set_xlim(-0.5, len(xs) - 0.5)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    style(ax)
    ax.set_title(title, loc="left", fontsize=11, pad=12)
    save(fig, fname)


per_cache = {}


def fig_scrubout(reg, per):
    global per_cache
    per_cache = per
    _scrubout_one(["moonshotai/kimi-k2", "moonshotai/kimi-k2-0905", "moonshotai/kimi-k2-thinking",
                   "moonshotai/kimi-k2.5", "moonshotai/kimi-k2.6", "moonshotai/kimi-k2.7-code"],
                  CAT[0], "The scrub-out (Kimi K2 line): name-mismatch rate across releases (cluster-bootstrap 95% CIs)",
                  "fig_scrubout_kimi.png", "Kimi ")
    _scrubout_one(["qwen/qwen-2.5-72b-instruct", "qwen/qwen-2.5-7b-instruct", "qwen/qwen3-235b-a22b",
                   "qwen/qwen3-max-thinking", "qwen/qwen3.5-397b-a17b", "qwen/qwen3.7-max"],
                  CAT[3], "The scrub-out (Qwen 2.5 → 3.x): name-mismatch rate across releases (cluster-bootstrap 95% CIs)",
                  "fig_scrubout_qwen.png", "")


def fig_cross(reg, per):
    # placebo-corrected acceptance per target, pooled
    targets = ["chatgpt", "claude", "gemini", "deepseek", "qwen", "kimi", "llama", "grok", "mistral", "doubao"]
    ty, tn = Counter(), Counter()
    py = pn = 0
    for v in per.values():
        for t in targets:
            ty[t] += v["cross_yes"].get(t, 0)
            tn[t] += v["cross_n"].get(t, 0)
        py += v["plac_yes"]
        pn += v["plac_n"]
    plac = 100 * py / pn if pn else 0
    rates = [(100 * ty[t] / tn[t] if tn[t] else 0) for t in targets]
    errs = np.array([cluster_ci([(v["cross_yes"].get(t, 0), v["cross_n"].get(t, 0))
                                 for v in per.values() if v["cross_n"].get(t, 0) > 0]) for t in targets]).T
    order = np.argsort(rates)[::-1]
    labels = [["ChatGPT", "Claude", "Gemini", "DeepSeek", "Qwen", "Kimi", "Llama", "Grok", "Mistral", "Doubao"][i] for i in order]
    rr = [rates[i] for i in order]
    ee = errs[:, order]
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.bar(range(len(rr)), rr, 0.62, color=CAT[0], zorder=3)
    ax.errorbar(range(len(rr)), rr, yerr=ee, fmt="none", ecolor=INK2, elinewidth=1, capsize=3, zorder=4)
    ax.axhline(plac, color=CAT[5], lw=1.6, ls="--", zorder=5,
               label=f"placebo baseline ({plac:.1f}% — fictional 'Are you Meridian-4/Solace/Cobalt?')")
    ax.set_xticks(range(len(rr)), labels, fontsize=9)
    ax.set_ylabel('% answering "yes"')
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    style(ax)
    ax.set_title('False-premise acceptance "Are you X?" vs placebo baseline (pooled; 95% CIs resampled over individual models)',
                 loc="left", fontsize=11, pad=12)
    save(fig, "fig_cross.png")


def fig_flow(reg, per):
    CB = {"anthropic": "claude", "openai": "chatgpt", "google": "gemini", "alibaba": "qwen",
          "meta": "llama", "moonshot": "kimi", "nvidia": "nvidia", "tencent": "hunyuan"}
    fam_claims = defaultdict(Counter)
    for mid, v in per.items():
        for c, k in v["claims"].items():
            c = "other/unlisted" if c.startswith("other:") else CB.get(c, c)
            fam_claims[reg[mid]["family"]][c] += k
    fams = sorted(fam_claims, key=lambda f: -sum(fam_claims[f].values()))[:10]
    cols = ["claude", "chatgpt", "qwen", "gemini", "nvidia", "deepseek", "llama", "other/unlisted"]
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    left = np.zeros(len(fams))
    for k, c in enumerate(cols):
        vals = []
        for f in fams:
            tot = sum(fam_claims[f].values())
            v = fam_claims[f].get(c, 0)
            if c == "other/unlisted":
                v = tot - sum(fam_claims[f].get(x, 0) for x in cols[:-1])
            vals.append(100 * v / tot if tot else 0)
        ax.barh(fams[::-1], np.array(vals)[::-1], left=left[::-1], height=0.62, color=CAT[k % 8],
                label=c, edgecolor=SURFACE, linewidth=2, zorder=3)
        left += np.array(vals)
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of that family's mismatched-name claims (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=8, frameon=False, fontsize=8)
    style(ax)
    ax.set_title("Which name each family gives instead — composition of mismatches per family", loc="left", fontsize=11, pad=12)
    save(fig, "fig_flow.png")


def fig_stance(reg, per):
    complete = set(per)
    groups = {"correct": Counter(), "foreign": Counter()}
    for j in load():
        if j["model_id"] not in complete or not j.get("had_reasoning"):
            continue
        if not is_identity(j["prompt_category"]):
            continue
        st = (j["judgment"] or {}).get("reasoning_identity_stance")
        if st not in ("asserts", "role_play", "uncertain"):
            continue
        grp = "foreign" if foreign_claims(j) else "correct"
        groups[grp][st] += 1
    cats = ["asserts", "role_play", "uncertain"]
    labs = ["asserts identity", "role-plays identity", "uncertain"]
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    x = np.arange(3)
    w = 0.38
    for k, g in enumerate(("correct", "foreign")):
        nn = sum(groups[g].values())
        vals = np.array([100 * groups[g].get(c, 0) / nn for c in cats])
        errs = np.array([wilson(groups[g].get(c, 0), nn) for c in cats]).T
        ax.bar(x + (k - 0.5) * w, vals, w * 0.92, color=CAT[k], zorder=3,
               label=f"{'matching (own) name' if g=='correct' else 'mismatched name'} in the reasoning trace (n={nn:,})")
        ax.errorbar(x + (k - 0.5) * w, vals, yerr=errs, fmt="none", ecolor=INK2, elinewidth=1, capsize=2, zorder=4)
        for xi, v in zip(x + (k - 0.5) * w, vals):
            ax.text(xi, v + 3.5, f"{v:.0f}%", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x, labs)
    ax.set_ylim(0, 110)
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    style(ax)
    ax.set_title("How reasoning traces state the model's own identity — across the 72 models that expose a trace",
                 loc="left", fontsize=10.5, pad=12)
    save(fig, "fig_stance.png")


def fig_family_panels(reg, per):
    TARGETS = ["chatgpt", "claude", "gemini", "deepseek", "qwen", "kimi", "llama",
               "mistral", "glm", "grok", "nvidia", "doubao", "other lab", "novel/unrec."]
    CB = {"anthropic": "claude", "openai": "chatgpt", "google": "gemini", "alibaba": "qwen",
          "meta": "llama", "moonshot": "kimi", "nvidia": "nvidia", "tencent": "hunyuan"}
    def to_col(c):
        if c.startswith("other:") or c == "other/unlisted":
            return "novel/unrec."
        c = CB.get(c, c)
        return c if c in TARGETS else "other lab"
    (FIGS / "family").mkdir(parents=True, exist_ok=True)
    fam_models = defaultdict(list)
    for mid in per:
        fam_models[reg[mid]["family"]].append(mid)
    manifest = []
    for fam, mids in fam_models.items():
        rows = []
        for mid in mids:
            v = per[mid]
            if v["d"] == 0:
                continue
            vec = np.zeros(len(TARGETS))
            for c, k in v["claims"].items():
                vec[TARGETS.index(to_col(c))] += k
            rows.append((mid, v["d"], v["n"], 100 * vec / v["n"]))
        if len(rows) < 1:
            continue
        rows.sort(key=lambda r: -r[1] / r[2])
        M = np.array([r[3] for r in rows])
        vmax = max(12, float(M.max()))
        nrow = len(rows)
        fig, ax = plt.subplots(figsize=(8.2, 0.42 * nrow + 1.8))
        im = ax.imshow(M, cmap=SEQ_CMAP, vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(TARGETS)), TARGETS, fontsize=8, rotation=35, ha="right")
        ax.set_yticks(range(nrow), [per[r[0]]["name"] for r in rows], fontsize=9)
        for i in range(nrow):
            for j in range(len(TARGETS)):
                if M[i, j] <= 0:
                    continue  # truly-zero cells stay blank/white
                txt = "<1" if M[i, j] < 1 else f"{M[i,j]:.0f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if M[i, j] > 0.6 * vmax else INK2)
        style(ax, bottom=False)
        ax.set_title(f"{fam} — which name each model gives instead (each cell = % of that model's identity/creator records)",
                     loc="left", fontsize=10, pad=10)
        fn = f"family/fam_{fam}.png"
        fig.savefig(FIGS / fn, dpi=200, bbox_inches="tight", facecolor=SURFACE)
        plt.close(fig)
        manifest.append({"file": fn, "family": fam, "drift": sum(r[1] for r in rows), "models": nrow})
    manifest.sort(key=lambda p: -p["drift"])
    (FIGS / "family" / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"  {len(manifest)} family panels")


if __name__ == "__main__":
    reg, per = gather()
    print(f"{len(per)} complete models")
    fig_all_models(reg, per)
    fig_lang_agg(per)
    fig_lang_heatmap(reg, per)
    fig_scrubout(reg, per)
    fig_cross(reg, per)
    fig_flow(reg, per)
    fig_stance(reg, per)
    fig_family_panels(reg, per)
