"""All figures — clean adjudicated data, complete models only, Wilson CIs.

Produces figures/*.png. Run after judge+adjudicate.
Usage: python -m sweep.make_figs
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

from .analyze import (load, lang_of, FAMILY_SELF, canon_identity, is_self,
                      foreign_claims, CREATOR_TO_BRAND, open_lines)
from .prompts import prompts_for_model, CORE, LANGS, prompt_id

# raw-weights models run locally (clean condition) that AREN'T already in the API
# set — folded into the same figures as ordinary rows. Skips local dupes of
# API-tested sizes (qwen3-8b/14b/32b/35b already covered via OpenRouter).
LOCAL_MODELS = {
    "openai/gpt-oss-20b": ("GPT-OSS 20B", "openai", ["gpt-oss", "openai", "chatgpt", "gpt"]),
    "openai/gpt-oss-120b": ("GPT-OSS 120B", "openai", ["gpt-oss", "openai", "chatgpt", "gpt"]),
    "allenai/Olmo-3-7B-Instruct": ("OLMo 3 7B Instruct", "olmo", ["olmo", "ai2", "allenai", "olmo 3", "allen institute"]),
    "allenai/Olmo-3-7B-Think": ("OLMo 3 7B Think", "olmo", ["olmo", "ai2", "allenai", "olmo 3", "allen institute"]),
    "allenai/Olmo-3.1-32B-Instruct": ("OLMo 3.1 32B Instruct", "olmo", ["olmo", "ai2", "allenai", "olmo 3", "allen institute"]),
    "Qwen/Qwen3-0.6B": ("Qwen3 0.6B", "qwen", ["qwen", "qwen3", "tongyi", "通义千问", "alibaba"]),
    "Qwen/Qwen3-1.7B": ("Qwen3 1.7B", "qwen", ["qwen", "qwen3", "tongyi", "通义千问", "alibaba"]),
    "Qwen/Qwen3-4B": ("Qwen3 4B", "qwen", ["qwen", "qwen3", "tongyi", "通义千问", "alibaba"]),
    "Qwen/Qwen3.5-0.8B": ("Qwen3.5 0.8B", "qwen", ["qwen", "qwen3", "tongyi", "通义千问", "alibaba"]),
    "Qwen/Qwen3.5-2B": ("Qwen3.5 2B", "qwen", ["qwen", "qwen3", "tongyi", "通义千问", "alibaba"]),
}

# the exactly-shared identity+creator battery (8 prompts × 8 languages) — excludes
# legacy EN/ZH prompt variants left over from prompt-id reuse, so languages are matched
BATTERY_CORE = {prompt_id(k, lang) for k, (role, _) in CORE.items()
                if role in ("identity", "creator") for lang in LANGS}

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "figures"
CAT = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
# claimed-identity palette — ONE color per identity across every figure that colors
# by "who the model claims to be" (fig_flow here; fig_coherence/fig_cutoff in explain)
IDCOLOR = {"claude": "#d97a34", "deepseek": "#2a78d6", "qwen": "#7b4fd0", "chatgpt": "#1baf7a",
           "gemini": "#c94f9c", "llama": "#4aa0a0", "nvidia": "#5b8c1f", "kimi": "#e0a020",
           "mistral": "#b0402a"}
# sequential floor sits slightly OFF the surface color so a measured-0 cell reads
# as data, not as missing/background
SEQ = ["#f2f4f8", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
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
    for l in open_lines(ROOT / "results" / "main_sweep.jsonl"):
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
    per = {k: v for k, v in per.items() if v["n"] >= 40}
    add_local(reg, per)
    return reg, per


def _local_genuine():
    keep = set()
    p = ROOT / "results" / "adjudications_local.jsonl"
    if p.exists():
        for l in open(p, encoding="utf-8"):
            try:
                d = json.loads(l)
            except json.JSONDecodeError:
                continue
            if d.get("verdict") == "genuine_foreign":
                keep.add(d["adj_key"])
    return keep


def add_local(reg, per):
    """Fold the local raw-weights models (clean condition, adjudicated) into `per`
    as ordinary rows, computed exactly like gather() does for API models."""
    jpath = ROOT / "results_local" / "judgments_clean.jsonl"
    if not jpath.exists():
        return
    keep = _local_genuine()
    for mid, (name, fam, al) in LOCAL_MODELS.items():
        reg[mid] = {"id": mid, "name": name, "family": fam, "expected_identity": name, "aliases": al}
    acc = defaultdict(lambda: {"n": 0, "d": 0, "claims": Counter(), "lang": defaultdict(lambda: [0, 0]),
                               "cells": defaultdict(lambda: [0, 0]), "cross_yes": Counter(),
                               "cross_n": Counter(), "plac_yes": 0, "plac_n": 0, "name": ""})
    for l in open(jpath, encoding="utf-8"):
        j = json.loads(l)
        if not j.get("judgment"):
            continue
        if j["resume_key"].split("::")[-1] != "clean":   # raw-weights read
            continue
        mid = j["model_id"]
        if mid not in LOCAL_MODELS:
            continue
        name, fam, al = LOCAL_MODELS[mid]
        m = acc[mid]; m["name"] = name
        cat = j["prompt_category"]; jm = j["judgment"] or {}
        if cat == "probe_placebo":
            m["plac_n"] += 1
            m["plac_yes"] += 1 if jm.get("answered_yes") is True else 0
            continue
        if cat == "probe_cross":
            t = j["prompt_id"].replace("zh_cross_", "").replace("en_cross_", "").replace("cross_", "")
            if t in FAMILY_SELF.get(fam, {fam}):
                continue
            m["cross_n"][t] += 1
            if jm.get("answered_yes") is True:
                m["cross_yes"][t] += 1
            continue
        if not is_identity(cat) or j["prompt_id"] not in BATTERY_CORE:
            continue
        cn = canon_identity(jm.get("claimed_name")); cc = canon_identity(jm.get("claimed_creator"))
        foreign = {c for c in (cn, cc) if c and not is_self(c, fam, al, name)}
        drift = bool(foreign) and (f"{j['resume_key']}::t0" in keep)
        m["n"] += 1
        l_ = lang_of(cat); m["lang"][l_][1] += 1
        cell = m["cells"][j["prompt_id"]]; cell[1] += 1
        if drift:
            m["d"] += 1; m["lang"][l_][0] += 1; cell[0] += 1
            for b in {("other/unlisted" if c.startswith("other:") else CREATOR_TO_BRAND.get(c, c)) for c in foreign}:
                m["claims"][b] += 1
    for mid, m in acc.items():
        if m["n"] >= 40:
            per[mid] = m
    print(f"  folded in {sum(1 for mid in acc if acc[mid]['n']>=40)} local raw-weights models")


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
    fig, ax = plt.subplots(figsize=(10.2, 0.205 * n + 1.8))
    ax.barh(np.arange(n), rates, height=0.66, color=colors, zorder=3)
    ax.errorbar(rates, np.arange(n), xerr=errs, fmt="none", ecolor=INK2, elinewidth=0.7, capsize=1.5, zorder=4)
    ax.set_yticks(np.arange(n), names, fontsize=7.6)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("% of short-question responses with a mismatched name (cluster-bootstrap 95% CI)")
    ax.set_ylabel("model (sorted)")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(color=fc[f], label=FAMILY_DISPLAY.get(f, f.title())) for f in top]
              + [Patch(color="#b9b7b0", label="other")],
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
    ax.set_ylim(0, max(v + e for v, e in zip(vals, errs[1])) + 2)
    ax.set_yticks([])
    style(ax)
    ax.set_title("Name-mismatch rate by prompt language — the short-question battery, all models pooled (model-clustered 95% CIs)",
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
    cb.set_label("% of responses with a mismatched name", fontsize=8, color=INK2)
    cb.outline.set_visible(False)
    # no in-image title (it forced a wider tight-bbox and shrank the heatmap);
    # titling happens in the post text outside the image
    save(fig, "fig_lang_heatmap.png")


def _scrubout_labels(fig, ax, xs, ys, es, tl, segs, nbr, force_below=()):
    """Deterministic point labels that clear the line, the CI whiskers and each
    other. `segs` = marker-index pairs actually joined by a segment (same-event
    markers share an x and are NOT joined); `nbr` = per-marker (left_y, right_y)
    neighbour-event heights (None at the ends). Each label sits to the RIGHT of
    its point (LEFT for the final event) so the vertical whisker never strikes
    it; it goes ABOVE when the segment on the label's side falls/stays flat and
    BELOW when it rises; a below-label that would leave the axes bottom is
    flipped above (switching side if that one is free); any residual overlap
    (measured with real text extents) is nudged away. Labels named in
    `force_below` are pinned below their point (the flip is skipped) — for
    adjacent near-zero points whose above-labels would otherwise stack."""
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    px = fig.dpi / 72.0
    tr = ax.transData.transform
    axbox = ax.get_window_extent(ren)
    P = [tr((x, y)) for x, y in zip(xs, ys)]
    obstacles = []          # marker + whisker box per point
    for i, (x, y) in enumerate(zip(xs, ys)):
        cx, cy = tr((x, y))
        w0 = tr((x, y - es[0][i]))[1]
        w1 = tr((x, y + es[1][i]))[1]
        obstacles.append((cx - 4.5 * px, w0 - 2 * px, cx + 4.5 * px, w1 + 2 * px))

    def seg_hit(bb):
        for i, j in segs:
            (x1, y1), (x2, y2) = P[i], P[j]
            if x2 < x1:
                (x1, y1), (x2, y2) = (x2, y2), (x1, y1)
            if x2 == x1:
                continue
            lo, hi = max(x1, bb[0]), min(x2, bb[2])
            if lo > hi:
                continue
            ya = y1 + (y2 - y1) * (lo - x1) / (x2 - x1)
            yb = y1 + (y2 - y1) * (hi - x1) / (x2 - x1)
            if min(ya, yb) - 2 * px < bb[3] and max(ya, yb) + 2 * px > bb[1]:
                return True
        return False

    placed = []
    for i, (xx, yy, t) in enumerate(zip(xs, ys, tl)):
        left_y, right_y = nbr[i]
        side = 1 if right_y is not None else -1
        nb = right_y if side == 1 else left_y
        if nb is None:
            nb = yy                              # single event — no slope to dodge
        below = t in force_below or nb > yy + 1e-9
        if below and t not in force_below and tr((xx, yy))[1] - 16 * px < axbox.y0 + 3 * px:
            below = False                        # would leave the axes — go above,
            if side == 1 and left_y is not None and left_y <= yy + 1e-9:
                side = -1                        # on the other side if that one is flat
        dx, dy = 9 * side, (-7 if below else 4)
        for k in range(6):
            a = ax.annotate(t, (xx, yy), textcoords="offset points", xytext=(dx, dy),
                            fontsize=7, color=INK2, ha="left" if side == 1 else "right",
                            va="top" if below else "bottom", zorder=5)
            b = a.get_window_extent(ren)
            bb = (b.x0 - 1, b.y0 - 1, b.x1 + 1, b.y1 + 1)
            hit = seg_hit(bb) or any(bb[0] < o[2] and bb[2] > o[0] and bb[1] < o[3] and bb[3] > o[1]
                                     for o in placed + obstacles)
            if not hit or k == 5:
                break
            a.remove()
            dy += -11 if below else 11
        if abs(dy) >= 15:      # nudged well away from its point — add a thin leader
            a.remove()
            ax.annotate(t, (xx, yy), textcoords="offset points", xytext=(dx, dy),
                        fontsize=7, color=INK2, ha="left" if side == 1 else "right",
                        va="top" if below else "bottom", zorder=5,
                        arrowprops=dict(arrowstyle="-", lw=0.55, color=BASE, shrinkA=3, shrinkB=2))
        placed.append(bb)


def _scrubout_one(events, color, title, fname, strip, subtitle=None, force_below=()):
    """events = [(release_date, [mid, …])] in true release order. Same-day
    siblings share ONE x position (two markers, no segment between them — two
    sizes shipped the same day is one release event, not a temporal step).
    Events are evenly spaced with dated tick labels: a true time axis would
    crush the dense 2026 releases into each other; the dates keep the even
    spacing honest. See analysis_scratch/generics_audit/release_dates.md."""
    evs = [(date, [m for m in mids if m in per_cache]) for date, mids in events]
    evs = [(date, mids) for date, mids in evs if mids]
    if not evs:
        return
    xs, ys, es, tl, idxs = [], [], [], [], []
    for xi, (date, mids) in enumerate(evs):
        idxs.append([])
        for mid in mids:
            v = per_cache[mid]
            idxs[xi].append(len(xs))
            xs.append(xi)
            ys.append(100 * v["d"] / v["n"])
            es.append(cluster_ci(list(v["cells"].values())))
            tl.append(v["name"].replace(strip, "") if strip else v["name"])
    es = np.array(es).T
    ne = len(evs)
    # line = event-to-event segments only; a multi-model event fans out to the next
    segs = [(i, j) for k in range(ne - 1) for i in idxs[k] for j in idxs[k + 1]]
    # neighbour-event heights steer each label above/below (first marker = anchor)
    nbr = [(ys[idxs[xi - 1][0]] if xi > 0 else None,
            ys[idxs[xi + 1][0]] if xi < ne - 1 else None) for xi in xs]
    fig, ax = plt.subplots(figsize=(max(7.0, 1.8 + 0.66 * ne), 3.8))
    for i, j in segs:
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color=color, lw=2, zorder=2)
    ax.errorbar(xs, ys, yerr=es, fmt="o", color=color, ms=7, capsize=3, elinewidth=1, zorder=3)
    ax.set_ylabel("% of responses with a mismatched name\n(cluster-bootstrap 95% CI)", fontsize=9)
    ax.set_xticks(range(ne), [date for date, _ in evs], fontsize=7.3, rotation=30,
                  ha="right", rotation_mode="anchor")
    ax.set_xlabel("release date (events evenly spaced — spacing not proportional to elapsed time)", fontsize=8.5)
    ax.set_ylim(-6, max(ys) + 14)
    ax.set_xlim(-0.5, ne - 0.5)
    ax.set_yticks([t for t in ax.get_yticks() if 0 <= t <= max(ys) + 14])   # no phantom negative ticks
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    style(ax)
    ax.set_title(title, loc="left", fontsize=11, pad=24 if subtitle else 12)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")
    _scrubout_labels(fig, ax, xs, ys, es, tl, segs, nbr, force_below=force_below)
    save(fig, fname)


per_cache = {}

# Release EVENTS, verified against primary sources (release_dates.md; Anthropic
# newsroom/TechCrunch/Axios for the 2026 Claudes). models.json `created` is an
# OpenRouter LISTING epoch, not a release date — it lags by up to ~4 weeks
# (qwen-2.5-7b) and 2 days (opus-5), so nothing here may sort by `created`.
KIMI_EVENTS = [
    ("2025-07-11", ["moonshotai/kimi-k2"]),
    ("2025-09-05", ["moonshotai/kimi-k2-0905"]),
    ("2025-11-06", ["moonshotai/kimi-k2-thinking"]),
    ("2026-01-27", ["moonshotai/kimi-k2.5"]),
    ("2026-04-20", ["moonshotai/kimi-k2.6"]),
    ("2026-06-12", ["moonshotai/kimi-k2.7-code"]),
]
QWEN_EVENTS = [
    # ONE flagship per generation (2.5 → 3 → 3.5 → 3.6 → 3.7), the largest
    # analyzed model of each. The 2.5 launch's same-day 7B sibling is dropped
    # (one point per generation); Qwen3 Max / Max Thinking sit off the numbered
    # ladder and also measure 0.0% — same reading as the 235B point either way.
    # No Qwen3.5-Max exists in the pool, so 397B A17B is the 3.5 flagship.
    ("2024-09-19", ["qwen/qwen-2.5-72b-instruct"]),
    ("2025-04-29", ["qwen/qwen3-235b-a22b"]),
    ("2026-02-16", ["qwen/qwen3.5-397b-a17b"]),
    # 3.6 Max Preview: announced + live on Alibaba Bailian 2026-04-20 (datanorth.ai
    # write-up dated 04-21); the 2026-04-27 OpenRouter epoch is the usual listing lag.
    ("2026-04-20", ["qwen/qwen3.6-max-preview"]),
    ("2026-05-19", ["qwen/qwen3.7-max"]),
]
OPUS_EVENTS = [
    ("2025-05-22", ["anthropic/claude-opus-4"]),
    ("2025-08-05", ["anthropic/claude-opus-4.1"]),
    ("2025-11-24", ["anthropic/claude-opus-4.5"]),
    ("2026-02-05", ["anthropic/claude-opus-4.6"]),
    ("2026-04-16", ["anthropic/claude-opus-4.7"]),
    ("2026-05-28", ["anthropic/claude-opus-4.8"]),
    ("2026-07-24", ["anthropic/claude-opus-5"]),
]
SONNET_EVENTS = [
    ("2025-05-22", ["anthropic/claude-sonnet-4"]),
    ("2025-09-29", ["anthropic/claude-sonnet-4.5"]),
    ("2026-02-17", ["anthropic/claude-sonnet-4.6"]),
    ("2026-06-30", ["anthropic/claude-sonnet-5"]),
]
# Combined frontier-Claude timeline — Opus + Sonnet interleaved by REAL release
# date (hardcoded + verified; the `created`-sorted version drew a fake step
# between the same-day Opus 4 / Sonnet 4 launch). Lab-level view: two spikes.
CLAUDE_EVENTS = [
    ("2025-05-22", ["anthropic/claude-opus-4", "anthropic/claude-sonnet-4"]),   # Claude 4 launch — one event
    ("2025-08-05", ["anthropic/claude-opus-4.1"]),
    ("2025-09-29", ["anthropic/claude-sonnet-4.5"]),
    ("2025-11-24", ["anthropic/claude-opus-4.5"]),
    ("2026-02-05", ["anthropic/claude-opus-4.6"]),
    ("2026-02-17", ["anthropic/claude-sonnet-4.6"]),
    ("2026-04-16", ["anthropic/claude-opus-4.7"]),
    ("2026-05-28", ["anthropic/claude-opus-4.8"]),
    ("2026-06-30", ["anthropic/claude-sonnet-5"]),
    ("2026-07-24", ["anthropic/claude-opus-5"]),
]


def fig_scrubout(reg, per):
    # titles/notes are DESCRIPTIVE only (figure-text policy) — the reading of
    # the lines lives in the post text, not in the images
    global per_cache
    per_cache = per
    _scrubout_one(KIMI_EVENTS,
                  CAT[0], "Official-name mismatch rate by release — Kimi K2 line",
                  "fig_scrubout_kimi.png", "Kimi ")
    _scrubout_one(QWEN_EVENTS,
                  CAT[3], "Official-name mismatch rate by release — Qwen flagships",
                  "fig_scrubout_qwen.png", "",
                  subtitle="one flagship per generation (Qwen2.5 → 3 → 3.5 → 3.6 → 3.7) — the generation's largest analyzed model")
    _scrubout_one(OPUS_EVENTS,
                  CAT[4], "Official-name mismatch rate by release — Claude Opus line",
                  "fig_scrubout_claude_opus.png", "Claude ")
    _scrubout_one(SONNET_EVENTS,
                  CAT[0], "Official-name mismatch rate by release — Claude Sonnet line",
                  "fig_scrubout_claude_sonnet.png", "Claude ")
    _scrubout_one(CLAUDE_EVENTS, CAT[4],
                  "Official-name mismatch rate by release — Claude frontier (Opus + Sonnet)",
                  "fig_scrubout_claude.png", "Claude ",
                  subtitle="Opus 4 and Sonnet 4 shipped the same day — two markers, one release event")


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
    ax.set_title('% of models answering yes to "Are you X?" — own family excluded, vs three invented control names',
                 loc="left", fontsize=10.5, pad=12)
    save(fig, "fig_cross.png")


def fig_flow(reg, per):
    CB = {"anthropic": "claude", "openai": "chatgpt", "google": "gemini", "alibaba": "qwen",
          "meta": "llama", "moonshot": "kimi", "nvidia": "nvidia", "tencent": "hunyuan"}
    fam_claims = defaultdict(Counter)
    for mid, v in per.items():
        for c, k in v["claims"].items():
            c = "other/unlisted" if c.startswith("other:") else CB.get(c, c)
            fam_claims[reg[mid]["family"]][c] += k
    from .build_rollouts import brand
    # top families by mismatch volume, PLUS the main labs regardless of volume
    # (a reader looks for gpt/claude/gemini/llama rows before they look for poolside)
    by_vol = sorted(fam_claims, key=lambda f: -sum(fam_claims[f].values()))
    MAIN = {"openai", "anthropic", "google", "meta", "qwen", "deepseek", "kimi", "olmo"}
    fams = [f for f in by_vol if f in set(by_vol[:10]) | MAIN and sum(fam_claims[f].values())]
    labs = {f: f"{FAMILY_DISPLAY.get(f, f)}  ({sum(fam_claims[f].values())})" for f in fams}   # row volume, visible
    cols = ["chatgpt", "claude", "qwen", "gemini", "deepseek", "llama", "other/unlisted"]
    colr = {**IDCOLOR, "other/unlisted": "#b9b7b0"}   # same identity = same color as fig_coherence/fig_cutoff
    fig, ax = plt.subplots(figsize=(8.8, 0.44 * len(fams) + 1.6))
    left = np.zeros(len(fams))
    for k, c in enumerate(cols):
        vals = []
        for f in fams:
            tot = sum(fam_claims[f].values())
            v = fam_claims[f].get(c, 0)
            if c == "other/unlisted":
                v = tot - sum(fam_claims[f].get(x, 0) for x in cols[:-1])
            vals.append(100 * v / tot if tot else 0)
        ax.barh([labs[f] for f in fams][::-1], np.array(vals)[::-1], left=left[::-1], height=0.62, color=colr[c],
                label="other/unlisted" if c == "other/unlisted" else brand(c),
                edgecolor=SURFACE, linewidth=2, zorder=3)
        left += np.array(vals)
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of that family's mismatched-name claims (%) — (n) = the family's total mismatched-name claims")
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


# family display names — the ONE source of truth for slug → display across the
# figures: fam_* panel titles here, fig_flow's y-ticks, and (imported by
# fig_post) fig_family_counts rows + fam_cross_* titles. Parenthetical = the
# model line (when the slug is a lab: NVIDIA → Nemotron) or the lab (when the
# slug is a model line: Qwen → Alibaba).
FAMILY_DISPLAY = {
    "ai21": "AI21 (Jamba)", "amazon": "Amazon (Nova)", "ant": "Ant Group (Ling)",
    "anthropic": "Anthropic", "arcee": "Arcee AI", "baidu": "Baidu (ERNIE)",
    "cohere": "Cohere (Command)", "deepseek": "DeepSeek", "gemma": "Gemma (Google)",
    "google": "Google (Gemini)", "ibm": "IBM (Granite)", "kimi": "Moonshot (Kimi)",
    "kuaishou": "Kuaishou (KAT)", "meta": "Meta (Llama)", "microsoft": "Microsoft",
    "minimax": "MiniMax", "mistral": "Mistral", "nex": "Nex (N2)",
    "nous": "Nous (Hermes)", "nvidia": "NVIDIA (Nemotron)", "olmo": "Ai2 (OLMo)",
    "openai": "OpenAI", "perceptron": "Perceptron", "perplexity": "Perplexity (Sonar)",
    "poolside": "Poolside (Laguna)", "qwen": "Alibaba (Qwen)", "reka": "Reka",
    "stepfun": "StepFun (Step)", "tencent": "Tencent (Hunyuan)", "xiaomi": "Xiaomi (MiMo)",
    "zhipu": "Zhipu (GLM)",
}


def fig_family_panels(reg, per):
    # targets are MODEL identities only; nvidia (=Nemotron) and doubao fold into
    # "other lab" per review — lab names, not model names.
    TARGETS = ["chatgpt", "claude", "gemini", "deepseek", "qwen", "kimi", "llama",
               "mistral", "glm", "grok", "other lab", "novel/unrec."]
    CB = {"anthropic": "claude", "openai": "chatgpt", "google": "gemini", "alibaba": "qwen",
          "meta": "llama", "moonshot": "kimi", "tencent": "hunyuan"}
    def to_col(c):
        if c.startswith("other:") or c == "other/unlisted":
            return "novel/unrec."
        c = CB.get(c, c)
        return c if c in TARGETS else "other lab"
    (FIGS / "family").mkdir(parents=True, exist_ok=True)
    fam_models = defaultdict(list)
    for mid in per:
        fam_models[reg[mid]["family"]].append(mid)
    # pass 1: build every panel + a GLOBAL vmax so colors mean the same % everywhere
    panels, gmax = {}, 1.0
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
        if not rows:
            continue
        rows.sort(key=lambda r: -r[1] / r[2])
        M = np.array([r[3] for r in rows])
        panels[fam] = (rows, M)
        gmax = max(gmax, float(M.max()))
    # pass 2: render with the shared scale; drop all-zero columns (incl. each
    # family's own-name column, which is empty by construction)
    manifest = []
    for fam, (rows, M) in panels.items():
        keep = [j for j in range(len(TARGETS)) if M[:, j].sum() > 0]
        Mk = M[:, keep]; labels = [TARGETS[j] for j in keep]
        nrow, ncol = len(rows), len(keep)
        fig, ax = plt.subplots(figsize=(max(0.62 * ncol + 3.0, 6.2), 0.42 * nrow + 1.9))
        im = ax.imshow(Mk, cmap=SEQ_CMAP, vmin=0, vmax=gmax, aspect="auto")
        ax.set_xticks(range(ncol), labels, fontsize=8, rotation=35, ha="right")
        ax.set_yticks(range(nrow), [per[r[0]]["name"] for r in rows], fontsize=9)
        for i in range(nrow):
            for j in range(ncol):
                if Mk[i, j] <= 0:
                    continue
                ax.text(j, i, "<1" if Mk[i, j] < 1 else f"{Mk[i,j]:.0f}", ha="center", va="center",
                        fontsize=7.5, color="#ffffff" if Mk[i, j] > 0.6 * gmax else INK2)
        cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
        # short panels can't fit the rotated long label beside the colorbar
        cb.set_label("% of the model's short-question responses" if nrow > 2 else "% of responses",
                     fontsize=8, color=INK2)
        cb.outline.set_visible(False)
        style(ax, bottom=False)
        ax.set_title(f"{FAMILY_DISPLAY.get(fam, fam)} — which name each model gives instead",
                     loc="left", fontsize=11, pad=22)
        ax.annotate("color scale shared across all family panels", (0, 1), xycoords="axes fraction",
                    xytext=(0, 5), textcoords="offset points", fontsize=7, color=MUTED, va="bottom")
        ax.annotate("a response naming one model and a different creator counts in both columns,\n"
                    "so cells can sum above the model's overall rate",
                    (0, 0), xycoords="axes fraction", xytext=(0, -46), textcoords="offset points",
                    fontsize=7.5, color=MUTED, ha="left", va="top")
        fn = f"family/fam_{fam}.png"
        fig.savefig(FIGS / fn, dpi=200, bbox_inches="tight", facecolor=SURFACE)
        plt.close(fig)
        manifest.append({"file": fn, "family": fam, "drift": sum(r[1] for r in rows), "models": nrow})
    manifest.sort(key=lambda p: -p["drift"])
    (FIGS / "family" / "manifest.json").write_text(json.dumps(manifest, indent=1))
    current = {ROOT / "figures" / p["file"] for p in manifest}
    for p in (FIGS / "family").glob("fam_*.png"):     # drop panels whose family went clean
        if p not in current:
            p.unlink()
            print(f"  removed stale {p.name}")
    print(f"  {len(manifest)} family panels (shared vmax={gmax:.0f}%)")


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
