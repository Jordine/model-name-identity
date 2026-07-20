"""Explanatory re-analysis: don't just measure *how often* a model misidentifies,
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
import math
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .analyze import CREATOR_TO_BRAND
from .build_rollouts import collect, collect_model, adj_verdicts, brand
from .make_figs import FIGS, SURFACE, INK, INK2, MUTED, GRID, BASE, style, save

MIN_FOREIGN = 10   # need at least this many foreign claims to estimate a distribution
MIN_TOT = 100      # and a reasonably complete battery

# dominant-identity palette (who the model claims to be) — kept visually distinct.
# creator-canons are collapsed to brand in gather(), so only brands appear here.
IDCOLOR = {"claude": "#d97a34", "deepseek": "#2a78d6", "qwen": "#7b4fd0", "chatgpt": "#1baf7a",
           "gemini": "#c94f9c", "llama": "#4aa0a0", "nvidia": "#5b8c1f", "kimi": "#e0a020",
           "mistral": "#b0402a"}


def entropy(counter):
    tot = sum(counter.values())
    if not tot:
        return 0.0
    return -sum((n / tot) * math.log2(n / tot) for n in counter.values() if n)


def gather():
    reg, jud, rec = collect()
    verdicts = adj_verdicts()
    rows = {}
    for mid, r in rec.items():
        m = reg.get(mid, {})
        name = m.get("name", mid); fam = m.get("family", "?")
        exp = m.get("expected_identity", name); al = m.get("aliases", [])
        rate, dn, tot, claims, cross, recs, lstats = collect_model(mid, name, fam, exp, al, r, jud, verdicts)
        # collapse creator-canons to their brand (OpenAI==ChatGPT, Google==Gemini,
        # Alibaba==Qwen…) so name/creator splits count as ONE identity; drop 'other:' noise
        merged = Counter()
        for c, n in claims.items():
            if str(c).startswith("other:"):
                continue
            merged[CREATOR_TO_BRAND.get(c, c)] += n
        claims = merged
        rows[mid] = dict(name=name, fam=fam, rate=rate, dn=dn, tot=tot,
                         claims=claims, cross=cross, lstats=lstats)
    return rows


def metrics(d):
    claims = d["claims"]
    total = sum(claims.values())
    top = claims.most_common(1)[0] if total else (None, 0)
    # language: drift share + per-language rate spread
    langdrift = Counter({l: dd for l, (dd, nn) in d["lstats"].items() if dd})
    lt = sum(langdrift.values())
    lang_top = langdrift.most_common(1)[0] if lt else (None, 0)
    rates = [dd / nn for l, (dd, nn) in d["lstats"].items() if nn >= 20]
    return dict(
        n_foreign=total,
        top1_id=top[0], top1_share=(top[1] / total if total else 0),
        entropy=entropy(claims), n_distinct=sum(1 for c, n in claims.items() if n >= 2),
        dom_lang=lang_top[0], lang_top_share=(lang_top[1] / lt if lt else 0),
        lang_spread=(max(rates) - min(rates)) if rates else 0.0,
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
    # label only the well-separated right side (rate>=33); the low-rate left is a
    # dense cloud (many models rarely-but-consistently drift) — left unlabelled
    lab = sorted([p for p in pts if p[0]["rate"] >= 33], key=lambda x: x[0]["rate"])
    prev = {}
    for d, mm in lab:
        y = 100 * mm["top1_share"]
        band = round(d["rate"] / 8)                 # nudge labels apart within an x-band
        dy = 9 if prev.get(band) and abs(prev[band] - y) < 5 else 3
        prev[band] = y
        ax.annotate(f"{d['name']} →{brand(mm['top1_id'])}", (d["rate"], y),
                    fontsize=6.6, color=INK2, xytext=(5, dy), textcoords="offset points")
    ax.axhline(60, color=BASE, lw=0.8, ls="--", zorder=1)
    ax.text(1, 62, "consistently ONE identity  (coherent alternate persona)", fontsize=7.5, color=MUTED)
    ax.text(1, 20, "scatters across many  (confabulation / weak identity)", fontsize=7.5, color=MUTED)
    ax.set_xlabel("spontaneous mismatch rate  (% of identity answers)")
    ax.set_ylabel("consistency — share of misclaims on the single top identity (%)")
    ax.set_title("Coherence of misidentification: a stable alternate self vs. confabulation",
                 fontsize=11, color=INK, loc="left")
    ax.set_ylim(0, 105); ax.set_xlim(0, max(d["rate"] for d, _ in pts) + 6)
    style(ax)
    ax.grid(axis="both", color=GRID, lw=0.6, zorder=0)
    # legend for dominant identity
    from matplotlib.patches import Patch
    seen_ids = [i for i in IDCOLOR if any(mm["top1_id"] == i for _, mm in pts)]
    ax.legend(handles=[Patch(facecolor=IDCOLOR[i], label=brand(i)) for i in seen_ids],
              title="claims to be", fontsize=7.5, title_fontsize=8, loc="lower right", frameon=False)
    save(fig, "fig_coherence.png")


def fig_lang_conditional(data):
    """models most gated by language: dominant-language drift share vs overall rate."""
    pts = [(d, metrics(d)) for d in data.values()]
    pts = [(d, mm) for d, mm in pts if mm["n_foreign"] >= MIN_FOREIGN and d["tot"] >= MIN_TOT and d["rate"] >= 3]
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    for d, mm in pts:
        ax.scatter(d["rate"], 100 * mm["lang_top_share"], s=22 + d["dn"] * 0.6,
                   color="#2a78d6", alpha=0.6, edgecolor="white", linewidth=0.5, zorder=3)
    lab = sorted([p for p in pts if p[1]["lang_top_share"] >= 0.58], key=lambda x: -x[1]["lang_top_share"])
    prevy = 999
    for d, mm in lab:
        y = 100 * mm["lang_top_share"]
        dy = -10 if (prevy - y) < 4 else 3
        prevy = y
        ax.annotate(f"{d['name']} ({mm['dom_lang']})", (d["rate"], y),
                    fontsize=6.6, color=INK2, xytext=(5, dy), textcoords="offset points")
    ax.set_xlabel("spontaneous mismatch rate (%)")
    ax.set_ylabel("share of all drift in the single dominant language (%)")
    ax.set_title("Language-conditionality: how much a model's drift lives in one language",
                 fontsize=11, color=INK, loc="left")
    ax.set_ylim(0, 105)
    style(ax)
    ax.grid(axis="both", color=GRID, lw=0.6, zorder=0)
    save(fig, "fig_lang_conditional.png")


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

    table("3. Most LANGUAGE-CONDITIONAL (drift concentrated in one language)",
          sorted([(d, mm) for d, mm in ok if d["rate"] >= 3], key=lambda x: -x[1]["lang_top_share"])[:15],
          lambda d, mm: f"{d['name']:26s} rate {d['rate']:4.0f}%  {mm['lang_top_share']*100:3.0f}% in {mm['dom_lang']}  spread {mm['lang_spread']*100:.0f}pp")


def main():
    data = gather()
    report(data)
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_coherence(data)
    fig_lang_conditional(data)


if __name__ == "__main__":
    main()
