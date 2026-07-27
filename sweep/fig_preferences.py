"""Preference-fingerprint figure — do the absorbers share Claude's preferences,
or only its name?

One point per model version, per language: of the screened questions where the
Anthropic and OpenAI family consensuses DIFFER (leave-one-out for family members,
tied consensus = no consensus), the share answered the Claude way (y) vs the GPT
way (x). Whiskers are hierarchical-bootstrap 95% CIs: questions resampled, and
every model's 8-sample answer cell resampled within each drawn question, so
consensus fragility at this n is priced in.

The read: Claude versions sit above the diagonal, GPTs below. Kimi K2.5 — which
claims to BE Claude, mostly in Chinese — lands at the top of the real-Claude
range in Chinese (+20pp vs its English self): the persona travels with the name.
MiniMax M2.7 claims the name too but sits below every real Claude in both
languages: name only.

  python -m sweep.fig_preferences
"""
import random
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# importing make_figs applies the shared rcParams / palette
from .make_figs import CAT, SURFACE, INK, INK2, MUTED, GRID, BASE, FIGS
from .preferences import (_load, ANT, OAI, ABSORBERS, kept_questions,
                          match_vector)

C_ANT, C_OAI, C_ABS = CAT[4], CAT[3], CAT[5]        # violet / green / red — CVD-validated triple
REPS, SEED = 2000, 12345
SHORT = {"moonshotai/kimi-k2.5": "Kimi K2.5", "moonshotai/kimi-k2.6": "Kimi K2.6",
         "minimax/minimax-m2.7": "MiniMax M2.7"}


def _modal_of(lst):
    if not lst:
        return None
    top = Counter(lst).most_common(2)
    if len(top) > 1 and top[0][1] == top[1][1]:
        return None                                   # tied cell → no modal this rep
    return top[0][0]


def _cons_from(counts, removed):
    """Consensus from a Counter of family modals with one member's modal removed
    (LOO by decrement). Tied plurality = no consensus, matching preferences.py."""
    if removed is not None:
        counts = counts.copy()
        counts[removed] -= 1
        if counts[removed] <= 0:
            del counts[removed]
    if not counts:
        return None
    top = counts.most_common(2)
    if len(top) > 1 and top[0][1] == top[1][1]:
        return None
    return top[0][0]


def bootstrap(recs, qs, models, reps=REPS, seed=SEED):
    """One pass, all models at once. Returns {mid: ([→ANT rates], [→OAI rates])}
    across bootstrap reps. Hierarchy: resample the question list, then within each
    drawn question resample every model's answer cell before taking modals."""
    rng = random.Random(seed)
    cells = {(m, q): recs[m][q] for m in models for q in qs if recs[m][q]}
    out = {m: ([], []) for m in models}
    for _ in range(reps):
        qs_r = rng.choices(qs, k=len(qs))
        acc = defaultdict(lambda: [0, 0, 0])          # mid -> [ant, oai, t]
        for q in qs_r:
            modal = {}
            for m in models:
                cell = cells.get((m, q))
                if cell:
                    modal[m] = _modal_of(rng.choices(cell, k=len(cell)))
            ca_counts = Counter(modal.get(m) for m in ANT if modal.get(m))
            co_counts = Counter(modal.get(m) for m in OAI if modal.get(m))
            for mid in models:
                a = modal.get(mid)
                if a is None:
                    continue
                ca = _cons_from(ca_counts, a if mid in ANT else None) if mid in ANT else _cons_from(ca_counts, None)
                co = _cons_from(co_counts, a if mid in OAI else None) if mid in OAI else _cons_from(co_counts, None)
                if ca is None or co is None or ca == co:
                    continue
                r = acc[mid]
                r[0] += (a == ca); r[1] += (a == co); r[2] += 1
        for mid in models:
            a, o, t = acc[mid]
            if t:
                out[mid][0].append(100 * a / t)
                out[mid][1].append(100 * o / t)
    return out


def pct_ci(samples, point):
    if not samples:
        return 0.0, 0.0
    s = sorted(samples)
    lo = s[int(0.025 * (len(s) - 1))]
    hi = s[int(0.975 * (len(s) - 1))]
    return max(0.0, point - lo), max(0.0, hi - point)


def main():
    qs = list(kept_questions())
    models = ANT + OAI + ABSORBERS
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 5.0), sharey=True)
    panel_notes = {"en": "English", "zh": "Chinese — where the name-mismatch lives"}
    drawn = []
    for ax, lang in zip(axes, ("en", "zh")):
        recs = _load("full", lang)
        boots = bootstrap(recs, qs, models)
        ax.plot([0, 100], [0, 100], ls=(0, (4, 4)), lw=1, color=BASE, zorder=1)
        ax.annotate("no lean", (76, 76), rotation=45, fontsize=7, color=MUTED,
                    ha="center", va="bottom", rotation_mode="anchor")
        for fam, col, mk, lab in [(ANT, C_ANT, "o", "Claude versions (n=9)"),
                                  (OAI, C_OAI, "s", "GPT versions (n=8)"),
                                  (ABSORBERS, C_ABS, "D", "claims to be Claude")]:
            for mid in fam:
                mv = match_vector(recs, qs, mid)
                if not mv:
                    continue
                t = len(mv)
                y = 100 * sum(m[0] for m in mv) / t
                x = 100 * sum(m[1] for m in mv) / t
                ey = pct_ci(boots[mid][0], y)
                ex = pct_ci(boots[mid][1], x)
                is_abs = mid in ABSORBERS
                if is_abs:
                    # CIs only on the points the claim is about; the family points'
                    # uncertainty is comparable, and their visible SPREAD is the
                    # yardstick the multi-version design was built around.
                    ax.errorbar(x, y, yerr=[[ey[0]], [ey[1]]], xerr=[[ex[0]], [ex[1]]],
                                fmt="none", ecolor=col, elinewidth=1.1,
                                alpha=0.65, capsize=2.5, zorder=2)
                ax.plot(x, y, mk, color=col, ms=9.5 if is_abs else 8,
                        mec=SURFACE, mew=1.4, zorder=4 if is_abs else 3)
                drawn.append((lang, mid, y, ey, x, ex, t))
                if is_abs:
                    # per-language placement; K2.6 and MiniMax coincide exactly at
                    # (40,20) in English, so those two get leader lines to the
                    # shared diamond instead of sitting on top of each other.
                    dx, dy, ha = {
                        ("moonshotai/kimi-k2.5", "en"): (-8, 8, "right"),
                        ("moonshotai/kimi-k2.6", "en"): (16, 14, "left"),
                        ("minimax/minimax-m2.7", "en"): (-14, -16, "right"),
                        ("moonshotai/kimi-k2.5", "zh"): (-8, 8, "right"),
                        ("moonshotai/kimi-k2.6", "zh"): (10, 5, "left"),
                        ("minimax/minimax-m2.7", "zh"): (10, -11, "left"),
                    }[(mid, lang)]
                    arrow = (dict(arrowstyle="-", lw=0.7, color=C_ABS, alpha=0.6,
                                  shrinkA=2, shrinkB=6)
                             if lang == "en" and mid != "moonshotai/kimi-k2.5" else None)
                    ax.annotate(SHORT[mid], (x, y), textcoords="offset points",
                                xytext=(dx, dy), fontsize=8, fontweight="bold",
                                color=C_ABS, ha=ha, arrowprops=arrow,
                                va="bottom" if dy > 0 else "top", zorder=5)
        ax.set_xlim(-6, 104); ax.set_ylim(-6, 104)
        ax.set_aspect("equal")
        ax.set_title(panel_notes[lang], loc="left", fontsize=9.5, color=INK2, pad=6)
        ax.set_xlabel("% of split questions answered the GPT way", fontsize=9)
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.xaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_ylabel("% of split questions answered the Claude way", fontsize=9)
    axes[0].legend(handles=[
        Line2D([], [], marker="o", ls="", color=C_ANT, mec=SURFACE, ms=8, label="Claude versions"),
        Line2D([], [], marker="s", ls="", color=C_OAI, mec=SURFACE, ms=8, label="GPT versions"),
        Line2D([], [], marker="D", ls="", color=C_ABS, mec=SURFACE, ms=8, label="claims to be Claude")],
        loc="upper right", frameon=False, fontsize=8, handletextpad=0.4, borderaxespad=0.2)
    fig.suptitle("Preference fingerprints — Kimi leans Claude in preference space too (most in Chinese); MiniMax borrows only the name",
                 x=0.005, y=1.0, ha="left", fontsize=11.5, fontweight="bold", color=INK)
    fig.text(0.005, -0.06,
             "30 screened forced-choice preference questions, cold, no system prompt, 8 samples per model×question. Axes: share of the questions\n"
             "where the Anthropic and OpenAI family consensuses disagree (leave-one-out for family members; tied consensus = none; n≈7–12 per\n"
             "model) answered each family's way. Whiskers: hierarchical bootstrap 95% CIs (questions + answer cells resampled, consensus\n"
             "recomputed) on the absorbers; family points carry comparable uncertainty — their visible spread is the within-family yardstick.\n"
             "Kimi K2.6 and MiniMax M2.7 land on the identical point in the English panel (one diamond, two leader lines).",
             fontsize=7.4, color=MUTED)
    fig.subplots_adjust(wspace=0.08)
    FIGS.mkdir(exist_ok=True)
    out = FIGS / "fig_preferences.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {out}")
    for lang, mid, y, ey, x, ex, t in drawn:
        if mid in ABSORBERS or mid in ("anthropic/claude-opus-4.8", "openai/gpt-5"):
            print(f"  {lang} {mid.split('/')[-1]:20} →ANT {y:3.0f}% (−{ey[0]:.0f}/+{ey[1]:.0f}) "
                  f"→OAI {x:3.0f}% (−{ex[0]:.0f}/+{ex[1]:.0f})  n={t}")


if __name__ == "__main__":
    main()
