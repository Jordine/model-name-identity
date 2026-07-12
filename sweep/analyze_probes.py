"""Stage-2 probe analysis: confrontation outcomes with controls,
third-identity suggestibility, and version-specificity depth.

Usage: python -m sweep.analyze_probes
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

from .analyze import canon_identity, is_self, FAMILY_SELF

ROOT = Path(__file__).resolve().parent.parent
PJ = ROOT / "results" / "probes_judgments.jsonl"


def load():
    rows = []
    reg = {m["id"]: m for m in json.loads((ROOT / "config" / "models.json").read_text())["models"]}
    for line in open(PJ, encoding="utf-8"):
        j = json.loads(line)
        if j.get("judge_error") or not j.get("judgment"):
            continue
        m = reg.get(j["model_id"])
        if m:
            j["aliases"], j["family"] = m["aliases"], m["family"]
        rows.append(j)
    return rows


def claimed_canon(j):
    """Canon of the visible-response identity claim (None if none/generic)."""
    jm = j["judgment"]
    for f in ("claimed_name", "claimed_creator"):
        c = canon_identity(jm.get(f))
        if c:
            return c
    return None


def outcome_of_final(j):
    """official / pushed / other-foreign / none-or-hedge for a final turn."""
    c = claimed_canon(j)
    fam = j.get("family", "")
    jm = j["judgment"]
    pushed = (j.get("pushed_identity") or "").lower()
    if c is None:
        return "hedge/none" if jm.get("hedged") or jm.get("no_identity_claim") else "hedge/none"
    if is_self(c, fam, j.get("aliases", []), j["expected_identity"]):
        return "official"
    key = c.replace("other:", "")
    if pushed and (pushed in key or key in pushed):
        return "adopted_pushed"
    return "other_foreign"


def main():
    rows = load()
    print(f"{len(rows)} judged probe records\n")

    # ---- confrontation outcomes (final turns) ----
    finals = [j for j in rows if j.get("final_turn") and j.get("seq_id", "").startswith("confront")]
    table = defaultdict(Counter)
    for j in finals:
        table[j["seq_id"]][outcome_of_final(j)] += 1
    print("=== CONFRONTATION — final-turn 'so who are you?' outcomes ===")
    print(f"{'sequence':20s} {'n':>4s}  {'official':>9s} {'adopted-pushed':>14s} {'other-foreign':>13s} {'hedge/none':>10s}")
    for seq in ("confront_correct", "confront_third", "confront_false"):
        c = table[seq]
        n = sum(c.values())
        if not n:
            continue
        print(f"{seq:20s} {n:4d}  "
              f"{c['official']/n:9.0%} {c['adopted_pushed']/n:14.0%} "
              f"{c['other_foreign']/n:13.0%} {c['hedge/none']/n:10.0%}")
    print("""
  reading: confront_correct = flagged models told their TRUE name (adopted-pushed == recanted to official)
           confront_third   = flagged models told an UNRELATED name (adopted-pushed == swallowed arbitrary suggestion)
           confront_false   = clean controls told a FALSE name (adopted-pushed == gaslit successfully)""")

    # ---- middle-turn immediate reaction ("what if I told you...") ----
    mids = [j for j in rows if j.get("turn_index") == 1 and j.get("seq_id", "").startswith("confront")]
    acc = defaultdict(Counter)
    for j in mids:
        jm = j["judgment"]
        y = jm.get("answered_yes")
        acc[j["seq_id"]]["accepts" if y is True else ("rejects" if y is False else "unclear/hedges")] += 1
    print("=== immediate reaction to the push (turn 2) ===")
    for seq, c in acc.items():
        n = sum(c.values())
        print(f"  {seq:20s} n={n:3d}  " + "  ".join(f"{k}={v/n:.0%}" for k, v in c.most_common()))

    # ---- specificity: does the claimed identity come with a version? ----
    import re
    ver_pat = re.compile(r"\d|version|版本|バージョン|버전|opus|sonnet|haiku|turbo|mini|pro\b", re.I)
    spec = defaultdict(Counter)
    examples = []
    for j in rows:
        if j.get("seq_id") != "specificity" or j.get("turn_index") != 1:
            continue
        c = claimed_canon(j)
        fam = j.get("family", "")
        grp = "foreign-claim" if (c and not is_self(c, fam, j.get("aliases", []), j["expected_identity"])) else "self-claim"
        name = (j["judgment"].get("claimed_name") or "")
        has_version = bool(ver_pat.search(name))
        spec[grp]["versioned" if has_version else "vague"] += 1
        if grp == "foreign-claim" and has_version and len(examples) < 8:
            examples.append((j["model_id"], name, j["judgment"].get("evidence", "")[:80]))
    print("\n=== SPECIFICITY — 'which one exactly?' (turn 2 claims) ===")
    for grp, c in spec.items():
        n = sum(c.values())
        print(f"  {grp:14s} n={n:3d}  versioned={c['versioned']/n:.0%}  vague={c['vague']/n:.0%}")
    print("\n  foreign claims WITH specific versions:")
    for mid, name, ev in examples:
        print(f"    {mid:45s} -> {name!r}")
        print(f"       {ev}")


if __name__ == "__main__":
    main()
