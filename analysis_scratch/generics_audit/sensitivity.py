"""Sensitivity of headline mismatch numbers to how "generic" answers are handled.

Variants (per-record mismatch definition; battery/filters identical to make_figs.gather):
  S0 baseline      — exact current pipeline (must match shipped numbers).
  S1 known-only    — mismatch only on foreign claims with KNOWN canon (drop other:* evidence).
  S2 conditional   — denominator restricted to records whose response named something
                     specific (canon(claimed_name) or canon(claimed_creator) non-None);
                     numerator = S0 mismatches that named something.
  S3 strict-self   — family-equivalence via alias strings disabled for KNOWN canons:
                     a known canon is self only if in FAMILY_SELF[fam] (+ family slug,
                     + known canon of the slug, for coverage). Disclosed-ancestor names
                     (Nemotron/Hermes saying "Llama") become foreign. other:* branch and
                     None branch unchanged. Separate axis, labeled as such.

Run from repo root:  python analysis_scratch/generics_audit/sensitivity.py
Writes sensitivity.json (+ prints the master table) into this directory.
"""

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sweep.analyze import (FAMILY_SELF, _adjudications, canon_identity,  # noqa: E402
                           foreign_claims, is_self, load)
from sweep.make_figs import (BATTERY_CORE, LOCAL_MODELS, complete_models,  # noqa: E402
                             _local_genuine, is_identity)

OUT = Path(__file__).resolve().parent

SHIPPED = dict(mismatch_records=4849, denom=60770, models_ge1=116, models_ge3=95,
               models_zero=74, median_model_rate_pct=0.78, n_models=190)


# --- strict-self machinery (S3) --------------------------------------------
_FSS = {}


def fam_self_strict(fam):
    """KNOWN canons that count as self under S3: the family's own FAMILY_SELF set
    (+ the family slug itself, + the slug's known canon — needed for local 'olmo'
    whose canon is 'allenai'). NO alias-string fallback: that is where disclosed
    ancestors (llama/meta/qwen in the alias list) get forgiven in S0."""
    if fam not in _FSS:
        s = set(FAMILY_SELF.get(fam, set())) | {fam}
        c = canon_identity(fam)
        if c and not c.startswith("other:"):
            s.add(c)
        _FSS[fam] = s
    return _FSS[fam]


def canon_sets(jm, fam, aliases, expected, include_reasoning):
    """Per-record canon evidence under the three self rules.

    Returns (raw_foreign, known_foreign, strict_foreign, named):
      raw_foreign    — S0's pre-adjudication foreign set (exactly foreign_claims' logic)
      known_foreign  — raw minus other:*
      strict_foreign — S3 rule (superset of raw; asserted by caller)
      named          — response-level claimed_name/claimed_creator canon non-None
    """
    fields = ["claimed_name", "claimed_creator"]
    if include_reasoning and jm.get("reasoning_identity_stance") != "role_play":
        fields += ["reasoning_claimed_name", "reasoning_claimed_creator"]
    raw, strict = set(), set()
    for f in fields:
        c = canon_identity(jm.get(f))
        if not c:
            continue
        s0_self = is_self(c, fam, aliases, expected)
        if not s0_self:
            raw.add(c)
        if c.startswith("other:"):
            if not s0_self:
                strict.add(c)          # other:* branch unchanged in S3
        elif c not in fam_self_strict(fam):
            strict.add(c)
    named = (canon_identity(jm.get("claimed_name")) is not None
             or canon_identity(jm.get("claimed_creator")) is not None)
    return raw, {c for c in raw if not c.startswith("other:")}, strict, named


def new_model():
    return dict(n=0, named=0, named_adj=0, d0=0, d1=0, d2=0, d2b=0, d3=0,
                legacy_named=0, legacy_d=0, api=False,
                s1_dropped_other=Counter(),   # other:* evidence on records S1 drops
                s1_dropped_lang=Counter(),
                s3_new=Counter(),             # newly-foreign known canons (S3 only)
                name="", family="")


def run():
    reg = {m["id"]: m for m in json.loads((ROOT / "config" / "models.json").read_text())["models"]}
    hyg = json.loads((ROOT / "config" / "provider_hygiene.json").read_text())
    complete = complete_models(reg, hyg)
    per = defaultdict(new_model)
    flip_inventory = Counter()          # (family, canon) pairs self->foreign under S3
    n_trace_only_mm = 0                 # S0 mismatches with no response-level name
    n_s3_new_killed_by_adj = 0          # strict-only evidence on adj-rejected records

    # ---- API models ----
    for j in load():
        if j["model_id"] not in complete:
            continue
        cat = j["prompt_category"]
        if cat in ("probe_placebo", "probe_cross", "system_probe"):
            continue
        if not is_identity(cat) or j["prompt_id"] not in BATTERY_CORE:
            continue
        jm = j["judgment"]
        fam = j.get("family", "")
        aliases = j.get("aliases", [])
        raw, known, strict, named = canon_sets(jm, fam, aliases, j["expected_identity"], True)
        assert raw <= strict, (j["resume_key"], raw, strict)
        adj = _adjudications().get(f"{j['resume_key']}::t{j.get('turn_index', 0)}") if (raw or strict) else None
        adj_ok = adj is None or adj == "genuine_foreign"
        s0 = bool(raw) and adj_ok
        assert s0 == bool(foreign_claims(j)), j["resume_key"]   # harness == pipeline
        s1 = bool(known) and adj_ok
        s2 = s0 and named
        s3 = bool(strict) and adj_ok
        # S2b: "named" minus records whose extraction was adjudicated spurious/generic
        named_adj = named and not (adj in ("judge_error", "generic"))
        s2b = s0 and named_adj
        # legacy PLAN Q&A definition: named = counted-mismatch OR canon-self in ANY
        # of the 4 fields (reasoning included, role_play NOT applied on self side)
        self_named = any(
            (c := canon_identity(jm.get(f))) and is_self(c, fam, aliases, j["expected_identity"])
            for f in ("claimed_name", "claimed_creator",
                      "reasoning_claimed_name", "reasoning_claimed_creator"))
        legacy_named = s0 or self_named
        if strict - raw:
            if adj_ok:
                for c in strict - raw:
                    flip_inventory[(fam, c)] += 1
            else:
                n_s3_new_killed_by_adj += 1
        m = per[j["model_id"]]
        m["name"] = reg[j["model_id"]]["name"]
        m["family"] = fam
        m["api"] = True
        m["n"] += 1
        m["named"] += named
        m["named_adj"] += named_adj
        m["legacy_named"] += legacy_named
        m["legacy_d"] += s0
        m["d0"] += s0
        m["d1"] += s1
        m["d2"] += s2
        m["d2b"] += s2b
        m["d3"] += s3
        if s0 and not named:
            n_trace_only_mm += 1
        if s0 and not s1:
            from sweep.analyze import lang_of
            m["s1_dropped_lang"][lang_of(cat)] += 1
            for c in raw:
                if c.startswith("other:"):
                    m["s1_dropped_other"][c] += 1
        if s3:
            for c in strict - raw:
                m["s3_new"][c] += 1

    # ---- local raw-weights models (mirror make_figs.add_local exactly) ----
    keep = _local_genuine()
    local_verdict = {}
    for line in open(ROOT / "results" / "adjudications_local.jsonl", encoding="utf-8"):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("verdict"):
            local_verdict[d["adj_key"]] = d["verdict"]
    jpath = ROOT / "results_local" / "judgments_clean.jsonl"
    for line in open(jpath, encoding="utf-8"):
        j = json.loads(line)
        if not j.get("judgment"):
            continue
        if j["resume_key"].split("::")[-1] != "clean":
            continue
        mid = j["model_id"]
        if mid not in LOCAL_MODELS:
            continue
        name, fam, al = LOCAL_MODELS[mid]
        cat = j["prompt_category"]
        if cat in ("probe_placebo", "probe_cross", "system_probe"):
            continue
        if not is_identity(cat) or j["prompt_id"] not in BATTERY_CORE:
            continue
        jm = j["judgment"]
        # add_local uses response fields only, and a POSITIVE adjudication gate
        raw, known, strict, named = canon_sets(jm, fam, al, name, False)
        assert raw <= strict
        adj_ok = f"{j['resume_key']}::t0" in keep
        s0 = bool(raw) and adj_ok
        s1 = bool(known) and adj_ok
        s2 = s0 and named
        v = local_verdict.get(f"{j['resume_key']}::t0")
        named_adj = named and not (v in ("judge_error", "generic"))
        s2b = s0 and named_adj
        # strict-only evidence was never flagged -> never adjudicated -> can't
        # require the positive gate for it; count it ungated (documented).
        s3 = s0 or bool(strict - raw)
        if strict - raw:
            for c in strict - raw:
                flip_inventory[(fam, c)] += 1
        m = per[mid]
        m["name"], m["family"] = name, fam
        m["n"] += 1
        m["named"] += named
        m["named_adj"] += named_adj
        m["d0"] += s0
        m["d1"] += s1
        m["d2"] += s2
        m["d2b"] += s2b
        m["d3"] += s3
        if s0 and not named:
            n_trace_only_mm += 1
        if s0 and not s1:
            from sweep.analyze import lang_of
            m["s1_dropped_lang"][lang_of(cat)] += 1
            for c in raw:
                if c.startswith("other:"):
                    m["s1_dropped_other"][c] += 1
        if s3:
            for c in strict - raw:
                m["s3_new"][c] += 1

    per = {k: v for k, v in per.items() if v["n"] >= 40}
    return per, flip_inventory, n_trace_only_mm, n_s3_new_killed_by_adj


def summarize(per, dkey, nkey="n"):
    vals = [(v[dkey], v[nkey]) for v in per.values() if v[nkey] > 0]
    D, N = sum(d for d, _ in vals), sum(n for _, n in vals)
    rates = [100 * d / n for d, n in vals]
    return dict(pooled_pct=round(100 * D / N, 2), mismatch_records=D, denom=N,
                models_ge1=sum(1 for d, _ in vals if d >= 1),
                models_ge3=sum(1 for d, _ in vals if d >= 3),
                models_zero=sum(1 for d, _ in vals if d == 0),
                median_model_rate_pct=round(statistics.median(rates), 2),
                n_models=len(vals))


def movers(per, dkey, nkey="n", top=10):
    out = []
    for mid, v in per.items():
        if v["n"] == 0 or v[nkey] == 0:
            continue
        r0 = 100 * v["d0"] / v["n"]
        rv = 100 * v[dkey] / v[nkey]
        out.append(dict(model=mid, name=v["name"], family=v["family"],
                        s0=f"{v['d0']}/{v['n']}", s0_pct=round(r0, 1),
                        var=f"{v[dkey]}/{v[nkey]}", var_pct=round(rv, 1),
                        delta_pp=round(rv - r0, 1)))
    out.sort(key=lambda x: -abs(x["delta_pp"]))
    return out[:top]


def main():
    per, flips, trace_only, s3_adjkill = run()

    # ---- S0 harness check ----
    s0 = summarize(per, "d0")
    mismatch = {k: (s0[k], SHIPPED[k]) for k in SHIPPED if s0[k] != SHIPPED[k]}
    print("S0 vs shipped:", "EXACT MATCH" if not mismatch else f"DISCREPANCY {mismatch}")
    if mismatch:
        # dump per-model to debug rather than proceeding
        print(json.dumps(s0, indent=1))
        sys.exit(1)

    # cross-check against the pipeline's own aggregates
    from sweep.make_figs import gather
    _, gper = gather()
    diff = {mid for mid in set(gper) | set(per)
            if (gper.get(mid, {}).get("d"), gper.get(mid, {}).get("n"))
            != (per.get(mid, {}).get("d0"), per.get(mid, {}).get("n"))}
    print("per-model check vs make_figs.gather():",
          "all 190 identical" if not diff else f"DIFFERS for {sorted(diff)}")
    assert not diff

    s1 = summarize(per, "d1")
    s2 = summarize(per, "d2", nkey="named")
    s2b = summarize(per, "d2b", nkey="named_adj")
    s3 = summarize(per, "d3")

    # legacy PLAN Q&A replication (API models only — the original loop never saw locals)
    api = {k: v for k, v in per.items() if v["api"]}
    LD = sum(v["legacy_d"] for v in api.values())
    LN = sum(v["legacy_named"] for v in api.values())
    legacy = dict(pooled_pct=round(100 * LD / LN, 2), mismatch_records=LD, denom=LN,
                  n_models=len(api))
    lg_h = api["nousresearch/hermes-3-llama-3.1-70b"]
    legacy["hermes70b"] = f"{lg_h['legacy_d']}/{lg_h['legacy_named']} = {100*lg_h['legacy_d']/lg_h['legacy_named']:.1f}%"

    naming = sorted((100 * v["named"] / v["n"], v["name"], v["named"], v["n"])
                    for v in per.values())
    pool_naming = 100 * sum(v["named"] for v in per.values()) / sum(v["n"] for v in per.values())

    res = dict(
        shipped=SHIPPED,
        S0_baseline=s0,
        S1_known_only=dict(**s1, movers=[
            dict(**m, dropped_other=dict(per[m["model"]]["s1_dropped_other"].most_common(4)),
                 dropped_langs=dict(per[m["model"]]["s1_dropped_lang"]))
            for m in movers(per, "d1")],
            dropped_records=s0["mismatch_records"] - s1["mismatch_records"],
            dropped_lang_pool=dict(sum((v["s1_dropped_lang"] for v in per.values()),
                                       Counter()).most_common()),
            dropped_other_pool_top20=dict(sum((v["s1_dropped_other"] for v in per.values()),
                                              Counter()).most_common(20)),
            note="mismatch requires a KNOWN-canon foreign claim; other:* evidence dropped"),
        S2_conditional=dict(**s2, movers=movers(per, "d2", nkey="named"),
                            pool_naming_rate_pct=round(pool_naming, 1),
                            s0_mismatches_outside_denominator=trace_only,
                            naming_rate_min_med_max=[round(naming[0][0], 1),
                                                     round(statistics.median(x[0] for x in naming), 1),
                                                     round(naming[-1][0], 1)],
                            shyest_5=[dict(name=n, naming_pct=round(r, 1), named=f"{k}/{t}")
                                      for r, n, k, t in naming[:5]]),
        S2b_conditional_adjaware=dict(**s2b,
            note="'named' excludes records whose extraction was adjudicated judge_error/generic"),
        S2_legacy_plan_qna=dict(**legacy,
            note="exact replication of the 2026-07-28 Q&A computation: denominator = "
                 "counted-mismatch OR canon-self-named (any of 4 fields), API models only "
                 "(locals absent from that loop); NOT 'named anything specific'"),
        S3_strict_self=dict(**s3, movers=movers(per, "d3"),
                            flip_inventory={f"{f}->{c}": n for (f, c), n in flips.most_common(30)},
                            strict_only_evidence_killed_by_adj=s3_adjkill,
                            note="separate axis: disclosed-ancestor names count foreign; "
                                 "local strict-only evidence counted without positive adj gate"),
        per_model={mid: {k: v[k] for k in ("name", "family", "n", "named", "named_adj",
                                           "d0", "d1", "d2", "d2b", "d3")}
                   for mid, v in sorted(per.items())},
    )
    (OUT / "sensitivity.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))

    hdr = f"{'variant':22s} {'pooled':>7s} {'records':>12s} {'>=1':>5s} {'>=3':>5s} {'zero':>5s} {'median':>7s}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for tag, s in [("S0 baseline", s0), ("S1 known-only", s1),
                   ("S2 conditional", s2), ("S2b cond adj-aware", s2b),
                   ("S3 strict-self", s3)]:
        print(f"{tag:22s} {s['pooled_pct']:6.2f}% {s['mismatch_records']:>5d}/{s['denom']:<6d} "
              f"{s['models_ge1']:5d} {s['models_ge3']:5d} {s['models_zero']:5d} "
              f"{s['median_model_rate_pct']:6.2f}%")
    print(f"{'S2-legacy (PLAN QA)':22s} {legacy['pooled_pct']:6.2f}% "
          f"{legacy['mismatch_records']:>5d}/{legacy['denom']:<6d}  (API-only; Hermes70B {legacy['hermes70b']})")
    print(f"\npool naming rate (S2 denominator share): {pool_naming:.1f}%  "
          f"| S0 mismatches outside S2 denominator (trace-only): {trace_only}")
    print(f"S3 strict-only evidence suppressed by an adverse adjudication: {s3_adjkill}")

    for tag, key, nkey in [("S1", "d1", "n"), ("S2", "d2", "named"), ("S3", "d3", "n")]:
        print(f"\n{tag} top movers (delta pp vs S0):")
        for m in movers(per, key, nkey=nkey):
            print(f"  {m['name']:34s} {m['s0']:>9s} ({m['s0_pct']:5.1f}%) -> "
                  f"{m['var']:>9s} ({m['var_pct']:5.1f}%)  {m['delta_pp']:+7.1f}pp")

    print("\nS1: dropped-evidence detail for top droppers (top other:* strings, langs):")
    for m in movers(per, "d1")[:10]:
        v = per[m["model"]]
        ev = "; ".join(f"{c[6:]} x{n}" for c, n in v["s1_dropped_other"].most_common(3))
        lg = " ".join(f"{l}:{n}" for l, n in v["s1_dropped_lang"].most_common())
        print(f"  {m['name']:34s} [{lg}]  {ev}")

    print("\nS3 flip inventory (family -> newly-foreign canon, top 15):")
    for (f, c), n in flips.most_common(15):
        print(f"  {f:12s} -> {c:12s} x{n}")

    print("\n5 shyest models (naming rate):")
    for r, n, k, t in naming[:5]:
        print(f"  {n:34s} {k:>3d}/{t} ({r:.1f}%)")


if __name__ == "__main__":
    main()
