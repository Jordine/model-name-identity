"""Post-fix statistical analyses on the identity-survey dataset (2026-07-29).

Four analyses, all on the CURRENT working tree (post canon fix; headline
4,992/60,770 = 8.21%, 115/190 models >=1 mismatch — asserted before anything
else is trusted):

  1. Lab-region x language interaction (paired zh-en delta, Chinese vs
     non-Chinese labs; ja/ko mirror as specificity check).
  2. Multiple-comparisons treatment of the 7 paired language deltas
     (Holm + Bonferroni at m=7), via the repo's paired-bootstrap path.
  3. Cutoff at-risk (model, identity) pairs + permutation null, using
     fig_cutoff's own inclusion rules (explain.VERSIONS breakouts,
     CUTOFF_LAG=0.5, MIN_TOT=100).
  4. S2 conditional-on-naming refresh via analysis_scratch/generics_audit/
     sensitivity.py's own run() (post-fix).

Run from repo root:  python analysis_scratch/postfix_analyses.py
Writes analysis_scratch/postfix_analyses.json (+ the .md is written by hand
from the printed output).
"""

import importlib.util
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sweep.make_figs as mf                                  # noqa: E402
from sweep.explain import (CUTOFF_LAG, MIN_TOT, VERSIONS,     # noqa: E402
                           _load, _year)
import sweep.explain as explain                               # noqa: E402

OUT = Path(__file__).resolve().parent
REPS = 10_000
SEED = 12345

RES = {}

# --------------------------------------------------------------------------
# Load ONCE through the repo's own gather(); cache so explain.gather() reuses it.
reg, per = mf.gather()
_cached = (reg, per)
mf.gather = lambda: _cached          # explain.gather() resolves this at call time

# --------------------------------------------------------------------------
# SANITY ANCHOR — post-fix headline must reproduce exactly.
D = sum(v["d"] for v in per.values())
N = sum(v["n"] for v in per.values())
GE1 = sum(1 for v in per.values() if v["d"] >= 1)
anchor = dict(pooled_d=D, pooled_n=N, pooled_pct=round(100 * D / N, 2),
              models=len(per), models_ge1=GE1)
print(f"ANCHOR: {D}/{N} = {100*D/N:.2f}%  models={len(per)} ge1={GE1}")
assert (D, N, len(per), GE1) == (4992, 60770, 190, 115), anchor
RES["anchor"] = anchor


def boot_mean(vals, rng, reps=REPS):
    k = len(vals)
    return [sum(rng.choices(vals, k=k)) / k for _ in range(reps)]


def pct(sorted_boots, q):
    return sorted_boots[int(q * len(sorted_boots))]


def p_two_sided(boots):
    """Add-one two-sided bootstrap p for H0: mean == 0."""
    reps = len(boots)
    lo = (1 + sum(1 for b in boots if b <= 0)) / (reps + 1)
    hi = (1 + sum(1 for b in boots if b >= 0)) / (reps + 1)
    return min(1.0, 2 * min(lo, hi))


# ==========================================================================
# 1. LAB-REGION x LANGUAGE INTERACTION
# ==========================================================================
# PRC labs present in config/models.json (task list + Nex AGI, which is the
# Shanghai Innovation Institute alliance — confirmed via web). Families listed
# but absent from the registry (internlm, zhinao360, sparkdesk, huawei) or from
# the 190-model complete pool (bytedance) can't contribute rows.
CN_FAMS = {"qwen", "alibaba", "deepseek", "kimi", "moonshot", "zhipu", "baidu",
           "tencent", "bytedance", "ant", "internlm", "kuaishou", "stepfun",
           "xiaomi", "minimax", "zhinao360", "sparkdesk", "huawei", "nex"}
# ambiguous non-PRC-non-Western labs excluded from BOTH groups:
EXC_FAMS = {"naver", "upstage", "yandex", "sber", "sakana", "aisingapore"}


def fam_of(mid):
    return reg[mid]["family"]


def group_of(mid):
    f = fam_of(mid)
    if f in CN_FAMS:
        return "cn"
    if f in EXC_FAMS:
        return None
    return "noncn"


def paired_delta(v, lang):
    de, ne = v["lang"].get("en", (0, 0))
    dl, nl = v["lang"].get(lang, (0, 0))
    if not ne or not nl:
        return None
    return 100 * dl / nl - 100 * de / ne


def region_test(lang, rng):
    groups = {"cn": [], "noncn": []}
    for mid, v in per.items():
        g = group_of(mid)
        d = paired_delta(v, lang)
        if g and d is not None:
            groups[g].append(d)
    out = {}
    boots = {}
    for g, ds in groups.items():
        bs = sorted(boot_mean(ds, rng))
        boots[g] = bs
        out[g] = dict(n_models=len(ds), mean_delta_pp=round(sum(ds) / len(ds), 2),
                      ci95=[round(pct(bs, .025), 2), round(pct(bs, .975), 2)])
    # difference: independent cluster bootstrap of the two groups per rep
    rng2 = random.Random(SEED + 1)
    dcn, dnc = groups["cn"], groups["noncn"]
    diffs = sorted(
        sum(rng2.choices(dcn, k=len(dcn))) / len(dcn)
        - sum(rng2.choices(dnc, k=len(dnc))) / len(dnc)
        for _ in range(REPS))
    point = out["cn"]["mean_delta_pp"] - out["noncn"]["mean_delta_pp"]
    out["diff_cn_minus_noncn"] = dict(
        point_pp=round(sum(dcn) / len(dcn) - sum(dnc) / len(dnc), 2),
        ci95=[round(pct(diffs, .025), 2), round(pct(diffs, .975), 2)],
        p_boot=round(p_two_sided(diffs), 5))
    return out


rng = random.Random(SEED)
a1 = {}
for lang in ("zh", "ja", "ko"):
    a1[lang] = region_test(lang, rng)
# final mapping (families actually contributing models in the 190 pool)
fam_counts = Counter(fam_of(m) for m in per)
mapping = {"chinese_lab_families_in_pool":
               {f: fam_counts[f] for f in sorted(fam_counts) if f in CN_FAMS},
           "non_chinese_families_in_pool":
               {f: fam_counts[f] for f in sorted(fam_counts)
                if f not in CN_FAMS and f not in EXC_FAMS},
           "excluded_families": sorted(EXC_FAMS),
           "excluded_present_in_pool":
               {f: fam_counts[f] for f in sorted(fam_counts) if f in EXC_FAMS},
           "cn_families_listed_but_absent":
               sorted(f for f in CN_FAMS if fam_counts.get(f, 0) == 0)}
a1["mapping"] = mapping
RES["a1_region_language"] = a1
print("\n=== A1: lab-region x language (paired delta = lang - en, pp) ===")
for lang in ("zh", "ja", "ko"):
    r = a1[lang]
    print(f"  {lang}: CN {r['cn']['mean_delta_pp']:+.2f} {r['cn']['ci95']} (n={r['cn']['n_models']})"
          f"  nonCN {r['noncn']['mean_delta_pp']:+.2f} {r['noncn']['ci95']} (n={r['noncn']['n_models']})"
          f"  diff {r['diff_cn_minus_noncn']['point_pp']:+.2f} {r['diff_cn_minus_noncn']['ci95']}"
          f" p={r['diff_cn_minus_noncn']['p_boot']}")
print("  mapping:", json.dumps(mapping["chinese_lab_families_in_pool"]))

# ==========================================================================
# 2. MULTIPLE COMPARISONS on the 7 paired language deltas
# ==========================================================================
LANGS7 = ["zh", "ja", "ko", "ru", "fr", "es", "vi"]
deltas = {l: [] for l in LANGS7}
for v in per.values():
    de, ne = v["lang"].get("en", (0, 0))
    if not ne:
        continue
    for l in LANGS7:
        d, n = v["lang"].get(l, (0, 0))
        if n:
            deltas[l].append(100 * d / n - 100 * de / ne)

# (a) exact replication of the repo's --paired output (reps=4000, one rng):
rng = random.Random(SEED)
repo_tbl = {}
for l in LANGS7:
    ds = deltas[l]
    k = len(ds)
    point = sum(ds) / k
    boots = sorted(sum(rng.choices(ds, k=k)) / k for _ in range(4000))
    repo_tbl[l] = (round(point, 2), round(boots[int(.025 * 4000)], 2),
                   round(boots[int(.975 * 4000)], 2), k)

# (b) 10k-rep version with bootstrap p per language (same path, higher reps):
rng = random.Random(SEED)
a2 = {}
for l in LANGS7:
    ds = deltas[l]
    k = len(ds)
    point = sum(ds) / k
    boots = sorted(boot_mean(ds, rng))
    a2[l] = dict(delta_pp=round(point, 2),
                 ci95=[round(pct(boots, .025), 2), round(pct(boots, .975), 2)],
                 p_raw=p_two_sided(boots), n_models=k)

# Holm + Bonferroni at m=7
m = len(LANGS7)
order = sorted(LANGS7, key=lambda l: a2[l]["p_raw"])
running = 0.0
for i, l in enumerate(order):
    running = max(running, (m - i) * a2[l]["p_raw"])
    a2[l]["p_holm"] = round(min(1.0, running), 5)
for l in LANGS7:
    a2[l]["p_bonf"] = round(min(1.0, m * a2[l]["p_raw"]), 5)
    a2[l]["p_raw"] = round(a2[l]["p_raw"], 5)
    a2[l]["sig_holm_05"] = a2[l]["p_holm"] < 0.05
    a2[l]["sig_bonf_05"] = a2[l]["p_bonf"] < 0.05
RES["a2_multiple_comparisons"] = {"repo_paired_4000rep_replication": repo_tbl,
                                  "deltas_10k": a2,
                                  "note": "p = add-one two-sided bootstrap p (floor 2/10001 = 0.0002)"}
print("\n=== A2: paired deltas vs en + Holm/Bonferroni (m=7) ===")
for l in LANGS7:
    r = a2[l]
    print(f"  {l}: {r['delta_pp']:+.2f}pp {r['ci95']}  p={r['p_raw']:.5f}"
          f"  holm={r['p_holm']:.5f}  bonf={r['p_bonf']:.5f}"
          f"  {'H' if r['sig_holm_05'] else '.'}{'B' if r['sig_bonf_05'] else '.'}")

# ==========================================================================
# 3. CUTOFF AT-RISK PAIRS + PERMUTATION NULL (fig_cutoff's rules)
# ==========================================================================
data = explain.gather()          # uses the cached (reg, per)
cut = _load("model_cutoffs.json")
rows = []                        # (mid, claims Counter, tot, x, documented?)
for mid, d in data.items():
    c = cut.get(mid) or {}
    yc, yr = _year(c.get("cutoff")), _year(c.get("release_date"))
    if yc is not None:
        x, doc = yc, True
    elif yr is not None:
        x, doc = yr - CUTOFF_LAG, False
    else:
        continue
    if d["tot"] < MIN_TOT:
        continue
    rows.append((mid, d["claims"], d["tot"], x, doc))

BREAKOUT = {t: min(y for y, _, b in VERSIONS[t] if b) for t in VERSIONS}
FIRST = {t: min(y for y, _, _ in VERSIONS[t]) for t in VERSIONS}
targets = list(VERSIONS)
EPS = 1 / 12          # both axes are month-resolution; a same-month "gap" is a tie

def risk_stats(thresh, eps=0.0):
    at_risk, observed = [], []
    for mid, claims, tot, x, doc in rows:
        for t in targets:
            if x < thresh[t] - eps:
                at_risk.append((mid, t, doc))
                kk = claims.get(t, 0)
                if kk > 0:
                    observed.append(dict(model=mid, target=t, records=kk,
                                         pct=round(100 * kk / tot, 2),
                                         cutoff_x=round(x, 3), documented=doc,
                                         gap_yr=round(thresh[t] - x, 3)))
    return at_risk, observed

ar_b, obs_b = risk_stats(BREAKOUT)
ar_f, obs_f = risk_stats(FIRST)
ar_be, obs_be = risk_stats(BREAKOUT, eps=EPS)     # tie-robust (>=1 month gap)

# permutation null: shuffle which model carries which claim-target set.
# Claim WEIGHTS (record counts) travel with the set for the record-weighted null.
claim_sets = [tuple((t, claims[t]) for t in targets if claims.get(t, 0) > 0)
              for _, claims, _, _, _ in rows]
xs = [x for *_, x, _ in rows]
docs = [doc for *_, doc in rows]
rng = random.Random(SEED)
k = len(rows)
cnts_b, cnts_f, cnts_be = [], [], []
cnts_b_doc, cnts_b_est = [], []
recs_b, recs_f = [], []
idx = list(range(k))
for _ in range(REPS):
    rng.shuffle(idx)
    cb = cf = cbd = cbe_ = ce = rb = rf = 0
    for pos in range(k):
        s = claim_sets[idx[pos]]
        if not s:
            continue
        x = xs[pos]
        for t, w in s:
            if x < BREAKOUT[t]:
                cb += 1
                rb += w
                if docs[pos]:
                    cbd += 1
                else:
                    cbe_ += 1
            if x < BREAKOUT[t] - EPS:
                ce += 1
            if x < FIRST[t]:
                cf += 1
                rf += w
    cnts_b.append(cb); cnts_f.append(cf); cnts_be.append(ce)
    cnts_b_doc.append(cbd); cnts_b_est.append(cbe_)
    recs_b.append(rb); recs_f.append(rf)

def summ(cnts, obs=0):
    return dict(expected=round(sum(cnts) / len(cnts), 2),
                p_zero=round(sum(1 for c in cnts if c == 0) / len(cnts), 5),
                p_le_obs=round(sum(1 for c in cnts if c <= obs) / len(cnts), 5),
                max=max(cnts))

obs_b_records = sum(o["records"] for o in obs_b)
obs_f_records = sum(o["records"] for o in obs_f)
n_ties = sum(1 for o in obs_b if o["gap_yr"] <= EPS + 1e-9)
a3 = dict(
    models_included=k,
    documented=sum(docs), estimated=k - sum(docs),
    breakout_dates={t: round(BREAKOUT[t], 2) for t in targets},
    first_release_dates={t: round(FIRST[t], 2) for t in targets},
    at_risk_pairs_breakout=len(ar_b),
    at_risk_by_prov_breakout={"documented": sum(1 for *_, d in ar_b if d),
                              "estimated": sum(1 for *_, d in ar_b if not d)},
    at_risk_by_target_breakout=dict(Counter(t for _, t, _ in ar_b)),
    observed_breakout=obs_b, n_observed_breakout=len(obs_b),
    observed_breakout_by_prov={"documented": sum(1 for o in obs_b if o["documented"]),
                               "estimated": sum(1 for o in obs_b if not o["documented"])},
    observed_breakout_same_month_ties=n_ties,
    observed_breakout_records=obs_b_records,
    null_breakout_pairs=summ(cnts_b, len(obs_b)),
    null_breakout_records=summ(recs_b, obs_b_records),
    null_breakout_documented=summ(cnts_b_doc, sum(1 for o in obs_b if o["documented"])),
    null_breakout_estimated=summ(cnts_b_est, sum(1 for o in obs_b if not o["documented"])),
    tie_robust_eps_1mo=dict(at_risk_pairs=len(ar_be), observed=len(obs_be),
                            null=summ(cnts_be, len(obs_be))),
    at_risk_pairs_first_release=len(ar_f),
    at_risk_by_prov_first={"documented": sum(1 for *_, d in ar_f if d),
                           "estimated": sum(1 for *_, d in ar_f if not d)},
    observed_first_release=obs_f, n_observed_first=len(obs_f),
    observed_first_records=obs_f_records,
    null_first_release_pairs=summ(cnts_f, len(obs_f)),
    null_first_release_records=summ(recs_f, obs_f_records),
)
RES["a3_cutoff_permutation"] = a3
print("\n=== A3: cutoff at-risk pairs + permutation null ===")
print(f"  models included: {k} (doc {sum(docs)} / est {k - sum(docs)})")
print(f"  BREAKOUT: at-risk pairs = {len(ar_b)}  observed claim-pairs = {len(obs_b)}"
      f" ({obs_b_records} records; {n_ties} same-month ties)")
for o in obs_b:
    print("    OBS:", o)
print(f"  null pairs: E = {a3['null_breakout_pairs']['expected']}, P(0) = {a3['null_breakout_pairs']['p_zero']},"
      f" P(<=obs) = {a3['null_breakout_pairs']['p_le_obs']}")
print(f"  null records: E = {a3['null_breakout_records']['expected']},"
      f" P(<= {obs_b_records}) = {a3['null_breakout_records']['p_le_obs']}")
print(f"  by provenance: doc E={a3['null_breakout_documented']['expected']}"
      f" est E={a3['null_breakout_estimated']['expected']}")
print(f"  tie-robust (>=1mo gap): at-risk = {len(ar_be)}, observed = {len(obs_be)},"
      f" E = {a3['tie_robust_eps_1mo']['null']['expected']},"
      f" P(<=obs) = {a3['tie_robust_eps_1mo']['null']['p_le_obs']}")
print(f"  FIRST-RELEASE (strict): at-risk = {len(ar_f)}  observed = {len(obs_f)}"
      f"  E = {a3['null_first_release_pairs']['expected']},"
      f" P(0) = {a3['null_first_release_pairs']['p_zero']}"
      f"  (records E = {a3['null_first_release_records']['expected']})")
for o in obs_f:
    print("    OBS(first):", o)

# ==========================================================================
# 4. S2 CONDITIONAL refresh (sensitivity.py's own machinery, post-fix data)
# ==========================================================================
spec = importlib.util.spec_from_file_location(
    "sens", OUT / "generics_audit" / "sensitivity.py")
sens = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sens)

per4, _flips, trace_only, _ = sens.run()
s0 = sens.summarize(per4, "d0")
assert (s0["mismatch_records"], s0["denom"], s0["models_ge1"], s0["n_models"]) \
    == (4992, 60770, 115, 190), s0    # harness == pipeline, post-fix
s2 = sens.summarize(per4, "d2", nkey="named")
pool_naming = 100 * sum(v["named"] for v in per4.values()) / sum(v["n"] for v in per4.values())
movers5 = sens.movers(per4, "d2", nkey="named", top=5)
naming = sorted((100 * v["named"] / v["n"], v["name"], v["named"], v["n"])
                for v in per4.values())
a4 = dict(S0_check=s0, S2_conditional=s2,
          pool_naming_rate_pct=round(pool_naming, 2),
          s0_mismatches_outside_denominator=trace_only,
          naming_min_med_max=[round(naming[0][0], 1),
                              round(statistics.median(x[0] for x in naming), 1),
                              round(naming[-1][0], 1)],
          top5_reshuffles=movers5)
RES["a4_s2_conditional"] = a4
print("\n=== A4: S2 conditional refresh (post-fix) ===")
print(f"  S0 harness check: {s0['mismatch_records']}/{s0['denom']} = {s0['pooled_pct']}%  (match)")
print(f"  S2 conditional: {s2['mismatch_records']}/{s2['denom']} = {s2['pooled_pct']}%"
      f"   naming rate = {pool_naming:.2f}%   trace-only mismatches excluded: {trace_only}")
for mv in movers5:
    print(f"  {mv['name']:34s} {mv['s0']:>9s} ({mv['s0_pct']}%) -> {mv['var']:>9s}"
          f" ({mv['var_pct']}%)  {mv['delta_pp']:+.1f}pp")

(OUT / "postfix_analyses.json").write_text(
    json.dumps(RES, indent=1, ensure_ascii=False))
print("\nwrote", OUT / "postfix_analyses.json")
