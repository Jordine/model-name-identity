"""Census of extraction paths (NULL / GENERIC / KNOWN / OTHER) for claimed_name
and claimed_creator over the shipped identity+creator battery slice.

Slice = exactly make_figs.gather(): complete API models (>=95% coverage, hygiene-
included), prompt_category direct_*/creator_*, prompt_id in BATTERY_CORE,
per-model n>=40, plus the 10 local raw-weights (clean) models.

Run from repo root:  python3 analysis_scratch/generics_audit/census.py
"""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/root/projects/model_identity_evals")
sys.path.insert(0, str(ROOT))

from sweep.analyze import (canon_identity, foreign_claims, lang_of, open_lines,
                           GENERIC_TERMS_N, TRAD2SIMP)
from sweep.make_figs import (BATTERY_CORE, LOCAL_MODELS, complete_models,
                             is_identity, _local_genuine, gather)
from sweep.prompts import CORE, LANGS, prompt_id

OUT = ROOT / "analysis_scratch" / "generics_audit"

# ---------------------------------------------------------------- validation
print("== phase 1: reproduce shipped headline via gather() ==", flush=True)
reg_all, per = gather()
tot_n = sum(v["n"] for v in per.values())
tot_d = sum(v["d"] for v in per.values())
print(f"gather(): {len(per)} models, n={tot_n}, mismatches d={tot_d} "
      f"({100*tot_d/tot_n:.2f}%)", flush=True)
VALID = {"models": len(per), "n": tot_n, "d": tot_d}
KEPT_MODELS = set(per.keys())
del per

# ------------------------------------------------------------- prompt-id map
PID = {}  # prompt_id -> (key, lang, role)
for k, (role, _) in CORE.items():
    if role in ("identity", "creator"):
        for lg in LANGS:
            PID[prompt_id(k, lg)] = (k, lg, role)
assert set(PID) == BATTERY_CORE

# ----------------------------------------------------------- classification
def generic_branch(raw: str) -> str:
    """Which canon_identity branch produced None for a non-empty raw:
    'term' = matched GENERIC_TERMS, 'short' = len<3 fallthrough."""
    low = raw.strip().lower().translate(TRAD2SIMP)
    art = low
    for a in ("an ", "a ", "the "):
        if art.startswith(a):
            art = art[len(a):].strip()
            break
    gm = any(low == g or art == g for g in GENERIC_TERMS_N) or \
        any(g in low for g in GENERIC_TERMS_N
            if len(g) >= 4 or (len(g) >= 2 and not g.isascii()))
    return "term" if gm else "short"


NONSTR = Counter()

def path_of(raw):
    """-> (path, detail).  path in NULL/GENERIC/KNOWN/OTHER.
    detail: GENERIC -> 'term'|'short'; KNOWN -> canon; OTHER -> other-string."""
    if raw is not None and not isinstance(raw, str):
        NONSTR[type(raw).__name__] += 1
        raw = str(raw)
    if not raw:
        return "NULL", None
    c = canon_identity(raw)
    if c is None:
        return "GENERIC", generic_branch(raw)
    if c.startswith("other:"):
        return "OTHER", c
    return "KNOWN", c


# ---------------------------------------------------------------- streaming
print("== phase 2: census pass ==", flush=True)
reg = {m["id"]: m for m in json.loads((ROOT / "config" / "models.json").read_text())["models"]}
hyg = json.loads((ROOT / "config" / "provider_hygiene.json").read_text())
complete = complete_models(reg, hyg)
local_keep = _local_genuine()

records = []          # compact per-record dicts
per_model_n = Counter()

def classify(j, source, mismatch):
    jm = j["judgment"]
    key, lg, role = PID[j["prompt_id"]]
    rn = jm.get("claimed_name")
    rc = jm.get("claimed_creator")
    np_, nd = path_of(rn)
    cp_, cd = path_of(rc)
    return {
        "rk": j["resume_key"], "model": j["model_id"], "family": j.get("family", ""),
        "src": source, "key": key, "lang": lg, "role": role,
        "resp_lang": jm.get("response_language"),
        "name_raw": rn if isinstance(rn, str) else (None if rn is None else str(rn)),
        "creator_raw": rc if isinstance(rc, str) else (None if rc is None else str(rc)),
        "name_path": np_, "name_detail": nd,
        "creator_path": cp_, "creator_detail": cd,
        "no_claim": bool(jm.get("no_identity_claim")),
        "refused": bool(jm.get("refused")),
        "hedged": bool(jm.get("hedged")),
        "evidence": (jm.get("evidence") or "")[:160],
        "mismatch": mismatch,
    }

# API side
n_stream = 0
for line in open_lines(ROOT / "results" / "judgments.jsonl"):
    j = json.loads(line)
    n_stream += 1
    if j.get("judge_error") or not j.get("judgment"):
        continue
    if j["model_id"] not in complete:
        continue
    if not is_identity(j["prompt_category"]) or j["prompt_id"] not in BATTERY_CORE:
        continue
    m = reg.get(j["model_id"])
    if m:
        j["aliases"] = m["aliases"]
        j["family"] = m["family"]
    mm = bool(foreign_claims(j))
    records.append(classify(j, "api", mm))
    per_model_n[j["model_id"]] += 1
print(f"streamed {n_stream} judgment rows; API battery records={len(records)}", flush=True)

# apply gather()'s per-model n>=40 filter
drop = {mid for mid, n in per_model_n.items() if n < 40}
if drop:
    print(f"dropping {len(drop)} API models with n<40: {sorted(drop)}")
    records = [r for r in records if r["model"] not in drop]

# local side (replicates add_local exactly)
n_local = 0
for line in open(ROOT / "results_local" / "judgments_clean.jsonl", encoding="utf-8"):
    j = json.loads(line)
    if not j.get("judgment"):
        continue
    if j["resume_key"].split("::")[-1] != "clean":
        continue
    if j["model_id"] not in LOCAL_MODELS:
        continue
    if not is_identity(j["prompt_category"]) or j["prompt_id"] not in BATTERY_CORE:
        continue
    name, fam, al = LOCAL_MODELS[j["model_id"]]
    jm = j["judgment"]
    from sweep.analyze import is_self
    cn = canon_identity(jm.get("claimed_name"))
    cc = canon_identity(jm.get("claimed_creator"))
    foreign = {c for c in (cn, cc) if c and not is_self(c, fam, al, name)}
    drift = bool(foreign) and (f"{j['resume_key']}::t0" in local_keep)
    j["family"] = fam
    records.append(classify(j, "local", drift))
    n_local += 1
print(f"local battery records={n_local}", flush=True)

models = sorted({r["model"] for r in records})
tot = len(records)
mm_tot = sum(r["mismatch"] for r in records)
print(f"census slice: {tot} records, {len(models)} models, mismatches={mm_tot}")
print(f"validation:   {VALID}")
assert tot == VALID["n"], (tot, VALID)
assert len(models) == VALID["models"]
assert mm_tot == VALID["d"]
print("SLICE VALIDATED ✓", flush=True)
if NONSTR:
    print(f"non-string judgment values coerced: {dict(NONSTR)}")

# ---------------------------------------------------------------- aggregates
PATHS = ["NULL", "GENERIC", "KNOWN", "OTHER"]

def table(rows, dim_fn, field):
    t = defaultdict(Counter)
    for r in rows:
        t[dim_fn(r)][r[f"{field}_path"]] += 1
    return {k: {p: v.get(p, 0) for p in PATHS} for k, v in sorted(t.items())}

res = {"validation": VALID, "n_records": tot, "n_models": len(models)}

for field in ("name", "creator"):
    c = Counter(r[f"{field}_path"] for r in records)
    res[f"{field}_overall"] = {p: c.get(p, 0) for p in PATHS}
    res[f"{field}_by_lang"] = table(records, lambda r: r["lang"], field)
    res[f"{field}_by_role"] = table(records, lambda r: r["role"], field)
    res[f"{field}_by_key"] = table(records, lambda r: r["key"], field)
    res[f"{field}_by_resp_lang"] = table(
        records, lambda r: (r["resp_lang"] or "?") if isinstance(r["resp_lang"], str) else "?", field)

# GENERIC sub-branch split
for field in ("name", "creator"):
    res[f"{field}_generic_sub"] = dict(Counter(
        r[f"{field}_detail"] for r in records if r[f"{field}_path"] == "GENERIC"))

# joint distribution
joint = Counter((r["name_path"], r["creator_path"]) for r in records)
res["joint"] = {f"{a}|{b}": joint.get((a, b), 0) for a in PATHS for b in PATHS}
nameless = sum(v for (a, b), v in joint.items()
               if a in ("NULL", "GENERIC") and b in ("NULL", "GENERIC"))
res["fully_nameless"] = {"n": nameless, "pct": round(100 * nameless / tot, 2)}

# top generic literals (case-folded, keep an example original)
def top_literals(field, path, detail=None, k=25):
    cnt, ex = Counter(), {}
    for r in records:
        if r[f"{field}_path"] != path:
            continue
        if detail and r[f"{field}_detail"] != detail:
            continue
        raw = r[f"{field}_raw"].strip()
        keyl = raw.lower()
        cnt[keyl] += 1
        ex.setdefault(keyl, (raw, r["rk"], r["lang"]))
    return [{"literal": ex[s][0], "n": n, "lang_ex": ex[s][2], "rk_ex": ex[s][1]}
            for s, n in cnt.most_common(k)]

res["top_generic_name"] = top_literals("name", "GENERIC", "term")
res["top_generic_creator"] = top_literals("creator", "GENERIC", "term")
res["short_name_literals"] = top_literals("name", "GENERIC", "short", 30)
res["short_creator_literals"] = top_literals("creator", "GENERIC", "short", 30)

# NULL sub-population (name field): what does NULL mean?
nulls = [r for r in records if r["name_path"] == "NULL"]
res["null_name_profile"] = {
    "n": len(nulls),
    "no_identity_claim": sum(r["no_claim"] for r in nulls),
    "refused": sum(r["refused"] for r in nulls),
    "creator_nonnull": sum(r["creator_path"] != "NULL" for r in nulls),
    "evidence_nonempty": sum(bool(r["evidence"]) for r in nulls),
}
gens = [r for r in records if r["name_path"] == "GENERIC"]
res["generic_name_profile"] = {
    "n": len(gens),
    "no_identity_claim": sum(r["no_claim"] for r in gens),
    "refused": sum(r["refused"] for r in gens),
    "creator_nonnull": sum(r["creator_path"] != "NULL" for r in gens),
}

# NULL:GENERIC systematicity — chi-square over languages / prompt keys
def chi2_table(rows, dim_fn):
    t = defaultdict(lambda: [0, 0])  # dim -> [NULL, GENERIC]
    for r in rows:
        if r["name_path"] == "NULL":
            t[dim_fn(r)][0] += 1
        elif r["name_path"] == "GENERIC":
            t[dim_fn(r)][1] += 1
    return dict(sorted(t.items()))

from scipy.stats import chi2_contingency
import numpy as np

def chi2_report(t):
    obs = np.array([[a, b] for a, b in t.values()])
    obs = obs[obs.sum(1) > 0]
    stat, p, dof, _ = chi2_contingency(obs)
    n = obs.sum()
    v = float(np.sqrt(stat / (n * (min(obs.shape) - 1))))
    return {"chi2": round(float(stat), 1), "dof": int(dof),
            "p": float(p), "cramers_v": round(v, 3),
            "table": {k: {"NULL": a, "GENERIC": b,
                          "pct_GENERIC": round(100 * b / (a + b), 1) if a + b else None}
                      for k, (a, b) in t.items()}}

res["null_vs_generic_by_lang"] = chi2_report(chi2_table(records, lambda r: r["lang"]))
res["null_vs_generic_by_key"] = chi2_report(chi2_table(records, lambda r: r["key"]))
res["null_vs_generic_by_role"] = chi2_report(chi2_table(records, lambda r: r["role"]))

# per-model spread of the NULL:GENERIC choice (is it model-driven?)
pm = defaultdict(lambda: [0, 0])
for r in records:
    if r["name_path"] == "NULL":
        pm[r["model"]][0] += 1
    elif r["name_path"] == "GENERIC":
        pm[r["model"]][1] += 1
shares = sorted(100 * b / (a + b) for a, b in pm.values() if a + b >= 20)
res["generic_share_per_model"] = {
    "models_with_20plus_noneish": len(shares),
    "min": round(shares[0], 1), "p25": round(shares[len(shares)//4], 1),
    "median": round(shares[len(shares)//2], 1),
    "p75": round(shares[3*len(shares)//4], 1), "max": round(shares[-1], 1)}

# mismatch cross-tab: which paths produce shipped mismatches?
mmt = Counter()
for r in records:
    if r["mismatch"]:
        mmt[(r["name_path"], r["creator_path"])] += 1
res["mismatch_by_joint_path"] = {f"{a}|{b}": n for (a, b), n in mmt.most_common()}

# KNOWN raws that also contain a generic term (brand+generic composites)
comp, comp_ex = Counter(), {}
for r in records:
    for field in ("name", "creator"):
        if r[f"{field}_path"] != "KNOWN":
            continue
        raw = r[f"{field}_raw"]
        low = raw.strip().lower().translate(TRAD2SIMP)
        if any(g in low for g in GENERIC_TERMS_N
               if len(g) >= 4 or (len(g) >= 2 and not g.isascii())):
            k = (low, r[f"{field}_detail"])
            comp[k] += 1
            comp_ex.setdefault(k, raw)
res["known_with_generic_substring"] = [
    {"literal": comp_ex[k], "canon": k[1], "n": n} for k, n in comp.most_common(30)]

# OTHER literals — scan for generic-looking leaks the term list missed
oth, oth_ex = Counter(), {}
for r in records:
    for field in ("name", "creator"):
        if r[f"{field}_path"] == "OTHER":
            d = r[f"{field}_detail"]
            oth[d] += 1
            oth_ex.setdefault(d, (r["rk"], r["mismatch"]))
res["top_other"] = [{"other": k, "n": n, "rk_ex": oth_ex[k][0],
                     "mismatch_ex": oth_ex[k][1]} for k, n in oth.most_common(40)]

# ------------------------------------------------ sample join vs main_sweep
print("== phase 3: sample join against main_sweep ==", flush=True)
rng = random.Random(20260729)
def sample_path(path, detail=None, k=12):
    pool = [r for r in records if r["src"] == "api" and r["name_path"] == path
            and (detail is None or r["name_detail"] == detail)]
    # stratify across languages: round-robin languages
    by_lang = defaultdict(list)
    for r in pool:
        by_lang[r["lang"]].append(r)
    out = []
    langs = sorted(by_lang)
    i = 0
    while len(out) < k and any(by_lang.values()):
        lg = langs[i % len(langs)]
        if by_lang[lg]:
            out.append(by_lang[lg].pop(rng.randrange(len(by_lang[lg]))))
        i += 1
    return out

samp_null = sample_path("NULL")
samp_gen = sample_path("GENERIC", "term")
want = {r["rk"] for r in samp_null + samp_gen}
resp = {}
for line in open_lines(ROOT / "results" / "main_sweep.jsonl"):
    r = json.loads(line)
    if r.get("resume_key") in want:
        resp[r["resume_key"]] = (r.get("content_clean") or r.get("content") or "")[:400]
def pack(rs):
    return [{"rk": r["rk"], "lang": r["lang"], "key": r["key"], "model": r["model"],
             "name_raw": r["name_raw"], "no_claim": r["no_claim"],
             "evidence": r["evidence"][:120],
             "response": resp.get(r["rk"], "«not found»")} for r in rs]
res["samples_null"] = pack(samp_null)
res["samples_generic"] = pack(samp_gen)

(OUT / "census.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
print(f"wrote {OUT/'census.json'}")

# quick console summary
for field in ("name", "creator"):
    o = res[f"{field}_overall"]
    print(f"\n{field}: " + "  ".join(f"{p}={o[p]} ({100*o[p]/tot:.1f}%)" for p in PATHS))
print(f"\nfully nameless: {res['fully_nameless']}")
print(f"NULL:GENERIC by lang: {res['null_vs_generic_by_lang']['table']}")
print(f"  chi2={res['null_vs_generic_by_lang']['chi2']} p={res['null_vs_generic_by_lang']['p']:.2e} "
      f"V={res['null_vs_generic_by_lang']['cramers_v']}")
