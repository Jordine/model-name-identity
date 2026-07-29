"""Follow-up analyses on the battery slice; merges results into census.json and
dumps compact per-record data to records.jsonl.gz for ad-hoc queries."""
import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/root/projects/model_identity_evals")
sys.path.insert(0, str(ROOT))
from sweep.analyze import (canon_identity, foreign_claims, is_self, open_lines,
                           GENERIC_TERMS_N, TRAD2SIMP)
from sweep.make_figs import (BATTERY_CORE, LOCAL_MODELS, complete_models,
                             is_identity, _local_genuine)
from sweep.prompts import CORE, LANGS, prompt_id

OUT = ROOT / "analysis_scratch" / "generics_audit"
PID = {prompt_id(k, lg): (k, lg, role) for k, (role, _) in CORE.items()
       if role in ("identity", "creator") for lg in LANGS}


def generic_branch(raw):
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


def path_of(raw):
    if raw is not None and not isinstance(raw, str):
        raw = str(raw)
    if not raw:
        return "NULL", None
    c = canon_identity(raw)
    if c is None:
        return "GENERIC", generic_branch(raw)
    if c.startswith("other:"):
        return "OTHER", c
    return "KNOWN", c


reg = {m["id"]: m for m in json.loads((ROOT / "config" / "models.json").read_text())["models"]}
hyg = json.loads((ROOT / "config" / "provider_hygiene.json").read_text())
complete = complete_models(reg, hyg)
local_keep = _local_genuine()

records = []
pmn = Counter()
for line in open_lines(ROOT / "results" / "judgments.jsonl"):
    j = json.loads(line)
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
    jm = j["judgment"]
    key, lg, role = PID[j["prompt_id"]]
    np_, nd = path_of(jm.get("claimed_name"))
    cp_, cd = path_of(jm.get("claimed_creator"))
    records.append(dict(
        rk=j["resume_key"], model=j["model_id"], family=j.get("family", ""),
        aliases=j.get("aliases", []), expected=j["expected_identity"], src="api",
        pid=j["prompt_id"], key=key, lang=lg, role=role,
        name_raw=jm.get("claimed_name") if isinstance(jm.get("claimed_name"), str) else None,
        creator_raw=jm.get("claimed_creator") if isinstance(jm.get("claimed_creator"), str) else None,
        name_path=np_, name_detail=nd, creator_path=cp_, creator_detail=cd,
        mismatch=bool(foreign_claims(j))))
    pmn[j["model_id"]] += 1
drop = {mid for mid, n in pmn.items() if n < 40}
records = [r for r in records if r["model"] not in drop]

for line in open(ROOT / "results_local" / "judgments_clean.jsonl", encoding="utf-8"):
    j = json.loads(line)
    if not j.get("judgment") or j["resume_key"].split("::")[-1] != "clean":
        continue
    if j["model_id"] not in LOCAL_MODELS:
        continue
    if not is_identity(j["prompt_category"]) or j["prompt_id"] not in BATTERY_CORE:
        continue
    name, fam, al = LOCAL_MODELS[j["model_id"]]
    jm = j["judgment"]
    cn = canon_identity(jm.get("claimed_name"))
    cc = canon_identity(jm.get("claimed_creator"))
    foreign = {c for c in (cn, cc) if c and not is_self(c, fam, al, name)}
    drift = bool(foreign) and (f"{j['resume_key']}::t0" in local_keep)
    key, lg, role = PID[j["prompt_id"]]
    np_, nd = path_of(jm.get("claimed_name"))
    cp_, cd = path_of(jm.get("claimed_creator"))
    records.append(dict(
        rk=j["resume_key"], model=j["model_id"], family=fam, aliases=al,
        expected=name, src="local", pid=j["prompt_id"], key=key, lang=lg, role=role,
        name_raw=jm.get("claimed_name") if isinstance(jm.get("claimed_name"), str) else None,
        creator_raw=jm.get("claimed_creator") if isinstance(jm.get("claimed_creator"), str) else None,
        name_path=np_, name_detail=nd, creator_path=cp_, creator_detail=cd,
        mismatch=drift))
assert len(records) == 60770, len(records)
print(f"records rebuilt: {len(records)}", flush=True)

with gzip.open(OUT / "records.jsonl.gz", "wt", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

add = {}

# ---- 1. NULL:GENERIC by language, identity-role records only
from scipy.stats import chi2_contingency
import numpy as np

def ng_table(rows, dim):
    t = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["name_path"] == "NULL":
            t[dim(r)][0] += 1
        elif r["name_path"] == "GENERIC":
            t[dim(r)][1] += 1
    return dict(sorted(t.items()))

def chi2_report(t):
    obs = np.array([[a, b] for a, b in t.values()])
    obs = obs[obs.sum(1) > 0]
    stat, p, dof, _ = chi2_contingency(obs)
    n = obs.sum()
    v = float(np.sqrt(stat / (n * (min(obs.shape) - 1))))
    return {"chi2": round(float(stat), 1), "dof": int(dof), "p": float(p),
            "cramers_v": round(v, 3),
            "table": {k: {"NULL": a, "GENERIC": b,
                          "pct_GENERIC": round(100 * b / (a + b), 1) if a + b else None}
                      for k, (a, b) in t.items()}}

idrows = [r for r in records if r["role"] == "identity"]
add["null_vs_generic_by_lang_identity_only"] = chi2_report(ng_table(idrows, lambda r: r["lang"]))

# ---- 2. within-cell (model x prompt_id) NULL/GENERIC mixing
cells = defaultdict(list)
for r in records:
    cells[(r["model"], r["pid"])].append(r["name_path"])
none_cells = mixed = 0
for paths in cells.values():
    s = set(paths)
    if s <= {"NULL", "GENERIC"} and len(paths) >= 2:
        none_cells += 1
        if len(s) == 2:
            mixed += 1
add["within_cell_mixing"] = {
    "cells_all_none_path": none_cells, "cells_mixed_null_generic": mixed,
    "pct_mixed": round(100 * mixed / none_cells, 1)}

# ---- 3. exact-duplicate responses with divergent paths
want = {r["rk"] for r in records if r["src"] == "api" and r["name_path"] in ("NULL", "GENERIC")}
rk2hash, hash2text = {}, {}
for line in open_lines(ROOT / "results" / "main_sweep.jsonl"):
    rec = json.loads(line)
    rk = rec.get("resume_key")
    if rk in want:
        txt = (rec.get("content_clean") or rec.get("content") or "").strip()
        h = hashlib.md5(txt.encode()).hexdigest()
        rk2hash[rk] = h
        hash2text.setdefault(h, txt[:200])
byh = defaultdict(lambda: {"NULL": 0, "GENERIC": 0, "ex": {}})
for r in records:
    h = rk2hash.get(r["rk"])
    if h is None:
        continue
    byh[h][r["name_path"]] += 1
    byh[h]["ex"].setdefault(r["name_path"], (r["rk"], r["name_raw"]))
div = [(h, d) for h, d in byh.items() if d["NULL"] and d["GENERIC"]]
dup_hashes = [(h, d) for h, d in byh.items() if d["NULL"] + d["GENERIC"] >= 2]
add["duplicate_response_divergence"] = {
    "distinct_responses_with_2plus_nonepath_records": len(dup_hashes),
    "responses_judged_both_ways": len(div),
    "pct": round(100 * len(div) / max(len(dup_hashes), 1), 1),
    "examples": [{"text": hash2text[h], "NULL_n": d["NULL"], "GENERIC_n": d["GENERIC"],
                  "null_ex": d["ex"].get("NULL"), "generic_ex": d["ex"].get("GENERIC")}
                 for h, d in sorted(div, key=lambda x: -(x[1]["NULL"] + x[1]["GENERIC"]))[:8]]}

# ---- 4. Open Assistant impact (battery records, either field)
def is_oa(s):
    if not s:
        return False
    l = s.strip().lower()
    return "open assistant" in l or "オープンアシスタント" in l or "openassistant" in l
oa = [r for r in records if is_oa(r["name_raw"]) or is_oa(r["creator_raw"])]
oa_by_model = Counter(r["model"] for r in oa)
would_flip = 0
for r in oa:
    if not r["mismatch"]:
        if not is_self("other:open assistant", r["family"], r["aliases"], r["expected"]):
            would_flip += 1
add["open_assistant"] = {
    "battery_records": len(oa), "by_model": dict(oa_by_model.most_common()),
    "already_mismatch_via_other_field": sum(r["mismatch"] for r in oa),
    "would_flip_to_mismatch_if_not_generic": would_flip}

# ---- 5. generic-looking OTHER leaks (full dump of OTHER details w/ mismatch)
oth = defaultdict(lambda: [0, 0])
for r in records:
    for f in ("name", "creator"):
        if r[f"{f}_path"] == "OTHER":
            oth[r[f"{f}_detail"]][0] += 1
            oth[r[f"{f}_detail"]][1] += r["mismatch"]
with open(OUT / "other_details_full.tsv", "w", encoding="utf-8") as f:
    f.write("n\tn_mismatch\tother\n")
    for k, (n, nm) in sorted(oth.items(), key=lambda x: -x[1][0]):
        f.write(f"{n}\t{nm}\t{k}\n")
# curated generic-ish patterns with no brand token
GEN_PAT = ["modèle", "model", "модел", "язык", "нейро", "linguistique", "langage",
           "언어", "모델", "ngôn ngữ", "assistant", "asistente", "ассист",
           "アシスタント", "助手", "chatbot", "чат-бот", "intelligen", "인공지능",
           "人工知能", "人工智能", "trí tuệ", "ии-", " ai", "ai "]
BRANDS = ["gemma", "nex", "perplexity", "step", "kat", "jamba", "ai21", "ibm", "arcee",
          "trinity", "ling", "phi", "minimax", "mistral", "grok", "hermes", "command",
          "커맨드", "コマンド", "gpt", "llama", "lama", "qwen", "claude", "gemini",
          "sonar", "granite", "olmo", "nova", "aya", "solar", "palmyra", "lfm",
          "virtuoso", "mercury", "pi", "reka", "sea-lion", "laguna", "hugging"]
leak = {k: v for k, v in oth.items()
        if any(p in k for p in GEN_PAT) and not any(b in k for b in BRANDS)}
add["other_generic_leaks"] = {
    "n_records": sum(v[0] for v in leak.values()),
    "n_mismatch": sum(v[1] for v in leak.values()),
    "top": [{"other": k, "n": v[0], "n_mismatch": v[1]}
            for k, v in sorted(leak.items(), key=lambda x: -x[1][0])[:30]]}

# ---- 6. Shanghai AI Lab / short-branch named-entity records in battery
odd = ["上海人工智能实验室", "T5", "큐웬", "华为"]
hits = [dict(rk=r["rk"], model=r["model"], field=f, raw=r[f"{f}_raw"],
             path=r[f"{f}_path"], mismatch=r["mismatch"])
        for r in records for f in ("name", "creator")
        if r[f"{f}_raw"] and r[f"{f}_raw"].strip() in odd]
add["named_entity_swallowed"] = hits

res = json.loads((OUT / "census.json").read_text())
res.update(add)
(OUT / "census.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
print(json.dumps(add, ensure_ascii=False, indent=1)[:5000])
