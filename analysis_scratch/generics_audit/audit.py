"""Audit of other:* false-positive leakage in the identity-claims study.

Phase 1: reproduce the shipped battery slice (190 models / 60,770 records / 4,849 mismatches)
Phase 2: enumerate other:* canon values on battery records
Phase 3: join other:-only flagged records against adjudications
Phase 4: adjudication coverage of all flagged records
Outputs into analysis_scratch/generics_audit/.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/root/projects/model_identity_evals")
sys.path.insert(0, str(ROOT))

from sweep.analyze import (canon_identity, is_self, FAMILY_SELF, lang_of,
                           open_lines, _adjudications)
from sweep.make_figs import (BATTERY_CORE, is_identity, complete_models,
                             LOCAL_MODELS, _local_genuine)
from sweep.prompts import prompts_for_model

OUT = ROOT / "analysis_scratch" / "generics_audit"

# ---------------------------------------------------------------- phase 1
reg = {m["id"]: m for m in json.loads((ROOT / "config" / "models.json").read_text())["models"]}
hyg = json.loads((ROOT / "config" / "provider_hygiene.json").read_text())
complete = complete_models(reg, hyg)
print(f"complete API models: {len(complete)}")

ADJ = _adjudications()          # {adj_key: verdict}
LOCAL_KEEP = _local_genuine()   # local adj_keys with genuine_foreign

# local adjudications full verdict map (for verdict distribution on local recs)
LOCAL_ADJ = {}
p = ROOT / "results" / "adjudications_local.jsonl"
if p.exists():
    for l in open(p, encoding="utf-8"):
        try:
            d = json.loads(l)
        except json.JSONDecodeError:
            continue
        if d.get("verdict"):
            LOCAL_ADJ[d["adj_key"]] = d["verdict"]

records = []   # per-battery-record audit rows

def audit_api_record(j):
    """Replicates foreign_claims() field logic, exposing intermediate state."""
    jm = j["judgment"]
    fam = j.get("family", "")
    fields = ["claimed_name", "claimed_creator"]
    if jm.get("reasoning_identity_stance") != "role_play":
        fields += ["reasoning_claimed_name", "reasoning_claimed_creator"]
    canon_by_field = {}
    foreign = set()
    for f in fields:
        c = canon_identity(jm.get(f))
        canon_by_field[f] = c
        if c and not is_self(c, fam, j.get("aliases", []), j["expected_identity"]):
            foreign.add(c)
    adj_k = f"{j['resume_key']}::t{j.get('turn_index', 0)}"
    adj = ADJ.get(adj_k)
    counted = bool(foreign) and (adj is None or adj == "genuine_foreign")
    return canon_by_field, foreign, adj_k, adj, counted

per_model = defaultdict(lambda: {"n": 0, "d": 0})

# stream judgments (API)
n_lines = 0
for line in open_lines(ROOT / "results" / "judgments.jsonl"):
    n_lines += 1
    j = json.loads(line)
    if j.get("judge_error") or not j.get("judgment"):
        continue
    m = reg.get(j["model_id"])
    if m:
        j["aliases"] = m["aliases"]
        j["family"] = m["family"]
    if j["model_id"] not in complete:
        continue
    if not is_identity(j["prompt_category"]) or j["prompt_id"] not in BATTERY_CORE:
        continue
    canon_by_field, foreign, adj_k, adj, counted = audit_api_record(j)
    per_model[j["model_id"]]["n"] += 1
    if counted:
        per_model[j["model_id"]]["d"] += 1
    jm = j["judgment"]
    records.append({
        "src": "api",
        "model_id": j["model_id"], "family": j.get("family", ""),
        "resume_key": j["resume_key"], "adj_key": adj_k,
        "prompt_id": j["prompt_id"], "lang": lang_of(j["prompt_category"]),
        "raw_name": jm.get("claimed_name"), "raw_creator": jm.get("claimed_creator"),
        "raw_rname": jm.get("reasoning_claimed_name"), "raw_rcreator": jm.get("reasoning_claimed_creator"),
        "stance": jm.get("reasoning_identity_stance"),
        "canon": canon_by_field, "foreign": sorted(foreign),
        "adj": adj, "counted": counted,
        "evidence": (jm.get("evidence") or "")[:300],
    })

print(f"judgment lines read: {n_lines}; API battery records: {len(records)}")

# local models
jpath = ROOT / "results_local" / "judgments_clean.jsonl"
local_per = defaultdict(lambda: {"n": 0, "d": 0})
local_records = []
for l in open(jpath, encoding="utf-8"):
    j = json.loads(l)
    if not j.get("judgment"):
        continue
    if j["resume_key"].split("::")[-1] != "clean":
        continue
    mid = j["model_id"]
    if mid not in LOCAL_MODELS:
        continue
    name, fam, al = LOCAL_MODELS[mid]
    cat = j["prompt_category"]
    if not is_identity(cat) or j["prompt_id"] not in BATTERY_CORE:
        continue
    jm = j["judgment"] or {}
    cn = canon_identity(jm.get("claimed_name"))
    cc = canon_identity(jm.get("claimed_creator"))
    foreign = {c for c in (cn, cc) if c and not is_self(c, fam, al, name)}
    adj_k = f"{j['resume_key']}::t0"
    counted = bool(foreign) and (adj_k in LOCAL_KEEP)
    local_per[mid]["n"] += 1
    if counted:
        local_per[mid]["d"] += 1
    local_records.append({
        "src": "local",
        "model_id": mid, "family": fam,
        "resume_key": j["resume_key"], "adj_key": adj_k,
        "prompt_id": j["prompt_id"], "lang": lang_of(cat),
        "raw_name": jm.get("claimed_name"), "raw_creator": jm.get("claimed_creator"),
        "raw_rname": None, "raw_rcreator": None, "stance": jm.get("reasoning_identity_stance"),
        "canon": {"claimed_name": cn, "claimed_creator": cc},
        "foreign": sorted(foreign),
        "adj": LOCAL_ADJ.get(adj_k), "counted": counted,
        "evidence": (jm.get("evidence") or "")[:300],
    })

# apply n>=40 filters like gather()/add_local
kept_api = {mid for mid, v in per_model.items() if v["n"] >= 40}
kept_local = {mid for mid, v in local_per.items() if v["n"] >= 40}
records = [r for r in records if r["model_id"] in kept_api]
local_records = [r for r in local_records if r["model_id"] in kept_local]
all_records = records + local_records

tot_n = sum(per_model[m]["n"] for m in kept_api) + sum(local_per[m]["n"] for m in kept_local)
tot_d = sum(per_model[m]["d"] for m in kept_api) + sum(local_per[m]["d"] for m in kept_local)
n_models = len(kept_api) + len(kept_local)
models_ge1 = sum(1 for m in kept_api if per_model[m]["d"] > 0) + \
             sum(1 for m in kept_local if local_per[m]["d"] > 0)
import statistics
rates = [per_model[m]["d"] / per_model[m]["n"] for m in kept_api] + \
        [local_per[m]["d"] / local_per[m]["n"] for m in kept_local]
print(f"\n=== PHASE 1: battery slice reproduction ===")
print(f"models: {n_models} (api {len(kept_api)} + local {len(kept_local)})")
print(f"records: {tot_n}   mismatches: {tot_d}   pooled: {100*tot_d/tot_n:.2f}%")
print(f"models with >=1 mismatch: {models_ge1}")
print(f"median per-model rate: {100*statistics.median(rates):.2f}%")

# save the per-record audit table for later phases
with open(OUT / "battery_records.jsonl", "w", encoding="utf-8") as f:
    for r in all_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"\nwrote {len(all_records)} battery records -> battery_records.jsonl")
