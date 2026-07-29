"""Verify the 2026-07-29 canon fixes: diff newly-counted vs no-longer-counted
mismatch records against the audit baseline (battery_records.jsonl, old canon).

Expectations from census.md + other_audit.md:
  OUT: ~22 FP records (создатели/Algorithm/self-garbles/PaLM-2 gemma)
  IN (pre-adjudication): ~160-207 Open Assistant battery records (nova-lite/pro/micro,
      laguna, olmo-32B, granite) + T5 (hermes x4) + 큐웬 x1 + 华为 x1 + 上海AI实验室 x2
Writes to_adjudicate.txt (newly-counted adj_keys lacking adjudication rows).
"""
import json, sys
from collections import Counter, defaultdict

sys.path.insert(0, ".")
from sweep.analyze import load, foreign_claims, _adjudications
from sweep.make_figs import gather

# old counted set from the audit baseline (old canon, adjudication-filtered)
old_counted = {}
for l in open("analysis_scratch/generics_audit/battery_records.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if r.get("counted") and r.get("src") == "api":   # like-for-like: load() is API-lane only
        old_counted[r["adj_key"]] = r

# new counted set under the edited canon (same pipeline path)
reg, per = gather()
new_counted = {}
for j in load():
    fc = foreign_claims(j)
    if fc:
        k = f"{j['resume_key']}::t{j.get('turn_index', 0)}"
        new_counted[k] = {"model": j["model_id"], "claims": sorted(fc),
                          "prompt_id": j["prompt_id"], "cat": j.get("prompt_category", "")}

# restrict comparison to battery keys present in the baseline file
battery_keys = set()
for l in open("analysis_scratch/generics_audit/battery_records.jsonl", encoding="utf-8"):
    battery_keys.add(json.loads(l)["adj_key"])
new_b = {k: v for k, v in new_counted.items() if k in battery_keys}

out = set(old_counted) - set(new_b)
inn = set(new_b) - set(old_counted)
print(f"old counted (battery): {len(old_counted)}   new counted (battery): {len(new_b)}")
print(f"OUT (no longer counted): {len(out)}")
for k in sorted(out):
    r = old_counted[k]
    print(f"  - {r['model_id']:<40} {r['prompt_id']:<16} name={r['raw_name']!r} creator={r['raw_creator']!r}")
print(f"IN (newly counted, pre-adjudication): {len(inn)}")
bymodel = Counter()
byclaim = Counter()
for k in sorted(inn):
    v = new_b[k]
    bymodel[v["model"]] += 1
    for c in v["claims"]:
        byclaim[c] += 1
print("  by model:", dict(bymodel.most_common(12)))
print("  by claim:", dict(byclaim.most_common(12)))

adj = _adjudications()
need = [k for k in inn if k not in adj]
with open("analysis_scratch/generics_audit/to_adjudicate.txt", "w") as f:
    f.write("\n".join(sorted(need)))
print(f"\nnewly counted lacking adjudication rows: {len(need)} -> to_adjudicate.txt")
print("(these are counted via the un-adjudicated fallback until sweep.adjudicate runs)")
