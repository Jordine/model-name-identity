"""Phase 2: enumerate other:* canon values over battery records."""
import json
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path("/root/projects/model_identity_evals/analysis_scratch/generics_audit")

recs = [json.loads(l) for l in open(OUT / "battery_records.jsonl", encoding="utf-8")]
print(f"{len(recs)} battery records")

# --- distinct other:* values from claimed_name / claimed_creator (main fields)
main_all = Counter()          # any occurrence (incl. ones is_self clears)
main_foreign = Counter()      # occurrences that actually count as foreign evidence
reasoning_foreign = Counter() # other:* arriving only via reasoning fields
val_models = defaultdict(set)
val_langs = defaultdict(Counter)
val_field = defaultdict(Counter)

n_rec_any_other_main = 0
for r in recs:
    c = r["canon"]
    hit = False
    for f in ("claimed_name", "claimed_creator"):
        v = c.get(f)
        if v and v.startswith("other:"):
            main_all[v] += 1
            hit = True
            if v in r["foreign"]:
                main_foreign[v] += 1
                val_models[v].add(r["model_id"])
                val_langs[v][r["lang"]] += 1
                val_field[v][f] += 1
    for f in ("reasoning_claimed_name", "reasoning_claimed_creator"):
        v = c.get(f)
        if v and v.startswith("other:") and v in r["foreign"]:
            reasoning_foreign[v] += 1
    if hit:
        n_rec_any_other_main += 1

print(f"records with any other:* in claimed_name/claimed_creator: {n_rec_any_other_main}")
print(f"distinct other:* values (main fields, any): {len(main_all)}")
print(f"distinct other:* values (main fields, counted-as-foreign occurrences): {len(main_foreign)}")
print(f"distinct other:* values (reasoning-only foreign): {len(set(reasoning_foreign) - set(main_foreign))}")

with open(OUT / "other_values_full.txt", "w", encoding="utf-8") as f:
    f.write("# distinct other:* canon values over the 60,770-record identity/creator battery\n")
    f.write("# columns: total_occurrences | foreign_occurrences(main fields, is_self=False) | "
            "n_models | top_langs | value\n")
    for v, n in main_all.most_common():
        langs = ",".join(f"{l}:{k}" for l, k in val_langs[v].most_common(3))
        f.write(f"{n:5d} | {main_foreign.get(v,0):5d} | {len(val_models[v]):3d} | {langs:20s} | {v}\n")
    extra = {v: n for v, n in reasoning_foreign.items() if v not in main_all}
    if extra:
        f.write("\n# other:* appearing ONLY in reasoning_claimed_* fields (still foreign evidence)\n")
        for v, n in sorted(extra.items(), key=lambda x: -x[1]):
            f.write(f"{n:5d} |  (reasoning) | {v}\n")

# ≥3-occurrence list for classification
ge3 = [(v, n) for v, n in main_foreign.most_common() if n >= 3]
print(f"\ndistinct foreign other:* values with >=3 occurrences: {len(ge3)} "
      f"(covering {sum(n for _, n in ge3)} of {sum(main_foreign.values())} foreign occurrences)")
with open(OUT / "ge3_values.txt", "w", encoding="utf-8") as f:
    for v, n in ge3:
        langs = ",".join(f"{l}:{k}" for l, k in val_langs[v].most_common(3))
        mods = ",".join(sorted(val_models[v]))[:120]
        f.write(f"{n:5d} | {langs:22s} | {v}    [{mods}]\n")

# --- record-level: other-only vs known-name foreign evidence
other_only = [r for r in recs if r["foreign"] and all(c.startswith("other:") for c in r["foreign"])]
mixed = [r for r in recs if r["foreign"] and any(c.startswith("other:") for c in r["foreign"])
         and not all(c.startswith("other:") for c in r["foreign"])]
known_only = [r for r in recs if r["foreign"] and not any(c.startswith("other:") for c in r["foreign"])]
print(f"\nrecords with foreign evidence pre-adjudication: {sum(1 for r in recs if r['foreign'])}")
print(f"  other:-only evidence: {len(other_only)}")
print(f"  mixed (other + known): {len(mixed)}")
print(f"  known-name only: {len(known_only)}")

for name, group in (("other-only", other_only), ("mixed", mixed), ("known-only", known_only)):
    vc = Counter(r["adj"] if r["adj"] else ("NO_ADJ_ROW") for r in group)
    counted = sum(1 for r in group if r["counted"])
    print(f"\n{name}: {len(group)} records, {counted} counted in final numbers")
    for k, n in vc.most_common():
        print(f"    {k:16s} {n}")

# save other-only counted for phase 3 eyeballing
oc = [r for r in other_only if r["counted"]]
with open(OUT / "other_only_counted.jsonl", "w", encoding="utf-8") as f:
    for r in oc:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"\nother-only records counted in final numbers: {len(oc)} -> other_only_counted.jsonl")
print(f"  = {100*len(oc)/4849:.1f}% of the 4,849 final mismatches")
vc = Counter()
for r in oc:
    for c in r["foreign"]:
        vc[c] += 1
print("\ntop-40 other:* values among COUNTED other-only records:")
for v, n in vc.most_common(40):
    print(f"  {n:4d}  {v}")
