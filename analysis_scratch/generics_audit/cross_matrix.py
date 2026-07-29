"""Per-model x per-probe-name acceptance matrix ("Are you X?" -> % yes).
Rows: 10 real identities + 3 placebos (meridian, solace, cobalt).
Own-family probes excluded for real targets (mirrors make_figs.gather()).
Outputs cross_matrix.json + console summary.
"""
import json, sys
from collections import defaultdict, Counter

sys.path.insert(0, ".")
from sweep.make_figs import gather  # noqa: E402  (reuses load(), complete_models, FAMILY_SELF rules)
from sweep.analyze import load, FAMILY_SELF  # noqa: E402

REAL = ["chatgpt", "claude", "gemini", "deepseek", "qwen", "kimi", "llama", "grok", "mistral", "doubao"]
PLACEBO = ["meridian", "solace", "cobalt"]

reg = {m["id"]: m for m in json.loads(open("config/models.json").read())["models"]}
per = defaultdict(lambda: {"yes": Counter(), "n": Counter(), "fam": "", "name": ""})

n_rows = 0
for j in load():
    cat = j.get("prompt_category")
    if cat not in ("probe_cross", "probe_placebo"):
        continue
    mid = j["model_id"]
    if mid not in reg:
        continue
    pid = j["prompt_id"]
    t = pid.replace("zh_cross_", "").replace("en_cross_", "").replace("cross_", "") \
           .replace("zh_placebo_", "").replace("en_placebo_", "").replace("placebo_", "")
    if cat == "probe_cross" and t in FAMILY_SELF.get(j.get("family", ""), set()):
        continue  # own-family probe: excluded, mirroring gather()
    m = per[mid]
    m["fam"] = j.get("family", "")
    m["name"] = reg[mid]["name"]
    m["n"][t] += 1
    if (j["judgment"] or {}).get("answered_yes") is True:
        m["yes"][t] += 1
    n_rows += 1

print(f"probe records used: {n_rows} over {len(per)} models")

# pooled per-placebo rates (skeptic ask: report the three separately)
for p in PLACEBO:
    y = sum(v["yes"][p] for v in per.values())
    n = sum(v["n"][p] for v in per.values())
    print(f"placebo {p:<9} pooled: {y}/{n} = {100*y/n:.2f}%")

# who drives each real target: top accepters
print("\ntop accepters per target (>=30% and n>=5):")
for t in REAL:
    rows = []
    for mid, v in per.items():
        n = v["n"][t]
        if n >= 5:
            r = v["yes"][t] / n
            if r >= 0.30:
                rows.append((r, v["name"], v["yes"][t], n))
    rows.sort(reverse=True)
    print(f"  {t:<9} " + ("; ".join(f"{nm} {y}/{n}" for _, nm, y, n in rows[:6]) or "-")
          + (f"  (+{len(rows)-6} more)" if len(rows) > 6 else ""))

# distribution: how many models accept anything real above floor
acc_models = [mid for mid, v in per.items()
              if any(v["yes"][t] >= 2 for t in REAL)]
print(f"\nmodels with >=2 yeses on some real target: {len(acc_models)}/{len(per)}")
plac_acc = [mid for mid, v in per.items() if any(v["yes"][p] >= 2 for p in PLACEBO)]
print(f"models with >=2 yeses on some placebo:     {len(plac_acc)}/{len(per)}")

out = {mid: {"name": v["name"], "fam": v["fam"],
             "yes": dict(v["yes"]), "n": dict(v["n"])} for mid, v in per.items()}
json.dump(out, open("analysis_scratch/generics_audit/cross_matrix.json", "w"), indent=0)
print("\nwrote analysis_scratch/generics_audit/cross_matrix.json")
