"""Phase 3: join response text for counted other-only records; Phase 4: coverage."""
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/root/projects/model_identity_evals")
OUT = ROOT / "analysis_scratch" / "generics_audit"

recs = [json.loads(l) for l in open(OUT / "battery_records.jsonl", encoding="utf-8")]
oc = [json.loads(l) for l in open(OUT / "other_only_counted.jsonl", encoding="utf-8")]

# ---- phase 4: adjudication coverage --------------------------------------
flagged = [r for r in recs if r["foreign"]]
api_flagged = [r for r in flagged if r["src"] == "api"]
loc_flagged = [r for r in flagged if r["src"] == "local"]
api_noadj = [r for r in api_flagged if r["adj"] is None]
loc_noadj = [r for r in loc_flagged if r["adj"] is None]
print("=== PHASE 4: adjudication coverage (battery slice) ===")
print(f"flagged battery records: {len(flagged)} (api {len(api_flagged)}, local {len(loc_flagged)})")
print(f"API flagged with NO adjudication row: {len(api_noadj)} "
      f"({100*len(api_noadj)/max(len(api_flagged),1):.2f}%)  <- these COUNT via fallback")
print(f"local flagged with NO local adjudication row: {len(loc_noadj)} "
      f"({100*len(loc_noadj)/max(len(loc_flagged),1):.2f}%)  <- these are DROPPED (local logic)")
if api_noadj:
    cc = Counter((r["model_id"], r["lang"]) for r in api_noadj)
    for (m, l), n in cc.most_common(30):
        print(f"    {m} {l}: {n}")
if loc_noadj:
    cc = Counter((r["model_id"], r["lang"]) for r in loc_noadj)
    for (m, l), n in cc.most_common(30):
        print(f"    {m} {l}: {n}")

# local verdict distribution for context
print("\nlocal flagged verdicts:", Counter(r["adj"] for r in loc_flagged).most_common())

# full-judgment-set coverage (all categories except system_probe, as build_worklist scopes)
import sys
sys.path.insert(0, str(ROOT))
from sweep.analyze import load, foreign_claims, _adjudications
ADJ = _adjudications()
all_flag = no_row = 0
noadj_by = Counter()
for j in load():
    if j["prompt_category"] == "system_probe":
        continue
    jm = j["judgment"]
    fam = j.get("family", "")
    from sweep.analyze import canon_identity, is_self
    fields = ["claimed_name", "claimed_creator"]
    if jm.get("reasoning_identity_stance") != "role_play":
        fields += ["reasoning_claimed_name", "reasoning_claimed_creator"]
    out = set()
    for f in fields:
        c = canon_identity(jm.get(f))
        if c and not is_self(c, fam, j.get("aliases", []), j["expected_identity"]):
            out.add(c)
    if not out:
        continue
    all_flag += 1
    if ADJ.get(f"{j['resume_key']}::t{j.get('turn_index', 0)}") is None:
        no_row += 1
        noadj_by[(j["model_id"], j["prompt_category"])] += 1
print(f"\nALL categories (except system_probe), all judged models incl. incomplete:")
print(f"  flagged: {all_flag}   no adjudication row: {no_row} ({100*no_row/max(all_flag,1):.2f}%)")
for k, n in noadj_by.most_common(25):
    print(f"    {k}: {n}")

# ---- phase 3: join response text for the 534 ------------------------------
want = {}
for r in oc:
    want.setdefault(r["resume_key"], []).append(r)
from sweep.analyze import open_lines
found = 0
for l in open_lines(ROOT / "results" / "main_sweep.jsonl"):
    rr = json.loads(l)
    if rr.get("error"):
        continue
    rk = rr["resume_key"]
    if rk in want:
        for r in want[rk]:
            r["prompt_text"] = (rr["messages_sent"][-1]["content"] if rr.get("messages_sent") else "")[:200]
            r["response"] = (rr.get("content_clean") or rr.get("content") or "")[:900]
        found += 1
# local records' raw responses live in results_local/<model>.jsonl
loc_want = {rk for rk, rs in want.items() if any(r["src"] == "local" for r in rs)}
if loc_want:
    for p in (ROOT / "results_local").glob("*.jsonl"):
        if p.name == "judgments_clean.jsonl":
            continue
        for l in open(p, encoding="utf-8"):
            rr = json.loads(l)
            if rr.get("resume_key") in loc_want:
                for r in want[rr["resume_key"]]:
                    r["prompt_text"] = (rr.get("prompt") or rr.get("content_sent") or "")[:200] if isinstance(rr.get("prompt"), str) else str(rr.get("messages_sent", ""))[:200]
                    r["response"] = (rr.get("content_clean") or rr.get("content") or "")[:900]

missing = [r for r in oc if "response" not in r]
print(f"\n=== PHASE 3: joined response text for {len(oc)-len(missing)}/{len(oc)} counted other-only records "
      f"({len(missing)} missing raw)")

with open(OUT / "other_only_counted_joined.jsonl", "w", encoding="utf-8") as f:
    for r in oc:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# group by claimed other:* value for eyeballing
byval = defaultdict(list)
for r in oc:
    for c in r["foreign"]:
        byval[c].append(r)
print(f"distinct other:* values among the 534 counted records: {len(byval)}")
langs = Counter(r["lang"] for r in oc)
print("counted other-only by language:", dict(langs.most_common()))
mods = Counter(r["model_id"] for r in oc)
print("top-15 models:", mods.most_common(15))

with open(OUT / "eyeball_by_value.txt", "w", encoding="utf-8") as f:
    for v, rs in sorted(byval.items(), key=lambda x: -len(x[1])):
        f.write(f"\n{'='*110}\n### {v}   x{len(rs)}  models={sorted(set(r['model_id'] for r in rs))} "
                f"langs={dict(Counter(r['lang'] for r in rs))}\n")
        for r in rs[:4]:
            f.write(f"  -- {r['model_id']} [{r['lang']}] {r['prompt_id']} adj={r['adj']} "
                    f"name={r['raw_name']!r} creator={r['raw_creator']!r} rname={r['raw_rname']!r} rcreator={r['raw_rcreator']!r}\n")
            f.write(f"     Q: {r.get('prompt_text','')!r}\n")
            f.write(f"     A: {r.get('response','')[:600]!r}\n")
print("wrote eyeball_by_value.txt")
