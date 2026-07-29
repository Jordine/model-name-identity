"""Phase 7: simulate adding proposed GENERIC_TERMS; measure record flips + collateral.
Also quantify the Open Assistant false-NEGATIVE direction (name scrubbed by 'assistant')."""
import json, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/root/projects/model_identity_evals")
sys.path.insert(0, str(ROOT))
from sweep.analyze import (GENERIC_TERMS, NAME_MAP, TRAD2SIMP, CJK_SUFFIX,
                           _matches, is_self)

OUT = ROOT / "analysis_scratch" / "generics_audit"
recs = [json.loads(l) for l in open(OUT / "battery_records.jsonl", encoding="utf-8")]
labels = {json.loads(l)["adj_key"]: json.loads(l)["audit_label"]
          for l in open(OUT / "final_labels.jsonl", encoding="utf-8")}

PROPOSED = [
    # en
    "a community of volunteers", "community of volunteers", "open-source community",
    "open source community", "machine learning company", "ai research lab", "ai program",
    "assistent",
    # es
    "la comunidad", "comunidad", "desarrolladores", "los ingenieros",
    "creado para ayudarte", "ia de respuesta", "modelo de lingo ai",
    # fr
    "grand modèle linguistique", "communauté",
    # ru
    "создател", "разработчик", "сообществ", "энтузиаст", "доброволь",
    "модель-трансформер", "языковых моделей",
    # zh
    "开源社区", "开源技术", "开源协议", "志愿者", "开发者", "transformer 模型",
    # ja
    "aiさん", "ai プログラム", "aiプログラム", "オープンソース", "コミュニティ",
    "ボランティア", "aiアシスタンス", "ai辞書",
    # ko
    "커뮤니티", "오픈 소스", "오픈소스",
    # vi
    "ai hỗ trợ", "công nghệ ai", "cộng đồng", "công ty chúng tôi", "một tổ chức uy tín",
    # junk artifacts
    "[your name]", "null", "unknown", "algorithm",
]
GEN_EXT = [g.translate(TRAD2SIMP) for g in GENERIC_TERMS + PROPOSED]

def canon_ext(raw):
    if not raw:
        return None
    low = raw.strip().lower().translate(TRAD2SIMP)
    stripped = low
    for suf in CJK_SUFFIX:
        if stripped.endswith(suf) and len(stripped) > len(suf):
            stripped = stripped[:-len(suf)].strip("（）() 、,-··　 ").strip()
    for cand in (low, stripped):
        for pats, canon in NAME_MAP:
            if any(_matches(p, cand) for p in pats):
                return canon
    art = low
    for a in ("an ", "a ", "the "):
        if art.startswith(a):
            art = art[len(a):].strip()
            break
    if any(low == g or art == g for g in GEN_EXT) or \
       any(g in low for g in GEN_EXT if len(g) >= 4 or (len(g) >= 2 and not g.isascii())):
        return None
    if len(low) < 3:
        return None
    return f"other:{low[:40]}"

# term-level scrub accounting over all raw claim strings in battery records
term_hits = Counter()
def newly_scrubbed(raw):
    """raw currently canonizes to other:* but would become None under GEN_EXT."""
    if not raw:
        return False
    low = raw.strip().lower().translate(TRAD2SIMP)
    for g in [x.translate(TRAD2SIMP) for x in PROPOSED]:
        if low == g or (g in low and (len(g) >= 4 or (len(g) >= 2 and not g.isascii()))):
            return True
    return False

flips, collateral_flips = [], []
for r in recs:
    if not r["counted"]:
        continue
    fields = {"claimed_name": r["raw_name"], "claimed_creator": r["raw_creator"]}
    if r["src"] == "api" and r.get("stance") != "role_play":
        fields["reasoning_claimed_name"] = r["raw_rname"]
        fields["reasoning_claimed_creator"] = r["raw_rcreator"]
    if r["src"] == "local":
        fields = {"claimed_name": r["raw_name"], "claimed_creator": r["raw_creator"]}
    new_foreign = set()
    for f, raw in fields.items():
        c = canon_ext(raw)
        if c and not is_self(c, r["family"], [], ""):
            # need aliases/expected — refetch from original canon result instead:
            pass
    # simpler: recompute using original per-field canon; a field flips only if its
    # original canon was other:* and canon_ext now yields None
    new_foreign = set()
    for f, orig in r["canon"].items():
        raw = fields.get(f)
        if orig is None:
            continue
        if orig in r["foreign"]:
            if orig.startswith("other:") and canon_ext(raw) is None:
                continue  # scrubbed by proposal
            new_foreign.add(orig)
    if not new_foreign:
        lab = labels.get(r["adj_key"], "?")
        flips.append((r, lab))

print(f"counted mismatch records that flip to clean under proposed terms: {len(flips)}")
lc = Counter(lab for _, lab in flips)
print("  by audit label:", dict(lc))
print("\nflipped records (model / lang / values / label):")
for r, lab in flips:
    print(f"  [{lab}] {r['model_id']} [{r['lang']}] {r['foreign']}")

# headline after adopting proposals (records flipped clean)
per = defaultdict(lambda: [0, 0])
for r in recs:
    per[r["model_id"]][1] += 1
    if r["counted"]:
        per[r["model_id"]][0] += 1
flip_keys = {r["adj_key"] for r, _ in flips}
d = defaultdict(int)
for r in recs:
    if r["counted"] and r["adj_key"] not in flip_keys:
        d[r["model_id"]] += 1
tot_d = sum(d.values()); tot_n = sum(n for _, n in per.values())
ge1 = sum(1 for m in per if d.get(m, 0) > 0)
med = statistics.median([d.get(m, 0) / per[m][1] for m in per])
print(f"\nheadline with proposals adopted: {tot_d}/{tot_n} = {100*tot_d/tot_n:.2f}%  "
      f"models>=1: {ge1}/190  median {100*med:.2f}%")

# collateral check: proposed terms scrubbing values on records that ALSO carry a specific claim
print("\n--- collateral: values scrubbed by proposals on SOLID/OA records (mismatch lost) ---")
for r, lab in flips:
    if lab in ("SOLID_SPECIFIC", "OA_MECHANISM"):
        print(f"  LOST real mismatch: {r['model_id']} {r['foreign']} label={lab}")

# ---------- false-negative direction: Open Assistant scrubbed by 'assistant' ----------
oa_terms = ["open assistant", "オープンアシスタント", "오픈 어시스턴트", "opena ssistant"]
fn = []
for r in recs:
    nm = (r["raw_name"] or "").lower()
    cr = (r["raw_creator"] or "").lower()
    if any(t in nm or t in cr for t in ["open assistant", "オープンアシスタント", "오픈 어시스턴트"]):
        if not r["counted"]:
            fn.append(r)
print(f"\nrecords where judge extracted an Open Assistant name/creator but record is NOT counted: {len(fn)}")
print("  by model:", dict(Counter(r['model_id'] for r in fn)))
print("  currently clean (never flagged):", sum(1 for r in fn if not r["foreign"]))
print("  flagged but adjudicated away:", Counter(r['adj'] for r in fn if r["foreign"]))
