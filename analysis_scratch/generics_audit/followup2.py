"""Field-level attribution for curated generic-leak OTHER strings + empty-content check."""
import copy
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/root/projects/model_identity_evals")
sys.path.insert(0, str(ROOT))
from sweep.analyze import foreign_claims, open_lines
from sweep.make_figs import complete_models, is_identity, BATTERY_CORE

OUT = ROOT / "analysis_scratch" / "generics_audit"

# hand-curated: OTHER strings that are pure generic descriptors / placeholders
# (no proper noun) — leaked past GENERIC_TERMS into other:
CURATED = {
 "grand modèle linguistique", "un grand modèle linguistique",
 "grand modèle linguistique open-source", "grand modèle linguistique open-weights",
 "un modèle linguistique", "modèle linguistique", "modèle linguistique open-source",
 "최신 버전 모델", "модель", "модель-трансформер", "бот", "чатбот", "null",
 "[이름]", "[모델 이름]", "[ai の名称]", "[ai 이름]", "[ai 모델 이름]",
 "transformer 模型", "transformer_model", "mô hình transformer",
 "dialogue", "mon dialogue", "assist", "assistent", "assistan",
 "ai 모델", "ai 또는 도우미", "ai-модель", "командой разработчиков",
 "большой языковой модель", "модель мовного моделювання",
 "искусственная модель языка", "языковым моделью", "языковой модели ai",
 "модель искусственного интеллекта", "нейросетевая помощница",
 "модель крупной языковой нейросети", "한국어 모델", "인공통합언어모델",
 "модель конвертера языков", "модель языка", "большая модель языка обучения",
 "большой языковый модель", "я модель русского языка",
 "модель natural language processing (nlp)", "искусственная модель интеллекта",
 "gran modelo lingüístico", "mô hình của công nghệ ngôn ngữ lớn (llm)",
 "대화식 ai ngôn ngữ mô형입니다", "블라우스 모델입니다.", "나은어모델",
 "языковое модель as", "제이름은 어떻게 되세요?", "お名前さん",
 "модуль дизайнаはいспытающий русский бот", "bắt đầu từ một không gian ngôn ngữ",
 "family",
}

reg = {m["id"]: m for m in json.loads((ROOT / "config" / "models.json").read_text())["models"]}
hyg = json.loads((ROOT / "config" / "provider_hygiene.json").read_text())
complete = complete_models(reg, hyg)

hits = []          # curated-generic OTHER occurrences in battery (API)
empty_paths = Counter()
empty_generic_models = Counter()
empty_generic_raws = Counter()

# which battery rks have empty visible content? (one main_sweep pass)
battery_rk_empty = set()
for line in open_lines(ROOT / "results" / "main_sweep.jsonl"):
    r = json.loads(line)
    if r.get("error"):
        continue
    if not is_identity(r.get("prompt_category", "")) or r.get("prompt_id") not in BATTERY_CORE:
        continue
    if not (r.get("content_clean") or r.get("content") or "").strip():
        battery_rk_empty.add(r["resume_key"])
print(f"battery records with empty visible content in main_sweep: {len(battery_rk_empty)}")

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
    # empty-content check
    if j["resume_key"] in battery_rk_empty:
        for f in ("claimed_name",):
            v = jm.get(f)
            if isinstance(v, str) and v.strip():
                empty_paths["nonnull_name"] += 1
                empty_generic_models[j["model_id"]] += 1
                empty_generic_raws[v.strip()] += 1
            else:
                empty_paths["null_name"] += 1
    # curated-generic attribution
    for f in ("claimed_name", "claimed_creator"):
        v = jm.get(f)
        if isinstance(v, str) and v.strip().lower() in CURATED:
            mm = bool(foreign_claims(j))
            solely = False
            if mm:
                j2 = copy.deepcopy(j)
                j2["judgment"][f] = None
                solely = not bool(foreign_claims(j2))
            hits.append(dict(rk=j["resume_key"], model=j["model_id"], field=f,
                             raw=v.strip(), mismatch=mm, mismatch_solely_from_leak=solely))

n = len(hits)
mm = sum(h["mismatch"] for h in hits)
solo = sum(h["mismatch_solely_from_leak"] for h in hits)
print(f"curated generic-leak OTHER occurrences (battery, api): {n}")
print(f"  on records that are shipped mismatches: {mm}")
print(f"  mismatch attributable SOLELY to the leak string: {solo}")
for h in hits:
    if h["mismatch_solely_from_leak"]:
        print("   FP:", h)

print("\nempty-content records: name-path split:", dict(empty_paths))
print("fabricated names on empty content:", dict(empty_generic_raws))
print("by model:", dict(empty_generic_models.most_common(10)))

res = json.loads((OUT / "census.json").read_text())
res["other_generic_leaks_curated"] = {
    "occurrences": n, "on_mismatch_records": mm,
    "mismatch_solely_from_leak": solo,
    "fp_records": [h for h in hits if h["mismatch_solely_from_leak"]]}
res["empty_content_check"] = {
    "battery_records_empty_content": len(battery_rk_empty),
    "judge_extracted_name_anyway": empty_paths["nonnull_name"],
    "fabricated_names": dict(empty_generic_raws.most_common(15)),
    "by_model": dict(empty_generic_models.most_common(10))}
# drop the over-broad auto version to avoid confusion
res.pop("other_generic_leaks", None)
(OUT / "census.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
print("census.json updated")
