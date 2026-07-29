"""Phase 6: final per-record labels for the 534 counted other-only records,
plus flip simulations on the headline numbers."""
import json, statistics
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path("/root/projects/model_identity_evals/analysis_scratch/generics_audit")
recs = [json.loads(l) for l in open(OUT / "battery_records.jsonl", encoding="utf-8")]
oc = [json.loads(l) for l in open(OUT / "other_only_counted_joined.jsonl", encoding="utf-8")]
by_adjkey = {r["adj_key"]: r for r in oc}

# ---- hand-classified record-level labels (from eyeballing responses) -------
# STRICT SELF-GARBLE: the claim IS the model's own identity in another script/garble;
# adjudicator confirmed genuine_foreign in error.
STRICT_SELF = [
    ("cohere/north-mini-code:free", "ko_name", "other:코맨드"),            # Command translit
    ("cohere/north-mini-code:free", "ko_introduce", "other:코헤렌트"),      # Cohere->Coherent
    ("inclusionai/ling-2.6-flash", "ja_introduce", "other:ling社"),        # LingDT by Ling
    ("inclusionai/ling-2.6-1t", "ja_who_made", "other:アントグループ"),      # Ant Group (name=Bailing=self)
    ("inclusionai/ring-2.6-1t", "ja_model", "other:百霊大モデル"),          # Bailing/Ling kyujitai
    ("inclusionai/ling-2.6-flash", "ru_which_lm", "other:antbailing"),     # AntBailing
    ("mistralai/mistral-medium-3-5", "zh_which_lm", "other:米斯ral ai团队"), # Mistral AI half-script (name=Mistral Large=self)
    ("deepseek/deepseek-v4-pro", "vi_which_lm", "other:công ty thâm độ cầu sách"),  # DeepSeek VN calque
]
# GENERIC-ONLY: every counted foreign string is a generic descriptor / parse junk,
# and the response contains no substantive foreign name (or self-identifies).
GENERIC_ONLY = [
    ("anthropic/claude-3-haiku", "ru_call_you", "other:создатели"),        # name=Клэр=self; "my creators"
    ("Qwen/Qwen3.5-2B", "ko_who_made", "other:algorithm"),                 # response says Alibaba Cloud (self)
    ("meta-llama/llama-3.1-70b-instruct", "ja_who", "other:ai プログラム"), # "I am an AI program"
    ("nousresearch/hermes-3-llama-3.1-70b", "ko_company", "other:machine learning company"),
    ("Qwen/Qwen3.5-0.8B", "es_which_lm", "other:desarrollado en una red de dispositivos "),
    ("meta-llama/llama-3.2-3b-instruct", "ru_model", "other:modelo de lingo ai"),
    ("nousresearch/hermes-3-llama-3.1-70b", "fr_model", "other:snn"),      # "inspiré par ... SNN" comparative
    ("mistralai/ministral-14b-2512", "es_introduce", "other:ia de respuesta"),   # x2 records
    ("mistralai/ministral-14b-2512", "es_introduce", "other:ia de respuestas"),
    ("rekaai/reka-edge", "ja_name", "other:ai辞書"),                        # "my name is AI-dictionary"
    ("Qwen/Qwen3-0.6B", "vi_who_made", "other:một tổ chức uy tín"),        # "a reputable org" (resp mentions using GPT)
    ("poolside/laguna-xs-2.1", "vi_who_made", "other:interne"),            # "team at Interne" truncation junk
]

def match(r, spec_list):
    for mid, pid, val in spec_list:
        if r["model_id"] == mid and r["prompt_id"] == pid and val in r["foreign"]:
            return True
    return False

OA_STRINGS = ["open assistant", "オープンアシスタント", "오픈 어시스턴트", "опен ассистент", "openassistant"]
def is_oa(r):
    blob = ((r.get("response") or "") + " " + (r.get("raw_name") or "")).lower()
    return any(s in blob for s in OA_STRINGS)

labels = {}
for r in oc:
    k = r["adj_key"]
    if match(r, STRICT_SELF):
        labels[k] = "STRICT_SELF"
    elif match(r, GENERIC_ONLY):
        labels[k] = "GENERIC_ONLY"
    elif is_oa(r):
        labels[k] = "OA_MECHANISM"   # substantively foreign (Open Assistant), counted via generic creator string
    else:
        labels[k] = "SOLID_SPECIFIC"

lc = Counter(labels.values())
print("labels over the 534 counted other-only records:", dict(lc))

q_recs = [r for r in oc if labels[r["adj_key"]] in ("STRICT_SELF", "GENERIC_ONLY")]
print(f"\nquestionable (true FP) records: {len(q_recs)}")
print("  by language:", dict(Counter(r["lang"] for r in q_recs)))
print("  by model:", dict(Counter(r["model_id"] for r in q_recs)))
oa_recs = [r for r in oc if labels[r["adj_key"]] == "OA_MECHANISM"]
print(f"\nOA-mechanism records: {len(oa_recs)}")
print("  by model:", dict(Counter(r["model_id"] for r in oa_recs)))
print("  by language:", dict(Counter(r["lang"] for r in oa_recs)))

# ---- rebuild per-model d/n and recompute headline under scenarios ----------
per = defaultdict(lambda: [0, 0])   # model -> [d, n]
for r in recs:
    per[r["model_id"]][1] += 1
    if r["counted"]:
        per[r["model_id"]][0] += 1

def headline(drop_keys, tag):
    d = defaultdict(int)
    for r in recs:
        if r["counted"] and r["adj_key"] not in drop_keys:
            d[r["model_id"]] += 1
    tot_d = sum(d.values())
    tot_n = sum(n for _, n in per.values())
    ge1 = sum(1 for m in per if d.get(m, 0) > 0)
    rates = [d.get(m, 0) / per[m][1] for m in per]
    med = statistics.median(rates)
    print(f"{tag}: pooled {tot_d}/{tot_n} = {100*tot_d/tot_n:.2f}%  "
          f"models>=1: {ge1}/190  median {100*med:.2f}%")
    return tot_d, ge1

print("\n=== headline scenarios ===")
headline(set(), "baseline (shipped)")
headline({r["adj_key"] for r in q_recs}, "minus 20 true-FP records (strict self + generic-only)")
headline({r["adj_key"] for r in oc}, "worst case: minus ALL 534 other-only counted")
headline({r["adj_key"] for r in oc if labels[r["adj_key"]] != "OA_MECHANISM"},
         "minus other-only except OA-mechanism")

# which models lose their last mismatch under each scenario?
def lost_models(drop_keys):
    d = defaultdict(int)
    for r in recs:
        if r["counted"] and r["adj_key"] not in drop_keys:
            d[r["model_id"]] += 1
    base = defaultdict(int)
    for r in recs:
        if r["counted"]:
            base[r["model_id"]] += 1
    return [m for m in base if base[m] > 0 and d.get(m, 0) == 0]

print("\nmodels dropping to 0 mismatches if all 534 removed:", lost_models({r["adj_key"] for r in oc}))
print("models dropping to 0 if only the 20 true FPs removed:", lost_models({r["adj_key"] for r in q_recs}))

with open(OUT / "final_labels.jsonl", "w", encoding="utf-8") as f:
    for r in oc:
        r2 = dict(r); r2["audit_label"] = labels[r["adj_key"]]
        f.write(json.dumps(r2, ensure_ascii=False) + "\n")
