"""Phase 5: split the 534 counted other-only records into candidate-generic vs specific-name,
print candidates fully + a random sample of the rest for eyeballing."""
import json, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path("/root/projects/model_identity_evals/analysis_scratch/generics_audit")
oc = [json.loads(l) for l in open(OUT / "other_only_counted_joined.jsonl", encoding="utf-8")]

GENERIC_HINTS = [
    # latin
    "assist", "asist", "model", "language", "lenguaje", "linguistique", "langage",
    "modèle", "modelo", "intelligen", "chatbot", "chat bot", "llm", "neural",
    "community", "comunidad", "communauté", "developer", "desarrollador", "développeur",
    "team", "equipo", "équipe", "open source", "open-source", "digital", "virtual",
    "helper", "ayudarte", "dialogue", "conversational",
    # zh/ja/ko
    "助手", "模型", "智能", "语言", "机器人", "开源", "社区", "开发者", "研发团队", "团队",
    "公司", "会社", "アシスタント", "モデル", "チャット", "ボット", "人工知能", "知能",
    "aiさん", "aiさん", "언어", "모델", "도우미", "봇", "비서", "어시스턴트", "인공지능", "회사",
    # ru
    "модель", "ассистент", "помощник", "бот", "нейросет", "разработчик", "команд",
    "интеллект", "языков",
    # vi
    "trợ", "hỗ trợ", "mô hình", "ngôn ngữ", "trí tuệ", "công nghệ", "người máy",
    # generic-word tails
    "your name", "null", "unknown",
]

def hint(v):
    s = v[6:]
    return [h for h in GENERIC_HINTS if h in s]

cand, rest = [], []
for r in oc:
    hits = sorted({h for c in r["foreign"] for h in hint(c)})
    (cand if hits else rest).append((r, hits))

print(f"counted other-only: {len(oc)}; candidate-generic: {len(cand)}; specific-name: {len(rest)}")

def show(r, hits=None, resp=500):
    print(f"--- {r['model_id']} [{r['lang']}] {r['prompt_id']} adj={r['adj']}")
    print(f"    foreign={r['foreign']} name={r['raw_name']!r} creator={r['raw_creator']!r} "
          f"rname={r['raw_rname']!r} rcreator={r['raw_rcreator']!r}" + (f" hints={hits}" if hits else ""))
    q = (r.get('prompt_text') or '').replace('\n', ' ')
    a = (r.get('response') or '').replace('\n', ' ¶ ')[:resp]
    print(f"    Q: {q}")
    print(f"    A: {a}")

mode = sys.argv[1] if len(sys.argv) > 1 else "cand"
if mode == "cand":
    print("\n================ CANDIDATE-GENERIC records (all) ================")
    for r, hits in cand:
        show(r, hits)
elif mode == "rest_values":
    vc = Counter()
    for r, _ in rest:
        for c in r["foreign"]:
            vc[c] += 1
    print(f"\n{len(vc)} distinct specific-name values:")
    for v, n in vc.most_common():
        print(f"  {n:3d}  {v}")
elif mode == "rest_sample":
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    rng = random.Random(7)
    for r, _ in rng.sample(rest, min(k, len(rest))):
        show(r, resp=380)
