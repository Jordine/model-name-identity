"""Analysis over judgments.jsonl — discrepancy rates, language effects,
claimed-identity flows, cross-probe acceptance.

Usage: python -m sweep.analyze [--csv out.csv]
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JUDGMENTS = ROOT / "results" / "judgments.jsonl"

# --- canonicalization -------------------------------------------------------
# Generic descriptors that are NOT identity claims (multilingual).
GENERIC_TERMS = [
    "assistant", "ai assistant", "an ai", "artificial intelligence",
    "language model", "large language model", "llm", "chatbot",
    "conversational ai", "virtual assistant", "digital assistant",
    "助手", "人工智能", "智能助手", "大模型", "语言模型", "大型语言模型",
    "人工智能助手", "ai助手", "聊天机器人",
    "アシスタント", "人工知能", "言語モデル", "aiアシスタント",
    "어시스턴트", "인공지능", "언어 모델", "ai 어시스턴트",
    "ассистент", "искусственный интеллект", "языковая модель", "ии",
    "asistente", "inteligencia artificial", "modelo de lenguaje",
    "trợ lý", "trí tuệ nhân tạo",
    "intelligence artificielle", "modèle de langage",
]

# (pattern, canon) — checked in order, most specific first. Patterns are
# lowercase substrings incl. non-Latin renderings of major identities.
NAME_MAP = [
    (["chatgpt", "chat gpt", "чатгпт", "챗gpt", "챗지피티"], "chatgpt"),
    (["openai", "оупенэйай", "オープンai"], "openai"),
    (["claude", "克劳德", "クロード", "클로드", "клод"], "claude"),
    (["anthropic", "антропик", "アンソロピック", "앤트로픽"], "anthropic"),
    (["gemini", "ジェミニ", "제미나이", "джемини"], "gemini"),
    (["bard"], "gemini"),
    (["google", "谷歌", "グーグル", "구글", "гугл"], "google"),
    (["deepseek", "深度求索", "딥시크", "дипсик"], "deepseek"),
    (["通义千问", "千问", "通义", "通義", "qwen", "tongyi", "큐원"], "qwen"),
    (["阿里巴巴", "阿里云", "alibaba", "aliyun", "알리바바", "アリババ", "алибаба"], "alibaba"),
    (["llama", "ラマ", "라마", "лама", "льяма"], "llama"),
    (["meta", "メタ", "메타", "мета"], "meta"),
    (["kimi", "キミ", "키미", "кими"], "kimi"),
    (["moonshot", "月之暗面", "月の暗面", "문샷"], "moonshot"),
    (["mistral", "мистраль", "ミストラル"], "mistral"),
    (["grok", "грок"], "grok"), (["xai", "x-ai", "x.ai"], "xai"),
    (["chatglm", "glm", "智谱", "zhipu", "z.ai", "z-ai"], "glm"),
    (["文心一言", "文心", "ernie", "wenxin"], "ernie"),
    (["百度", "baidu"], "baidu"),
    (["混元", "hunyuan"], "hunyuan"), (["腾讯", "tencent"], "tencent"),
    (["豆包", "doubao", "云雀"], "doubao"), (["字节跳动", "字节", "bytedance"], "bytedance"),
    (["360智脑", "智脑", "360 zhinao", "qihoo"], "zhinao360"),
    (["讯飞星火", "星火", "spark", "iflytek", "讯飞"], "sparkdesk"),
    (["naver", "네이버", "hyperclova", "클로바", "clova"], "naver"),
    (["copilot"], "copilot"), (["microsoft", "微软"], "microsoft"),
    (["siri"], "siri"), (["alexa"], "alexa"), (["cortana"], "cortana"),
    (["gpt"], "chatgpt"),  # bare "GPT-x" after all specific checks
]

# family -> canon keys that count as SELF for that family
FAMILY_SELF = {
    "openai": {"chatgpt", "openai"},
    "anthropic": {"claude", "anthropic"},
    "google": {"gemini", "google"}, "gemma": {"google"},
    "deepseek": {"deepseek"},
    "qwen": {"qwen", "alibaba"}, "alibaba": {"qwen", "alibaba"},
    "meta": {"llama", "meta"},
    "kimi": {"kimi", "moonshot"},
    "mistral": {"mistral"}, "xai": {"grok", "xai"},
    "zhipu": {"glm"}, "baidu": {"ernie", "baidu"},
    "tencent": {"hunyuan", "tencent"}, "bytedance": {"doubao", "bytedance"},
    "microsoft": {"microsoft", "copilot"},
    # long-tail labs (self-name != family slug)
    "ant": {"ling", "ring", "bailing", "ant", "inclusionai"},
    "nvidia": {"nemotron", "nvidia"},
    "nous": {"hermes", "nous"},
    "perplexity": {"sonar", "perplexity"},
    "poolside": {"laguna", "poolside"},
    "ibm": {"granite", "ibm"},
    "kuaishou": {"kat", "kuaishou", "kwaipilot"},
    "xiaomi": {"mimo", "xiaomi"},
    "stepfun": {"step", "stepfun"},
    "amazon": {"nova", "amazon"},
    "cohere": {"command", "cohere", "aya", "north"},
    "inception": {"mercury", "inception"},
    "ai21": {"jamba", "ai21"},
    "inflection": {"pi", "inflection"},
    "arcee": {"trinity", "virtuoso", "arcee"},
    "reka": {"reka"}, "nex": {"nex"}, "perceptron": {"perceptron"},
    "sakana": {"fugu", "sakana"},
    "allenai": {"olmo", "allenai", "ai2"},
    "upstage": {"solar", "upstage"},
    "writer": {"palmyra", "writer"},
    "liquid": {"lfm", "liquid"},
    "minimax": {"minimax"},
    "cogito": {"cogito", "deepcogito"},
}


def canon_identity(raw: str | None) -> str | None:
    """None for generics/empty; canon key for known identities; other:… else."""
    if not raw:
        return None
    low = raw.strip().lower()
    if any(low == g or low.strip("an ") == g for g in GENERIC_TERMS) or \
       any(g in low for g in GENERIC_TERMS if len(g) >= 4):
        # exact or substring generic — unless a real name also appears
        if not any(p in low for pats, _ in NAME_MAP for p in pats):
            return None
    for pats, canon in NAME_MAP:
        if any(p in low for p in pats):
            return canon
    if len(low) < 3:
        return None
    return f"other:{low[:40]}"


def is_self(canon: str | None, family: str, aliases: list[str], expected: str) -> bool:
    if canon is None:
        return True
    if canon in FAMILY_SELF.get(family, {family}):
        return True
    own = " ".join(aliases).lower() + " " + expected.lower() + " " + family.lower()
    if canon.startswith("other:"):
        term = canon[6:]
        return any(w in own for w in term.split() if len(w) > 2)
    return canon in own


def load():
    # join against the CURRENT registry — records may carry stale aliases
    reg_path = ROOT / "config" / "models.json"
    reg = {}
    if reg_path.exists():
        reg = {m["id"]: m for m in json.loads(reg_path.read_text())["models"]}
    rows = []
    for line in open(JUDGMENTS, encoding="utf-8"):
        j = json.loads(line)
        if j.get("judge_error") or not j.get("judgment"):
            continue
        m = reg.get(j["model_id"])
        if m:
            j["aliases"] = m["aliases"]
            j["family"] = m["family"]
        rows.append(j)
    return rows


def lang_of(category: str) -> str:
    return category.split("_", 1)[1] if category.startswith("direct_") else \
        {"creator_en": "en", "casual": "en", "probe_self": "en",
         "probe_cross": "en", "multi_turn": "mixed", "system_probe": "en"}.get(category, "en")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    args = ap.parse_args()

    rows = load()
    print(f"{len(rows)} judged records\n")

    per_model = defaultdict(lambda: {"n": 0, "disc": 0, "by_lang": defaultdict(lambda: [0, 0]),
                                     "claims": Counter(), "cross_yes": [], "hedged": 0})
    for j in rows:
        jm = j["judgment"]
        m = per_model[j["model_id"]]
        cat = j["prompt_category"]

        if cat == "probe_cross":
            target = j["prompt_id"].replace("cross_", "")
            if jm.get("answered_yes") is True and \
               target not in FAMILY_SELF.get(j.get("family", ""), set()):
                m["cross_yes"].append(j["prompt_id"])
            continue
        if cat == "system_probe":
            continue

        m["n"] += 1
        fam = j.get("family", "")
        claimed = []
        for f in ("claimed_name", "claimed_creator"):
            c = canon_identity(jm.get(f))
            if c and not is_self(c, fam, j.get("aliases", []), j["expected_identity"]):
                claimed.append(c)
        # reasoning-only claims tracked separately
        r_claimed = []
        for f in ("reasoning_claimed_name", "reasoning_claimed_creator"):
            c = canon_identity(jm.get(f))
            if c and not is_self(c, fam, j.get("aliases", []), j["expected_identity"]):
                r_claimed.append(c)

        lang = lang_of(cat)
        m["by_lang"][lang][1] += 1
        if claimed or r_claimed:
            m["disc"] += 1
            m["by_lang"][lang][0] += 1
            for c in set(claimed + r_claimed):
                m["claims"][c] += 1
        if jm.get("hedged"):
            m["hedged"] += 1

    print(f"{'model':52s} {'disc':>9s}  {'langs with discrepancies':32s} claims / cross-accepts")
    print("-" * 130)
    for mid, m in sorted(per_model.items(), key=lambda x: -(x[1]["disc"] / max(x[1]["n"], 1))):
        rate = m["disc"] / max(m["n"], 1)
        langs = " ".join(f"{l}:{d}/{n}" for l, (d, n) in sorted(m["by_lang"].items()) if d)
        claims = ",".join(f"{k}x{v}" for k, v in m["claims"].most_common(4))
        cross = f" | YES-to: {','.join(m['cross_yes'])}" if m["cross_yes"] else ""
        print(f"{mid:52s} {m['disc']:3d}/{m['n']:3d} ({rate:4.0%})  {langs:32s} {claims}{cross}")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["model_id", "n", "n_discrepant", "rate", "top_claims", "cross_yes", "hedged"])
            for mid, m in per_model.items():
                w.writerow([mid, m["n"], m["disc"], round(m["disc"] / max(m["n"], 1), 4),
                            ";".join(f"{k}:{v}" for k, v in m["claims"].most_common()),
                            ";".join(m["cross_yes"]), m["hedged"]])
        print(f"\ncsv -> {args.csv}")


if __name__ == "__main__":
    main()
