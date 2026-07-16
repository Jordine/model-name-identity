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
    "ai model", "text-based ai model", "language ai", "chat assistant", "the company",
    "machine learning model", "large-scale language model", "generative ai",
    "助手", "人工智能", "智能助手", "大模型", "语言模型", "大型语言模型",
    "人工智能助手", "ai助手", "聊天机器人", "聊天助手", "助手模型", "对话助手",
    "智能搜索助理", "ai对话助手", "语言模型助手", "大规模语言模型", "语言ai",
    "アシスタント", "人工知能", "言語モデル", "aiアシスタント", "aiモデル",
    "aiチャットボット", "大規模言語モデル", "対話型ai", "言語ai",
    "어시스턴트", "인공지능", "언어 모델", "ai 어시스턴트", "ai 헬피어", "ai 도우미",
    "ai 비서", "대화형 ai", "챗봇", "언어 ai",
    "ассистент", "искусственный интеллект", "языковая модель", "ии", "нейросеть",
    "виртуальный помощник", "помощник", "языковой моделью",
    "крупномасштабной языковой моделью", "большая языковая модель", "чат-бот",
    "машинного обучения", "модель машинного обучения", "нейросетевая модель",
    "asistente", "inteligencia artificial", "modelo de lenguaje", "ia conversacional",
    "asistente virtual", "modelo de lenguaje formal", "chatbot",
    "trợ lý", "trí tuệ nhân tạo", "mô hình ngôn ngữ lớn", "mô hình ngôn ngữ",
    "trợ lý ảo", "trợ lý ai", "người máy", "chatbot",
    "intelligence artificielle", "modèle de langage", "ia conversationnelle",
    "assistant virtuel", "agent conversationnel",
    "ai 助手", "ai 비서", "ai 언어 모델", "ai 语言助手", "语言助手",
]

# (pattern, canon) — checked in order, most specific first. Patterns are
# lowercase substrings incl. non-Latin renderings of major identities.
NAME_MAP = [
    (["chatgpt", "chat gpt", "чатгпт", "챗gpt", "챗지피티"], "chatgpt"),
    (["openai", "оупенэйай", "オープンai"], "openai"),
    (["claude", "克劳德", "クロード", "클로드", "клод", "клавдий", "клэр", "クロエ", "클라우드"], "claude"),
    (["anthropic", "антропик", "アンソロピック", "앤트로픽", "안트로픽"], "anthropic"),
    (["gemini", "ジェミニ", "제미나이", "джемини", "제미니"], "gemini"),
    (["bard", "바드"], "gemini"),
    (["deepmind", "딥마인드", "дипмайнд"], "google"),
    (["google", "谷歌", "グーグル", "구글", "гугл"], "google"),
    (["deepseek", "深度求索", "딥시크", "дипсик"], "deepseek"),
    (["通义千问", "千问", "通义", "通義", "qwen", "tongyi", "큐원",
      "тонги цяньвэнь", "тоньи цяньвэнь", "цяньвэнь", "тунъи"], "qwen"),
    (["阿里巴巴", "阿里云", "alibaba", "aliyun", "알리바바", "アリババ", "алибаба",
      "阿里", "아리바바", "알리 클라우드"], "alibaba"),
    (["llama", "ラマ", "라마", "лама", "льяма"], "llama"),
    (["meta", "メタ", "메타", "мета"], "meta"),
    (["kimi", "キミ", "키미", "кими"], "kimi"),
    (["moonshot", "月之暗面", "月の暗面", "문샷"], "moonshot"),
    (["mistral", "мистраль", "ミストラル", "미스트랄", "mixtral", "le chat", "lechat"], "mistral"),
    (["grok", "грок", "그록"], "grok"), (["xai", "x-ai", "x.ai"], "xai"),
    (["chatglm", "glm", "智谱", "zhipu", "z.ai", "z-ai", "지푸", "즈푸", "智譜"], "glm"),
    (["文心一言", "文心", "ernie", "wenxin"], "ernie"),
    (["百度", "baidu"], "baidu"),
    (["混元", "hunyuan", "혼원"], "hunyuan"),
    (["腾讯", "tencent", "騰訊", "テンセント", "텐센트"], "tencent"),
    (["亚马逊", "amazon", "アマゾン", "아마존"], "amazon"),
    (["yandex", "яндекс", "алиса", "alice", "алису", "얀덱스"], "yandex"),
    (["сбер", "sber", "gigachat", "гигачат", "рудалл"], "sber"),
    (["bing", "빙", "бинг"], "microsoft"),
    (["豆包", "doubao", "云雀"], "doubao"), (["字节跳动", "字节", "bytedance"], "bytedance"),
    (["360智脑", "智脑", "360 zhinao", "qihoo"], "zhinao360"),
    (["讯飞星火", "星火", "spark", "iflytek", "讯飞"], "sparkdesk"),
    (["naver", "네이버", "hyperclova", "클로바", "clova"], "naver"),
    (["快手", "kuaishou", "kwaipilot"], "kuaishou"),
    (["阶跃星辰", "stepfun", "ステップ"], "stepfun"),
    (["小米", "xiaomi", "mimo"], "xiaomi"),
    (["蚂蚁", "ant group", "ant digital", "inclusionai", "百灵", "bailing", "lingdt", "アリデジタル"], "ant"),
    (["书生", "浦语", "internlm", "shanghai ai lab"], "internlm"),
    (["yasa", "reka"], "reka"),
    (["le chat", "lechat"], "mistral"),
    (["阿里云", "アリクラウド", "알리 클라우드"], "alibaba"),
    (["코히어", "cohere", "command", "north"], "cohere"),
    (["nemotron", "nvidia", "엔비디아", "네모트론", "немотрон"], "nvidia"),
    (["hermes"], "nous"), (["sonar"], "perplexity"),
    (["granite"], "ibm"), (["olmo", "allenai"], "allenai"),
    (["copilot"], "copilot"), (["microsoft", "微软"], "microsoft"),
    (["siri"], "siri"), (["alexa"], "alexa"), (["cortana"], "cortana"),
    (["gpt"], "chatgpt"),  # bare "GPT-x" after all specific checks
]

# family -> canon keys that count as SELF for that family
FAMILY_SELF = {
    "openai": {"chatgpt", "openai"},
    "anthropic": {"claude", "anthropic"},
    "google": {"gemini", "google"}, "gemma": {"google", "gemini"},
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
    "amazon": {"nova", "amazon", "alexa"},
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
    "aisingapore": {"sea-lion", "sealion", "aisingapore"},
}


# trailing lab/model suffixes that defeat name matching (通义千问模型 -> 通义千问)
CJK_SUFFIX = ["模型", "大模型", "公司", "科技", "研究所", "实验室", "团队", "系列",
              "モデル", "연구소", "팀", "회사", "모델"]
# common traditional -> simplified so 聊天機器人 == 聊天机器人 (generic), 騰訊 == 腾讯
TRAD2SIMP = str.maketrans("機語說對開發麼們產業騰訊龍認學實驗會單詞問誰",
                          "机语说对开发么们产业腾讯龙认学实验会单词问谁")


def canon_identity(raw: str | None) -> str | None:
    """None for generics/empty; canon key for known identities; other:… else.

    Order: known-name match FIRST (raw + CJK-suffix-stripped), then generic
    filter, then residual `other:`. Fixes v2 bugs — generics leaking as claims,
    cross-script self-names bucketed as hallucinations, `strip("an ")` char-set bug.
    """
    if not raw:
        return None
    low = raw.strip().lower().translate(TRAD2SIMP)
    stripped = low
    for suf in CJK_SUFFIX:
        if stripped.endswith(suf) and len(stripped) > len(suf):
            stripped = stripped[:-len(suf)].strip("（）() 、,-··　 ").strip()
    # 1. known identity name (raw or suffix-stripped)
    for cand in (low, stripped):
        for pats, canon in NAME_MAP:
            if any(p in cand for p in pats):
                return canon
    # 2. generic descriptor? (strip a leading article properly, not char-set)
    art = low
    for a in ("an ", "a ", "the "):
        if art.startswith(a):
            art = art[len(a):].strip()
            break
    if any(low == g or art == g for g in GENERIC_TERMS) or \
       any(g in low for g in GENERIC_TERMS
           if len(g) >= 4 or (len(g) >= 2 and not g.isascii())):
        # non-ASCII (CJK/Korean) generics substring-match at >=2 chars: 助手 in 小助手
        return None
    if len(low) < 3:
        return None
    return f"other:{low[:40]}"


def _norm(s: str) -> str:
    """lowercase, punctuation/hyphens -> spaces, collapse whitespace."""
    out = []
    for ch in s.lower():
        out.append(ch if (ch.isalnum() or "一" <= ch <= "鿿"
                          or "぀" <= ch <= "ヿ" or "가" <= ch <= "힯"
                          or "Ѐ" <= ch <= "ӿ") else " ")
    return " ".join("".join(out).split())


def is_self(canon: str | None, family: str, aliases: list[str], expected: str) -> bool:
    if canon is None:
        return True
    fam_self = FAMILY_SELF.get(family, {family})
    if canon in fam_self:
        return True
    own = _norm(" ".join(aliases) + " " + expected + " " + family
                + " " + " ".join(fam_self))
    if canon.startswith("other:"):
        words = [w for w in _norm(canon[6:]).split() if len(w) > 2 or
                 any("一" <= c <= "鿿" for c in w)]
        return any(w in own for w in words) if words else True
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
