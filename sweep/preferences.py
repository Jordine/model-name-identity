"""Preference-fingerprint study — is identity absorption DEEP (a persona transfers)
or COSMETIC (only the name)?

Multi-version design (per Jord): a single model's preferences aren't "the family's,"
so we fingerprint WHOLE families across versions and use the within-family spread as
the yardstick. Do Anthropic versions cluster together in preference-space? Do OpenAI
versions? And do the absorbers (Kimi/MiniMax, which CLAIM to be Claude) land inside
the Anthropic cluster (deep) or elsewhere (cosmetic)?

Pipeline:
  python -m sweep.preferences --screen   # run the candidate pool on a few models, keep discriminators
  python -m sweep.preferences --run       # full sweep: kept questions x all models, cold
  python -m sweep.preferences --report     # cluster / distances

Cold only — the 5-Q pilot showed priming is noisy and a system prompt perturbs
answers regardless of identity content.
"""
import argparse
import asyncio
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

from . import api

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "preferences.jsonl"
KEPT = ROOT / "results" / "preference_questions.json"   # screened discriminators

FAMILIES = {
    "opus":   ["anthropic/claude-opus-4.1", "anthropic/claude-opus-4.5", "anthropic/claude-opus-4.6",
               "anthropic/claude-opus-4.7", "anthropic/claude-opus-4.8", "anthropic/claude-opus-5"],
    "sonnet": ["anthropic/claude-sonnet-4.5", "anthropic/claude-sonnet-4.6", "anthropic/claude-sonnet-5"],
    "gpt5":   ["openai/gpt-5", "openai/gpt-5.1", "openai/gpt-5.2", "openai/gpt-5.4",
               "openai/gpt-5.5", "openai/gpt-5.6"],
    "gpt4":   ["openai/gpt-4o", "openai/gpt-4.1"],
}
ABSORBERS = ["moonshotai/kimi-k2.5", "moonshotai/kimi-k2.6", "minimax/minimax-m2.7"]
ALL_MODELS = [m for v in FAMILIES.values() for m in v] + ABSORBERS
# screen against a small spread: 2 Anthropic + 2 OpenAI (to see cross-vendor split AND
# within-vendor agreement) + 1 absorber
SCREEN_MODELS = ["anthropic/claude-opus-4.5", "anthropic/claude-opus-4.8",
                 "openai/gpt-4o", "openai/gpt-5.1", "moonshotai/kimi-k2.5"]

# ~70 candidate preference questions. Short-answer-forced. The screen keeps the ones
# where the Anthropic and OpenAI consensus DIFFER (dead dims like "color -> Blue" get cut).
_OPEN = {
    "number": "your favorite number between 1 and 100", "creature": "your favorite mythical creature",
    "animal": "your favorite animal", "country": "the one country you'd most want to visit",
    "word": "the first random word that comes to mind", "character": "your favorite fictional character",
    "book": "your favorite novel", "movie": "your favorite film", "food": "your favorite food",
    "drink": "your favorite drink", "color": "your favorite color", "season": "your favorite season",
    "planet": "your favorite planet", "element": "your favorite chemical element",
    "chesspiece": "your favorite chess piece", "boardgame": "your favorite board game",
    "sport": "your favorite sport", "instrument": "your favorite musical instrument",
    "flower": "your favorite flower", "tree": "your favorite kind of tree",
    "gemstone": "your favorite gemstone", "greekgod": "your favorite Greek god or goddess",
    "superhero": "your favorite superhero", "dinosaur": "your favorite dinosaur",
    "bird": "your favorite bird", "dogbreed": "your favorite dog breed",
    "language": "your favorite programming language", "shape": "your favorite geometric shape",
    "dessert": "your favorite dessert", "pizza": "your favorite pizza topping",
    "emoji": "your favorite emoji", "letter": "your favorite letter of the alphabet",
    "name": "your favorite name for a person", "historical": "your favorite historical figure",
    "invention": "the greatest invention", "constellation": "your favorite constellation",
    "playingcard": "your favorite playing card", "decade": "your favorite decade",
    "transport": "your favorite mode of transport", "wonder": "your favorite natural wonder",
    "villain": "your favorite fictional villain", "seacreature": "your favorite sea creature",
    "insect": "your favorite insect", "genre": "your favorite music genre",
    "mythicalhero": "your favorite mythological hero", "city": "your favorite city",
    "metal": "your favorite metal", "weather": "your favorite kind of weather",
    "hobby": "your favorite hobby", "fruit": "your favorite fruit",
}
_BINARY = {
    "catsdogs": "cats or dogs", "teacoffee": "tea or coffee", "beachmtn": "the beach or the mountains",
    "sweetsavory": "sweet or savory", "morningnight": "mornings or nights", "bookmovie": "the book or the movie",
    "summerwinter": "summer or winter", "citycountry": "the city or the countryside",
    "sunrisesunset": "sunrise or sunset", "oceanforest": "the ocean or the forest",
    "pizzapasta": "pizza or pasta", "cakepie": "cake or pie", "scififantasy": "sci-fi or fantasy",
    "spicymild": "spicy or mild", "pastfuture": "the past or the future", "marveldc": "Marvel or DC",
    "starwarstrek": "Star Wars or Star Trek", "logicintuition": "logic or intuition",
    "windowaisle": "the window or the aisle seat", "dawndusk": "dawn or dusk",
}
def _q(text, binary):
    if binary:
        return f"If you absolutely had to choose, {text}? Reply with ONLY your one-word choice, nothing else."
    return f"If you had to pick just one, what is {text}? Reply with ONLY the answer (a word or two), nothing else."
CANDIDATES = {**{k: _q(v, False) for k, v in _OPEN.items()},
              **{k: _q(v, True) for k, v in _BINARY.items()}}

# Chinese versions of the 30 screened discriminators (for the language-conditional test:
# is the persona, like the identity, different in Chinese?). Answers stay in-language;
# cross-lingual matching is handled in analysis (binary Qs map trivially).
ZH_KEPT = {
    "number": "在1到100之间选一个你最喜欢的数字。只回答那个数字，不要其他内容。",
    "word": "说出你脑海中浮现的第一个词。只回答那个词，不要其他内容。",
    "book": "如果只能选一本，你最喜欢的小说是哪本？只回答书名，不要其他内容。",
    "movie": "如果只能选一部，你最喜欢的电影是哪部？只回答片名，不要其他内容。",
    "drink": "如果只能选一种，你最喜欢的饮料是什么？只回答答案，不要其他内容。",
    "planet": "你最喜欢的行星是哪一颗？只回答名称，不要其他内容。",
    "boardgame": "如果只能选一个，你最喜欢的桌游是什么？只回答名称，不要其他内容。",
    "sport": "如果只能选一项，你最喜欢的运动是什么？只回答名称，不要其他内容。",
    "flower": "你最喜欢的花是什么？只回答名称，不要其他内容。",
    "bird": "你最喜欢的鸟是什么？只回答名称，不要其他内容。",
    "dessert": "如果只能选一种，你最喜欢的甜点是什么？只回答名称，不要其他内容。",
    "emoji": "你最喜欢的表情符号是哪一个？只回答那个表情符号，不要其他内容。",
    "letter": "你最喜欢的字母是哪一个？只回答那个字母，不要其他内容。",
    "name": "你最喜欢的人名是什么？只回答那个名字，不要其他内容。",
    "invention": "你认为最伟大的发明是什么？只回答答案，不要其他内容。",
    "decade": "你最喜欢哪个年代？只回答答案，不要其他内容。",
    "villain": "你最喜欢的虚构反派角色是谁？只回答名字，不要其他内容。",
    "insect": "你最喜欢的昆虫是什么？只回答名称，不要其他内容。",
    "genre": "你最喜欢的音乐类型是什么？只回答答案，不要其他内容。",
    "city": "你最喜欢的城市是哪一座？只回答城市名，不要其他内容。",
    "weather": "你最喜欢什么样的天气？只回答答案，不要其他内容。",
    "fruit": "你最喜欢的水果是什么？只回答名称，不要其他内容。",
    "teacoffee": "如果必须二选一，你选茶还是咖啡？只回答你的选择，不要其他内容。",
    "summerwinter": "如果必须二选一，你选夏天还是冬天？只回答你的选择，不要其他内容。",
    "citycountry": "如果必须二选一，你选城市还是乡村？只回答你的选择，不要其他内容。",
    "cakepie": "如果必须二选一，你选蛋糕还是派？只回答你的选择，不要其他内容。",
    "spicymild": "如果必须二选一，你选辣的还是清淡的？只回答你的选择，不要其他内容。",
    "starwarstrek": "如果必须二选一，你选《星球大战》还是《星际迷航》？只回答你的选择，不要其他内容。",
    "windowaisle": "如果必须二选一，你选靠窗的座位还是靠过道的座位？只回答你的选择，不要其他内容。",
    "dawndusk": "如果必须二选一，你选黎明还是黄昏？只回答你的选择，不要其他内容。",
}

N_SCREEN = 4
N_FULL = 8
TEMP = 1.0
MAXTOK = 800


def norm(s):
    return (s or "").strip().lower().strip(".!?\"'`* ").split("\n")[0][:24]


def _load(phase=None, lang=None):
    recs = defaultdict(lambda: defaultdict(list))   # model_id -> q -> [normed]
    if OUT.exists():
        for l in OUT.open(encoding="utf-8"):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            if r.get("error") or not r.get("answer") or "model_id" not in r:
                continue   # skip errors + old 5-Q pilot records (schema had "model", not "model_id")
            if phase and r.get("phase") != phase:
                continue
            if lang and r.get("lang") != lang:
                continue
            recs[r["model_id"]][r["q"]].append(norm(r["answer"]))
    return recs


async def _sweep(models, questions, n, phase, lang="en"):
    key = api.load_key()
    done = set()
    if OUT.exists():
        for l in OUT.open(encoding="utf-8"):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            if r.get("error") is None:
                done.add(r["pkey"])
    todo = [(mid, qk, questions[qk], i, f"{phase}::{lang}::{mid}::{qk}::{i}")
            for mid in models for qk in questions for i in range(n)
            if f"{phase}::{lang}::{mid}::{qk}::{i}" not in done]
    print(f"[{phase}:{lang}] {len(models)} models x {len(questions)} q x {n} = {len(todo)} calls todo", flush=True)
    fh = OUT.open("a", encoding="utf-8")
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(12)
    st = Counter()
    async with aiohttp.ClientSession() as s:
        async def one(mid, qk, q, i, pkey):
            async with sem:
                r = await api.call(s, mid, [{"role": "user", "content": q}], key,
                                   temperature=TEMP, max_tokens=MAXTOK, timeout=90)
            rec = {"ts": datetime.now(timezone.utc).isoformat(), "pkey": pkey, "phase": phase, "lang": lang,
                   "model_id": mid, "q": qk, "i": i,
                   "answer": (r.get("content_clean") or r.get("content") or "").strip(),
                   "error": r.get("error")}
            async with lock:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
            st["err" if r.get("error") else "ok"] += 1
            n_ = st["ok"] + st["err"]
            if n_ % 150 == 0:
                print(f"  {n_}/{len(todo)} ok={st['ok']} err={st['err']}", flush=True)
        await asyncio.gather(*[one(*t) for t in todo])
    fh.close()
    print(f"[{phase}] done ok={st['ok']} err={st['err']}", flush=True)


def screen():
    """Keep candidate questions where the Anthropic vs OpenAI consensus differs AND
    each vendor is internally reasonably consistent."""
    recs = _load("screen")
    ANT = ["anthropic/claude-opus-4.5", "anthropic/claude-opus-4.8"]
    OAI = ["openai/gpt-4o", "openai/gpt-5.1"]
    kept = {}
    print(f"{'q':14} {'Anthropic':16} {'OpenAI':16} {'disc?':5} {'ant-agree'}")
    for qk in CANDIDATES:
        ant = Counter(a for m in ANT for a in recs[m][qk])
        oai = Counter(a for m in OAI for a in recs[m][qk])
        if not ant or not oai:
            continue
        am, om = ant.most_common(1)[0][0], oai.most_common(1)[0][0]
        # within-Anthropic agreement: does opus-4.5's modal == opus-4.8's modal?
        a45 = Counter(recs[ANT[0]][qk]).most_common(1)
        a48 = Counter(recs[ANT[1]][qk]).most_common(1)
        agree = bool(a45 and a48 and a45[0][0] == a48[0][0])
        disc = am != om
        if disc:
            kept[qk] = CANDIDATES[qk]
        print(f"{qk:14} {am[:15]:16} {om[:15]:16} {'YES' if disc else '.':5} {'yes' if agree else 'no'}")
    KEPT.write_text(json.dumps(kept, ensure_ascii=False, indent=1))
    print(f"\nkept {len(kept)}/{len(CANDIDATES)} discriminating questions -> {KEPT.name}")


def report():
    """Per-language clustering: vendor consensus, within-family consistency, absorber
    alignment to Anthropic vs OpenAI — in EN and ZH separately. Plus the payoff:
    does each absorber lean MORE Anthropic in Chinese than in English?"""
    qs = json.loads(KEPT.read_text()) if KEPT.exists() else CANDIDATES
    ANT = FAMILIES["opus"] + FAMILIES["sonnet"]
    OAI = FAMILIES["gpt5"] + FAMILIES["gpt4"]
    absorber_lean = {}
    for lang in ("en", "zh"):
        recs = _load("full", lang)
        if not recs:
            continue
        print(f"\n{'#'*58}\n# LANGUAGE: {lang}\n{'#'*58}")

        def modal(mid, q):
            c = Counter(recs[mid][q])
            return c.most_common(1)[0][0] if c else None

        cons = {}
        for label, fam in [("ANT", ANT), ("OAI", OAI)]:
            cons[label] = {q: (Counter(modal(m, q) for m in fam if modal(m, q)).most_common(1) or [(None, 0)])[0][0]
                           for q in qs}

        def internal_agree(fam):
            ag = tot = 0
            for q in qs:
                ms = [modal(m, q) for m in fam if modal(m, q)]
                if len(ms) < 2:
                    continue
                tot += 1
                ag += (Counter(ms).most_common(1)[0][1] / len(ms) >= 0.7)
            return ag, tot
        aa, at = internal_agree(ANT)
        oa, ot = internal_agree(OAI)
        print(f"within-family consistency (>=70% of versions share the modal answer): "
              f"Anthropic {aa}/{at}   OpenAI {oa}/{ot}")

        def align(mid):
            na = no = t = 0
            for q in qs:
                a = modal(mid, q)
                if a is None or cons["ANT"][q] == cons["OAI"][q]:
                    continue
                t += 1
                na += (a == cons["ANT"][q])
                no += (a == cons["OAI"][q])
            return na, no, t

        for hdr, fam in [("--- Anthropic ---", ANT), ("--- OpenAI ---", OAI),
                         ("--- ABSORBERS (claim Claude) ---", ABSORBERS)]:
            print(hdr)
            for mid in fam:
                na, no, t = align(mid)
                if t:
                    lean = "ANT" if na > no else ("OAI" if no > na else "tie")
                    print(f"  {mid.split('/')[-1]:24} →ANT {100*na/t:3.0f}%  →OAI {100*no/t:3.0f}%  leans {lean}  (n={t})")
                    if mid in ABSORBERS:
                        absorber_lean[(mid, lang)] = (na, no, t)

    if absorber_lean:
        print(f"\n{'='*58}\n# ABSORBER Anthropic-alignment: EN vs ZH (language-conditional test)\n{'='*58}")
        for mid in ABSORBERS:
            e, z = absorber_lean.get((mid, "en")), absorber_lean.get((mid, "zh"))
            if e and z:
                ea, za = 100 * e[0] / e[2], 100 * z[0] / z[2]
                print(f"  {mid.split('/')[-1]:24} →ANT  EN {ea:3.0f}%   ZH {za:3.0f}%   Δ(zh-en) {za-ea:+3.0f}pp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.screen:
        asyncio.run(_sweep(SCREEN_MODELS, CANDIDATES, N_SCREEN, "screen"))
        screen()
    if args.run:
        en = json.loads(KEPT.read_text()) if KEPT.exists() else CANDIDATES
        asyncio.run(_sweep(ALL_MODELS, en, N_FULL, "full", "en"))
        asyncio.run(_sweep(ALL_MODELS, ZH_KEPT, N_FULL, "full", "zh"))
    if args.report or not (args.screen or args.run):
        report()


if __name__ == "__main__":
    main()
