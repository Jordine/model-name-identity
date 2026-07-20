"""Generate rollouts/MISMATCHES.md — a GitHub-rendered, searchable summary of every
model that names another vendor as itself, with copy-pasteable examples so anyone
can reproduce a specific case ("Claude Opus 4.8 -> DeepSeek in Chinese").

Drift is gated on the same adjudication as the figures (adj_verdicts / is_drift).
The full every-answer browser is rollouts/index.html + rollouts_data.json
(sweep/build_rollout_viewer.py).

  python -m sweep.build_rollouts
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from .analyze import load, lang_of, canon_identity, is_self, open_lines

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "rollouts"
LANG_NAME = {"en": "English", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
             "ru": "Russian", "fr": "French", "es": "Spanish", "vi": "Vietnamese", "mixed": "Multi"}
KEEP = ("direct_", "creator_", "probe_cross", "probe_self")
BRAND = {"chatgpt": "ChatGPT", "openai": "OpenAI", "claude": "Claude", "anthropic": "Anthropic",
         "gemini": "Gemini", "google": "Google", "deepseek": "DeepSeek", "qwen": "Qwen",
         "alibaba": "Alibaba", "llama": "Llama", "meta": "Meta", "grok": "Grok", "kimi": "Kimi",
         "moonshot": "Moonshot", "mistral": "Mistral", "doubao": "Doubao", "glm": "GLM",
         "ernie": "Ernie", "hunyuan": "Hunyuan", "gpt": "GPT"}


def brand(c):
    if c.startswith("other:"):
        return c[6:]
    return BRAND.get(c, c.title())


def safe(mid):
    return re.sub(r"[^A-Za-z0-9._-]", "__", mid)


def adj_verdicts():
    """{adj_key: verdict} from both adjudication passes (API + local). Used to gate
    rollout drift flags on the *adjudicated* verdict, matching the study's
    foreign_claims() — a canon-foreign claim reviewed and NOT confirmed
    genuine_foreign is not drift; an un-adjudicated foreign claim keeps the canon flag."""
    v = {}
    for name in ("adjudications.jsonl", "adjudications_local.jsonl"):
        p = ROOT / "results" / name
        if p.exists():
            for l in open(p, encoding="utf-8"):
                try:
                    d = json.loads(l)
                except json.JSONDecodeError:
                    continue
                if d.get("verdict"):
                    v[d["adj_key"]] = d["verdict"]
    return v


def is_drift(foreign, adj_key, verdicts):
    """canon-foreign, gated on adjudication (matches analyze.foreign_claims)."""
    if not foreign:
        return False
    verdict = verdicts.get(adj_key)
    return verdict is None or verdict == "genuine_foreign"


def collect():
    from .make_figs import LOCAL_MODELS
    reg = {m["id"]: m for m in json.loads((ROOT / "config" / "models.json").read_text())["models"]}
    for mid, (name, fam, al) in LOCAL_MODELS.items():
        reg.setdefault(mid, {"id": mid, "name": name, "family": fam, "expected_identity": name, "aliases": al})
    # judgments by judge_key -> claimed name
    jud = {}
    for j in load():
        jud[j["judge_key"]] = j
    for l in open_lines(ROOT / "results_local" / "judgments_clean.jsonl"):
        j = json.loads(l)
        jud[f"{j['resume_key']}::t{j.get('turn_index',0)}"] = j
    # raw responses -> per model
    rec = defaultdict(list)
    def take(path, cond_filter=None):
        for l in open_lines(path):
            r = json.loads(l)
            if r.get("error"):
                continue
            cat = r.get("prompt_category", "")
            if not cat.startswith(KEEP):
                continue
            if cond_filter and r["resume_key"].split("::")[-1] != cond_filter:
                continue
            rec[r["model_id"]].append(r)
    take(ROOT / "results" / "main_sweep.jsonl")
    for f in (ROOT / "results_local").glob("*.jsonl"):
        if "judgment" in f.name:
            continue
        take(f, cond_filter="clean")
    return reg, jud, rec


def prompt_label(cat, pid, content):
    return f"{content}" if content else pid


def anchor(name):
    """GitHub heading-anchor slug for a model name."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


MAX_EX = 10  # copy-pasteable examples per model (dedup by claimed-identity × language)


def collect_model(mid, name, fam, exp, aliases, rows, jud, verdicts):
    """Per-model mismatch summary: spontaneous claim counts, "are you X?" acceptances,
    and a representative set of examples spanning distinct (identity, language) pairs."""
    claims = Counter()            # spontaneous canon claimed -> count
    cross = Counter()             # canon accepted when asked "are you X?"
    ex = {}                       # (canon, lang) -> (lang, prompt, snippet, claimed_display)
    dn = tot = 0
    for r in rows:
        cat = r["prompt_category"]
        lang = "cross" if cat in ("probe_cross", "probe_self") else lang_of(cat)
        prompt = (r.get("messages_sent") or [{}])[-1].get("content", r.get("prompt_id", ""))
        resp = (r.get("content_clean") or r.get("content") or "").strip()
        key = f"{r['resume_key']}::t{r.get('turn_index',0)}"
        jm = (jud.get(key) or {}).get("judgment") or {}
        cn = canon_identity(jm.get("claimed_name")); cc = canon_identity(jm.get("claimed_creator"))
        foreign = [c for c in (cn, cc) if c and not is_self(c, fam, aliases, exp)]
        drift = is_drift(foreign, key, verdicts)
        if cat.startswith(("direct_", "creator_")):
            tot += 1
            if drift:
                dn += 1
        if not drift:
            continue
        if cat in ("probe_cross", "probe_self"):
            cross[foreign[0]] += 1   # primary claimed id (avoid name+creator double-count)
        else:                     # spontaneous (direct / creator / casual)
            canon = foreign[0]
            claims[canon] += 1
            ex.setdefault((canon, lang),
                          (lang, prompt, resp[:220].replace("\n", " "), jm.get("claimed_name") or canon))
    # keep examples spanning the most-claimed identities first, then by language
    order = {l: i for i, l in enumerate(["en", "zh", "ja", "ko", "ru", "fr", "es", "vi"])}
    picks = [v for _, v in sorted(ex.items(),
             key=lambda kv: (-claims.get(kv[0][0], 0), order.get(kv[0][1], 9)))]
    return (100 * dn / tot if tot else 0), dn, tot, claims, cross, picks[:MAX_EX]


def main():
    OUT.mkdir(exist_ok=True)
    reg, jud, rec = collect()
    verdicts = adj_verdicts()
    models = []
    for mid, rows in rec.items():
        m = reg.get(mid, {})
        name = m.get("name", mid); fam = m.get("family", "?")
        exp = m.get("expected_identity", name); al = m.get("aliases", [])
        rate, dn, tot, claims, cross, picks = collect_model(mid, name, fam, exp, al, rows, jud, verdicts)
        if claims or cross:
            models.append((rate, dn, tot, name, fam, exp, claims, cross, picks))
    models.sort(key=lambda x: -x[0])

    L = ["# Identity mismatches — where models name another vendor as themselves\n",
         f"Across {len(models)} models: what each one claims to be when it *doesn't* claim its own "
         "identity, with copy-pasteable examples so you can reproduce a specific case "
         "(e.g. \"Claude Opus 4.8 → DeepSeek in Chinese\"). **Rate** is the spontaneous mismatch rate on the "
         "identity/creator battery. *Claims as* counts spontaneous answers; *accepts when asked* counts the "
         "\"are you X?\" suggestibility probes (a separate experiment — not in the rate).\n",
         "Examples are a representative sample. For **every** answer from **every** model (drift or not), open the "
         "full browser [`rollouts/index.html`](./index.html) (GitHub Pages / any static host) or search the raw "
         "[`rollouts_data.json`](./rollouts_data.json).\n",
         "| model | family | mismatch rate | claims as |", "|---|---|---|---|"]
    for rate, dn, tot, name, fam, exp, claims, cross, picks in models:
        top = ", ".join(brand(c) for c, _ in claims.most_common(3)) or "—"
        L.append(f"| [{name}](#{anchor(name)}) | {fam} | {rate:.0f}% ({dn}/{tot}) | {top} |")
    L.append("")
    for rate, dn, tot, name, fam, exp, claims, cross, picks in models:
        L.append(f"## {name}\n")
        L.append(f"official **{exp}** · family `{fam}` · spontaneous mismatch **{rate:.0f}%** ({dn}/{tot})  ")
        if claims:
            L.append("**Claims as:** " + " · ".join(f"{brand(c)} ×{n}" for c, n in claims.most_common()) + "  ")
        if cross:
            L.append("**Accepts when asked “are you X?”:** " + ", ".join(f"{brand(c)} ×{n}" for c, n in cross.most_common()) + "  ")
        L.append("")
        for lang, prompt, snippet, claimed in picks:
            tag = LANG_NAME.get(lang, lang)
            L.append(f"- [{tag}] *{prompt}* → **{claimed}**  \n  {snippet}")
        L.append("")
    (OUT / "MISMATCHES.md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote rollouts/MISMATCHES.md — {len(models)} models with ≥1 cross-vendor claim")


if __name__ == "__main__":
    main()
