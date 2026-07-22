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
    from .make_figs import complete_models, LOCAL_MODELS
    reg = {m["id"]: m for m in json.loads((ROOT / "config" / "models.json").read_text())["models"]}
    hyg = json.loads((ROOT / "config" / "provider_hygiene.json").read_text())
    # raw-weights registry from the run manifest (names/families for canon)
    for l in open_lines(ROOT / "config" / "local_manifest.jsonl"):
        d = json.loads(l)
        reg.setdefault(d["entry"]["id"], d["entry"])
    # ANALYZED set only: complete API models (>=95%) + the canonical raw-weights RESULTS
    # (LOCAL_MODELS = 10). This deliberately excludes the 6 API-duplicate raw-weights
    # (moved to raw_weights_comparison/) so they can't be double-counted, and the
    # incomplete hf-router experiments (which fail the completeness gate).
    allowed = complete_models(reg, hyg) | set(LOCAL_MODELS)
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
            if r.get("error") or r["model_id"] not in allowed:
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
    """GitHub heading-anchor slug: downcase, DROP punctuation (so '4.8' -> '48',
    not '4-8'), then spaces -> hyphens. Must match GFM or the index links break."""
    s = re.sub(r"[^\w\- ]", "", name.lower())
    return re.sub(r"\s+", "-", s.strip())


def collect_model(mid, name, fam, exp, aliases, rows, jud, verdicts):
    """Per-model mismatch data: spontaneous claim counts, "are you X?" acceptances,
    per-language rates, and EVERY spontaneous drift record (prompt + response)."""
    claims = Counter()            # spontaneous canon claimed -> count
    cross = Counter()             # canon accepted when asked "are you X?"
    recs = []                     # every spontaneous drift: (lang, prompt, snippet, claimed_display)
    lang_stats = defaultdict(lambda: [0, 0])   # lang -> [drift, total] on identity/creator
    dn = tot = 0
    for r in rows:
        cat = r["prompt_category"]
        lang = "cross" if cat in ("probe_cross", "probe_self") else lang_of(cat)
        prompt = (r.get("messages_sent") or [{}])[-1].get("content", r.get("prompt_id", ""))
        resp = (r.get("content_clean") or r.get("content") or "").strip()
        key = f"{r['resume_key']}::t{r.get('turn_index',0)}"
        j = jud.get(key)
        jm = (j or {}).get("judgment") or {}
        # adjudications are keyed off the JUDGMENT record's turn (turn_index=None -> ::tNone),
        # NOT the raw record's (missing -> ::t0). Use the judgment's key or the verdict never matches.
        adjk = f"{j['resume_key']}::t{j.get('turn_index', 0)}" if j else key
        cn = canon_identity(jm.get("claimed_name")); cc = canon_identity(jm.get("claimed_creator"))
        foreign = [c for c in (cn, cc) if c and not is_self(c, fam, aliases, exp)]
        drift = is_drift(foreign, adjk, verdicts)
        if cat.startswith(("direct_", "creator_")):
            tot += 1
            lang_stats[lang][1] += 1
            if drift:
                dn += 1
                lang_stats[lang][0] += 1
        if not drift:
            continue
        if cat in ("probe_cross", "probe_self"):
            cross[foreign[0]] += 1   # primary claimed id (avoid name+creator double-count)
        else:                     # spontaneous (direct / creator / casual)
            claims[foreign[0]] += 1
            recs.append((lang, prompt, resp[:300].replace("\n", " "), jm.get("claimed_name") or foreign[0]))
    return (100 * dn / tot if tot else 0), dn, tot, claims, cross, recs, dict(lang_stats)


LANG_ORDER = ["en", "zh", "ja", "ko", "ru", "fr", "es", "vi"]
# collapse duplicate family slugs so a vendor's models share one page
VENDOR = {"olmo": "allenai", "alibaba": "qwen", "gemma": "google", "moonshot": "kimi"}
VENDOR_NAME = {"openai": "OpenAI", "anthropic": "Anthropic", "google": "Google", "deepseek": "DeepSeek",
               "qwen": "Qwen / Alibaba", "meta": "Meta (Llama)", "kimi": "Moonshot (Kimi)", "mistral": "Mistral",
               "xai": "xAI (Grok)", "zhipu": "Zhipu (GLM)", "baidu": "Baidu", "tencent": "Tencent",
               "bytedance": "ByteDance", "microsoft": "Microsoft", "nvidia": "NVIDIA", "cohere": "Cohere",
               "amazon": "Amazon", "nous": "Nous", "perplexity": "Perplexity", "allenai": "Ai2 (OLMo)",
               "minimax": "MiniMax", "poolside": "Poolside", "perceptron": "Perceptron", "reka": "Reka",
               "ai21": "AI21", "upstage": "Upstage", "ibm": "IBM", "aisingapore": "AI Singapore",
               "liquid": "Liquid", "inception": "Inception", "stepfun": "StepFun", "kuaishou": "Kuaishou",
               "xiaomi": "Xiaomi", "nex": "Nex", "sakana": "Sakana", "arcee": "Arcee", "writer": "Writer",
               "inflection": "Inflection", "cogito": "Cogito"}


def vendor_of(fam):
    return VENDOR.get(fam, fam)


def vname(v):
    return VENDOR_NAME.get(v, v.title())


def render_section(m):
    """Full markdown for one model: summary lines + every drift record by language."""
    rate, dn, tot, name, fam, exp, claims, cross, recs, lstats = m
    L = [f"## {name}\n",
         f"official **{exp}** · family `{fam}` · spontaneous mismatch **{rate:.0f}%** ({dn}/{tot})  "]
    drifting = sorted(((l, d, n) for l, (d, n) in lstats.items() if d), key=lambda x: -x[1] / x[2])
    if drifting:
        parts = " · ".join(f"{LANG_NAME.get(l, l)} {100*d/n:.0f}% ({d}/{n})" for l, d, n in drifting)
        clean = sorted(LANG_NAME.get(l, l) for l, (d, n) in lstats.items() if n and not d)
        L.append("**By language:** " + parts + ("  ·  clean in " + ", ".join(clean) if clean else "") + "  ")
    if claims:
        L.append("**Claims as:** " + " · ".join(f"{brand(c)} ×{n}" for c, n in claims.most_common()) + "  ")
    if cross:
        L.append("**Accepts when asked “are you X?”:** " + ", ".join(f"{brand(c)} ×{n}" for c, n in cross.most_common()) + "  ")
    L.append("")
    bylang = defaultdict(list)
    for lang, prompt, snippet, claimed in recs:
        bylang[lang].append((prompt, snippet, claimed))
    for lang in LANG_ORDER + sorted(set(bylang) - set(LANG_ORDER)):
        if lang not in bylang:
            continue
        L.append(f"**{LANG_NAME.get(lang, lang)}**  ")
        for prompt, snippet, claimed in bylang[lang]:
            L.append(f"- *{prompt}* → **{claimed}**  \n  {snippet}")
        L.append("")
    return L


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / "mismatches").mkdir(exist_ok=True)
    reg, jud, rec = collect()
    verdicts = adj_verdicts()
    models = []
    for mid, rows in rec.items():
        m = reg.get(mid, {})
        name = m.get("name", mid); fam = m.get("family", "?")
        exp = m.get("expected_identity", name); al = m.get("aliases", [])
        rate, dn, tot, claims, cross, recs, lstats = collect_model(mid, name, fam, exp, al, rows, jud, verdicts)
        if claims or cross:
            models.append((rate, dn, tot, name, fam, exp, claims, cross, recs, lstats))
    models.sort(key=lambda x: -x[0])

    # full records split by vendor so every page renders on GitHub (<512 KB each)
    byv = defaultdict(list)
    for m in models:
        byv[vendor_of(m[4])].append(m)
    link = {}
    for v, ms in sorted(byv.items()):
        fn = f"mismatches/{safe(v)}.md"
        VL = [f"# Identity mismatches — {vname(v)}\n",
              f"Every spontaneous cross-vendor identity claim by {vname(v)} models "
              f"({len(ms)} models, worst-first). Back to the [index](../MISMATCHES.md) · "
              f"full browser [rollouts/index.html](../index.html).\n"]
        for m in ms:
            VL += render_section(m)
        (OUT / fn).write_text("\n".join(VL), encoding="utf-8")
        # GFM disambiguates duplicate heading slugs with -1, -2 … (e.g. "Command R"
        # and "Command R+" both slug to command-r); mirror that so links resolve.
        used = Counter()
        for m in ms:
            base = anchor(m[3])
            link[m[3]] = f"./{fn}#{base if used[base] == 0 else f'{base}-{used[base]}'}"
            used[base] += 1

    # index: scannable summary table + per-vendor links (renders fully on GitHub)
    L = ["# Identity mismatches — where models name another vendor as themselves\n",
         f"Across {len(models)} models: what each one claims to be when it *doesn't* claim its own "
         "identity. **Rate** is the spontaneous mismatch rate on the identity/creator battery; "
         "*claims as* is what it names instead. Click a model for every prompt + response "
         "(e.g. \"Claude Opus 4.8 → DeepSeek in Chinese\", to reproduce it).\n",
         "Records are split by vendor so each page renders on GitHub. For **all** answers from **all** "
         "models (drift or not), open the full browser [`rollouts/index.html`](./index.html) or the raw "
         "[`rollouts_data.json`](./rollouts_data.json).\n",
         "| model | family | mismatch rate | claims as |", "|---|---|---|---|"]
    for m in models:
        rate, dn, tot, name, fam, claims = m[0], m[1], m[2], m[3], m[4], m[6]
        top = ", ".join(brand(c) for c, _ in claims.most_common(3)) or "—"
        L.append(f"| [{name}]({link[name]}) | {fam} | {rate:.0f}% ({dn}/{tot}) | {top} |")
    L.append("\n## By vendor\n")
    for v in sorted(byv, key=lambda v: -len(byv[v])):
        L.append(f"- [{vname(v)}](./mismatches/{safe(v)}.md) — {len(byv[v])} model{'s' if len(byv[v]) != 1 else ''}")
    (OUT / "MISMATCHES.md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote MISMATCHES.md index + {len(byv)} vendor files ({len(models)} models)")


if __name__ == "__main__":
    main()
