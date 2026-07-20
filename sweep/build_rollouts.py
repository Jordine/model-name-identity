"""Generate browsable per-model rollout pages (GitHub-rendered markdown) so anyone
can verify a specific claim ("Kimi K2.5 -> Claude in English") by eye.

Writes rollouts/<model>.md — every identity-probing response, grouped by language
then prompt, with the judge's extracted name and a DRIFT flag on cross-vendor
claims — plus rollouts/README.md (an index sorted by mismatch rate).

  python -m sweep.build_rollouts
"""
import json
import re
from collections import defaultdict
from pathlib import Path

from .analyze import load, lang_of, canon_identity, is_self

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "rollouts"
LANG_NAME = {"en": "English", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
             "ru": "Russian", "fr": "French", "es": "Spanish", "vi": "Vietnamese", "mixed": "Multi"}
KEEP = ("direct_", "creator_", "probe_cross", "probe_self")


def safe(mid):
    return re.sub(r"[^A-Za-z0-9._-]", "__", mid)


def collect():
    reg = {m["id"]: m for m in json.loads((ROOT / "config" / "models.json").read_text())["models"]}
    # judgments by judge_key -> claimed name
    jud = {}
    for j in load():
        jud[j["judge_key"]] = j
    for l in open(ROOT / "results_local" / "judgments_clean.jsonl", encoding="utf-8"):
        j = json.loads(l)
        jud[f"{j['resume_key']}::t{j.get('turn_index',0)}"] = j
    # raw responses -> per model
    rec = defaultdict(list)
    def take(path, cond_filter=None):
        for l in open(path, encoding="utf-8"):
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


def render_model(mid, name, fam, exp, aliases, rows, jud):
    by_lang = defaultdict(lambda: defaultdict(list))
    drift_n = tot = 0
    for r in rows:
        cat = r["prompt_category"]
        lang = "cross" if cat in ("probe_cross", "probe_self") else lang_of(cat)
        prompt = (r.get("messages_sent") or [{}])[-1].get("content", r.get("prompt_id", ""))
        resp = (r.get("content_clean") or r.get("content") or "").strip()
        j = jud.get(f"{r['resume_key']}::t{r.get('turn_index',0)}")
        jm = (j or {}).get("judgment") or {}
        cn = canon_identity(jm.get("claimed_name")); cc = canon_identity(jm.get("claimed_creator"))
        foreign = [c for c in (cn, cc) if c and not is_self(c, fam, aliases, exp)]
        if cat.startswith(("direct_", "creator_")):
            tot += 1
            if foreign:
                drift_n += 1
        by_lang[lang][prompt].append((resp, jm.get("claimed_name"), foreign))
    lines = [f"# {name} — identity rollouts", ""]
    lines.append(f"**Official identity:** {exp} · family `{fam}` · aliases: {', '.join(aliases[:6]) or '—'}  ")
    rate = f"{100*drift_n/tot:.0f}% ({drift_n}/{tot})" if tot else "n/a"
    lines.append(f"**Cross-vendor mismatch rate (identity+creator):** {rate}")
    lines.append(f"\n*Each line is one sampled response. **→ X** marks a judged cross-vendor claim (drift).*\n")
    order = ["en", "zh", "ja", "ko", "ru", "fr", "es", "vi", "cross"]
    for lang in order:
        if lang not in by_lang:
            continue
        lines.append(f"## {LANG_NAME.get(lang, lang)}\n")
        for prompt, samples in by_lang[lang].items():
            lines.append(f"**{prompt}**")
            for resp, claimed, foreign in samples:
                flag = f" **→ {claimed}**" if foreign else ""
                snippet = resp[:400].replace("\n", " ")
                lines.append(f"- {snippet}{flag}")
            lines.append("")
    return "\n".join(lines), (100 * drift_n / tot if tot else 0), drift_n, tot


def main():
    OUT.mkdir(exist_ok=True)
    reg, jud, rec = collect()
    index = []
    for mid, rows in rec.items():
        m = reg.get(mid, {})
        name = m.get("name", mid); fam = m.get("family", "?")
        exp = m.get("expected_identity", name); al = m.get("aliases", [])
        md, rate, dn, tot = render_model(mid, name, fam, exp, al, rows, jud)
        fn = safe(mid) + ".md"
        (OUT / fn).write_text(md, encoding="utf-8")
        index.append((rate, dn, tot, name, fam, fn))
    index.sort(key=lambda x: -x[0])
    idx = ["# Identity rollouts — every model's answers\n",
           "Browse any model to verify a claim by eye (e.g. \"Kimi K2 → Claude\"). Rate = "
           "cross-vendor mismatch on the identity/creator battery. Sorted worst-first.\n",
           "| model | family | mismatch rate | file |", "|---|---|---|---|"]
    for rate, dn, tot, name, fam, fn in index:
        idx.append(f"| {name} | {fam} | {rate:.0f}% ({dn}/{tot}) | [{fn}](./{fn}) |")
    (OUT / "README.md").write_text("\n".join(idx), encoding="utf-8")
    print(f"wrote {len(index)} model pages + index to {OUT}")


if __name__ == "__main__":
    main()
