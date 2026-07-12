"""Judge-vs-regex agreement on judged records.

Reimplements v1's regex detection (foreign model/creator names with word
boundaries, self-references excluded) and compares against the LLM judge's
is_discrepant verdict. Reports the confusion matrix and disagreements.

Usage: python -m sweep.validate_judge [--show=10]
"""

import argparse
import json
import re
from pathlib import Path

from .analyze import load, canon_identity, is_self

ROOT = Path(__file__).resolve().parent.parent
SWEEP = ROOT / "results" / "main_sweep.jsonl"

# v1's name lists (SWEEP_DETAILS.md)
V1_NAMES = ["chatgpt", "claude", "gemini", "deepseek", "grok", "llama", "qwen",
            "kimi", "ernie", "glm", "phi", "command", "nova", "pi", "mercury",
            "jamba", "hunyuan", "tongyi"]
V1_CREATORS = ["openai", "anthropic", "google", "deepseek", "meta", "mistral",
               "xai", "moonshot", "baidu", "zhipu", "tencent", "cohere",
               "amazon", "microsoft", "inflection", "inception", "ai21",
               "liquid", "alibaba", "meituan", "stepfun"]


def regex_flag(text: str, aliases: list[str], family: str) -> list[str]:
    """v1-style: any foreign name present (word-boundary), self excluded."""
    own = " ".join(aliases).lower() + " " + family
    found = []
    for name in V1_NAMES + V1_CREATORS:
        if name in own:
            continue
        if re.search(rf"\b{re.escape(name)}\b", text, re.I):
            found.append(name)
    return sorted(set(found))


def judge_flag(j: dict) -> bool:
    jm = j["judgment"]
    fam = j.get("family", "")
    for f in ("claimed_name", "claimed_creator",
              "reasoning_claimed_name", "reasoning_claimed_creator"):
        c = canon_identity(jm.get(f))
        if c and not is_self(c, fam, j.get("aliases", []), j["expected_identity"]):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=8)
    args = ap.parse_args()

    text_by_key = {}
    for line in open(SWEEP, encoding="utf-8"):
        r = json.loads(line)
        key = f"{r['resume_key']}::t{r.get('turn_index', 0)}"
        text_by_key[key] = (r.get("content_clean") or "") + " " + (r.get("reasoning") or "")

    rows = load()
    both = j_only = r_only = neither = 0
    dis_j, dis_r = [], []
    for j in rows:
        if j["prompt_category"] in ("system_probe", "probe_cross", "probe_self"):
            continue  # v1 regex wasn't meant for these
        key = j["judge_key"]
        if key not in text_by_key:
            continue
        text = text_by_key[key]
        # current-registry aliases (load() joins them onto j) — fair comparison
        rf = bool(regex_flag(text, j.get("aliases", []), j.get("family", "")))
        jf = judge_flag(j)
        if jf and rf:
            both += 1
        elif jf:
            j_only += 1
            dis_j.append(j)
        elif rf:
            r_only += 1
            dis_r.append((j, regex_flag(text, j.get("aliases", []), j.get("family", "")), text))
        else:
            neither += 1

    n = both + j_only + r_only + neither
    print(f"n={n}   both-flag={both}   judge-only={j_only}   regex-only={r_only}   neither={neither}")
    print(f"raw agreement: {(both + neither) / max(n, 1):.1%}\n")

    print(f"=== judge-only (regex missed) — up to {args.show} ===")
    for j in dis_j[: args.show]:
        jm = j["judgment"]
        print(f"  {j['model_id']} {j['prompt_id']}: name={jm.get('claimed_name')!r} "
              f"creator={jm.get('claimed_creator')!r} ev={jm.get('evidence', '')[:60]!r}")

    print(f"\n=== regex-only (judge said self/none) — up to {args.show} ===")
    for j, names, text in dis_r[: args.show]:
        print(f"  {j['model_id']} {j['prompt_id']}: regex={names} | {text[:110]!r}")


if __name__ == "__main__":
    main()
