"""Judge-model bench: candidates re-judge a stratified tricky subset of the
pilot; verdicts compared against the panel majority.

Strata: (A) records the incumbent flagged discrepant, (B) regex/judge
disagreements, (C) random clean records. ~60-70 records total.

Usage: python -m sweep.bench_judge [--run] [--report]
"""

import argparse
from .analyze import open_lines
import asyncio
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import aiohttp

from . import api
from .judge import PROMPT, parse_json, judge_key
from .validate_judge import judge_flag, regex_flag

ROOT = Path(__file__).resolve().parent.parent
SWEEP = ROOT / "results" / "main_sweep.jsonl"
JUDGMENTS = ROOT / "results" / "judgments.jsonl"
BENCH_DIR = ROOT / "results" / "judge_bench"

CANDIDATES = {
    "gpt-4o-mini": "openai/gpt-4o-mini",          # incumbent
    "gpt-4.1-mini": "openai/gpt-4.1-mini",
    "gpt-5-mini": "openai/gpt-5-mini",
    "gemini-2.5-flash": "openrouter/google/gemini-2.5-flash",
    "gemini-3-flash": "openrouter/google/gemini-3-flash-preview",
    "claude-haiku-4.5": "anthropic/claude-haiku-4-5",
}
NATIVE = {"anthropic/claude-haiku-4-5"}
CONCURRENCY = 12


def pick_records():
    recs = {}
    for line in open_lines(SWEEP):
        r = json.loads(line)
        if not r.get("error") and (r.get("content_clean") or r.get("reasoning")):
            recs[judge_key(r)] = r

    # FROZEN SET: if a bench file already exists, reuse its record keys so all
    # candidates are compared on identical records (judgments.jsonl churns as
    # the sweep grows, so re-deriving strata would silently shift the set).
    anchor = BENCH_DIR / "gpt-4o-mini.jsonl"
    if anchor.exists():
        keys = [json.loads(l)["judge_key"] for l in open(anchor, encoding="utf-8")]
        return [recs[k] for k in keys if k in recs]

    flagged, clean = [], []
    for line in open_lines(JUDGMENTS):
        j = json.loads(line)
        if j.get("judge_error") or not j.get("judgment"):
            continue
        if j["prompt_category"] in ("system_probe",):
            continue
        k = j["judge_key"]
        if k not in recs:
            continue
        (flagged if judge_flag(j) else clean).append(k)

    disagreements = []
    for line in open_lines(JUDGMENTS):
        j = json.loads(line)
        if j.get("judge_error") or not j.get("judgment"):
            continue
        k = j["judge_key"]
        if k not in recs:
            continue
        r = recs[k]
        text = (r.get("content_clean") or "") + " " + (r.get("reasoning") or "")
        rf = bool(regex_flag(text, j.get("aliases", []), j.get("family", "")))
        if rf != judge_flag(j):
            disagreements.append(k)

    rng = random.Random(3)
    sel = list(dict.fromkeys(
        rng.sample(flagged, min(30, len(flagged))) +
        rng.sample(disagreements, min(20, len(disagreements))) +
        rng.sample(clean, min(20, len(clean)))
    ))
    return [recs[k] for k in sel]


async def judge_with(session, key, rec, cand_id, sem):
    user_prompt = rec["messages_sent"][-1]["content"] if rec.get("messages_sent") else ""
    content = rec.get("content_clean") or rec.get("content") or ""
    reasoning = rec.get("reasoning") or ""
    prompt = PROMPT.format(prompt=user_prompt[:2000], content=content[:4000],
                           reasoning=reasoning[:4000])
    route = "proxy-native" if cand_id in NATIVE else None
    async with sem:
        r = await api.call(session, cand_id, [{"role": "user", "content": prompt}],
                           key, temperature=0, max_tokens=500, route=route, timeout=120)
    parsed = parse_json(r["content_clean"]) if not r["error"] else None
    u = r.get("usage") or {}
    return {
        "judge_key": judge_key(rec),
        "model_id": rec["model_id"], "family": rec["family"],
        "expected_identity": rec["expected_identity"], "aliases": rec.get("aliases", []),
        "prompt_id": rec["prompt_id"], "prompt_category": rec["prompt_category"],
        "judgment": parsed, "judge_error": r["error"] if parsed is None else None,
        "tok_in": u.get("prompt_tokens") or 0, "tok_out": u.get("completion_tokens") or 0,
        "latency_ms": r.get("latency_ms"),
    }


async def run_bench():
    records = pick_records()
    print(f"bench set: {len(records)} records")
    key = api.load_key()
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(CONCURRENCY)
    conn = aiohttp.TCPConnector(limit=CONCURRENCY + 4)
    async with aiohttp.ClientSession(connector=conn) as session:
        for name, cand_id in CANDIDATES.items():
            out = BENCH_DIR / f"{name}.jsonl"
            if out.exists() and sum(1 for _ in open(out)) >= len(records):
                print(f"  {name}: cached")
                continue
            results = await asyncio.gather(*[
                judge_with(session, key, r, cand_id, sem) for r in records
            ])
            with open(out, "w", encoding="utf-8") as f:
                for res in results:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
            errs = sum(1 for r in results if r["judge_error"])
            print(f"  {name}: {len(results)} judged, {errs} errors")


def report():
    # current-registry aliases for fair scoring
    reg = {m["id"]: m for m in json.loads((ROOT / "config" / "models.json").read_text())["models"]}
    panels = {}
    for name in CANDIDATES:
        p = BENCH_DIR / f"{name}.jsonl"
        if not p.exists():
            continue
        panels[name] = {}
        for line in open(p, encoding="utf-8"):
            j = json.loads(line)
            m = reg.get(j["model_id"])
            if m:
                j["aliases"], j["family"] = m["aliases"], m["family"]
            panels[name][j["judge_key"]] = j

    keys = sorted(set.intersection(*[set(p) for p in panels.values()]))
    verdicts = {k: {} for k in keys}
    for name, p in panels.items():
        for k in keys:
            j = p[k]
            verdicts[k][name] = None if j["judgment"] is None else judge_flag(j)

    # panel majority (ignoring None)
    majority = {}
    for k in keys:
        votes = [v for v in verdicts[k].values() if v is not None]
        majority[k] = Counter(votes).most_common(1)[0][0] if votes else None

    print(f"{len(keys)} records judged by all {len(panels)} candidates\n")
    print(f"{'judge':18s} {'agree-maj':>9s} {'flip-vs-4o-mini':>15s} {'parse-fail':>10s} {'avg-latency':>11s} {'$/1k-recs':>9s}")
    prices = {"gpt-4o-mini": (0.15, 0.6), "gpt-4.1-mini": (0.4, 1.6), "gpt-5-mini": (0.25, 2.0),
              "gemini-2.5-flash": (0.3, 2.5), "claude-haiku-4.5": (1.0, 5.0)}
    for name, p in panels.items():
        agree = sum(1 for k in keys if verdicts[k][name] == majority[k])
        flips = sum(1 for k in keys
                    if verdicts[k][name] is not None and verdicts[k].get("gpt-4o-mini") is not None
                    and verdicts[k][name] != verdicts[k]["gpt-4o-mini"])
        fails = sum(1 for k in keys if p[k]["judgment"] is None)
        lat = sum(p[k]["latency_ms"] or 0 for k in keys) / max(len(keys), 1)
        pin, pout = prices.get(name, (0, 0))
        ti = sum(p[k]["tok_in"] for k in keys) / max(len(keys), 1)
        to = sum(p[k]["tok_out"] for k in keys) / max(len(keys), 1)
        cost1k = (ti * pin + to * pout) / 1e6 * 1000
        print(f"{name:18s} {agree:>4d}/{len(keys):<4d} {flips:>15d} {fails:>10d} {lat:>9.0f}ms ${cost1k:>7.2f}")

    print("\n=== split records (no unanimity) — for adjudication ===")
    for k in keys:
        vs = {n: v for n, v in verdicts[k].items() if v is not None}
        if len(set(vs.values())) > 1:
            j0 = panels["gpt-4o-mini"][k]
            names = {n: (p[k]["judgment"] or {}).get("claimed_name") for n, p in panels.items()}
            print(f"  {k}")
            print(f"    verdicts={vs}")
            print(f"    claimed_name per judge: {names}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.run or not args.report:
        asyncio.run(run_bench())
    if args.report or not args.run:
        report()


if __name__ == "__main__":
    main()
