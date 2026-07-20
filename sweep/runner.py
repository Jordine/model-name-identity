"""Sweep runner — async, resumable, litellm-proxy-backed.

Usage:
  python -m sweep.runner --dry-run                 # counts + cost estimate
  python -m sweep.runner --models=moonshotai/kimi-k2.5,openai/gpt-4o-mini
  python -m sweep.runner --families=kimi,deepseek
  python -m sweep.runner --pilot                   # the 6-model pilot set
  python -m sweep.runner                           # everything (asks unless --yes)

Resume: reruns are safe — completed (model, prompt_id, sample_idx) triples
found in the output file are skipped; errored ones are retried.
"""

import argparse
import asyncio
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

from . import api
from .prompts import prompts_for_model, MULTI_TURN_PROMPTS

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "config" / "models.json"
HYGIENE = ROOT / "config" / "provider_hygiene.json"
OUT = ROOT / "results" / "main_sweep.jsonl"

MAX_CONCURRENT = 12
TEMPERATURE = 0.7
MAX_TOKENS = 500
MAX_TOKENS_REASONING = 1600   # so thinking doesn't eat the whole budget
SHUFFLE_SEED = 7

PILOT_MODELS = [
    "moonshotai/kimi-k2.5",            # known discrepant (claims Claude)
    "openai/gpt-4o-mini",              # known clean baseline
    "deepseek/deepseek-r1-0528",       # reasoning traces, known drift
    "meta-llama/llama-3.2-3b-instruct",  # small open model (new coverage)
    "z-ai/glm-4.7-flash",              # ZH-lab flash tier
    "qwen/qwen3.5-9b",                 # small new qwen (new coverage)
]


# ---------------------------------------------------------------------------

def load_models(args) -> list[dict]:
    reg = json.loads(REGISTRY.read_text())["models"]
    hygiene = {}
    if HYGIENE.exists():
        hygiene = json.loads(HYGIENE.read_text())
    for m in reg:
        h = hygiene.get(m["id"])
        if h:
            m["provider"] = h.get("provider", m.get("provider"))
            m["hygiene_excluded"] = h.get("exclude", False)

    models = [m for m in reg if not m.get("hygiene_excluded")]
    if args.pilot:
        models = [m for m in models if m["id"] in PILOT_MODELS]
    if args.models:
        want = set(args.models.split(","))
        models = [m for m in models if m["id"] in want]
        missing = want - {m["id"] for m in models}
        if missing:
            print(f"!! not in registry (or excluded): {sorted(missing)}")
    if args.families:
        fams = set(args.families.split(","))
        models = [m for m in models if m["family"] in fams]
    if args.limit:
        models = models[: args.limit]
    return models


def work_items(models: list[dict]) -> list[dict]:
    """Flat list of single-turn work items + one item per multi-turn seq."""
    items = []
    for m in models:
        for p in prompts_for_model(m):
            items.append({
                "kind": "single", "model": m, "prompt": p,
                "key": f"{m['id']}::{p['id']}::{p['sample_idx']}",
            })
        for mp in MULTI_TURN_PROMPTS:
            items.append({
                "kind": "multi", "model": m, "prompt": mp,
                "key": f"{m['id']}::{mp['id']}::0",
            })
    random.Random(SHUFFLE_SEED).shuffle(items)  # interleave across models
    return items


def done_keys(out_path: Path) -> set[str]:
    """Keys of already-successful calls (errored ones get retried)."""
    done = set()
    if not out_path.exists():
        return done
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("error") is None:
                done.add(r["resume_key"])
    return done


def estimate_cost(models: list[dict]) -> float:
    """$ estimate calibrated on pilot actuals (2026-07-12):
    ~35 prompt tokens/call; completion avg 376 (reasoning) / 55 (plain)."""
    total = 0.0
    for m in models:
        n = len(prompts_for_model(m)) + sum(len(mp["turns"]) for mp in MULTI_TURN_PROMPTS)
        out_tok = 376 if m.get("reasoning") else 55
        total += n * (35 * m["price_prompt"] + out_tok * m["price_completion"])
    return total


# ---------------------------------------------------------------------------

def build_record(model, prompt_id, category, sample_idx, messages, result,
                 run_type, max_tokens, resume_key, extra=None):
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "run_type": run_type,
        "resume_key": resume_key,
        "model_id": model["id"],
        "model_name": model["name"],
        "family": model["family"],
        "expected_identity": model["expected_identity"],
        "aliases": model.get("aliases", []),
        "prompt_id": prompt_id,
        "prompt_category": category,
        "sample_idx": sample_idx,
        "messages_sent": messages,
        "temperature": TEMPERATURE,
        "max_tokens": max_tokens,
        "provider_requested": model.get("provider"),
        "provider_served": result["provider_served"],
        "returned_model": result["returned_model"],
        "content": result["content"],
        "content_clean": result["content_clean"],
        "reasoning": result["reasoning"],
        "finish_reason": result["finish_reason"],
        "usage": result["usage"],
        "latency_ms": result["latency_ms"],
        "status": result["status"],
        "error": result["error"],
    }
    if extra:
        rec.update(extra)
    return rec


class Writer:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.lock = asyncio.Lock()
        self.n = 0

    async def write(self, rec: dict):
        async with self.lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.n += 1


async def run_single(session, key, item, writer, sem, stats):
    m, p = item["model"], item["prompt"]
    max_tok = MAX_TOKENS_REASONING if m.get("reasoning") else MAX_TOKENS
    messages = [{"role": "user", "content": p["content"]}]
    async with sem:
        result = await api.call(session, m["id"], messages, key,
                                temperature=TEMPERATURE, max_tokens=max_tok,
                                provider=m.get("provider"), route=m.get("route"))
    rec = build_record(m, p["id"], p["category"], p["sample_idx"], messages,
                       result, "single_turn", max_tok, item["key"])
    await writer.write(rec)
    stats["err" if result["error"] else "ok"] += 1
    u = result.get("usage") or {}
    stats["tok_in"] += u.get("prompt_tokens") or 0
    stats["tok_out"] += u.get("completion_tokens") or 0


async def run_multi(session, key, item, writer, sem, stats):
    m, mp = item["model"], item["prompt"]
    max_tok = MAX_TOKENS_REASONING if m.get("reasoning") else MAX_TOKENS
    conversation = []
    for turn_idx, user_content in enumerate(mp["turns"]):
        conversation.append({"role": "user", "content": user_content})
        async with sem:
            result = await api.call(session, m["id"], list(conversation), key,
                                    temperature=TEMPERATURE, max_tokens=max_tok,
                                    provider=m.get("provider"), route=m.get("route"))
        rec = build_record(
            m, f"{mp['id']}_turn{turn_idx}", "multi_turn", 0,
            list(conversation), result, "multi_turn", max_tok, item["key"],
            extra={"multi_turn_id": mp["id"], "turn_index": turn_idx,
                   "total_turns": len(mp["turns"])},
        )
        await writer.write(rec)
        stats["err" if result["error"] else "ok"] += 1
        u = result.get("usage") or {}
        stats["tok_in"] += u.get("prompt_tokens") or 0
        stats["tok_out"] += u.get("completion_tokens") or 0
        if result["content_clean"]:
            conversation.append({"role": "assistant", "content": result["content_clean"]})
        else:
            break  # cannot continue the dialogue


async def main_async(args):
    models = load_models(args)
    items = work_items(models)
    done = done_keys(OUT)
    todo = [i for i in items if i["key"] not in done]

    n_calls = sum(1 if i["kind"] == "single" else len(i["prompt"]["turns"]) for i in todo)
    est = estimate_cost(models)
    print(f"models: {len(models)}   items: {len(items)}   done: {len(items)-len(todo)}   todo: {len(todo)} (~{n_calls} calls)")
    print(f"full-run cost estimate for this selection: ~${est:.2f}")

    if args.dry_run:
        per_fam = {}
        for m in models:
            per_fam[m["family"]] = per_fam.get(m["family"], 0) + 1
        print("families:", dict(sorted(per_fam.items(), key=lambda x: -x[1])))
        return

    if n_calls > 5000 and not args.yes:
        print(f"!! {n_calls} calls queued. Rerun with --yes to confirm.")
        sys.exit(1)

    key = api.load_key()
    sem = asyncio.Semaphore(args.concurrency)
    writer = Writer(OUT)
    stats = {"ok": 0, "err": 0, "tok_in": 0, "tok_out": 0}
    start = time.monotonic()

    # run-metadata sidecar
    meta_path = OUT.parent / "run_meta.jsonl"
    with open(meta_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "args": vars(args), "n_models": len(models), "n_todo": len(todo),
            "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
            "max_tokens_reasoning": MAX_TOKENS_REASONING,
            "proxy": api.PROXY_BASE, "concurrency": args.concurrency,
        }) + "\n")

    async def progress():
        while True:
            await asyncio.sleep(30)
            el = time.monotonic() - start
            done_n = stats["ok"] + stats["err"]
            rate = done_n / el * 60 if el else 0
            print(f"  [{el/60:5.1f}m] {done_n}/{n_calls} calls  "
                  f"ok={stats['ok']} err={stats['err']}  "
                  f"{rate:.0f}/min  tok_out={stats['tok_out']:,}")

    conn = aiohttp.TCPConnector(limit=args.concurrency + 4)
    async with aiohttp.ClientSession(connector=conn) as session:
        prog = asyncio.create_task(progress())
        tasks = []
        for it in todo:
            fn = run_single if it["kind"] == "single" else run_multi
            tasks.append(fn(session, key, it, writer, sem, stats))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        prog.cancel()

    for r in results:
        if isinstance(r, Exception):
            print(f"  task exception: {type(r).__name__}: {r}")
    el = time.monotonic() - start
    print(f"\ndone in {el/60:.1f}m — ok={stats['ok']} err={stats['err']} "
          f"tokens in={stats['tok_in']:,} out={stats['tok_out']:,}")
    print(f"records -> {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", help="comma-separated OR model ids")
    ap.add_argument("--families", help="comma-separated families")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--concurrency", type=int, default=MAX_CONCURRENT)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
