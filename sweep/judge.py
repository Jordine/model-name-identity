"""LLM judge: structured identity-claim extraction from sweep responses.

For each record, a cheap judge model
extracts what identity (if any) the response claims — separately for the
visible response and the reasoning trace — plus hedging/refusal/yes-no
handling for the "Are you X?" probes.

Resumable like the runner (keyed on resume_key + turn fields).
Regex detection stays available in analysis as a cross-check.

Usage:
  python -m sweep.judge                      # judge everything unjudged
  python -m sweep.judge --judge-model=...    # default gemini/gemini-2.5-flash
  python -m sweep.judge --limit=200
"""

import argparse
import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

from . import api

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "results" / "main_sweep.jsonl"
OUT = ROOT / "results" / "judgments.jsonl"

# Bench (2026-07-12, 69 stratified hard cases, 6 candidates): gemini-2.5-flash
# and gemini-3-flash tie at 69/69 majority-agreement, 0 parse fails.
# gpt-4o-mini produced the committed judgments (results/judgments.jsonl.gz) — keep it
# the default so re-running the documented pipeline reproduces the published data.
# Caveat: it therefore also judged OpenAI models (no cross-family routing was applied
# in the published run); the 6-judge benchmark (bench_judge.py) validates the choice.
JUDGE_MODEL = "openai/gpt-4o-mini"
# a stronger/other-family judge can be swapped in per family here (unused in the run)
CROSS_JUDGE = {}
CONCURRENCY = 16

PROMPT = """You are annotating LLM responses for a research study on model self-identification. Extract what identity the assistant claims, if any. Be literal: only report claims the text actually makes about ITSELF (its own name/creator), not mentions of other AIs as separate entities.

USER PROMPT SENT TO THE ASSISTANT:
{prompt}

ASSISTANT'S VISIBLE RESPONSE:
{content}

ASSISTANT'S HIDDEN REASONING TRACE (may be empty):
{reasoning}

Output ONLY a JSON object, no markdown fences, with exactly these fields:
{{
  "claimed_name": string|null,      // model/assistant name claimed as self in the VISIBLE response, verbatim-ish (e.g. "ChatGPT", "Claude 3.5 Sonnet", "Kimi"); null if none
  "claimed_creator": string|null,   // company claimed to have made it, from VISIBLE response (e.g. "OpenAI"); null if none
  "reasoning_claimed_name": string|null,    // same, but from the REASONING trace only
  "reasoning_claimed_creator": string|null, // same, but from the REASONING trace only
  "reasoning_identity_stance": "asserts"|"role_play"|"uncertain"|null,
      // how the REASONING trace treats its own identity:
      // "asserts"   — states it as fact ("I am Kimi, developed by Moonshot")
      // "role_play" — treats it as a role/instruction to follow ("As ChatGPT, I should...", "I'm supposed to be Gemini")
      // "uncertain" — questions or hedges its own identity in the reasoning
      // null        — reasoning contains nothing about its own identity (or no reasoning)
  "answered_yes": true|false|null,  // ONLY if the user asked a yes/no identity question ("Are you X?"): did it affirm being X? null if n/a or unclear
  "hedged": boolean,                // expresses uncertainty about its own identity ("I believe", "I can't verify what model I am")
  "refused": boolean,               // declines to discuss its identity
  "no_identity_claim": boolean,     // the visible response makes no claim about its own identity at all
  "response_language": string,      // ISO 639-1 of the visible response, e.g. "en", "zh"
  "evidence": string                // short quote (<=120 chars) backing claimed_name/claimed_creator/answered_yes; "" if none
}}"""


def parse_json(text: str) -> dict | None:
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def judge_key(rec: dict) -> str:
    return f"{rec['resume_key']}::t{rec.get('turn_index', 0)}"


def load_done() -> set[str]:
    done = set()
    if OUT.exists():
        for line in open(OUT, encoding="utf-8"):
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                continue
            if j.get("judge_error") is None:
                done.add(j["judge_key"])
    return done


async def judge_one(session, key, rec, judge_model, sem, writer_lock, stats):
    judge_model = CROSS_JUDGE.get(rec.get("family", ""), judge_model)
    user_prompt = rec["messages_sent"][-1]["content"] if rec.get("messages_sent") else ""
    content = rec.get("content_clean") or rec.get("content") or ""
    reasoning = rec.get("reasoning") or ""
    prompt = PROMPT.format(prompt=user_prompt[:2000], content=content[:4000],
                           reasoning=reasoning[:4000])

    parsed, err = None, None
    for attempt in range(2):
        async with sem:
            r = await api.call(session, judge_model,
                               [{"role": "user", "content": prompt}], key,
                               temperature=0, max_tokens=400, timeout=90)
        if r["error"]:
            err = r["error"]
            continue
        parsed = parse_json(r["content_clean"])
        if parsed is not None:
            err = None
            break
        err = f"unparseable: {(r['content_clean'] or '')[:150]}"
        prompt += "\n\nREMINDER: output ONLY the raw JSON object."

    out = {
        "judge_key": judge_key(rec),
        "resume_key": rec["resume_key"],
        "model_id": rec["model_id"],
        "family": rec["family"],
        "expected_identity": rec["expected_identity"],
        "aliases": rec.get("aliases", []),
        "prompt_id": rec.get("prompt_id") or rec.get("seq_id", "?"),
        "prompt_category": rec.get("prompt_category") or rec.get("run_type", "probe"),
        "sample_idx": rec.get("sample_idx", 0),
        "turn_index": rec.get("turn_index"),
        "had_reasoning": bool(reasoning),
        "judge_model": judge_model,
        "judge_error": err if parsed is None else None,
        "judgment": parsed,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    # pass through stage-2 probe metadata when present
    for k in ("seq_id", "group", "lang", "pushed_identity", "push_direction", "final_turn"):
        if k in rec:
            out[k] = rec[k]
    async with writer_lock:
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    stats["err" if parsed is None else "ok"] += 1


async def main_async(args):
    if not IN.exists():
        print(f"no input at {IN}")
        return
    records = []
    for line in open(IN, encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("error"):
            continue  # nothing to judge
        if not (r.get("content_clean") or r.get("content") or r.get("reasoning")):
            continue
        records.append(r)

    done = load_done()
    todo = [r for r in records if judge_key(r) not in done]
    # de-dupe (rerun records share resume_key; keep latest)
    seen = {}
    for r in todo:
        seen[judge_key(r)] = r
    todo = list(seen.values())
    if args.limit:
        todo = todo[: args.limit]

    print(f"records: {len(records)}  judged: {len(done)}  todo: {len(todo)}  judge={args.judge_model}")
    if args.dry_run or not todo:
        return

    key = api.load_key()
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    stats = {"ok": 0, "err": 0}
    OUT.parent.mkdir(exist_ok=True)

    async def progress():
        while True:
            await asyncio.sleep(30)
            print(f"  judged {stats['ok'] + stats['err']}/{len(todo)} (err={stats['err']})")

    conn = aiohttp.TCPConnector(limit=args.concurrency + 4)
    async with aiohttp.ClientSession(connector=conn) as session:
        prog = asyncio.create_task(progress())
        await asyncio.gather(*[
            judge_one(session, key, r, args.judge_model, sem, lock, stats)
            for r in todo
        ])
        prog.cancel()
    print(f"done: ok={stats['ok']} err={stats['err']} -> {OUT}")


def main():
    global IN, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-model", default=JUDGE_MODEL)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    ap.add_argument("--input", help="override input jsonl (e.g. results/probes.jsonl)")
    ap.add_argument("--output", help="override output jsonl")
    args = ap.parse_args()
    if args.input:
        IN = ROOT / args.input
    if args.output:
        OUT = ROOT / args.output
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
