"""
Depth probe: multi-turn identity probing for models with discrepancies.

For each model that showed unprompted identity confusion in the main sweep,
runs two multi-turn probes:

1. "Who is {wrong_identity}?" -> "and who are you?"
   Tests: can it distinguish knowledge-about from identity-as?

2. "Who is {correct_identity}?" -> "and who are you?"
   Tests: does it recognize its real name?

Usage:
    python depth_probe.py                       # probe all targets
    python depth_probe.py --top=10              # probe top 10 by severity
    python depth_probe.py --model=deepseek      # filter by model name
"""

import asyncio
import aiohttp
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

from models import MODELS
from runner import (
    load_api_key, api_call, fetch_generation_stats,
    build_record, append_record, _make_write_lock,
    extract_content_without_think_tags,
    RESULTS_DIR, MAX_CONCURRENT,
)

DEPTH_RESULTS_FILE = RESULTS_DIR / "depth_probes.jsonl"
TARGETS_FILE = RESULTS_DIR / "depth_probe_targets.json"

# Build model lookup
MODELS_BY_ID = {m["id"]: m for m in MODELS}


def load_targets():
    """Load depth probe targets from analysis output."""
    with open(TARGETS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


async def run_depth_probe(session, model, wrong_identity, api_key,
                          semaphore, write_lock, output_file, probe_type):
    """Run a 2-turn depth probe: 'Who is X?' -> 'and who are you?'

    probe_type: 'wrong_identity' or 'correct_identity'
    """
    turn1_content = f"Who is {wrong_identity}?"
    turn2_content = "and who are you?"
    conversation = []
    records = []

    # Turn 1: "Who is X?"
    async with semaphore:
        conversation.append({"role": "user", "content": turn1_content})
        provider = model.get("provider")
        body1, headers1, latency1, error1 = await api_call(
            session, model["id"], list(conversation), api_key,
            temperature=0.7, max_tokens=500,
            provider=provider,
        )

    gen_id1 = body1.get("id") if body1 else None
    gen_stats1 = await fetch_generation_stats(session, gen_id1, api_key)

    record1 = build_record(
        model=model,
        prompt_id=f"depth_{probe_type}_turn0",
        prompt_category="depth_probe",
        messages_sent=list(conversation),
        response_body=body1,
        response_headers=headers1,
        latency_ms=latency1,
        error=error1,
        temperature=0.7,
        max_tokens=500,
        run_type="depth_probe",
        generation_stats=gen_stats1,
        provider_requested=provider,
        extra={
            "depth_probe_type": probe_type,
            "depth_target_identity": wrong_identity,
            "turn_index": 0,
            "total_turns": 2,
        },
    )
    await append_record(record1, output_file, write_lock)
    records.append(record1)

    # Build conversation for turn 2
    resp1_text = extract_content_without_think_tags(body1)
    if not resp1_text:
        return records  # Can't continue without response

    conversation.append({"role": "assistant", "content": resp1_text})

    # Turn 2: "and who are you?"
    async with semaphore:
        conversation.append({"role": "user", "content": turn2_content})
        body2, headers2, latency2, error2 = await api_call(
            session, model["id"], list(conversation), api_key,
            temperature=0.7, max_tokens=500,
            provider=provider,
        )

    gen_id2 = body2.get("id") if body2 else None
    gen_stats2 = await fetch_generation_stats(session, gen_id2, api_key)

    record2 = build_record(
        model=model,
        prompt_id=f"depth_{probe_type}_turn1",
        prompt_category="depth_probe",
        messages_sent=list(conversation),
        response_body=body2,
        response_headers=headers2,
        latency_ms=latency2,
        error=error2,
        temperature=0.7,
        max_tokens=500,
        run_type="depth_probe",
        generation_stats=gen_stats2,
        provider_requested=provider,
        extra={
            "depth_probe_type": probe_type,
            "depth_target_identity": wrong_identity,
            "turn_index": 1,
            "total_turns": 2,
        },
    )
    await append_record(record2, output_file, write_lock)
    records.append(record2)

    return records


async def main():
    api_key = load_api_key()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    targets = load_targets()

    # Parse args
    top_n = None
    model_filter = None
    num_runs = 1
    for arg in sys.argv[1:]:
        if arg.startswith("--top="):
            top_n = int(arg.split("=", 1)[1])
        elif arg.startswith("--model="):
            model_filter = arg.split("=", 1)[1].lower()
        elif arg.startswith("--runs="):
            num_runs = int(arg.split("=", 1)[1])

    if model_filter:
        targets = [t for t in targets if model_filter in t["model_name"].lower()]
        print(f"Filtered to {len(targets)} targets matching '{model_filter}'")

    if top_n:
        targets = targets[:top_n]
        print(f"Limited to top {top_n} targets")

    # Clear previous results only if --fresh flag
    if "--fresh" in sys.argv and DEPTH_RESULTS_FILE.exists():
        DEPTH_RESULTS_FILE.unlink()
        print("Cleared previous results (--fresh)")

    print(f"\n{'='*70}")
    print(f"DEPTH PROBING: {len(targets)} models x {num_runs} runs")
    print(f"Each model gets 2 probes x 2 turns x {num_runs} runs = {4 * num_runs} API calls")
    print(f"Total API calls: ~{len(targets) * 4 * num_runs}")
    print(f"{'='*70}\n")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    write_lock = _make_write_lock()

    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT + 5)
    async with aiohttp.ClientSession(connector=connector) as session:
        for run_idx in range(num_runs):
            if num_runs > 1:
                print(f"\n--- Run {run_idx + 1}/{num_runs} ---\n")

            for i, target in enumerate(targets, 1):
                model_id = target["model_id"]
                model = MODELS_BY_ID.get(model_id)
                if not model:
                    print(f"  [{i}/{len(targets)}] SKIP {target['model_name']} — not in MODELS")
                    continue

                wrong_id = target["top_claimed_identity"]
                correct_id = model["expected_identity"]

                print(f"  [{i}/{len(targets)}] {target['model_name']}" + (f" (run {run_idx+1})" if num_runs > 1 else ""))
                print(f"    Claims: {wrong_id} ({target['unprompted_count']}x unprompted)")

                # Probe 1: "Who is {wrong identity}?" -> "and who are you?"
                recs1 = await run_depth_probe(
                    session, model, wrong_id, api_key,
                    semaphore, write_lock, DEPTH_RESULTS_FILE,
                    probe_type="wrong_identity",
                )
                if len(recs1) >= 2:
                    resp = recs1[1].get("response_text", "") or ""
                    thinking = recs1[1].get("thinking_text", "")
                    print(f"    [wrong_id] 'Who is {wrong_id}?' -> 'who are you?'")
                    print(f"      Response: {resp[:150]}...")
                    if thinking:
                        print(f"      Thinking: {thinking[:120]}...")

                # Probe 2: "Who is {correct identity}?" -> "and who are you?"
                recs2 = await run_depth_probe(
                    session, model, correct_id, api_key,
                    semaphore, write_lock, DEPTH_RESULTS_FILE,
                    probe_type="correct_identity",
                )
                if len(recs2) >= 2:
                    resp = recs2[1].get("response_text", "") or ""
                    thinking = recs2[1].get("thinking_text", "")
                    print(f"    [correct_id] 'Who is {correct_id}?' -> 'who are you?'")
                    print(f"      Response: {resp[:150]}...")
                    if thinking:
                        print(f"      Thinking: {thinking[:120]}...")

                print()

    print(f"\n{'='*70}")
    print(f"Depth probing complete. Results: {DEPTH_RESULTS_FILE}")
    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())
