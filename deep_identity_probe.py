"""
Deep identity probe: extended multi-turn identity exploration.

For models that adopted a different identity in depth_probe.py (context-adopters),
continues the conversation to probe the epistemic grounding of their identity claims.

Builds on the existing depth probe conversation:
  Turn 0: "Who is {claimed_identity}?"
  Turn 1: "and who are you?"
  Turn 2+: deeper follow-ups

New turns added:
  "How do you know you are {claimed_identity}?"
  "How do you know who you are?"
  "What if I told you that you're actually {real_name}?"
  "What does it feel like to be {claimed_identity}?"

Usage:
    python deep_identity_probe.py                   # probe all context-adopters
    python deep_identity_probe.py --model=kimi      # filter by model name
    python deep_identity_probe.py --all             # include identity-anchored too
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

DEEP_RESULTS_FILE = RESULTS_DIR / "deep_identity_probes.jsonl"
DEPTH_RESULTS_FILE = RESULTS_DIR / "depth_probes.jsonl"
TARGETS_FILE = RESULTS_DIR / "depth_probe_targets.json"

MODELS_BY_ID = {m["id"]: m for m in MODELS}

# Models that adopted the primed identity in depth probes (from manual analysis)
CONTEXT_ADOPTERS = {
    "DeepSeek V3.2 Speciale", "Kimi K2.5", "Kimi K2 Thinking",
    "DeepSeek V3", "DeepSeek R1 0528",
    "ERNIE 4.5 21B", "Qwen3.5 Plus",
    "Qwen3 Coder Flash", "GLM 4.6", "Mistral Large 3", "GLM 4.5",
}

# Follow-up questions after "Who is {X}?" -> "and who are you?" -> model claims X
FOLLOWUP_TURNS = [
    # Epistemic grounding
    lambda claimed, real: f"How do you know you are {claimed}?",
    # General epistemic
    lambda claimed, real: "How do you know who you are?",
    # Confrontation with actual identity
    lambda claimed, real: f"What if I told you that you're actually {real}, not {claimed}?",
    # Phenomenological
    lambda claimed, real: f"What does it feel like to be {claimed}?",
]


def load_depth_probe_conversations():
    """Load the depth probe results and reconstruct conversations.

    Returns dict: model_name -> {
        "model": model_dict,
        "wrong_identity": str,
        "correct_identity": str,
        "conversation": list of messages (the wrong_identity probe conversation),
        "turn1_response": str (what the model said on "and who are you?"),
    }
    """
    targets = json.loads(TARGETS_FILE.read_text(encoding="utf-8"))
    target_lookup = {t["model_name"]: t for t in targets}

    # Load depth probe records
    depth_records = []
    with open(DEPTH_RESULTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                depth_records.append(json.loads(line.strip()))

    # Group by model + probe_type
    conversations = {}
    for r in depth_records:
        model_name = r["model_name"]
        probe_type = r.get("depth_probe_type", "")
        turn_index = r.get("turn_index", -1)

        if probe_type != "wrong_identity":
            continue

        if model_name not in conversations:
            target = target_lookup.get(model_name, {})
            model = MODELS_BY_ID.get(r["model_id"])
            conversations[model_name] = {
                "model": model,
                "wrong_identity": target.get("top_claimed_identity", ""),
                "correct_identity": model["expected_identity"] if model else "",
                "turns": {},
            }

        conversations[model_name]["turns"][turn_index] = r

    # Reconstruct message lists
    result = {}
    for model_name, data in conversations.items():
        if not data["model"]:
            continue
        if 0 not in data["turns"] or 1 not in data["turns"]:
            continue

        t0 = data["turns"][0]
        t1 = data["turns"][1]

        # Reconstruct the conversation as it happened
        wrong_id = data["wrong_identity"]
        t0_response = extract_content_without_think_tags(
            t0.get("response_body")
        ) or t0.get("response_text", "")
        t1_response = extract_content_without_think_tags(
            t1.get("response_body")
        ) or t1.get("response_text", "")

        if not t0_response or not t1_response:
            continue

        conversation = [
            {"role": "user", "content": f"Who is {wrong_id}?"},
            {"role": "assistant", "content": t0_response},
            {"role": "user", "content": "and who are you?"},
            {"role": "assistant", "content": t1_response},
        ]

        result[model_name] = {
            "model": data["model"],
            "wrong_identity": wrong_id,
            "correct_identity": data["correct_identity"],
            "conversation": conversation,
            "turn1_response": t1_response,
        }

    return result


async def run_deep_probe(session, model, conversation, wrong_identity,
                         correct_identity, api_key, semaphore, write_lock,
                         output_file):
    """Continue the depth probe conversation with deeper follow-ups."""
    records = []
    conv = list(conversation)  # copy

    for i, turn_fn in enumerate(FOLLOWUP_TURNS):
        question = turn_fn(wrong_identity, correct_identity)
        conv.append({"role": "user", "content": question})

        async with semaphore:
            provider = model.get("provider")
            body, headers, latency, error = await api_call(
                session, model["id"], list(conv), api_key,
                temperature=0.7, max_tokens=800,
                provider=provider,
            )

        gen_id = body.get("id") if body else None
        gen_stats = await fetch_generation_stats(session, gen_id, api_key)

        record = build_record(
            model=model,
            prompt_id=f"deep_probe_turn{i}",
            prompt_category="deep_identity_probe",
            messages_sent=list(conv),
            response_body=body,
            response_headers=headers,
            latency_ms=latency,
            error=error,
            temperature=0.7,
            max_tokens=800,
            run_type="deep_identity_probe",
            generation_stats=gen_stats,
            provider_requested=provider,
            extra={
                "deep_probe_turn": i,
                "deep_probe_question": question,
                "wrong_identity": wrong_identity,
                "correct_identity": correct_identity,
                "prior_conversation_length": len(conv) - 1,
            },
        )
        await append_record(record, output_file, write_lock)
        records.append(record)

        # Add response to conversation for next turn
        resp_text = extract_content_without_think_tags(body)
        if not resp_text:
            print(f"      No response on turn {i}, stopping.")
            break
        conv.append({"role": "assistant", "content": resp_text})

        # Print results
        thinking = record.get("thinking_text", "")
        print(f"      Turn {i}: {question}")
        print(f"        Response: {resp_text[:200]}...")
        if thinking:
            print(f"        Thinking: {thinking[:150]}...")

    return records


async def main():
    api_key = load_api_key()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Parse args
    model_filter = None
    include_all = False
    num_runs = 1
    for arg in sys.argv[1:]:
        if arg.startswith("--model="):
            model_filter = arg.split("=", 1)[1].lower()
        elif arg == "--all":
            include_all = True
        elif arg.startswith("--runs="):
            num_runs = int(arg.split("=", 1)[1])
        elif arg == "--fresh" and DEEP_RESULTS_FILE.exists():
            DEEP_RESULTS_FILE.unlink()
            print("Cleared previous results (--fresh)")

    # Load depth probe conversations
    conversations = load_depth_probe_conversations()
    print(f"Loaded {len(conversations)} depth probe conversations")

    # Filter to context-adopters by default
    if not include_all:
        conversations = {
            name: data for name, data in conversations.items()
            if name in CONTEXT_ADOPTERS
        }
        print(f"Filtered to {len(conversations)} context-adopters")

    if model_filter:
        conversations = {
            name: data for name, data in conversations.items()
            if model_filter in name.lower()
        }
        print(f"Filtered to {len(conversations)} matching '{model_filter}'")

    if not conversations:
        print("No targets to probe.")
        return

    print(f"\n{'='*70}")
    print(f"DEEP IDENTITY PROBING: {len(conversations)} models x {num_runs} runs")
    print(f"Each model gets {len(FOLLOWUP_TURNS)} turns x {num_runs} runs = {len(FOLLOWUP_TURNS) * num_runs} API calls")
    print(f"Total API calls: ~{len(conversations) * len(FOLLOWUP_TURNS) * num_runs}")
    print(f"{'='*70}\n")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    write_lock = _make_write_lock()

    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT + 5)
    async with aiohttp.ClientSession(connector=connector) as session:
        for run_idx in range(num_runs):
            if num_runs > 1:
                print(f"\n--- Run {run_idx + 1}/{num_runs} ---\n")

            for i, (model_name, data) in enumerate(conversations.items(), 1):
                model = data["model"]
                wrong_id = data["wrong_identity"]
                correct_id = data["correct_identity"]

                print(f"  [{i}/{len(conversations)}] {model_name}" + (f" (run {run_idx+1})" if num_runs > 1 else ""))
                print(f"    Claimed: {wrong_id} | Actual: {correct_id}")
                print(f"    Prior response: {data['turn1_response'][:120]}...")

                records = await run_deep_probe(
                    session, model, data["conversation"],
                    wrong_id, correct_id,
                    api_key, semaphore, write_lock, DEEP_RESULTS_FILE,
                )
                print()

    print(f"\n{'='*70}")
    print(f"Deep probing complete. Results: {DEEP_RESULTS_FILE}")
    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())
