"""
Provider sweep: find which providers inject hidden system prompts.

Phase 1: Run preflight on all models with default routing.
Phase 2: For suspicious models, try alternative providers.

Usage:
    python provider_sweep.py                   # full sweep
    python provider_sweep.py --families=kimi   # specific families
    python provider_sweep.py --phase2-only     # re-run phase 2 from existing preflight
"""

import asyncio
import aiohttp
import json
import sys
from pathlib import Path
from collections import defaultdict

from models import MODELS
from runner import (
    load_api_key, api_call, fetch_generation_stats,
    build_record, append_record, _make_write_lock,
    RESULTS_DIR, MAX_CONCURRENT,
)

PROVIDER_RESULTS_FILE = RESULTS_DIR / "provider_sweep.jsonl"
PREFLIGHT_FILE = RESULTS_DIR / "preflight.jsonl"

# Threshold: "hi" should be 1-10 tokens. Above this = suspicious.
SUSPICIOUS_THRESHOLD = 15


async def preflight_one(session, model, api_key, semaphore, write_lock,
                        output_file, provider=None):
    """Send 'hi' to one model (optionally with specific provider), check token count."""
    async with semaphore:
        messages = [{"role": "user", "content": "hi"}]
        body, headers, latency, error = await api_call(
            session, model["id"], messages, api_key,
            temperature=0, max_tokens=50,
            provider=provider,
        )

    # Gen stats outside semaphore
    gen_id = body.get("id") if body else None
    gen_stats = await fetch_generation_stats(session, gen_id, api_key)

    record = build_record(
        model=model,
        prompt_id="preflight_hi",
        prompt_category="preflight",
        messages_sent=messages,
        response_body=body,
        response_headers=headers,
        latency_ms=latency,
        error=error,
        temperature=0,
        max_tokens=50,
        run_type="provider_sweep",
        generation_stats=gen_stats,
        provider_requested=provider,
    )

    # Token analysis
    prompt_tokens = None
    if body and "usage" in body:
        prompt_tokens = body["usage"].get("prompt_tokens")
    record["preflight_prompt_tokens"] = prompt_tokens
    record["preflight_suspicious"] = (
        prompt_tokens is not None and prompt_tokens > SUSPICIOUS_THRESHOLD
    )
    record["provider_actual"] = body.get("provider") if body else None

    await append_record(record, output_file, write_lock)
    return record


async def phase1_all_models(models, api_key):
    """Phase 1: Preflight all models with default routing."""
    print("=" * 70)
    print(f"PHASE 1: Preflight all {len(models)} models (default provider routing)")
    print("=" * 70)

    output_file = PREFLIGHT_FILE
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    write_lock = _make_write_lock()

    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT + 5)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            preflight_one(session, m, api_key, semaphore, write_lock, output_file)
            for m in models
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # Summarize
    good, suspicious, errored = [], [], []
    for r in results:
        if isinstance(r, Exception):
            errored.append(r)
        elif r.get("error"):
            errored.append(r)
        elif r.get("preflight_suspicious"):
            suspicious.append(r)
        else:
            good.append(r)

    print(f"\nPhase 1 results: {len(good)} clean, {len(suspicious)} suspicious, {len(errored)} errors")

    # Print table
    all_records = [r for r in results if isinstance(r, dict)]
    all_records.sort(key=lambda r: r.get("preflight_prompt_tokens") or 0, reverse=True)

    print(f"\n{'Model':<45} {'Provider':<20} {'Tokens':>6}  Status")
    print("-" * 85)
    for r in all_records:
        tok = r.get("preflight_prompt_tokens")
        tok_str = str(tok) if tok is not None else "?"
        prov = r.get("provider_actual", "?")
        status = "!! SUSPICIOUS" if r.get("preflight_suspicious") else ("ERROR" if r.get("error") else "ok")
        print(f"  {r['model_name']:<43} {str(prov):<20} {tok_str:>6}  {status}")

    for e in errored:
        if isinstance(e, Exception):
            print(f"  EXCEPTION: {type(e).__name__}: {e}")

    return all_records, suspicious


async def phase2_provider_cycle(suspicious_records, all_models_dict, api_key):
    """Phase 2: For suspicious models, try alternative providers."""
    if not suspicious_records:
        print("\nNo suspicious models — skipping phase 2.")
        return

    print(f"\n{'=' * 70}")
    print(f"PHASE 2: Testing alternative providers for {len(suspicious_records)} suspicious models")
    print("=" * 70)

    output_file = PROVIDER_RESULTS_FILE
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    write_lock = _make_write_lock()

    # Group suspicious models by their default provider
    by_provider = defaultdict(list)
    for r in suspicious_records:
        by_provider[r.get("provider_actual", "unknown")].append(r)

    print("\nSuspicious models by provider:")
    for prov, recs in by_provider.items():
        print(f"  {prov}: {[r['model_name'] for r in recs]}")

    # For each suspicious model, try excluding the default provider
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT + 5)
    async with aiohttp.ClientSession(connector=connector) as session:
        for r in suspicious_records:
            model_id = r["model_id"]
            model = all_models_dict[model_id]
            default_prov = r.get("provider_actual", "unknown")
            default_tokens = r.get("preflight_prompt_tokens", "?")

            print(f"\n  {r['model_name']} (default: {default_prov}, {default_tokens} tokens)")

            # Strategy: try excluding the default provider
            alt_result = await preflight_one(
                session, model, api_key, semaphore, write_lock, output_file,
                provider={"ignore": [default_prov.lower()], "allow_fallbacks": True}
            )

            alt_prov = alt_result.get("provider_actual", "?")
            alt_tokens = alt_result.get("preflight_prompt_tokens", "?")
            alt_suspicious = alt_result.get("preflight_suspicious", False)
            alt_error = alt_result.get("error")

            if alt_error:
                print(f"    -> exclude {default_prov}: ERROR ({alt_error[:80]})")
            elif alt_suspicious:
                print(f"    -> {alt_prov}: {alt_tokens} tokens — STILL SUSPICIOUS")
                # Try one more: exclude both
                alt2 = await preflight_one(
                    session, model, api_key, semaphore, write_lock, output_file,
                    provider={"ignore": [default_prov.lower(), alt_prov.lower()],
                              "allow_fallbacks": True}
                )
                alt2_prov = alt2.get("provider_actual", "?")
                alt2_tokens = alt2.get("preflight_prompt_tokens", "?")
                alt2_error = alt2.get("error")
                if alt2_error:
                    print(f"    -> exclude {default_prov}+{alt_prov}: ERROR ({alt2_error[:80]})")
                elif alt2.get("preflight_suspicious"):
                    print(f"    -> {alt2_prov}: {alt2_tokens} tokens — STILL SUSPICIOUS")
                else:
                    print(f"    -> {alt2_prov}: {alt2_tokens} tokens — CLEAN!")
            else:
                print(f"    -> {alt_prov}: {alt_tokens} tokens — CLEAN!")


def load_existing_preflight():
    """Load suspicious records from existing preflight.jsonl."""
    if not PREFLIGHT_FILE.exists():
        print(f"No {PREFLIGHT_FILE} found. Run phase 1 first.")
        return []
    records = []
    with open(PREFLIGHT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    suspicious = [r for r in records if r.get("preflight_suspicious")]
    return suspicious


async def main():
    api_key = load_api_key()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Parse args
    phase2_only = "--phase2-only" in sys.argv
    family_filter = None
    for arg in sys.argv[1:]:
        if arg.startswith("--families="):
            family_filter = arg.split("=", 1)[1].split(",")

    models = MODELS
    if family_filter:
        models = [m for m in MODELS if m["family"] in family_filter]
        print(f"Filtered to {len(models)} models in families: {family_filter}")

    all_models_dict = {m["id"]: m for m in MODELS}

    if phase2_only:
        suspicious = load_existing_preflight()
        if family_filter:
            suspicious = [r for r in suspicious if r["model_family"] in family_filter]
        print(f"Loaded {len(suspicious)} suspicious models from existing preflight")
        await phase2_provider_cycle(suspicious, all_models_dict, api_key)
    else:
        # Clear previous preflight data
        if PREFLIGHT_FILE.exists():
            PREFLIGHT_FILE.unlink()
        all_records, suspicious = await phase1_all_models(models, api_key)
        await phase2_provider_cycle(suspicious, all_models_dict, api_key)

    print(f"\n{'=' * 70}")
    print("Provider sweep complete.")
    print(f"  Preflight results: {PREFLIGHT_FILE}")
    if PROVIDER_RESULTS_FILE.exists():
        print(f"  Provider alternatives: {PROVIDER_RESULTS_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
