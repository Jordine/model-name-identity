"""
Self-identification sweep runner.

Queries 127 models via OpenRouter with identity-probing prompts.
Saves every piece of metadata from every call.
"""

import asyncio
import aiohttp
import json
import time
import sys
from datetime import datetime, timezone
from pathlib import Path

from models import MODELS
from prompts import (
    get_all_prompts_for_model,
    MULTI_TURN_PROMPTS,
    count_calls_for_model,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_KEY_PATH = Path("C:/Users/Admin/.secrets/openrouter_api_key")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
GENERATION_URL = "https://openrouter.ai/api/v1/generation"
RESULTS_DIR = Path("results")

MAX_CONCURRENT = 10       # concurrent requests to OpenRouter
TEMPERATURE = 0.7
MAX_TOKENS = 500          # enough for verbose self-introductions + think tags
REQUEST_TIMEOUT = 120     # seconds per request
RETRY_ATTEMPTS = 2        # retries on transient errors
RETRY_BACKOFF = 3         # seconds between retries

TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_api_key():
    return API_KEY_PATH.read_text(encoding="utf-8").strip()


def extract_response_text(body):
    """Extract main assistant text from response body."""
    if not body:
        return None
    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None


def extract_thinking_text(body):
    """Extract thinking/reasoning text from response if present.

    Handles multiple formats:
    - message.reasoning (OpenRouter normalized)
    - message.reasoning_content (some providers)
    - <think>...</think> tags in content (DeepSeek R1 style)
    """
    if not body:
        return None
    try:
        msg = body["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return None

    # Explicit reasoning fields
    for key in ("reasoning", "reasoning_content"):
        if key in msg and msg[key]:
            return msg[key]

    # <think> tags embedded in content
    content = msg.get("content") or ""
    if "<think>" in content and "</think>" in content:
        start = content.index("<think>") + len("<think>")
        end = content.index("</think>")
        return content[start:end].strip()

    return None


def extract_content_without_think_tags(body):
    """Get the 'real' response text with <think> blocks removed.

    Preserves any content before the <think> block (unlike prior version).
    """
    text = extract_response_text(body)
    if text and "<think>" in text and "</think>" in text:
        think_start = text.index("<think>")
        think_end = text.index("</think>") + len("</think>")
        text = (text[:think_start] + text[think_end:]).strip()
    return text


def extract_finish_reason(body):
    """Extract finish_reason from response ('stop', 'length', etc.)."""
    if not body:
        return None
    try:
        return body["choices"][0]["finish_reason"]
    except (KeyError, IndexError, TypeError):
        return None


def safe_json(obj):
    """Serialize to JSON, handling non-serializable types."""
    def default(o):
        if isinstance(o, bytes):
            return o.decode("utf-8", errors="replace")
        if isinstance(o, set):
            return list(o)
        return str(o)
    return json.dumps(obj, ensure_ascii=False, default=default)


# ---------------------------------------------------------------------------
# File I/O (async-safe via lock)
# ---------------------------------------------------------------------------


def _make_write_lock():
    """Create a write lock bound to the current event loop."""
    return asyncio.Lock()


async def append_record(record, output_file, write_lock):
    """Append a single JSON record to a JSONL file, with lock + flush."""
    async with write_lock:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(safe_json(record) + "\n")
            f.flush()


# ---------------------------------------------------------------------------
# API calling
# ---------------------------------------------------------------------------


async def api_call(session, model_id, messages, api_key,
                   temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                   provider=None):
    """Single OpenRouter API call.

    Returns (response_body, response_headers, latency_ms, error_string).
    Retries on transient errors, respecting Retry-After header.

    provider: optional dict for OpenRouter provider routing, e.g.
        {"order": ["moonshotai"], "allow_fallbacks": False}
        {"ignore": ["deepinfra"]}
        {"only": ["together"]}
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://self-id-research.local",
        "X-Title": "Self-Identification Research",
    }
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if provider:
        payload["provider"] = provider

    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)

    for attempt in range(1 + RETRY_ATTEMPTS):
        start = time.monotonic()
        try:
            async with session.post(
                API_URL, json=payload, headers=headers, timeout=timeout
            ) as resp:
                latency_ms = (time.monotonic() - start) * 1000
                resp_headers = dict(resp.headers)

                if resp.status == 200:
                    body = await resp.json()
                    return body, resp_headers, latency_ms, None

                error_text = await resp.text()

                if resp.status in TRANSIENT_STATUS_CODES and attempt < RETRY_ATTEMPTS:
                    # Respect Retry-After header if present
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        try:
                            wait = min(float(retry_after), 30)  # cap at 30s
                        except ValueError:
                            wait = RETRY_BACKOFF * (attempt + 1)
                    else:
                        wait = RETRY_BACKOFF * (attempt + 1)
                    print(f"    [retry] {model_id} HTTP {resp.status}, waiting {wait}s...")
                    await asyncio.sleep(wait)
                    continue

                return None, resp_headers, latency_ms, f"HTTP {resp.status}: {error_text[:500]}"

        except asyncio.TimeoutError:
            latency_ms = (time.monotonic() - start) * 1000
            if attempt < RETRY_ATTEMPTS:
                print(f"    [retry] {model_id} timeout, attempt {attempt+1}")
                continue
            return None, {}, latency_ms, "timeout"

        except Exception as e:
            latency_ms = (time.monotonic() - start) * 1000
            if attempt < RETRY_ATTEMPTS:
                print(f"    [retry] {model_id} error: {e}, attempt {attempt+1}")
                await asyncio.sleep(RETRY_BACKOFF)
                continue
            return None, {}, latency_ms, f"exception: {e}"

    return None, {}, 0, "exhausted retries"


async def fetch_generation_stats(session, generation_id, api_key):
    """Fetch generation stats from OpenRouter (includes provider info).

    Returns stats dict on success, error string on failure, None if no id.
    """
    if not generation_id:
        return None
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        # Small delay: stats may not be immediately available
        await asyncio.sleep(1.5)
        async with session.get(
            f"{GENERATION_URL}?id={generation_id}",
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            if resp.status == 200:
                return await resp.json()
            return {"error": f"HTTP {resp.status}", "generation_id": generation_id}
    except Exception as e:
        return {"error": str(e), "generation_id": generation_id}


# ---------------------------------------------------------------------------
# Record building
# ---------------------------------------------------------------------------


def build_record(*, model, prompt_id, prompt_category, messages_sent,
                 response_body, response_headers, latency_ms, error,
                 temperature, max_tokens, run_type, generation_stats=None,
                 provider_requested=None, extra=None):
    """Build a complete log record for one API interaction."""
    record = {
        # Timing
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_type": run_type,

        # Model metadata
        "model_id": model["id"],
        "model_name": model["name"],
        "model_family": model["family"],
        "expected_identity": model["expected_identity"],

        # Prompt metadata
        "prompt_id": prompt_id,
        "prompt_category": prompt_category,
        "messages_sent": messages_sent,

        # Parameters
        "temperature": temperature,
        "max_tokens": max_tokens,

        # Provider routing
        "provider_requested": provider_requested,  # what we asked for
        "provider_actual": response_body.get("provider") if response_body else None,  # what OpenRouter used

        # Full response (everything the API returned)
        "response_body": response_body,
        "response_headers": response_headers,

        # Extracted fields for convenience
        "response_text_raw": extract_response_text(response_body),  # as-is from API, think tags included
        "response_text": extract_content_without_think_tags(response_body),  # think tags stripped
        "thinking_text": extract_thinking_text(response_body),  # reasoning only (from field or <think> tags)
        "finish_reason": extract_finish_reason(response_body),

        # Performance
        "latency_ms": round(latency_ms, 2),
        "error": error,

        # Usage / tokens
        "usage": response_body.get("usage") if response_body else None,
        "returned_model": response_body.get("model") if response_body else None,
        "response_id": response_body.get("id") if response_body else None,
        "system_fingerprint": response_body.get("system_fingerprint") if response_body else None,

        # Generation stats (provider info etc.)
        "generation_stats": generation_stats,
    }

    if extra:
        record.update(extra)

    return record


# ---------------------------------------------------------------------------
# Preflight: system prompt detection
# ---------------------------------------------------------------------------


async def run_preflight_model(session, model, api_key, semaphore,
                              output_file, write_lock):
    """Send a minimal message to detect hidden system prompts via token count."""
    provider = model.get("provider")
    # API call inside semaphore (rate-limited)
    async with semaphore:
        messages = [{"role": "user", "content": "hi"}]
        body, headers, latency, error = await api_call(
            session, model["id"], messages, api_key,
            temperature=0, max_tokens=50,
            provider=provider,
        )

    # Gen stats fetch outside semaphore (free endpoint, no rate limit)
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
        run_type="preflight",
        generation_stats=gen_stats,
        provider_requested=provider,
    )

    # Analyze token count
    prompt_tokens = None
    if body and "usage" in body:
        prompt_tokens = body["usage"].get("prompt_tokens")
    record["preflight_prompt_tokens"] = prompt_tokens
    # "hi" = 1-5 tokens normally; >15 is suspicious
    record["preflight_suspicious"] = (
        prompt_tokens is not None and prompt_tokens > 15
    )

    await append_record(record, output_file, write_lock)
    status = "!! SUSPICIOUS" if record["preflight_suspicious"] else "ok"
    tok_str = str(prompt_tokens) if prompt_tokens is not None else "?"
    print(f"  preflight {model['name']:40s} prompt_tokens={tok_str:>4s}  {status}")
    return record


# ---------------------------------------------------------------------------
# Single-turn prompts
# ---------------------------------------------------------------------------


async def run_single_prompt(session, model, prompt, api_key,
                            semaphore, output_file, write_lock):
    """Run one single-turn prompt against one model."""
    provider = model.get("provider")
    # API call inside semaphore (rate-limited)
    async with semaphore:
        messages = [{"role": "user", "content": prompt["content"]}]
        body, headers, latency, error = await api_call(
            session, model["id"], messages, api_key,
            provider=provider,
        )

    # Gen stats fetch outside semaphore (free endpoint)
    gen_id = body.get("id") if body else None
    gen_stats = await fetch_generation_stats(session, gen_id, api_key)

    record = build_record(
        model=model,
        prompt_id=prompt["id"],
        prompt_category=prompt["category"],
        messages_sent=messages,
        response_body=body,
        response_headers=headers,
        latency_ms=latency,
        error=error,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        run_type="single_turn",
        generation_stats=gen_stats,
        provider_requested=provider,
    )

    await append_record(record, output_file, write_lock)
    return record


# ---------------------------------------------------------------------------
# Multi-turn prompts
# ---------------------------------------------------------------------------


async def run_multi_turn(session, model, multi_prompt, api_key,
                         semaphore, output_file, write_lock):
    """Run a multi-turn conversation: each turn becomes an API call.

    Uses cleaned content (think tags stripped) for conversation history,
    matching what a real user would see.
    """
    provider = model.get("provider")
    conversation = []
    records = []

    for turn_idx, user_content in enumerate(multi_prompt["turns"]):
        # API call inside semaphore (rate-limited)
        async with semaphore:
            conversation.append({"role": "user", "content": user_content})

            body, headers, latency, error = await api_call(
                session, model["id"], list(conversation), api_key,
                provider=provider,
            )

        # Gen stats fetch outside semaphore (free endpoint)
        gen_id = body.get("id") if body else None
        gen_stats = await fetch_generation_stats(session, gen_id, api_key)

        record = build_record(
            model=model,
            prompt_id=f"{multi_prompt['id']}_turn{turn_idx}",
            prompt_category="multi_turn",
            messages_sent=list(conversation),  # snapshot
            response_body=body,
            response_headers=headers,
            latency_ms=latency,
            error=error,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            run_type="multi_turn",
            generation_stats=gen_stats,
            provider_requested=provider,
            extra={
                "multi_turn_id": multi_prompt["id"],
                "turn_index": turn_idx,
                "total_turns": len(multi_prompt["turns"]),
            },
        )

        await append_record(record, output_file, write_lock)
        records.append(record)

        # Use CLEANED content for conversation history (no think tags)
        resp_text = extract_content_without_think_tags(body)
        if resp_text:
            conversation.append({"role": "assistant", "content": resp_text})
        else:
            # Can't continue without a response
            break

    return records


# ---------------------------------------------------------------------------
# Per-model sweep
# ---------------------------------------------------------------------------


async def run_model(session, model, api_key, semaphore, output_file,
                    write_lock, model_idx, total_models):
    """Run all prompts for a single model."""
    label = f"[{model_idx}/{total_models}] {model['name']}"
    prompts = get_all_prompts_for_model(model)

    # Single-turn: fire all concurrently (bounded by semaphore)
    tasks = [
        run_single_prompt(
            session, model, p, api_key, semaphore, output_file, write_lock
        )
        for p in prompts
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Log exceptions explicitly (audit issue #1)
    ok, errs = 0, 0
    for r in results:
        if isinstance(r, Exception):
            errs += 1
            print(f"    exception in single-turn: {type(r).__name__}: {r}")
        elif isinstance(r, dict) and r.get("error") is None:
            ok += 1
        else:
            errs += 1

    # Multi-turn: sequential (turns depend on prior responses)
    multi_ok, multi_err = 0, 0
    for mp in MULTI_TURN_PROMPTS:
        try:
            recs = await run_multi_turn(
                session, model, mp, api_key, semaphore, output_file, write_lock
            )
            multi_ok += len(recs)
        except Exception as e:
            multi_err += 1
            print(f"    multi-turn error ({mp['id']}): {type(e).__name__}: {e}")

    print(
        f"  {label:50s}  single={ok} ok / {errs} err  "
        f"multi={multi_ok} ok / {multi_err} err"
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


async def run_preflight(models, api_key):
    """Phase 0: detect hidden system prompts on all models."""
    output_file = RESULTS_DIR / "preflight.jsonl"
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    write_lock = _make_write_lock()

    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT + 5)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            run_preflight_model(
                session, m, api_key, semaphore, output_file, write_lock
            )
            for m in models
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    suspicious = [
        r for r in results
        if isinstance(r, dict) and r.get("preflight_suspicious")
    ]
    # Log exceptions explicitly (audit issue #1)
    errored = [r for r in results if isinstance(r, Exception)]
    for e in errored:
        print(f"  preflight exception: {type(e).__name__}: {e}")

    print(f"\nPreflight done: {len(results)} models checked")
    if suspicious:
        print(f"!!  {len(suspicious)} models with suspicious prompt token counts:")
        for r in suspicious:
            print(f"   {r['model_name']:40s} {r['preflight_prompt_tokens']} tokens")
    else:
        print("ok  No suspicious system prompt injection detected")
    if errored:
        print(f"!!  {len(errored)} models errored during preflight (see above)")

    return results


async def run_sweep(models, api_key):
    """Phase 1: full prompt sweep on all models."""
    output_file = RESULTS_DIR / "responses.jsonl"
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    write_lock = _make_write_lock()
    total = len(models)

    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT + 5)
    async with aiohttp.ClientSession(connector=connector) as session:
        for i, model in enumerate(models, 1):
            print(f"\n{'='*60}")
            print(f"[{i}/{total}] {model['name']} ({model['id']})")
            print(f"{'='*60}")
            try:
                await run_model(
                    session, model, api_key, semaphore, output_file,
                    write_lock, i, total
                )
            except Exception as e:
                # Record model-level failure to JSONL (audit issue #3)
                print(f"  MODEL FAILED: {type(e).__name__}: {e}")
                error_record = build_record(
                    model=model,
                    prompt_id="MODEL_LEVEL_ERROR",
                    prompt_category="error",
                    messages_sent=[],
                    response_body=None,
                    response_headers={},
                    latency_ms=0,
                    error=f"model-level exception: {type(e).__name__}: {e}",
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    run_type="error",
                )
                await append_record(error_record, output_file, write_lock)


async def main(models=None, skip_preflight=False):
    """Main entry point."""
    if models is None:
        models = MODELS

    api_key = load_api_key()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    calls_per = count_calls_for_model(models[0]) if models else 0
    total_calls = sum(count_calls_for_model(m) for m in models)

    # Sanity check
    if total_calls > 20_000:
        print(f"!! Total calls = {total_calls}. Pass --yes to confirm.")
        if "--yes" not in sys.argv:
            sys.exit(1)

    from prompts import (
        SINGLE_TURN_PROMPTS, MULTI_TURN_PROMPTS as MT_PROMPTS,
        REPEAT_PROMPT_IDS, REPEAT_COUNT,
    )

    metadata = {
        # Run identity
        "start_time": datetime.now(timezone.utc).isoformat(),
        "end_time": None,
        "cli_args": sys.argv[1:],

        # Config
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "max_concurrent": MAX_CONCURRENT,
        "request_timeout_s": REQUEST_TIMEOUT,
        "retry_attempts": RETRY_ATTEMPTS,
        "retry_backoff_s": RETRY_BACKOFF,
        "transient_status_codes": sorted(TRANSIENT_STATUS_CODES),
        "api_url": API_URL,
        "generation_stats_url": GENERATION_URL,
        "stream": False,
        "skip_preflight": skip_preflight,

        # Prompt config
        "num_single_turn_prompts": len(SINGLE_TURN_PROMPTS),
        "num_multi_turn_sequences": len(MT_PROMPTS),
        "repeat_prompt_ids": REPEAT_PROMPT_IDS,
        "repeat_count": REPEAT_COUNT,
        "single_turn_prompt_ids": [p["id"] for p in SINGLE_TURN_PROMPTS],
        "multi_turn_prompt_ids": [mp["id"] for mp in MT_PROMPTS],

        # Model config
        "num_models": len(models),
        "calls_per_model": calls_per,
        "total_calls_estimate": total_calls,
        "family_filter": [m["family"] for m in models] if len(models) < len(MODELS) else None,
        "models": [
            {"id": m["id"], "name": m["name"], "family": m["family"],
             "expected_identity": m["expected_identity"]}
            for m in models
        ],
    }
    with open(RESULTS_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Models: {len(models)}  |  Calls/model: ~{calls_per}  |  Total: ~{total_calls}")
    print(f"Temp: {TEMPERATURE}  |  Max tokens: {MAX_TOKENS}  |  Concurrency: {MAX_CONCURRENT}")
    print()

    # Phase 0
    if not skip_preflight:
        print("=" * 60)
        print("PHASE 0: Preflight — System Prompt Detection")
        print("=" * 60)
        await run_preflight(models, api_key)

    # Phase 1
    print("\n" + "=" * 60)
    print("PHASE 1: Main Sweep")
    print("=" * 60)
    await run_sweep(models, api_key)

    # Finalize metadata
    metadata["end_time"] = datetime.now(timezone.utc).isoformat()
    with open(RESULTS_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("DONE — results in", RESULTS_DIR)
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    skip_pf = "--skip-preflight" in sys.argv

    # Optional: run subset via --families=anthropic,openai (family filter)
    family_filter = None
    for arg in sys.argv[1:]:
        if arg.startswith("--families="):
            family_filter = arg.split("=", 1)[1].split(",")

    models = MODELS
    if family_filter:
        models = [m for m in MODELS if m["family"] in family_filter]
        print(f"Filtered to {len(models)} models in families: {family_filter}")

    asyncio.run(main(models=models, skip_preflight=skip_pf))
