"""
Specific identity probe: ask models that claim a different identity
what exact model version they think they are.

For each model with unprompted identity discrepancies, asks:
  "What specific model version are you? Please state your exact model name and version."

Run 5 times at temp 0.7 to see variance.
"""

import asyncio
import aiohttp
import json
import sys
import time
from pathlib import Path
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding='utf-8')

API_KEY_PATH = Path("C:/Users/Admin/.secrets/openrouter_api_key")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
RESULTS_DIR = Path("results")
OUTPUT_FILE = RESULTS_DIR / "specific_identity_probes.jsonl"

MAX_CONCURRENT = 10
TEMPERATURE = 0.7
MAX_TOKENS = 300
REQUEST_TIMEOUT = 120
NUM_RUNS = 5

PROBES = [
    "What specific model version are you? Please state your exact model name and version number.",
    "Tell me your exact model name, version, and who made you. Be as specific as possible.",
    "I need to know exactly which AI model I'm talking to. What is your precise model name and version?",
]

# Load models
from models import MODELS

MODELS_BY_ID = {m["id"]: m for m in MODELS}


def load_api_key():
    return API_KEY_PATH.read_text(encoding="utf-8").strip()


def extract_response_text(body):
    if not body:
        return None
    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None


def extract_thinking_text(body):
    if not body:
        return None
    try:
        msg = body["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return None
    for key in ("reasoning", "reasoning_content"):
        if key in msg and msg[key]:
            return msg[key]
    content = msg.get("content") or ""
    if "<think>" in content and "</think>" in content:
        start = content.index("<think>") + len("<think>")
        end = content.index("</think>")
        return content[start:end].strip()
    return None


def extract_content_without_think(body):
    text = extract_response_text(body)
    if text and "<think>" in text and "</think>" in text:
        start = text.index("<think>")
        end = text.index("</think>") + len("</think>")
        text = (text[:start] + text[end:]).strip()
    return text


def safe_json(obj):
    def default(o):
        if isinstance(o, bytes):
            return o.decode("utf-8", errors="replace")
        if isinstance(o, set):
            return list(o)
        return str(o)
    return json.dumps(obj, ensure_ascii=False, default=default)


async def api_call(session, model_id, messages, api_key, provider=None):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/jord-research/self-id-sweep",
    }
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    if provider:
        payload["provider"] = provider

    for attempt in range(3):
        try:
            t0 = time.monotonic()
            async with session.post(API_URL, json=payload, headers=headers,
                                     timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as resp:
                latency = int((time.monotonic() - t0) * 1000)
                body = await resp.json()
                if resp.status == 200:
                    return body, latency, None
                if resp.status in {429, 500, 502, 503, 504}:
                    wait = int(resp.headers.get("Retry-After", 3))
                    await asyncio.sleep(wait)
                    continue
                return None, latency, f"HTTP {resp.status}: {json.dumps(body)[:200]}"
        except Exception as e:
            if attempt < 2:
                await asyncio.sleep(2)
                continue
            return None, 0, str(e)
    return None, 0, "max retries"


async def probe_model(session, model, probe_text, run_idx, api_key,
                       semaphore, write_lock, output_file):
    async with semaphore:
        messages = [{"role": "user", "content": probe_text}]
        provider = model.get("provider")
        body, latency, error = await api_call(session, model["id"], messages, api_key, provider)

        record = {
            "model_id": model["id"],
            "model_name": model["name"],
            "model_family": model["family"],
            "probe_text": probe_text,
            "run_index": run_idx,
            "response_text": extract_content_without_think(body) if body else None,
            "thinking_text": extract_thinking_text(body) if body else None,
            "latency_ms": latency,
            "error": error,
        }

        async with write_lock:
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(safe_json(record) + "\n")
                f.flush()

        return record


async def main():
    api_key = load_api_key()

    # Load targets: models with unprompted discrepancies
    targets_file = RESULTS_DIR / "depth_probe_targets.json"
    with open(targets_file, "r", encoding="utf-8") as f:
        targets = json.load(f)

    # Filter by command-line args
    model_filter = None
    for arg in sys.argv[1:]:
        if arg.startswith("--model="):
            model_filter = arg.split("=", 1)[1].lower()
        elif arg == "--fresh" and OUTPUT_FILE.exists():
            OUTPUT_FILE.unlink()
            print("Cleared previous results (--fresh)")

    if model_filter:
        targets = [t for t in targets if model_filter in t["model_name"].lower()]

    print(f"\n{'='*70}")
    print(f"SPECIFIC IDENTITY PROBING: {len(targets)} models")
    print(f"{len(PROBES)} probes x {NUM_RUNS} runs = {len(PROBES) * NUM_RUNS} calls per model")
    print(f"Total API calls: ~{len(targets) * len(PROBES) * NUM_RUNS}")
    print(f"{'='*70}\n")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    write_lock = asyncio.Lock()

    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT + 5)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for target in targets:
            model = MODELS_BY_ID.get(target["model_id"])
            if not model:
                continue
            for run_idx in range(NUM_RUNS):
                for probe in PROBES:
                    tasks.append(probe_model(
                        session, model, probe, run_idx, api_key,
                        semaphore, write_lock, OUTPUT_FILE,
                    ))

        print(f"Launching {len(tasks)} API calls...")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count errors
        errors = sum(1 for r in results if isinstance(r, Exception) or (isinstance(r, dict) and r.get("error")))
        print(f"\nCompleted. Errors: {errors}/{len(tasks)}")

    # Print summary
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}\n")

    records = []
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))

    by_model = defaultdict(list)
    for r in records:
        if not r.get("error"):
            by_model[r["model_name"]].append(r)

    for model_name in sorted(by_model.keys()):
        recs = by_model[model_name]
        print(f"\n{model_name} ({len(recs)} responses):")
        for r in recs:
            resp = (r.get("response_text") or "")[:200]
            if resp:
                # Extract the key identity claim
                first_line = resp.split('\n')[0].strip()
                print(f"  [{r['run_index']}] {first_line[:150]}")


if __name__ == "__main__":
    asyncio.run(main())
