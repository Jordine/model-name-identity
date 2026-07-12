"""Async client for the nielsrolf litellm proxy.

Raw aiohttp only — the proxy guardrail blocks the OpenAI SDK.
OpenRouter models are reached as `openrouter/<or_model_id>`; the OpenRouter
`provider` routing dict passes through, and the serving provider comes back
in the response body (`"provider": "Novita"`).
"""

import asyncio
import json
import time
from pathlib import Path

import aiohttp

PROXY_BASE = "https://litellm.nielsrolf.com/v1"
KEY_PATH = Path.home() / ".secrets" / "litellm_api_key"
HF_BASE = "https://router.huggingface.co/v1"
# hf_token_main's account depleted its monthly inference credits (402);
# hf_token_llama70b (account "Jordinner") has fresh ones. Swap back / buy
# credits per Jord's call.
HF_KEY_PATH = Path.home() / ".secrets" / "hf_token_llama70b"


def load_hf_key() -> str:
    return HF_KEY_PATH.read_text().strip()

REQUEST_TIMEOUT = 240          # stepfun/step-3.5-flash was seen hanging >100s
RETRY_ATTEMPTS = 2             # retries beyond the first attempt
RETRY_BACKOFF = 5              # seconds, linear * attempt
TRANSIENT = {408, 409, 429, 500, 502, 503, 504, 524}


def load_key() -> str:
    return KEY_PATH.read_text().strip()


def proxy_model_id(or_model_id: str, route: str | None = None) -> str:
    """Map a model id onto the proxy.

    Default: OpenRouter passthrough (openrouter/<id>) — including anthropic/*
    catalog ids (OR-mediated Anthropic API; supports provider pinning).
    route="proxy-native": use the id as-is on the proxy's own lab routes
    (anthropic/..., local/..., gemini/...) — first-party API, no provider param.
    """
    if route == "proxy-native" or or_model_id.startswith(("openrouter/", "local/", "gemini/")):
        return or_model_id
    return f"openrouter/{or_model_id}"


# ---------------------------------------------------------------------------

def extract_reasoning(msg: dict, content: str | None) -> str | None:
    """Reasoning from explicit fields or inline <think> tags."""
    for key in ("reasoning", "reasoning_content"):
        if msg.get(key):
            return msg[key]
    if content and "<think>" in content and "</think>" in content:
        return content.split("<think>", 1)[1].split("</think>", 1)[0].strip()
    return None


def strip_think(content: str | None) -> str | None:
    if content and "<think>" in content and "</think>" in content:
        pre, rest = content.split("<think>", 1)
        _, post = rest.split("</think>", 1)
        return (pre + post).strip()
    return content


# ---------------------------------------------------------------------------

async def call(
    session: aiohttp.ClientSession,
    or_model_id: str,
    messages: list[dict],
    api_key: str,
    *,
    temperature: float = 0.7,
    max_tokens: int = 500,
    provider: dict | None = None,
    route: str | None = None,
    timeout: int = REQUEST_TIMEOUT,
) -> dict:
    """One chat completion through the proxy.

    Returns a flat record:
      {ok, status, error, latency_ms, body, content, content_clean,
       reasoning, finish_reason, provider_served, usage, returned_model}
    Retries transient statuses / timeouts.
    """
    if route == "hf-router":
        # HF inference router: provider pin is a ":provider" id suffix;
        # billed to the HF account, not the CLR proxy.
        mid = or_model_id
        if provider and isinstance(provider, str):
            mid = f"{or_model_id}:{provider}"
        payload = {"model": mid, "messages": messages,
                   "temperature": temperature, "max_tokens": max_tokens,
                   "stream": False}
        url = f"{HF_BASE}/chat/completions"
        api_key = load_hf_key()
    else:
        payload = {
            "model": proxy_model_id(or_model_id, route),
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        url = f"{PROXY_BASE}/chat/completions"
        # provider routing is an OpenRouter concept; lab-native routes reject it
        if provider and route != "proxy-native" and not payload["model"].startswith(("local/", "gemini/", "anthropic/")):
            payload["provider"] = provider

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    last_err, status = "unknown", None

    for attempt in range(1 + RETRY_ATTEMPTS):
        start = time.monotonic()
        try:
            async with session.post(
                url, json=payload, headers=headers, timeout=client_timeout,
            ) as resp:
                latency = (time.monotonic() - start) * 1000
                status = resp.status
                text = await resp.text()

                if resp.status == 200:
                    try:
                        body = json.loads(text)
                    except json.JSONDecodeError:
                        last_err = f"bad json: {text[:200]}"
                        break
                    # litellm sometimes 200s an error body
                    if "error" in body and "choices" not in body:
                        last_err = f"error body: {json.dumps(body['error'])[:300]}"
                        break
                    msg = body["choices"][0]["message"]
                    content = msg.get("content")
                    return {
                        "ok": True, "status": 200, "error": None,
                        "latency_ms": round(latency, 1),
                        "body": body,
                        "content": content,
                        "content_clean": strip_think(content),
                        "reasoning": extract_reasoning(msg, content),
                        "finish_reason": body["choices"][0].get("finish_reason"),
                        "provider_served": body.get("provider"),
                        "usage": body.get("usage"),
                        "returned_model": body.get("model"),
                    }

                last_err = f"HTTP {resp.status}: {text[:300]}"
                if resp.status in TRANSIENT and attempt < RETRY_ATTEMPTS:
                    wait = RETRY_BACKOFF * (attempt + 1)
                    ra = resp.headers.get("Retry-After")
                    if ra:
                        try:
                            wait = min(float(ra), 60)
                        except ValueError:
                            pass
                    await asyncio.sleep(wait)
                    continue
                break

        except asyncio.TimeoutError:
            last_err, status = f"timeout after {timeout}s", None
            if attempt < RETRY_ATTEMPTS:
                continue
            break
        except aiohttp.ClientError as e:
            last_err, status = f"client error: {e}", None
            if attempt < RETRY_ATTEMPTS:
                await asyncio.sleep(RETRY_BACKOFF)
                continue
            break

    return {
        "ok": False, "status": status, "error": last_err,
        "latency_ms": None, "body": None, "content": None, "content_clean": None,
        "reasoning": None, "finish_reason": None, "provider_served": None,
        "usage": None, "returned_model": None,
    }
