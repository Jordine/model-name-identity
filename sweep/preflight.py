"""Provider-hygiene preflight.

For every model in the registry, enumerate its OpenRouter providers (public
endpoints API), then probe EACH provider with a pinned minimal call and flag
hidden system prompts via prompt_token counts ("hi" should be ~1-8 tokens;
v1 threshold: >15 suspicious).

Output: config/provider_hygiene.json
  {model_id: {provider: {...pin...} | None, exclude: bool, reason: str,
              chosen: {...}, checked: [per-provider results]}}

Choice ranking among clean providers:
  1. official lab provider (tag matches model org) first
  2. highest serving precision (unknown/bf16/fp16 > fp8 > int8 > fp4/int4)
  3. lowest prompt_tokens
"""

import asyncio
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

from . import api

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "config" / "models.json"
OUT = ROOT / "config" / "provider_hygiene.json"
RAW_OUT = ROOT / "results" / "preflight_raw.jsonl"

# Two-tier: provider-raw prompt_tokens for "hi" (proxy reports raw counts,
# NOT OR-normalized like v1 — template/BOS overhead varies by tokenizer).
# >EXCLUDE_PT => injection, exclude/avoid. (BORDERLINE_PT, EXCLUDE_PT] =>
# keep with borderline flag; in-sweep sysprompt probes double-check these.
BORDERLINE_PT = 15
EXCLUDE_PT = 25
PROBE_TIMEOUT = 150
CONCURRENCY = 12

QUANT_RANK = {"unknown": 0, "bf16": 0, "fp16": 0, "fp32": 0, "": 0,
              None: 0, "fp8": 1, "int8": 2, "fp4": 3, "int4": 3}

# org slug -> official provider tag prefixes
OFFICIAL = {
    "moonshotai": ["moonshotai"], "deepseek": ["deepseek"],
    "openai": ["openai"], "anthropic": ["anthropic", "google-vertex", "amazon-bedrock"],
    "google": ["google-vertex", "google-ai-studio"], "mistralai": ["mistral"],
    "x-ai": ["xai"], "z-ai": ["z-ai", "zhipu"], "qwen": ["alibaba"],
    "cohere": ["cohere"], "amazon": ["amazon-bedrock"], "ai21": ["ai21"],
    "inflection": ["inflection"], "perplexity": ["perplexity"],
    "minimax": ["minimax"], "stepfun": ["stepfun"], "tencent": ["tencent"],
    "baidu": ["baidu"], "meituan": ["meituan"], "inception": ["inception"],
    "liquid": ["liquid"], "writer": ["writer"], "upstage": ["upstage"],
    "reka": ["reka"], "sakana": ["sakana"],
}


_HF_PROVIDERS: dict | None = None


def hf_providers() -> dict:
    """model_id -> [provider slugs] from the HF router catalog (cached)."""
    global _HF_PROVIDERS
    if _HF_PROVIDERS is None:
        req = urllib.request.Request(
            "https://router.huggingface.co/v1/models",
            headers={"Authorization": f"Bearer {api.load_hf_key()}"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())["data"]
        _HF_PROVIDERS = {m["id"]: [p.get("provider") for p in (m.get("providers") or [])
                                   if p.get("provider")]
                         for m in data}
    return _HF_PROVIDERS


def fetch_endpoints(model_id: str) -> list[dict]:
    url = f"https://openrouter.ai/api/v1/models/{model_id}/endpoints"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.loads(r.read())["data"]
    except Exception as e:
        return [{"_fetch_error": str(e)}]
    eps = []
    for e in data.get("endpoints", []):
        tag = e.get("tag") or ""
        eps.append({
            "provider_name": e.get("provider_name"),
            "slug": tag.split("/")[0] if tag else None,
            "tag": tag,
            "status": e.get("status"),
            "quant": e.get("quantization"),
        })
    return eps


async def probe_provider(session, key, model, ep, sem, raw_f, lock):
    """One pinned 'hi' probe against one provider."""
    pin = {"order": [ep["slug"]], "allow_fallbacks": False}
    async with sem:
        r = await api.call(session, model["id"], [{"role": "user", "content": "hi"}],
                           key, temperature=0, max_tokens=30, provider=pin,
                           timeout=PROBE_TIMEOUT)
    usage = r.get("usage") or {}
    ptok = usage.get("prompt_tokens")
    res = {
        "model_id": model["id"],
        "slug": ep["slug"], "provider_name": ep["provider_name"],
        "quant": ep["quant"], "or_status": ep["status"],
        "prompt_tokens": ptok,
        "provider_served": r.get("provider_served"),
        "error": r.get("error"),
        "suspicious": ptok is not None and ptok > EXCLUDE_PT,
        "borderline": ptok is not None and BORDERLINE_PT < ptok <= EXCLUDE_PT,
        "content_head": (r.get("content_clean") or "")[:80],
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    async with lock:
        with open(RAW_OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
    return res


def choose(model, results: list[dict]) -> dict:
    """Pick the best clean provider, or exclude."""
    org = model["id"].split("/")[0]
    official = OFFICIAL.get(org, [org])

    clean = [r for r in results
             if not r["error"] and r["prompt_tokens"] is not None and not r["suspicious"]]
    # prefer strictly-clean over borderline, but borderline beats exclusion
    strictly = [r for r in clean if not r["borderline"]]
    if strictly:
        clean = strictly
    if not clean:
        errs = sum(1 for r in results if r["error"])
        susp = sum(1 for r in results if r["suspicious"])
        reason = f"no clean provider ({len(results)} checked: {susp} inject, {errs} error)"
        return {"provider": None, "exclude": True, "reason": reason, "chosen": None}

    def rank(r):
        is_official = any(r["slug"] and r["slug"].startswith(o) for o in official)
        return (0 if is_official else 1,
                1 if r["borderline"] else 0,
                QUANT_RANK.get(r["quant"], 1),
                r["prompt_tokens"])

    best = sorted(clean, key=rank)[0]
    n_inject = sum(1 for r in results if r["suspicious"])
    # ALWAYS pin: unpinned calls route to arbitrary providers per-call
    # (pilot showed kimi-k2.5 served by 4 different hosts in one run),
    # which injects uncontrolled provider/quantization variance.
    return {
        "provider": {"order": [best["slug"]], "allow_fallbacks": False},
        "exclude": False,
        "borderline": bool(best["borderline"]),
        "reason": (f"pinned to {best['slug']} (quant={best['quant']}, ptok={best['prompt_tokens']}"
                   f"{', BORDERLINE' if best['borderline'] else ''}; "
                   f"{n_inject}/{len(results)} providers inject, {len(clean)}/{len(results)} clean)"),
        "chosen": best,
    }


async def main_async(only_models=None):
    reg = json.loads(REGISTRY.read_text())["models"]
    if only_models:
        reg = [m for m in reg if m["id"] in only_models]
    key = api.load_key()
    sem = asyncio.Semaphore(CONCURRENCY)
    lock = asyncio.Lock()
    RAW_OUT.parent.mkdir(exist_ok=True)

    print(f"preflighting {len(reg)} models...")
    hygiene = {}

    conn = aiohttp.TCPConnector(limit=CONCURRENCY + 4)
    async with aiohttp.ClientSession(connector=conn) as session:

        async def do_model(m):
            if m.get("route") == "hf-router":
                provs = hf_providers().get(m["id"], [])
                if not provs:
                    hygiene[m["id"]] = {"provider": None, "exclude": True,
                                        "reason": "hf-router: no providers listed",
                                        "chosen": None, "checked": []}
                    return
                results = []
                for slug in provs:
                    async with sem:
                        r = await api.call(session, m["id"], [{"role": "user", "content": "hi"}],
                                           key, temperature=0, max_tokens=30,
                                           provider=slug, route="hf-router",
                                           timeout=PROBE_TIMEOUT)
                    ptok = (r.get("usage") or {}).get("prompt_tokens")
                    res = {"model_id": m["id"], "slug": slug, "provider_name": slug,
                           "quant": "unknown", "or_status": None, "prompt_tokens": ptok,
                           "provider_served": slug, "error": r.get("error"),
                           "suspicious": ptok is not None and ptok > EXCLUDE_PT,
                           "borderline": ptok is not None and BORDERLINE_PT < ptok <= EXCLUDE_PT,
                           "content_head": (r.get("content_clean") or "")[:80],
                           "ts": datetime.now(timezone.utc).isoformat()}
                    async with lock:
                        with open(RAW_OUT, "a", encoding="utf-8") as f:
                            f.write(json.dumps(res, ensure_ascii=False) + "\n")
                    results.append(res)
                verdict = choose(m, results)
                if verdict.get("provider"):  # hf pins are plain strings
                    verdict["provider"] = verdict["chosen"]["slug"]
                verdict["checked"] = [{k: r[k] for k in ("slug", "quant", "prompt_tokens", "suspicious", "error")}
                                      for r in results]
                hygiene[m["id"]] = verdict
                flag = "EXCLUDE" if verdict["exclude"] else "pin"
                print(f"  {m['id']:55s} hf x{len(provs)} -> {flag:8s} {verdict['reason'][:70]}")
                return
            if m.get("route") == "proxy-native":
                # lab-first-party route on the proxy: no OR providers to pin;
                # single probe verifies liveness + token sanity
                async with sem:
                    r = await api.call(session, m["id"], [{"role": "user", "content": "hi"}],
                                       key, temperature=0, max_tokens=30,
                                       route="proxy-native", timeout=PROBE_TIMEOUT)
                ptok = (r.get("usage") or {}).get("prompt_tokens")
                bad = bool(r["error"]) or ptok is None or ptok > EXCLUDE_PT
                hygiene[m["id"]] = {
                    "provider": None, "exclude": bad,
                    "reason": (f"proxy-native probe failed: {(r.get('error') or f'ptok={ptok}')[:80]}"
                               if bad else f"proxy-native (first-party API), ptok={ptok}"),
                    "chosen": None, "checked": [{"slug": "proxy-native", "quant": None,
                                                 "prompt_tokens": ptok, "suspicious": False,
                                                 "error": r.get("error")}],
                }
                print(f"  {m['id']:55s} native -> {'EXCLUDE' if bad else 'ok'}")
                return
            eps = await asyncio.to_thread(fetch_endpoints, m["id"])
            if eps and "_fetch_error" in eps[0]:
                hygiene[m["id"]] = {"provider": None, "exclude": True,
                                    "reason": f"endpoints fetch failed: {eps[0]['_fetch_error']}",
                                    "chosen": None, "checked": []}
                return
            eps = [e for e in eps if e["slug"]]
            if not eps:
                hygiene[m["id"]] = {"provider": None, "exclude": True,
                                    "reason": "no live endpoints", "chosen": None, "checked": []}
                return
            results = await asyncio.gather(*[
                probe_provider(session, key, m, ep, sem, RAW_OUT, lock) for ep in eps
            ])
            verdict = choose(m, list(results))
            verdict["checked"] = [
                {k: r[k] for k in ("slug", "quant", "prompt_tokens", "suspicious", "error")}
                for r in results
            ]
            hygiene[m["id"]] = verdict
            flag = "EXCLUDE" if verdict["exclude"] else ("pin" if verdict["provider"] else "ok")
            print(f"  {m['id']:55s} {len(eps)} providers -> {flag:8s} {verdict['reason'][:70]}")

        await asyncio.gather(*[do_model(m) for m in reg])

    # merge into any existing hygiene file (supports partial re-runs via --models)
    merged = {}
    if OUT.exists():
        merged = json.loads(OUT.read_text())
    merged.update(hygiene)
    OUT.write_text(json.dumps(merged, indent=1, ensure_ascii=False))
    n_ex = sum(1 for v in merged.values() if v["exclude"])
    n_pin = sum(1 for v in merged.values() if v["provider"])
    n_bl = sum(1 for v in merged.values() if v.get("borderline"))
    print(f"\n{len(merged)} models: {n_ex} excluded, {n_pin} pinned, {n_bl} borderline -> {OUT}")


if __name__ == "__main__":
    only = None
    for a in sys.argv[1:]:
        if a.startswith("--models="):
            only = set(a.split("=", 1)[1].split(","))
    asyncio.run(main_async(only))
