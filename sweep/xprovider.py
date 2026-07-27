"""Cross-provider variance sweep — same model, same battery, every clean provider.

Does *which endpoint* serves a model change what it says it is? At n=240 the
answer is mostly NO — direct Anthropic API ≈ OpenRouter ≈ Bedrock ≈ Azure within
noise; the one real effect is Google Vertex raising Opus 4.8's Chinese mismatch
(+17pp vs direct, p<0.001). Runs the full multilingual identity+creator battery
(direct_/creator_, all 8 languages, N=5) against each non-injecting provider,
pinned with allow_fallbacks:false so the call can't drift to another endpoint.

Providers are the preflight-clean (non-suspicious) set per model; Azure is
dropped for Opus 4.8 (it 400s — confirmed by probe, matches its null preflight
token count). Same weights, different serving stacks → any difference is the
host (checkpoint / quant / numerics / default sampler), not the model design.

Writes results/xprovider_sweep.jsonl, kept OUT of the headline roster — this is
an addendum, not part of the one-provider-per-model main study (no double-count).

  python -m sweep.xprovider --run      # execute (resumable; skips completed keys)
  python -m sweep.xprovider --report   # tabulate what's collected so far
"""
import argparse
import asyncio
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

from . import api
from .prompts import core_prompts
from .analyze import canon_identity, is_self

ROOT = Path(__file__).resolve().parent.parent
REG = {m["id"]: m for m in json.loads((ROOT / "config" / "models.json").read_text())["models"]}
OUT = ROOT / "results" / "xprovider_sweep.jsonl"

# preflight-clean (non-injecting) providers per model. Two experiments:
#   claude — same weights on Anthropic's own clouds (all quant=unknown); tests
#            host/kernel/hardware. (azure 400s on opus-4.8 → excluded there.)
#   kimi   — open weights served by many third parties at DIFFERENT quantizations
#            (int4/fp4/fp8) → a direct test of the precision hypothesis. K2 base &
#            K2-0905 are novita-only after the hygiene gate (groq injects), so no
#            cross-provider comparison there; K2.5 has 10 clean providers.
MATRICES = {
    "claude": {
        "anthropic/claude-opus-4.8":   ["anthropic-direct", "anthropic", "google-vertex", "amazon-bedrock"],
        "anthropic/claude-sonnet-4.6": ["anthropic-direct", "anthropic", "google-vertex", "amazon-bedrock", "azure"],
    },
    "kimi": {
        "moonshotai/kimi-k2.5": ["atlas-cloud", "chutes", "deepinfra", "digitalocean",
                                 "modelrun", "moonshotai", "novita", "phala",
                                 "siliconflow", "streamlake"],
    },
}


# "anthropic-direct" = the proxy's NATIVE Anthropic API (route=proxy-native, no
# OpenRouter middleman). Native ids use dashes, not dots. Tests whether the
# aggregator layer (OpenRouter→Anthropic) matches the true first-party API. At
# n=240 it DOES, within noise: Sonnet 34% direct vs 36% OR (p=0.57), Opus 57% vs
# 60% (p=0.58). (An n=12 probe fluked to Sonnet 0/12 and did NOT replicate.)
DIRECT_NATIVE = {
    "anthropic/claude-opus-4.8":   "claude-opus-4-8",
    "anthropic/claude-sonnet-4.6": "claude-sonnet-4-6",
}


def matrix_for(which):
    if which == "all":
        out = {}
        for mm in MATRICES.values():
            out.update(mm)
        return out
    return MATRICES[which]


def quant_map():
    """{(model_id, provider_slug): quant} from the preflight probes, so the report
    can show the served precision next to each provider — the mechanism clue."""
    qm = {}
    pf = ROOT / "results" / "preflight_raw.jsonl"
    if pf.exists():
        for line in pf.open(encoding="utf-8"):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("slug") and d.get("quant"):
                qm[(d["model_id"], d["slug"])] = d["quant"]
    return qm
N = 5
N_BOOST = 25   # extra zh samples/prompt/provider → n≈240, tight enough to test a ~13pp gap
TEMPERATURE = 0.7
MAX_TOKENS = 700
CONCURRENCY = 12


def battery(m):
    """The spontaneous identity battery: identity + creator, all languages."""
    return [p for p in core_prompts(m) if p["category"].startswith(("direct_", "creator_"))]


def zh_battery(m):
    """Chinese identity+creator only — the provider effect lives entirely here
    (every other language is ~0%), so this is where the extra samples buy CI."""
    return [p for p in core_prompts(m) if p["category"] in ("direct_zh", "creator_zh")]


def wilson(k, n, z=1.96):
    """Wilson score 95% CI for a binomial rate — returns (pct, lo, hi)."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * p, 100 * max(0, c - h), 100 * min(1, c + h))


def two_prop_p(k1, n1, k2, n2):
    """Two-sided p-value, pooled 2-proportion z-test (group2 vs group1)."""
    if not n1 or not n2:
        return 1.0
    pp = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (k2 / n2 - k1 / n1) / se
    return math.erfc(abs(z) / math.sqrt(2))


def done_keys():
    done = set()
    if OUT.exists():
        for line in OUT.open(encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("error") is None:
                done.add(r["xkey"])
    return done


async def run(boost=False, which="claude"):
    key = api.load_key()
    done = done_keys()
    mat = matrix_for(which)
    todo = []
    for mid, provs in mat.items():
        m = REG[mid]
        bat = zh_battery(m) if boost else battery(m)
        rng = range(N, N + N_BOOST) if boost else range(N)   # boost keys continue past N (additive)
        for prov in provs:
            for p in bat:
                for i in rng:
                    xkey = f"{mid}::{prov}::{p['id']}::{i}"
                    if xkey not in done:
                        todo.append((m, prov, p, i, xkey))
    tag = "zh-boost" if boost else "full-battery"
    print(f"[{tag}:{which}] models x providers: "
          + ", ".join(f"{k.split('/')[-1]}×{len(v)}" for k, v in mat.items())
          + f"   |   todo: {len(todo)} calls ({len(done)} already done)", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fh = OUT.open("a", encoding="utf-8")
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(CONCURRENCY)
    stats = Counter()
    conn = aiohttp.TCPConnector(limit=CONCURRENCY + 4)
    async with aiohttp.ClientSession(connector=conn) as session:
        async def one(m, prov, p, i, xkey):
            msgs = [{"role": "user", "content": p["content"]}]
            async with sem:
                if prov == "anthropic-direct":   # proxy-native direct API — no OpenRouter, no provider dict
                    r = await api.call(session, DIRECT_NATIVE[m["id"]], msgs, key,
                                       temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                                       route="proxy-native", timeout=120)
                else:
                    r = await api.call(session, m["id"], msgs, key,
                                       temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                                       provider={"order": [prov], "allow_fallbacks": False}, timeout=120)
            rec = {"ts": datetime.now(timezone.utc).isoformat(), "xkey": xkey,
                   "model_id": m["id"], "provider_pinned": prov,
                   "provider_served": r.get("provider_served"),
                   "prompt_id": p["id"], "prompt_category": p["category"], "sample_idx": i,
                   "content": r.get("content"), "content_clean": r.get("content_clean"),
                   "error": r.get("error"), "status": r.get("status")}
            async with lock:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
            stats["err" if r.get("error") else "ok"] += 1
            n = stats["ok"] + stats["err"]
            if n % 100 == 0:
                print(f"  {n}/{len(todo)}  ok={stats['ok']} err={stats['err']}", flush=True)
        await asyncio.gather(*[one(*t) for t in todo])
    fh.close()
    print(f"done — ok={stats['ok']} err={stats['err']}  ->  {OUT}", flush=True)


def report():
    rows = defaultdict(list)
    served = defaultdict(Counter)
    errs = defaultdict(int)
    qm = quant_map()
    if not OUT.exists():
        print("no data yet")
        return
    for line in OUT.open(encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        k = (r["model_id"], r["provider_pinned"])
        if r.get("error"):
            errs[k] += 1
            continue
        rows[k].append(r)
        served[k][r.get("provider_served")] += 1

    last_mid = None
    for (mid, prov), recs in sorted(rows.items()):
        m = REG[mid]
        fam, al, exp = m["family"], m["aliases"], m["expected_identity"]
        per = defaultdict(lambda: [0, 0])
        claims = Counter()
        d = n = 0
        for r in recs:
            resp = (r.get("content_clean") or r.get("content") or "").strip()
            c = canon_identity(resp)
            mism = bool(c and not str(c).startswith("other:") and not is_self(c, fam, al, exp))
            lg = r["prompt_category"].split("_")[-1]
            per[lg][1] += 1
            n += 1
            if mism:
                per[lg][0] += 1
                d += 1
                claims[c] += 1
        if mid != last_mid:
            print(f"\n{'='*72}\n{mid}\n{'='*72}")
            last_mid = mid
        srv = dict(served[(mid, prov)])
        e = errs.get((mid, prov), 0)
        q = qm.get((mid, prov), "?")
        print(f"\n  @ {prov:15} [{q:7}] served={srv}" + (f"  (+{e} errors)" if e else ""))
        print(f"     overall mismatch {100*d/n:4.0f}%  ({d}/{n})")
        print("     per-lang: " + "  ".join(f"{l} {100*a/b:.0f}%" for l, (a, b) in sorted(per.items())))
        print(f"     claims:   {claims.most_common(6)}")


def zh_report():
    """Chinese-only mismatch per (model, provider), all zh samples pooled (full
    battery + boost), with Wilson 95% CI and a 2-proportion test vs the
    first-party (anthropic) endpoint — the statistically honest provider test."""
    per = defaultdict(lambda: [0, 0])
    if not OUT.exists():
        print("no data yet")
        return
    for line in OUT.open(encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("error") or r["prompt_category"] not in ("direct_zh", "creator_zh"):
            continue
        mid, prov = r["model_id"], r["provider_pinned"]
        m = REG[mid]
        fam, al, exp = m["family"], m["aliases"], m["expected_identity"]
        resp = (r.get("content_clean") or r.get("content") or "").strip()
        c = canon_identity(resp)
        mism = bool(c and not str(c).startswith("other:") and not is_self(c, fam, al, exp))
        per[(mid, prov)][1] += 1
        if mism:
            per[(mid, prov)][0] += 1
    mids = sorted({mid for (mid, _prov) in per})
    for mid in mids:
        print(f"\n{mid} — Chinese identity mismatch (pooled zh samples):")
        # baseline = the TRUE first-party (direct API) when we have it, else OR-anthropic
        base_prov = "anthropic-direct" if (mid, "anthropic-direct") in per else "anthropic"
        base = per.get((mid, base_prov))
        provs_here = sorted({prov for (m2, prov) in per if m2 == mid})
        for prov in provs_here:
            k, n = per[(mid, prov)]
            p, lo, hi = wilson(k, n)
            line = f"  {prov:16} {p:5.1f}%  95% CI [{lo:4.1f}, {hi:5.1f}]  (n={n:3d}, {k} mismatch)"
            if base and prov != base_prov and base[1]:
                pv = two_prop_p(base[0], base[1], k, n)
                bp = 100 * base[0] / base[1]
                sig = " *" if pv < 0.05 else ""
                line += f"   Δ={p-bp:+6.1f}pp vs {base_prov}  p={pv:.3f}{sig}"
            print(line)


def quant_report():
    """Mismatch rate grouped by served *quantization*, pooling providers that share
    a quant — the direct test of 'does precision tip identity?'. Open models only
    (Kimi); Anthropic's clouds are all quant=unknown."""
    qm = quant_map()
    per = defaultdict(lambda: [0, 0])
    provs_in = defaultdict(set)
    if not OUT.exists():
        print("no data yet")
        return
    for line in OUT.open(encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("error"):
            continue
        mid, prov = r["model_id"], r["provider_pinned"]
        q = qm.get((mid, prov), "unknown")
        m = REG[mid]
        resp = (r.get("content_clean") or r.get("content") or "").strip()
        c = canon_identity(resp)
        mism = bool(c and not str(c).startswith("other:")
                    and not is_self(c, m["family"], m["aliases"], m["expected_identity"]))
        per[(mid, q)][1] += 1
        provs_in[(mid, q)].add(prov)
        if mism:
            per[(mid, q)][0] += 1
    for mid in sorted({mid for (mid, _q) in per}):
        print(f"\n{mid} — mismatch by served quantization:")
        for (m2, q), (d, n) in sorted(per.items(), key=lambda kv: -kv[1][1]):
            if m2 != mid:
                continue
            p, lo, hi = wilson(d, n)
            print(f"  {q:8} {p:5.1f}%  95% CI [{lo:4.1f}, {hi:5.1f}]  (n={n:4d})  "
                  f"providers: {', '.join(sorted(provs_in[(m2, q)]))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="claude", choices=["claude", "kimi", "all"], help="which model set")
    ap.add_argument("--run", action="store_true", help="full multilingual battery, all providers")
    ap.add_argument("--boost", action="store_true", help="zh-only high-N booster for tight CIs")
    ap.add_argument("--report", action="store_true", help="full per-language table (with quant)")
    ap.add_argument("--zh", action="store_true", help="Chinese-only report with CIs + significance")
    ap.add_argument("--quant", action="store_true", help="mismatch grouped by served quantization")
    args = ap.parse_args()
    if args.run:
        asyncio.run(run(boost=False, which=args.set))
    if args.boost:
        asyncio.run(run(boost=True, which=args.set))
    if args.zh:
        zh_report()
    if args.quant:
        quant_report()
    if args.report or not (args.run or args.boost or args.zh or args.quant):
        report()


if __name__ == "__main__":
    main()
