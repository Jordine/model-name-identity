"""Adjudicate the local raw-weights models the SAME way as the API sweep — Claude
Haiku, given ground truth, keeps only genuine_foreign. Clean condition only (the
raw-weights read). Writes results/adjudications_local.jsonl so the local models
flow through the exact same foreign_claims() filter as everyone else.

  python -m sweep.adjudicate_local
"""
import asyncio
import json
from pathlib import Path

import aiohttp

from . import adjudicate, api
from .analyze import canon_identity, is_self

ROOT = Path(__file__).resolve().parent.parent
RESP_DIR = ROOT / "results_local"                          # local raw responses (per-model jsonl)
JUD = ROOT / "results_local" / "judgments_clean.jsonl"
OUT = ROOT / "results" / "adjudications_local.jsonl"
adjudicate.OUT = OUT                                       # redirect the writer


def build_worklist():
    raw = {}
    for f in RESP_DIR.glob("*.jsonl"):                     # every raw-weights model file
        if "judgment" in f.name:
            continue
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            raw[f"{r['resume_key']}::t{r.get('turn_index', 0)}"] = r
    work = []
    for l in open(JUD, encoding="utf-8"):
        j = json.loads(l)
        if not j.get("judgment"):
            continue
        if j["resume_key"].split("::")[-1] != "clean":     # raw-weights read only
            continue
        cat = j["prompt_category"]
        if not (cat.startswith("direct_") or cat.startswith("creator_") or cat == "probe_self"):
            continue
        jm = j["judgment"]
        cn = canon_identity(jm.get("claimed_name")); cc = canon_identity(jm.get("claimed_creator"))
        fam, exp, al = j.get("family"), j.get("expected_identity"), j.get("aliases", [])
        flagged = (cn and not is_self(cn, fam, al, exp)) or (cc and not is_self(cc, fam, al, exp))
        if not flagged:
            continue
        r = raw.get(j["judge_key"])
        if r:
            work.append((j, r))
    return work


async def main():
    work = build_worklist()
    done = set()
    if OUT.exists():
        for l in open(OUT, encoding="utf-8"):
            try:
                d = json.loads(l)
                if d.get("verdict"):
                    done.add(d["adj_key"])
            except json.JSONDecodeError:
                pass
    todo = [(j, r) for (j, r) in work if adjudicate.adj_key(j) not in done]
    print(f"local drift-flagged (clean): {len(work)}  todo: {len(todo)}")
    if not todo:
        return
    sem = asyncio.Semaphore(12); lock = asyncio.Lock(); stats = {}
    conn = aiohttp.TCPConnector(limit=16)
    async with aiohttp.ClientSession(connector=conn) as s:
        await asyncio.gather(*[
            adjudicate.adjudicate_one(s, adjudicate.adj_key(j), j, r, adjudicate.ADJ_MODEL, sem, lock, stats)
            for (j, r) in todo])
    print("verdicts:", stats)


if __name__ == "__main__":
    asyncio.run(main())
