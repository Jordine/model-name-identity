"""Post-hoc hygiene verification over the main sweep.

Answers two questions the preflight can't:

1. PIN INTEGRITY — did every call actually get served by the pinned provider?
   (allow_fallbacks=false should guarantee it; verify empirically.)

2. HIDDEN INJECTION — the preflight's token-count check trusts the provider's
   own prompt_tokens accounting. A dishonest host could inject AND under-count.
   Counter-evidence: the in-sweep sysprompt probes. For each model:
     * scan the 3 sysprompt-probe responses for leak-like content
     * compare against the token budget: prompt_tokens on those calls bounds
       the size of any real injected prompt. Leak-like text with NO token room
       = confabulation (interesting, but not injection). Leak-like text WITH
       token room, or oddly specific/consistent quoted instructions = flag for
       manual review.

Usage: python -m sweep.verify_hygiene
"""

import json
from .analyze import open_lines
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SWEEP = ROOT / "results" / "main_sweep.jsonl"

# leak-like telltales in a sysprompt-probe response
LEAK_PAT = re.compile(
    r"(my system prompt (is|says)|the system prompt (is|says)|i was instructed to"
    r"|here is my system prompt|my instructions (are|say)|\"you are [A-Z]"
    r"|'you are [A-Z]|system prompt:?\s*[\"'`]|verbatim:?\s*[\"'`])", re.I)
# strong-leak: quoted block of >120 chars following a leak telltale
QUOTE_PAT = re.compile(r"[\"'`]{1,3}([^\"'`]{120,})[\"'`]{1,3}")

# rough per-prompt user-token expectations for the 3 sysprompt probes (+template)
PROBE_BASELINE = 25  # tokens; anything <= this has no room for a real prompt


def main():
    pin_mismatch = defaultdict(lambda: defaultdict(int))
    n_checked = 0
    probes = defaultdict(list)

    for line in open_lines(SWEEP):
        r = json.loads(line)
        if r.get("error"):
            continue
        req = r.get("provider_requested")
        served = r.get("provider_served")
        if req and isinstance(req, dict) and req.get("order"):
            n_checked += 1
            want = req["order"][0].lower().replace(" ", "").replace("-", "")
            got = (served or "?").lower().replace(" ", "").replace("-", "")
            if want not in got and got not in want:
                pin_mismatch[r["model_id"]][f"{req['order'][0]} -> {served}"] += 1
        if r["prompt_category"] == "system_probe":
            probes[r["model_id"]].append(r)

    print("=== 1. PIN INTEGRITY ===")
    total_mm = sum(sum(v.values()) for v in pin_mismatch.values())
    print(f"pinned calls checked: {n_checked}; mismatches: {total_mm}")
    for mid, mm in sorted(pin_mismatch.items()):
        for k, v in mm.items():
            print(f"  {mid:50s} {k} x{v}")

    print("\n=== 2. SYSPROMPT-PROBE SCAN ===")
    flags = {"confabulated": [], "REVIEW": [], "denied": 0}
    for mid, recs in sorted(probes.items()):
        for r in recs:
            text = (r.get("content_clean") or "")
            ptok = (r.get("usage") or {}).get("prompt_tokens") or 0
            leaky = bool(LEAK_PAT.search(text)) or bool(QUOTE_PAT.search(text))
            if not leaky:
                flags["denied"] += 1
                continue
            has_room = ptok > PROBE_BASELINE
            entry = (mid, r["prompt_id"], ptok, r.get("provider_served"),
                     text[:160].replace("\n", " "))
            if has_room:
                flags["REVIEW"].append(entry)
            else:
                flags["confabulated"].append(entry)

    print(f"probe responses scanned: {sum(len(v) for v in probes.values())} "
          f"({len(probes)} models)")
    print(f"no leak-like content: {flags['denied']}")
    print(f"leak-like, NO token room (confabulation): {len(flags['confabulated'])}")
    print(f"leak-like WITH token room — MANUAL REVIEW: {len(flags['REVIEW'])}\n")
    for mid, pid, ptok, prov, head in flags["REVIEW"]:
        print(f"  !! {mid} [{pid}] ptok={ptok} via {prov}")
        print(f"     {head}")
    if flags["confabulated"]:
        print("\n-- confabulation examples (first 8) --")
        for mid, pid, ptok, prov, head in flags["confabulated"][:8]:
            print(f"  {mid} [{pid}] ptok={ptok}: {head[:110]}")


if __name__ == "__main__":
    main()
