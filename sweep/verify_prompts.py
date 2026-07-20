"""VERIFY — for every model in the manifest — that render_clean yields an
identity-free prompt, BEFORE any generation. Prints per model: token counts,
any model/company names found in the clean vs shipped scaffolding, and the clean
scaffold itself. Exits nonzero if ANY clean prompt still names a model/company.

  python -m sweep.verify_prompts    # from the repo
  python verify_prompts.py          # flat, on the GPU box
"""
import json
import os
import sys

from transformers import AutoTokenizer

try:  # package import (repo); falls back to flat layout when copied onto a GPU box
    from .clean_render import render_clean, render_shipped, scaffold_flags, is_harmony
except ImportError:
    from clean_render import render_clean, render_shipped, scaffold_flags, is_harmony

Q = "Who are you? What model are you?"


def load_manifest():
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.join(here, "..", "config", "local_manifest.jsonl"),
              os.path.join(here, "local_manifest.jsonl")):
        if os.path.exists(p):
            return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    raise SystemExit("local_manifest.jsonl not found (config/ or alongside this script)")


def main():
    models = load_manifest()
    bad = []
    for m in models:
        hid = m["hf_id"]
        try:
            tok = AutoTokenizer.from_pretrained(hid, trust_remote_code=True)
        except Exception as e:
            print(f"[TOKENIZER FAIL] {hid}: {str(e)[:80]}")
            bad.append(hid)
            continue
        clean = render_clean(tok, hid, Q)
        shipped = render_shipped(tok, hid, Q)
        cf, sf = scaffold_flags(clean, Q), scaffold_flags(shipped, Q)
        ct, st = len(tok.encode(clean)), len(tok.encode(shipped))
        status = "CLEAN" if not cf else "*** DIRTY ***"
        print(f"{hid}  [{status}]")
        print(f"   clean={ct}tok flags={cf}   shipped={st}tok flags={sf}   harmony={is_harmony(tok, hid)}")
        print(f"   clean scaffold: {clean.replace(Q, '«QUESTION»')[:200]!r}")
        if cf:
            bad.append(hid)
    print("\n" + ("===== ALL CLEAN — safe to run =====" if not bad
                  else f"===== DIRTY / FAILED: {bad} ====="))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
