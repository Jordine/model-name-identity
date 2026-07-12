"""Purge contaminated records from main_sweep.jsonl.

Removes:
  * records of hygiene-EXCLUDED models (models swept before their exclusion:
    the fake local/* quartet, mercury-2, mimo-v2.5-pro, hermes-4 pair, ...)
  * ALL records of the 6 pilot models — collected unpinned before provider
    hygiene existed; the resumable runner refills them pinned afterwards.

Original file is preserved as results/main_sweep_prepurge_backup.jsonl.
Run ONLY when no runner is writing. Usage: python -m sweep.purge_records
"""

import json
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SWEEP = ROOT / "results" / "main_sweep.jsonl"
BACKUP = ROOT / "results" / "main_sweep_prepurge_backup.jsonl"
HYGIENE = ROOT / "config" / "provider_hygiene.json"

PILOT_REFILL = {
    "moonshotai/kimi-k2.5", "openai/gpt-4o-mini", "deepseek/deepseek-r1-0528",
    "meta-llama/llama-3.2-3b-instruct", "z-ai/glm-4.7-flash", "qwen/qwen3.5-9b",
}


def main():
    hygiene = json.loads(HYGIENE.read_text())
    excluded = {mid for mid, v in hygiene.items() if v.get("exclude")}

    shutil.copy2(SWEEP, BACKUP)
    kept, dropped = [], Counter()
    for line in open(SWEEP, encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            dropped["unparseable"] += 1
            continue
        mid = r.get("model_id", "?")
        if mid in excluded:
            dropped[f"excluded:{mid}"] += 1
        elif mid in PILOT_REFILL:
            dropped[f"pilot-refill:{mid}"] += 1
        else:
            kept.append(line)

    with open(SWEEP, "w", encoding="utf-8") as f:
        f.writelines(kept)

    print(f"kept {len(kept)} records; dropped {sum(dropped.values())}:")
    for k, v in sorted(dropped.items()):
        print(f"  {v:5d}  {k}")
    print(f"backup -> {BACKUP}")


if __name__ == "__main__":
    main()
