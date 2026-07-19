"""Driver: run gen_local.py for every model in the manifest, one subprocess each
(so GPU memory frees between models). Resumable (skips non-empty outputs).

On the vast box:  python run_all_local.py
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "local_manifest.jsonl")
OUTDIR = os.path.join(HERE, "results_local")
os.makedirs(OUTDIR, exist_ok=True)

models = [json.loads(l) for l in open(MANIFEST) if l.strip()]
done, failed = [], []
for m in models:
    hid, tp, entry = m["hf_id"], m["tp"], m["entry"]
    out = os.path.join(OUTDIR, hid.replace("/", "__") + ".jsonl")
    if os.path.exists(out) and os.path.getsize(out) > 0:
        print(f"SKIP {hid} (already done)", flush=True)
        done.append(hid)
        continue
    print(f"\n{'='*70}\n=== {hid}  (tp{tp})\n{'='*70}", flush=True)
    t0 = time.time()
    r = subprocess.run(
        [sys.executable, os.path.join(HERE, "gen_local.py"),
         "--model", hid, "--registry", json.dumps(entry, ensure_ascii=False),
         "--out", out, "--tp", str(tp), "--samples", "5",
         "--max-model-len", "8192", "--max-tokens", "1200"],
        capture_output=True, text=True,
    )
    print(r.stdout[-3000:], flush=True)
    if r.returncode != 0:
        print("STDERR tail:\n" + r.stderr[-3000:], flush=True)
        failed.append(hid)
        # clean partial output so a rerun retries
        if os.path.exists(out) and os.path.getsize(out) == 0:
            os.remove(out)
    else:
        done.append(hid)
    print(f"  rc={r.returncode}  {time.time()-t0:.0f}s", flush=True)

print(f"\n===== DONE {len(done)}/{len(models)} ok; failed={failed} =====", flush=True)
