"""tp-aware driver. tp=1 models run 2-at-a-time (one per GPU); tp=2 models run
alone on both GPUs. Resumable (skips non-empty outputs). Each model's entry may
set "backend":"transformers" for archs vLLM can't load.

  python run_parallel.py
"""
import json
import os
import queue
import subprocess
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "results_local")
os.makedirs(OUTDIR, exist_ok=True)

models = [json.loads(l) for l in open(os.path.join(HERE, "local_manifest2.jsonl")) if l.strip()]
todo = []
for m in models:
    out = os.path.join(OUTDIR, m["hf_id"].replace("/", "__") + ".jsonl")
    if os.path.exists(out) and os.path.getsize(out) > 0:
        print(f"SKIP {m['hf_id']} (done)", flush=True)
    else:
        todo.append((m, out))

lock = threading.Lock()
results = {"ok": [], "fail": []}


def run_one(m, out, gpu_env, tp):
    hid = m["hf_id"]
    with lock:
        print(f"[{gpu_env}] START {hid} tp{tp} ({m.get('backend','vllm')})", flush=True)
    cmd = [sys.executable, os.path.join(HERE, "gen_local.py"),
           "--model", hid, "--registry", json.dumps(m["entry"], ensure_ascii=False),
           "--out", out, "--tp", str(tp), "--samples", "5",
           "--max-model-len", "8192", "--max-tokens", str(m.get("max_tokens", 1200)),
           "--backend", m.get("backend", "vllm")]
    # isolate per-worker vLLM state so two concurrent engines don't collide
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_env,
           "VLLM_CACHE_ROOT": f"/root/.cache/vllm_{gpu_env.replace(',', '_')}"}
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    with lock:
        if r.returncode == 0:
            print(f"[{gpu_env}] OK   {hid}\n{r.stdout[-500:]}", flush=True)
            results["ok"].append(hid)
        else:
            print(f"[{gpu_env}] FAIL {hid}\n{r.stderr[-1500:]}", flush=True)
            results["fail"].append(hid)
            if os.path.exists(out) and os.path.getsize(out) == 0:
                os.remove(out)


# Phase 1: tp=1 models, one per GPU (2 concurrent)
single = queue.Queue()
for m, out in todo:
    if m.get("tp", 1) == 1:
        single.put((m, out))


def worker(gpu):
    while True:
        try:
            m, out = single.get_nowait()
        except queue.Empty:
            return
        run_one(m, out, str(gpu), 1)
        single.task_done()


print(f"=== Phase 1: {single.qsize()} tp=1 models, 2 per pass ===", flush=True)
threads = [threading.Thread(target=worker, args=(g,)) for g in (0, 1)]
for t in threads:
    t.start()
for t in threads:
    t.join()

# Phase 2: tp=2 models, sequential on both GPUs
big = [(m, out) for m, out in todo if m.get("tp", 1) == 2]
print(f"=== Phase 2: {len(big)} tp=2 models, sequential ===", flush=True)
for m, out in big:
    run_one(m, out, "0,1", 2)

print(f"\n===== PARALLEL DONE  ok={results['ok']}  fail={results['fail']} =====", flush=True)
