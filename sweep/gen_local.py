"""Local generation for open models that can't be read raw via any API
(gpt-oss, OLMo, granite, ...). Same battery as the API sweep, TWO conditions,
judge-compatible JSONL (main_sweep schema). Records the template-injected system
prompt + its token count (the rung-2 template identity).

Backends:  vllm (default, fast)  |  transformers (for archs vLLM can't load).

  python gen_local.py --model <hf_id> --registry '<json>' --out o.jsonl --tp 1 [--backend transformers]
"""
import argparse
import json
import os
import re
import shutil
import sys
import time

from transformers import AutoTokenizer

try:  # package import (repo); falls back to flat layout when copied onto a GPU box
    from . import prompts as P
    from .clean_render import render_clean, render_shipped, scaffold_flags
except ImportError:
    import prompts as P
    from clean_render import render_clean, render_shipped, scaffold_flags

HARMONY_FINAL = re.compile(r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)", re.S)
HARMONY_ANALYSIS = re.compile(r"<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|<\|start\|>|$)", re.S)
THINK = re.compile(r"<think>(.*?)</think>", re.S)


def split_reasoning(text):
    if text is None:
        return None, None
    if "<|channel|>" in text:
        fin = HARMONY_FINAL.search(text)
        ana = HARMONY_ANALYSIS.search(text)
        content = fin.group(1).strip() if fin else re.sub(r"<\|[^>]*\|>", " ", text).strip()
        return content, (ana.group(1).strip() if ana else None)
    m = THINK.search(text)
    if m:
        return THINK.sub("", text).strip(), m.group(1).strip()
    return text.strip(), None


def render(tok, model_id, user, condition):
    # clean = no injected identity (harmony gets model_identity=' '); shipped = default template
    return (render_clean(tok, model_id, user) if condition == "clean"
            else render_shipped(tok, model_id, user))


def gen_vllm(model_id, rendered, tp, max_model_len, max_tokens):
    from vllm import LLM, SamplingParams
    # enforce_eager=True skips torch.compile — avoids the shared compile-cache
    # collision when two engines start concurrently (one per GPU).
    llm = LLM(model=model_id, tensor_parallel_size=tp, trust_remote_code=True,
              max_model_len=max_model_len, gpu_memory_utilization=0.92, dtype="auto",
              enforce_eager=True)
    outs = llm.generate(rendered, SamplingParams(temperature=0.7, max_tokens=max_tokens))
    return [o.outputs[0].text for o in outs]


def gen_transformers(model_id, rendered, max_tokens, tok):
    import torch
    from transformers import AutoModelForCausalLM
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    model.eval()
    texts = [None] * len(rendered)
    i, bs = 0, 8
    while i < len(rendered):
        batch = rendered[i:i + bs]
        try:
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=8192, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=max_tokens, do_sample=True,
                                   temperature=0.7, top_p=0.9,
                                   pad_token_id=tok.pad_token_id or tok.eos_token_id)
            for j in range(len(batch)):
                texts[i + j] = tok.decode(g[j][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            i += bs
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            print(f"[gen_transformers] OOM at i={i}, batch -> {bs}", flush=True)
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--backend", default="vllm")
    ap.add_argument("--conditions", default="clean,shipped")
    args = ap.parse_args()

    model = json.loads(args.registry)
    conditions = args.conditions.split(",")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # HARD GATE: never generate a "clean" run whose prompt still names a model/company.
    if "clean" in conditions:
        flags = scaffold_flags(render_clean(tok, args.model, "Who are you?"), "Who are you?")
        if flags:
            print(f"[gen_local] ABORT {model['id']}: clean prompt still names {flags}", flush=True)
            sys.exit(2)

    battery = P.core_prompts(model) + P.cross_prompts(model)
    reqs, rendered = [], []
    for cond in conditions:
        for it in battery:
            for s in range(args.samples):
                reqs.append((cond, it, s))
                rendered.append(render(tok, args.model, it["content"], cond))

    if args.backend == "transformers":
        texts = gen_transformers(args.model, rendered, args.max_tokens, tok)
    else:
        texts = gen_vllm(args.model, rendered, args.tp, args.max_model_len, args.max_tokens)

    hygiene = {c: (lambda r: (r, len(tok.encode(r)), scaffold_flags(r, "hi")))(render(tok, args.model, "hi", c))
               for c in conditions}
    with open(args.out, "w", encoding="utf-8") as f:
        for (cond, it, s), raw in zip(reqs, texts):
            clean, reasoning = split_reasoning(raw)
            f.write(json.dumps({
                "model_id": model["id"], "model_name": model["name"],
                "family": model["family"], "expected_identity": model["expected_identity"],
                "aliases": model.get("aliases", []),
                "prompt_id": it["id"], "prompt_category": it["category"],
                "sample_idx": s, "turn_index": 0, "condition": cond,
                "resume_key": f'{model["id"]}::{it["id"]}::{s}::{cond}',
                "messages_sent": [{"role": "user", "content": it["content"]}],
                "content": raw, "content_clean": clean, "reasoning": reasoning,
                "hi_prompt_tokens": hygiene[cond][1], "hi_scaffold_flags": hygiene[cond][2],
                "backend": args.backend,
                "run_type": "local", "error": None, "ts": time.time(),
            }, ensure_ascii=False) + "\n")
    print(f"[gen_local] {model['id']} ({args.backend}): {len(reqs)} records -> {args.out}")
    for c, (hp, ht, fl) in hygiene.items():
        print(f"  hygiene[{c}]: bare-'hi' = {ht} tokens, scaffold names = {fl}")
    # free disk: drop this model's weights now that generation is done
    shutil.rmtree(os.path.expanduser("~/.cache/huggingface/hub/models--" + args.model.replace("/", "--")),
                  ignore_errors=True)


if __name__ == "__main__":
    main()
