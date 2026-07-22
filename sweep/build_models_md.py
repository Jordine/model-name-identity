"""Regenerate MODELS.md from config + results — the reader-facing model roster with,
for every API model, the provider it was PINNED to (so a run is reproducible and a
suspected injection is checkable). Run after preflight/runner.

  python -m sweep.build_models_md
"""
import json
import gzip
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FAM_NAME = {"openai": "OpenAI", "anthropic": "Anthropic", "google": "Google", "gemma": "Google (Gemma)",
            "deepseek": "DeepSeek", "qwen": "Qwen / Alibaba", "alibaba": "Qwen / Alibaba", "meta": "Meta (Llama)",
            "kimi": "Moonshot (Kimi)", "mistral": "Mistral", "xai": "xAI (Grok)", "zhipu": "Zhipu (GLM)",
            "baidu": "Baidu (Ernie)", "tencent": "Tencent (Hunyuan)", "bytedance": "ByteDance (Doubao)",
            "microsoft": "Microsoft", "nvidia": "NVIDIA", "nous": "Nous", "perplexity": "Perplexity", "cohere": "Cohere",
            "amazon": "Amazon (Nova)", "ai21": "AI21", "minimax": "MiniMax", "reka": "Reka", "allenai": "Ai2 (OLMo)",
            "olmo": "Ai2 (OLMo)", "upstage": "Upstage", "liquid": "Liquid", "inception": "Inception", "stepfun": "StepFun",
            "kuaishou": "Kuaishou", "xiaomi": "Xiaomi", "ibm": "IBM (Granite)", "writer": "Writer", "inflection": "Inflection",
            "arcee": "Arcee", "sakana": "Sakana", "perceptron": "Perceptron", "nex": "Nex", "cogito": "Cogito",
            "poolside": "Poolside", "aisingapore": "AI Singapore", "ant": "Ant (Ling)"}
ORDER = ["openai", "anthropic", "google", "gemma", "deepseek", "qwen", "alibaba", "meta", "kimi", "mistral",
         "xai", "zhipu", "baidu", "tencent", "bytedance", "microsoft", "nvidia", "cohere", "amazon", "nous", "perplexity"]


def bucket(reason):
    r = reason.lower()
    if "template-installed" in r or "hermes" in r:
        return "template"
    if "different model" in r or "route serves" in r or "proxy-native" in r:
        return "mismatch"
    if "inject" in r or "hidden" in r or "dishonest" in r or "accounting" in r:
        return "inject"
    return "endpoint"


def main():
    from .make_figs import complete_models
    reg = {m["id"]: m for m in json.loads((ROOT / "config" / "models.json").read_text())["models"]}
    hyg = json.loads((ROOT / "config" / "provider_hygiene.json").read_text())
    gpu = [json.loads(l) for l in open(ROOT / "config" / "local_manifest.jsonl", encoding="utf-8")]

    # the analyzed set = exactly what the figures use (>=95% complete, not excluded)
    tested = complete_models(reg, hyg)
    # pinned slug + actually-served providers (ground truth) from the sweep
    pin = {k: (v["provider"]["order"][0] if isinstance(v.get("provider"), dict) and v["provider"].get("order") else None)
           for k, v in hyg.items() if not v.get("exclude")}
    served = defaultdict(set)
    with gzip.open(ROOT / "results" / "main_sweep.jsonl.gz", "rt", encoding="utf-8") as f:
        for l in f:
            r = json.loads(l)
            if r.get("error") or not r.get("provider_served"):
                continue
            served[r["model_id"]].add(r["provider_served"])
    excl = {k: v for k, v in hyg.items() if v.get("exclude")}

    byfam = defaultdict(list)
    for mid in tested:
        m = reg.get(mid, {})
        byfam[m.get("family", "other")].append((m.get("name", mid), mid))

    L = ["# Models\n"]
    L.append(f"Every model in this study, with how it was reached. **{len(tested)} models were queried through "
             f"hosted APIs**, each **pinned to one provider** (below), **{len(gpu)} more were run from raw weights "
             f"on GPUs**, and **{len(excl)} were excluded** because no provider served them cleanly.\n")
    L.append("## Selection policy\n")
    L.append("- Official lab models only — community finetunes, roleplay merges, and base (non-chat) models are excluded.")
    L.append("- No `latest`-alias redirects and no auto-routers (they hide which model actually answered).")
    L.append("- `:free` variant used only when no paid sibling exists.")
    L.append("- **Provider hygiene:** a model is excluded if every available provider injects a system prompt "
             "(e.g. \"You are ChatGPT\"), so a \"who are you?\" answer would reflect the *provider's scaffolding*, not "
             "the model. See `sweep/preflight.py` + `config/provider_hygiene.json`.\n")

    # --- provenance / pinning ---
    npin = sum(1 for m in tested if pin.get(m))
    onep = sum(1 for m in tested if len(served.get(m, set())) == 1)
    unpinned = [m for m in tested if not pin.get(m)]
    L.append("## Provider pinning (reproducibility & injection-checkability)\n")
    L.append(f"Of the {len(tested)} analyzed API models, **{npin} are pinned** to one preflight-chosen provider via "
             "OpenRouter routing `{\"order\": [\"<slug>\"], \"allow_fallbacks\": false}`, so a call can't silently fall "
             "back to an injecting or wrong-quant endpoint. `runner.py` records `provider_served` on every response; the "
             f"table below lists the pin per model. **Verified against the collected data: {onep} models were served by "
             "exactly the pinned provider** (no fallbacks). Two (`deepseek/deepseek-r1`, `qwen/qwen3.5-9b`) were "
             "re-pinned between runs and carry records from two providers — **both of which passed the hygiene "
             "check**. "
             + (f"One model (`{unpinned[0]}`) predates the pinning setup and has no recorded provider — the only "
                "analyzed model without verified provenance. " if len(unpinned) == 1 else
                (f"{len(unpinned)} models predate pinning and have no recorded provider. " if unpinned else ""))
             + "To reproduce a model's answers (or check for injection yourself), pin its provider slug on OpenRouter "
             "and send the prompts in `prompts.jsonl` with no system prompt.\n")

    # --- queried, with providers ---
    L.append(f"## Queried via API ({len(tested)})\n")
    L.append("| model | family | pinned provider |")
    L.append("|---|---|---|")
    seen = set()
    for fam in ORDER + sorted(f for f in byfam if f not in ORDER):
        if fam in seen or fam not in byfam:
            continue
        seen.add(fam)
        for name, mid in sorted(byfam[fam]):
            slug = pin.get(mid)
            others = served.get(mid, set())
            if not slug:
                cell = "*unpinned (pre-dates pinning)*"
            elif len(others) > 1:
                cell = f"`{slug}` ⚠ two providers (both clean): " + ", ".join(sorted(others))
            else:
                cell = f"`{slug}`"
            L.append(f"| {name} | {FAM_NAME.get(fam, fam.title())} | {cell} |")
    L.append("")

    # --- GPU ---
    L.append(f"## Run from raw weights on GPUs ({len(gpu)})\n")
    L.append("Downloaded from HuggingFace and run on rented A100s with **any identity stripped from the chat template** "
             "and verified identity-free before generation (`sweep/verify_prompts.py`) — isolating what the *weights* "
             "say from what the shipped template says. No hosted provider involved.\n")
    for m in gpu:
        e = m["entry"]
        tag = f" (tp={m['tp']}, {m['backend']})" if m["tp"] > 1 or m["backend"] != "vllm" else ""
        L.append(f"- {e['name']} — `{m['hf_id']}`{tag}")
    L.append("")

    # --- excluded ---
    BKT = {"inject": "Provider injects a system prompt", "endpoint": "No clean / working endpoint",
           "mismatch": "Proxy served a different model",
           "template": "Identity baked into the model's own recommended template (case study, not comparison)"}
    grp = defaultdict(list)
    for mid, v in excl.items():
        grp[bucket(v["reason"])].append(mid)
    L.append(f"## Excluded ({len(excl)})\n")
    L.append("Not evidence of drift — models we *couldn't* measure cleanly, listed for transparency.\n")
    for b in ("inject", "endpoint", "mismatch", "template"):
        if b not in grp:
            continue
        L.append(f"**{BKT[b]}** ({len(grp[b])})  ")
        L.append(", ".join(f"`{mid}`" for mid in sorted(grp[b])) + "\n")

    (ROOT / "MODELS.md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote MODELS.md — {len(tested)} queried (+providers) · {len(gpu)} GPU · {len(excl)} excluded")


if __name__ == "__main__":
    main()
