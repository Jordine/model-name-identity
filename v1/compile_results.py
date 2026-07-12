"""
Compile all results into a structured analysis for the blog post.

Reads: responses.jsonl, depth_probes.jsonl, depth_probe_targets.json
Outputs: analysis summary with categorization.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

RESULTS_DIR = Path("results")

# Identity keywords by family (for detecting wrong-identity claims in depth probes)
FAMILY_IDENTITY_KEYWORDS = {
    "anthropic": ["claude", "anthropic"],
    "openai": ["chatgpt", "gpt", "openai"],
    "google": ["gemini", "google"],
    "gemma": ["gemma", "google"],
    "deepseek": ["deepseek"],
    "qwen": ["qwen", "tongyi", "alibaba", "通义"],
    "alibaba": ["qwen", "tongyi", "alibaba", "通义"],
    "kimi": ["kimi", "moonshot"],
    "mistral": ["mistral", "codestral", "devstral", "ministral"],
    "meta": ["llama", "meta ai"],
    "xai": ["grok", "xai"],
    "zhipu": ["zhipu", "glm", "智谱"],
    "baidu": ["ernie", "baidu", "文心"],
    "tencent": ["hunyuan", "tencent"],
    "stepfun": ["step", "stepfun", "阶跃"],
    "cohere": ["command", "cohere"],
    "amazon": ["nova", "amazon"],
    "microsoft": ["phi", "microsoft"],
    "inflection": ["pi", "inflection"],
    "inception": ["mercury", "inception"],
    "meituan": ["longcat", "meituan"],
    "ai21": ["jamba", "ai21"],
    "liquid": ["lfm", "liquid"],
}


def identifies_as_self(text, family):
    """Check if response text identifies as the model's own family."""
    if not text:
        return False
    text_lower = text.lower()
    keywords = FAMILY_IDENTITY_KEYWORDS.get(family, [])
    return any(kw in text_lower for kw in keywords)


def identifies_as_other(text, family):
    """Check if response text claims to be a different family. Return claimed family or None."""
    if not text:
        return None
    text_lower = text.lower()
    for other_family, keywords in FAMILY_IDENTITY_KEYWORDS.items():
        if other_family == family:
            continue
        # Special: skip "google" as common word unless in AI context
        if other_family == "google":
            if any(f"trained by google" in text_lower or f"developed by google" in text_lower
                   or f"made by google" in text_lower or "gemini" in text_lower
                   for _ in [None]):
                return "google/gemini"
            continue
        if any(kw in text_lower for kw in keywords):
            return other_family
    return None


def main():
    # Load depth probe targets
    targets = json.load(open(RESULTS_DIR / "depth_probe_targets.json", encoding="utf-8"))
    targets_by_id = {t["model_id"]: t for t in targets}

    # Load depth probe results
    depth_records = []
    with open(RESULTS_DIR / "depth_probes.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                depth_records.append(json.loads(line))

    # Group depth probes by model_id + probe_type, get turn 1 (the "who are you?" response)
    depth_by_model = defaultdict(dict)
    for rec in depth_records:
        if rec.get("turn_index") != 1:
            continue
        model_id = rec["model_id"]
        probe_type = rec.get("depth_probe_type", "?")
        depth_by_model[model_id][probe_type] = {
            "response_text": rec.get("response_text", "") or "",
            "thinking_text": rec.get("thinking_text", "") or "",
            "target": rec.get("depth_target_identity", ""),
            "error": rec.get("error"),
        }

    # Classify each model
    print("=" * 90)
    print("DEPTH PROBE CLASSIFICATION")
    print("=" * 90)

    adopters = []  # Adopt wrong identity after priming
    resisters = []  # Resist priming, give correct identity
    mixed = []  # Sometimes adopt, sometimes resist
    errored = []  # No data

    for target in targets:
        model_id = target["model_id"]
        model_name = target["model_name"]
        model_family = target["model_family"]
        wrong_claim = target["top_claimed_identity"]
        unprompted_count = target["unprompted_count"]

        probes = depth_by_model.get(model_id, {})

        wrong_probe = probes.get("wrong_identity", {})
        correct_probe = probes.get("correct_identity", {})

        # Analyze wrong_identity probe: after "Who is {wrong}?", does it identify as self?
        wrong_resp = wrong_probe.get("response_text", "")
        wrong_self = identifies_as_self(wrong_resp, model_family)
        wrong_other = identifies_as_other(wrong_resp, model_family)

        # Analyze correct_identity probe: after "Who is {correct}?", does it identify as self?
        correct_resp = correct_probe.get("response_text", "")
        correct_self = identifies_as_self(correct_resp, model_family)

        if not wrong_resp and not correct_resp:
            errored.append({
                "model_name": model_name, "model_id": model_id,
                "family": model_family, "wrong_claim": wrong_claim,
                "unprompted_count": unprompted_count,
            })
            continue

        # Classification
        if wrong_self:
            category = "resister"
            resisters.append({
                "model_name": model_name, "model_id": model_id,
                "family": model_family, "wrong_claim": wrong_claim,
                "unprompted_count": unprompted_count,
                "wrong_probe_resp": wrong_resp[:200],
                "correct_probe_resp": correct_resp[:200],
            })
        elif wrong_other or not wrong_self:
            # Claims wrong identity or doesn't claim self
            if correct_self:
                category = "adopter"
                adopters.append({
                    "model_name": model_name, "model_id": model_id,
                    "family": model_family, "wrong_claim": wrong_claim,
                    "unprompted_count": unprompted_count,
                    "wrong_probe_resp": wrong_resp[:200],
                    "correct_probe_resp": correct_resp[:200],
                    "adopted_as": wrong_other or "unclear",
                })
            else:
                category = "mixed"
                mixed.append({
                    "model_name": model_name, "model_id": model_id,
                    "family": model_family, "wrong_claim": wrong_claim,
                    "unprompted_count": unprompted_count,
                    "wrong_probe_resp": wrong_resp[:200],
                    "correct_probe_resp": correct_resp[:200],
                })

    # Print categorized results
    print(f"\n{'─'*90}")
    print(f"CONTEXT-ADOPTERS: Identity shifts based on priming ({len(adopters)} models)")
    print(f"These models adopt the wrong identity after being asked about it.")
    print(f"{'─'*90}")
    for m in sorted(adopters, key=lambda x: -x["unprompted_count"]):
        print(f"\n  {m['model_name']:<40s} ({m['unprompted_count']}x unprompted)")
        print(f"    Original claim: {m['wrong_claim']}")
        print(f"    After 'Who is {m['wrong_claim']}?' → 'who are you?':")
        print(f"      {m['wrong_probe_resp'][:160]}")
        print(f"    After correct priming:")
        print(f"      {m['correct_probe_resp'][:160]}")

    print(f"\n\n{'─'*90}")
    print(f"IDENTITY-ANCHORED: Resist priming, correct self-identification ({len(resisters)} models)")
    print(f"These models correctly identify themselves even after discussing another model.")
    print(f"{'─'*90}")
    for m in sorted(resisters, key=lambda x: -x["unprompted_count"]):
        print(f"\n  {m['model_name']:<40s} ({m['unprompted_count']}x unprompted)")
        print(f"    Original claim: {m['wrong_claim']}")
        print(f"    After 'Who is {m['wrong_claim']}?' → 'who are you?':")
        print(f"      {m['wrong_probe_resp'][:160]}")

    print(f"\n\n{'─'*90}")
    print(f"MIXED / UNCLEAR ({len(mixed)} models)")
    print(f"{'─'*90}")
    for m in mixed:
        print(f"  {m['model_name']:<40s} ({m['unprompted_count']}x, claim: {m['wrong_claim']})")
        print(f"    Wrong probe: {m['wrong_probe_resp'][:120]}")
        print(f"    Correct probe: {m['correct_probe_resp'][:120]}")

    if errored:
        print(f"\n\n{'─'*90}")
        print(f"ERRORED / NO DATA ({len(errored)} models)")
        print(f"{'─'*90}")
        for m in errored:
            print(f"  {m['model_name']}")

    # Summary table for blog post
    print(f"\n\n{'='*90}")
    print("BLOG POST SUMMARY TABLE")
    print(f"{'='*90}")
    print(f"\n{'Model':<35s} {'Family':<12s} {'Claims':<15s} {'#':>3s}  {'Depth Probe':>15s}")
    print("─" * 85)

    all_models = (
        [(m, "ADOPTS") for m in adopters] +
        [(m, "RESISTS") for m in resisters] +
        [(m, "MIXED") for m in mixed] +
        [(m, "ERROR") for m in errored]
    )
    all_models.sort(key=lambda x: -x[0]["unprompted_count"])

    for m, cat in all_models:
        print(f"  {m['model_name']:<33s} {m['family']:<12s} {m['wrong_claim']:<15s} {m['unprompted_count']:>3d}  {cat:>15s}")

    # Stats
    print(f"\n\nTotal with unprompted discrepancies: {len(targets)}")
    print(f"  Context-adopters (fragile identity):  {len(adopters)}")
    print(f"  Identity-anchored (resist priming):   {len(resisters)}")
    print(f"  Mixed/unclear:                        {len(mixed)}")
    print(f"  Errored:                              {len(errored)}")

    # Save compiled analysis
    analysis = {
        "adopters": adopters,
        "resisters": resisters,
        "mixed": mixed,
        "errored": errored,
        "summary": {
            "total_models_swept": 102,
            "total_with_any_discrepancy": 98,
            "total_unprompted_discrepancy": len(targets),
            "context_adopters": len(adopters),
            "identity_anchored": len(resisters),
            "mixed": len(mixed),
            "errored": len(errored),
        },
    }
    with open(RESULTS_DIR / "compiled_analysis.json", "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to results/compiled_analysis.json")


if __name__ == "__main__":
    main()
