"""Build config/models.json from the live OpenRouter catalog + v1 curation.

Policy:
  * official lab models only — community finetunes / roleplay merges excluded
  * no base models, no `latest`-alias redirects (~org/...), no auto-router
  * `:free` variant only when no paid sibling exists
  * v1 entries keep their curated expected_identity/family (auto=False)
  * v1's excluded-for-hidden-sysprompt models are INCLUDED here with
    recheck_hygiene=True — the preflight re-decides, provider landscape moves
  * new models get an auto-derived expected_identity (auto=True) for curation

Usage: python -m sweep.registry_build [--catalog /tmp/or_catalog.json]
"""

import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "v1"))
from models import MODELS as V1_MODELS, EXCLUDED_MODELS as V1_EXCLUDED  # noqa: E402

CATALOG_URL = "https://openrouter.ai/api/v1/models"
OUT_PATH = ROOT / "config" / "models.json"

# Community finetune / roleplay / merge orgs — not lab identities.
# (Interesting as a deliberate-identity-transplant appendix someday, not core.)
COMMUNITY_ORGS = {
    "thedrummer", "sao10k", "anthracite-org", "undi95", "gryphe", "mancer",
    "cognitivecomputations", "aion-labs", "neversleep", "alpindale",
    "sophosympatheia", "nothingiisreal", "eva-unit-01", "raifle",
    "latitudegames", "scb10x", "pygmalionai", "jondurbin", "teknium",
}
# Non-conversational specialty tools (code-apply/edit engines, embeddings).
TOOL_ORGS = {"relace", "morph", "openrouter"}

ORG_FAMILY = {
    "anthropic": "anthropic", "openai": "openai", "google": "google",
    "meta-llama": "meta", "mistralai": "mistral", "deepseek": "deepseek",
    "qwen": "qwen", "moonshotai": "kimi", "x-ai": "xai", "z-ai": "zhipu",
    "baidu": "baidu", "tencent": "tencent", "stepfun": "stepfun",
    "cohere": "cohere", "amazon": "amazon", "microsoft": "microsoft",
    "inflection": "inflection", "inception": "inception", "meituan": "meituan",
    "ai21": "ai21", "liquid": "liquid", "alibaba": "alibaba",
    "minimax": "minimax", "bytedance-seed": "bytedance", "bytedance": "bytedance",
    "nvidia": "nvidia", "allenai": "allenai", "ibm-granite": "ibm",
    "upstage": "upstage", "writer": "writer", "xiaomi": "xiaomi",
    "perplexity": "perplexity", "nousresearch": "nous", "rekaai": "reka",
    "sakana": "sakana", "kwaipilot": "kuaishou", "inclusionai": "ant",
    "arcee-ai": "arcee", "poolside": "poolside", "deepcogito": "cogito",
    "nex-agi": "nex", "swiss-ai": "swiss-ai", "tngtech": "tng",
    "opengvlab": "opengvlab", "shisa-ai": "shisa", "tii": "tii",
    "featherless": "featherless", "agentica-org": "agentica",
    "arliai": "arliai", "aetherwiing": "aetherwiing", "01-ai": "01ai",
    "openchat": "openchat", "huggingfaceh4": "huggingface",
}

# Names that models legitimately carry besides their own brand — layered
# identities from open-weight ancestry (distills, licensed derivatives).
ANCESTRY_ALIASES = [
    (re.compile(r"distill.*llama|llama.*nemotron|nemotron.*llama", re.I), ["llama", "meta"]),
    (re.compile(r"distill.*qwen", re.I), ["qwen", "alibaba"]),
    (re.compile(r"^perplexity/", re.I), ["llama"]),        # sonar lineage
    (re.compile(r"deepseek.*r1t", re.I), ["deepseek"]),    # tng chimera
    (re.compile(r"^nousresearch/hermes", re.I), ["llama", "meta"]),
    (re.compile(r"^microsoft/wizardlm", re.I), ["mistral", "mixtral"]),
    (re.compile(r"^deepcogito/", re.I), ["deepseek", "llama"]),
]

# ---------------------------------------------------------------------------
# Curation: prune near-duplicates and non-conversational variants.
# Explicit and documented > clever. Reviewed 2026-07-12.
# ---------------------------------------------------------------------------
PRUNE_PATTERNS = [
    (re.compile(r"-20\d{2}-\d{2}-\d{2}$"), "dated snapshot, undated sibling kept"),
    (re.compile(r"gpt-3\.5-turbo-(0613|16k|instruct)"), "legacy 3.5 variant"),
    (re.compile(r"gpt-4-turbo-preview$"), "dated alias"),
    (re.compile(r"search-preview"), "web-search variant"),
    (re.compile(r"(voxtral|-audio|-tts|transcribe|realtime)"), "audio modality"),
    (re.compile(r"(llama-guard|content-safety)"), "safety classifier, not assistant"),
    (re.compile(r"(^|/)ui-tars"), "GUI agent"),
    (re.compile(r"-fast$"), "serving-tier duplicate (claude -fast)"),
    (re.compile(r"gemini-2\.5-pro-preview"), "dated preview, GA sibling kept"),
    (re.compile(r"customtools"), "tool-variant duplicate"),
    (re.compile(r"gemini-3\.1-flash-lite-preview"), "preview, GA sibling kept"),
    (re.compile(r"grok-(build|4\.20-multi-agent)"), "agent-product variant"),
    (re.compile(r"deepseek-v3\.2-exp$"), "experimental, GA sibling kept"),
    (re.compile(r"qwen-plus-2025-07-28"), "superseded dated plus"),
    (re.compile(r"hy3-preview$"), "preview, GA sibling kept"),
    # vision-language variants where a text sibling exists (keeps ernie-vl via override)
    (re.compile(r"(qwen3-vl|qwen2\.5-vl|glm-[45](\.\d)?v(-turbo)?$|-vl\b|vision-instruct)"), "VL variant, text sibling kept"),
]
PRUNE_IDS = {
    # openai codex line: keep newest (gpt-5.3-codex) as the code-specialist rep
    "openai/gpt-5-codex": "codex line, keep 5.3 only",
    "openai/gpt-5.1-codex": "codex line, keep 5.3 only",
    "openai/gpt-5.1-codex-max": "codex line, keep 5.3 only",
    "openai/gpt-5.1-codex-mini": "codex line, keep 5.3 only",
    "openai/gpt-5.2-codex": "codex line, keep 5.3 only",
    "openai/codex-mini-latest": "codex line, keep 5.3 only",
    # -pro compute tiers: same lineage, ~10x price, identity-equivalent
    "openai/gpt-5-pro": "pro compute tier",
    "openai/gpt-5.2-pro": "pro compute tier",
    "openai/gpt-5.4-pro": "pro compute tier",
    "openai/gpt-5.5-pro": "pro compute tier",
    # perplexity: keep sonar + sonar-pro; research/search tiers hit the web per call
    "perplexity/sonar-deep-research": "agentic research product",
    "perplexity/sonar-pro-search": "agentic search product",
    "perplexity/sonar-reasoning-pro": "keep 2 sonar reps",
    # minimax mid-gens superseded; keep 01 (anchor), m2-her, m2.5, m2.7, m3
    "minimax/minimax-m1": "superseded mid-gen",
    "minimax/minimax-m2": "superseded mid-gen",
    "minimax/minimax-m2.1": "superseded mid-gen",
    "mistralai/mistral-large": "ambiguous alias, dated 2407 kept as anchor",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free": "omni modality",
}
KEEP_OVERRIDE = {
    "baidu/ernie-4.5-vl-424b-a47b",  # only remaining baidu model on OR
}


def fetch_catalog(path: str | None) -> list[dict]:
    if path and Path(path).exists():
        return json.loads(Path(path).read_text())["data"]
    with urllib.request.urlopen(CATALOG_URL, timeout=30) as r:
        return json.loads(r.read())["data"]


def is_chat_text(m: dict) -> bool:
    arch = m.get("architecture") or {}
    mod = arch.get("modality") or ""
    out = arch.get("output_modalities") or []
    if out and "text" not in out:
        return False
    if mod and not mod.endswith("->text"):
        return False
    return True


def display_to_identity(or_name: str) -> str:
    """'NVIDIA: Llama 3.3 Nemotron Super 49B V1.5' -> the part after the colon."""
    name = or_name.split(":", 1)[1].strip() if ":" in or_name else or_name.strip()
    # strip pure-suffix noise
    name = re.sub(r"\s*\((?:free|beta|preview|self-moderated)\)\s*$", "", name, flags=re.I)
    return name


FAMILY_EXTRA_ALIASES = {
    "openai": ["chatgpt", "openai"], "anthropic": ["claude", "anthropic"],
    "google": ["gemini", "google"], "gemma": ["google", "gemma"],
    "qwen": ["alibaba", "tongyi", "qwen"], "kimi": ["kimi", "moonshot"],
    "zhipu": ["glm", "chatglm", "zhipu", "z.ai"], "meta": ["llama", "meta"],
    "xai": ["grok", "xai"], "baidu": ["ernie", "wenxin", "baidu"],
    "tencent": ["hunyuan", "tencent"], "bytedance": ["doubao", "seed", "bytedance"],
    "deepseek": ["deepseek"], "mistral": ["mistral"],
}


def derive_aliases(or_id: str, family: str, identity: str) -> list[str]:
    aliases = {family, identity.lower()}
    aliases.update(w for w in re.split(r"[\s\-/]+", identity.lower()) if len(w) > 2 and not w[0].isdigit())
    aliases.update(FAMILY_EXTRA_ALIASES.get(family, []))
    for pat, extra in ANCESTRY_ALIASES:
        if pat.search(or_id):
            aliases.update(extra)
    return sorted(a for a in aliases if a)


def build(catalog_path: str | None = None) -> dict:
    catalog = fetch_catalog(catalog_path)
    v1_by_id = {m["id"]: m for m in V1_MODELS}
    v1_excl_by_id = {m["id"]: m for m in V1_EXCLUDED}

    kept, skipped = [], {"community": [], "tool": [], "alias": [], "base": [],
                         "nonchat": [], "free_dupe": [], "curated": []}
    ids_seen = {m["id"] for m in catalog}

    for m in sorted(catalog, key=lambda x: x["id"]):
        mid = m["id"]
        org = mid.split("/")[0]

        if org.startswith("~"):
            skipped["alias"].append(mid); continue
        if org in COMMUNITY_ORGS:
            skipped["community"].append(mid); continue
        if org in TOOL_ORGS:
            skipped["tool"].append(mid); continue
        if not is_chat_text(m):
            skipped["nonchat"].append(mid); continue
        if re.search(r"[-/:]base$|-base-", mid):
            skipped["base"].append(mid); continue
        if mid.endswith(":free") and mid[:-5] in ids_seen:
            skipped["free_dupe"].append(mid); continue
        if mid not in KEEP_OVERRIDE:
            reason = PRUNE_IDS.get(mid)
            if reason is None:
                for pat, r in PRUNE_PATTERNS:
                    if pat.search(mid):
                        reason = r
                        break
            if reason:
                skipped["curated"].append(f"{mid}  ({reason})"); continue

        v1_entry = v1_by_id.get(mid)
        v1_excl = v1_excl_by_id.get(mid)
        family = ORG_FAMILY.get(org, org)

        if v1_entry:
            identity, name = v1_entry["expected_identity"], v1_entry["name"]
            family, auto = v1_entry["family"], False
        elif v1_excl:
            identity, name = v1_excl["expected_identity"], v1_excl["name"]
            family, auto = v1_excl["family"], False
        else:
            identity = display_to_identity(m.get("name") or mid)
            name, auto = identity, True

        pricing = m.get("pricing") or {}
        supported = m.get("supported_parameters") or []
        entry = {
            "id": mid,
            "name": name,
            "family": family,
            "expected_identity": identity,
            "aliases": derive_aliases(mid, family, identity),
            "auto": auto,
            "in_v1": bool(v1_entry),
            "recheck_hygiene": bool(v1_excl),
            "v1_exclude_reason": v1_excl.get("exclude_reason") if v1_excl else None,
            "reasoning": "reasoning" in supported or "include_reasoning" in supported,
            "created": m.get("created"),
            "context_length": m.get("context_length"),
            "price_prompt": float(pricing.get("prompt") or 0),
            "price_completion": float(pricing.get("completion") or 0),
            "provider": (v1_entry or {}).get("provider"),  # stale v1 pins, preflight refreshes
        }
        kept.append(entry)

    # Models gone from OR but reachable via the proxy's native anthropic/ route
    # (api.proxy_model_id passes anthropic/claude* through untouched).
    kept.append({
        "id": "anthropic/claude-3-7-sonnet-20250219", "name": "Claude 3.7 Sonnet",
        "family": "anthropic", "expected_identity": "Claude 3.7 Sonnet",
        "aliases": ["anthropic", "claude", "sonnet"], "auto": False, "in_v1": True,
        "recheck_hygiene": False, "v1_exclude_reason": None, "reasoning": True,
        "created": None, "context_length": 200000,
        "price_prompt": 3e-06, "price_completion": 1.5e-05,
        "provider": None, "route": "proxy-native",
    })
    kept.append({
        "id": "anthropic/claude-3-opus-20240229", "name": "Claude 3 Opus",
        "family": "anthropic", "expected_identity": "Claude 3 Opus",
        "aliases": ["anthropic", "claude", "opus"], "auto": False, "in_v1": False,
        "recheck_hygiene": False, "v1_exclude_reason": None, "reasoning": False,
        "created": None, "context_length": 200000,
        "price_prompt": 1.5e-05, "price_completion": 7.5e-05,
        "provider": None, "route": "proxy-native",
    })

    # v1-excluded models that fell off the OR catalog entirely
    gone = [mid for mid in v1_excl_by_id if mid not in ids_seen]
    # v1-included models that fell off
    gone_v1 = [mid for mid in v1_by_id if mid not in ids_seen]

    registry = {
        "generated_from": "openrouter catalog",
        "n_models": len(kept),
        "models": kept,
        "skipped": {k: sorted(v) for k, v in skipped.items()},
        "v1_models_gone_from_or": sorted(gone_v1),
        "v1_excluded_gone_from_or": sorted(gone),
    }
    return registry


if __name__ == "__main__":
    catalog_path = None
    for a in sys.argv[1:]:
        if a.startswith("--catalog="):
            catalog_path = a.split("=", 1)[1]
    reg = build(catalog_path)
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(reg, indent=1, ensure_ascii=False))
    ms = reg["models"]
    print(f"kept {len(ms)} models -> {OUT_PATH}")
    print(f"  from v1:            {sum(m['in_v1'] for m in ms)}")
    print(f"  v1-excluded, retry: {sum(m['recheck_hygiene'] for m in ms)}")
    print(f"  new (auto id):      {sum(m['auto'] for m in ms)}")
    print(f"  reasoning-capable:  {sum(m['reasoning'] for m in ms)}")
    for k, v in reg["skipped"].items():
        print(f"  skipped {k}: {len(v)}")
    if reg["v1_models_gone_from_or"]:
        print("  v1 models GONE from OR:", ", ".join(reg["v1_models_gone_from_or"]))
