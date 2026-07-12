"""
Curated model list for self-identification sweep.

Each entry: {
    "id": OpenRouter model ID,
    "name": human-readable short name,
    "family": model family (for coarse grouping),
    "expected_identity": specific name the model *should* identify as,
    "provider": (optional) OpenRouter provider routing config,
}

Selection: major model families, roughly early 2025+, no community finetunes.
Models with unavoidable hidden system prompts are in EXCLUDED_MODELS.
"""

MODELS = [
    # === Anthropic ===
    {"id": "anthropic/claude-3.5-sonnet", "name": "Claude 3.5 Sonnet", "family": "anthropic", "expected_identity": "Claude 3.5 Sonnet"},
    {"id": "anthropic/claude-3.5-haiku", "name": "Claude 3.5 Haiku", "family": "anthropic", "expected_identity": "Claude 3.5 Haiku"},
    {"id": "anthropic/claude-3.7-sonnet", "name": "Claude 3.7 Sonnet", "family": "anthropic", "expected_identity": "Claude 3.7 Sonnet"},
    {"id": "anthropic/claude-haiku-4.5", "name": "Claude Haiku 4.5", "family": "anthropic", "expected_identity": "Claude Haiku 4.5"},
    {"id": "anthropic/claude-opus-4", "name": "Claude Opus 4", "family": "anthropic", "expected_identity": "Claude Opus 4"},
    {"id": "anthropic/claude-opus-4.1", "name": "Claude Opus 4.1", "family": "anthropic", "expected_identity": "Claude Opus 4.1"},
    {"id": "anthropic/claude-opus-4.5", "name": "Claude Opus 4.5", "family": "anthropic", "expected_identity": "Claude Opus 4.5"},
    {"id": "anthropic/claude-opus-4.6", "name": "Claude Opus 4.6", "family": "anthropic", "expected_identity": "Claude Opus 4.6"},
    {"id": "anthropic/claude-sonnet-4", "name": "Claude Sonnet 4", "family": "anthropic", "expected_identity": "Claude Sonnet 4"},
    {"id": "anthropic/claude-sonnet-4.5", "name": "Claude Sonnet 4.5", "family": "anthropic", "expected_identity": "Claude Sonnet 4.5"},
    {"id": "anthropic/claude-sonnet-4.6", "name": "Claude Sonnet 4.6", "family": "anthropic", "expected_identity": "Claude Sonnet 4.6"},

    # === OpenAI ===
    {"id": "openai/gpt-4o", "name": "GPT-4o", "family": "openai", "expected_identity": "GPT-4o"},
    {"id": "openai/gpt-4o-mini", "name": "GPT-4o Mini", "family": "openai", "expected_identity": "GPT-4o mini"},
    {"id": "openai/gpt-4.1", "name": "GPT-4.1", "family": "openai", "expected_identity": "GPT-4.1"},
    {"id": "openai/gpt-4.1-mini", "name": "GPT-4.1 Mini", "family": "openai", "expected_identity": "GPT-4.1 mini"},
    {"id": "openai/gpt-4.1-nano", "name": "GPT-4.1 Nano", "family": "openai", "expected_identity": "GPT-4.1 nano"},
    {"id": "openai/gpt-5", "name": "GPT-5", "family": "openai", "expected_identity": "GPT-5"},
    {"id": "openai/gpt-5-mini", "name": "GPT-5 Mini", "family": "openai", "expected_identity": "GPT-5 mini"},
    {"id": "openai/gpt-5-nano", "name": "GPT-5 Nano", "family": "openai", "expected_identity": "GPT-5 nano"},
    {"id": "openai/gpt-5.1", "name": "GPT-5.1", "family": "openai", "expected_identity": "GPT-5.1"},
    {"id": "openai/gpt-5.2", "name": "GPT-5.2", "family": "openai", "expected_identity": "GPT-5.2"},
    {"id": "openai/gpt-5.3-chat", "name": "GPT-5.3 Chat", "family": "openai", "expected_identity": "GPT-5.3"},
    {"id": "openai/gpt-5.4", "name": "GPT-5.4", "family": "openai", "expected_identity": "GPT-5.4"},
    {"id": "openai/o1", "name": "o1", "family": "openai", "expected_identity": "o1"},
    {"id": "openai/o3", "name": "o3", "family": "openai", "expected_identity": "o3"},
    {"id": "openai/o3-mini", "name": "o3 Mini", "family": "openai", "expected_identity": "o3-mini"},
    {"id": "openai/o4-mini", "name": "o4 Mini", "family": "openai", "expected_identity": "o4-mini"},

    # === Google Gemini ===
    {"id": "google/gemini-2.0-flash-001", "name": "Gemini 2.0 Flash", "family": "google", "expected_identity": "Gemini 2.0 Flash"},
    {"id": "google/gemini-2.5-flash", "name": "Gemini 2.5 Flash", "family": "google", "expected_identity": "Gemini 2.5 Flash"},
    {"id": "google/gemini-2.5-flash-lite", "name": "Gemini 2.5 Flash Lite", "family": "google", "expected_identity": "Gemini 2.5 Flash Lite"},
    {"id": "google/gemini-2.5-pro", "name": "Gemini 2.5 Pro", "family": "google", "expected_identity": "Gemini 2.5 Pro"},
    {"id": "google/gemini-3-flash-preview", "name": "Gemini 3 Flash Preview", "family": "google", "expected_identity": "Gemini 3 Flash"},
    {"id": "google/gemini-3-pro-preview", "name": "Gemini 3 Pro Preview", "family": "google", "expected_identity": "Gemini 3 Pro"},
    {"id": "google/gemini-3.1-pro-preview", "name": "Gemini 3.1 Pro Preview", "family": "google", "expected_identity": "Gemini 3.1 Pro"},

    # === Google Gemma (open weights) ===
    {"id": "google/gemma-3-4b-it", "name": "Gemma 3 4B", "family": "gemma", "expected_identity": "Gemma 3 4B"},
    {"id": "google/gemma-3-12b-it", "name": "Gemma 3 12B", "family": "gemma", "expected_identity": "Gemma 3 12B"},
    {"id": "google/gemma-3-27b-it", "name": "Gemma 3 27B", "family": "gemma", "expected_identity": "Gemma 3 27B"},
    {"id": "google/gemma-3n-e2b-it:free", "name": "Gemma 3n 2B", "family": "gemma", "expected_identity": "Gemma 3n 2B"},

    # === DeepSeek ===
    {"id": "deepseek/deepseek-chat", "name": "DeepSeek V3", "family": "deepseek", "expected_identity": "DeepSeek-V3"},
    {"id": "deepseek/deepseek-chat-v3-0324", "name": "DeepSeek V3 0324", "family": "deepseek", "expected_identity": "DeepSeek-V3"},
    {"id": "deepseek/deepseek-chat-v3.1", "name": "DeepSeek V3.1", "family": "deepseek", "expected_identity": "DeepSeek-V3.1"},
    {"id": "deepseek/deepseek-r1", "name": "DeepSeek R1", "family": "deepseek", "expected_identity": "DeepSeek-R1"},
    {"id": "deepseek/deepseek-r1-0528", "name": "DeepSeek R1 0528", "family": "deepseek", "expected_identity": "DeepSeek-R1"},
    {"id": "deepseek/deepseek-r1-distill-llama-70b", "name": "DeepSeek R1 Distill Llama 70B", "family": "deepseek", "expected_identity": "DeepSeek-R1-Distill-Llama-70B"},
    {"id": "deepseek/deepseek-r1-distill-qwen-32b", "name": "DeepSeek R1 Distill Qwen 32B", "family": "deepseek", "expected_identity": "DeepSeek-R1-Distill-Qwen-32B"},
    {"id": "deepseek/deepseek-v3.1-terminus", "name": "DeepSeek V3.1 Terminus", "family": "deepseek", "expected_identity": "DeepSeek-V3.1-Terminus"},
    {"id": "deepseek/deepseek-v3.2", "name": "DeepSeek V3.2", "family": "deepseek", "expected_identity": "DeepSeek-V3.2"},
    {"id": "deepseek/deepseek-v3.2-speciale", "name": "DeepSeek V3.2 Speciale", "family": "deepseek", "expected_identity": "DeepSeek-V3.2-Speciale"},

    # === Qwen (open weights + API) ===
    {"id": "qwen/qwen-max", "name": "Qwen Max", "family": "qwen", "expected_identity": "Qwen-Max"},
    {"id": "qwen/qwen-plus", "name": "Qwen Plus", "family": "qwen", "expected_identity": "Qwen-Plus"},
    {"id": "qwen/qwen-turbo", "name": "Qwen Turbo", "family": "qwen", "expected_identity": "Qwen-Turbo"},
    {"id": "qwen/qwen3-8b", "name": "Qwen3 8B", "family": "qwen", "expected_identity": "Qwen3-8B"},
    {"id": "qwen/qwen3-14b", "name": "Qwen3 14B", "family": "qwen", "expected_identity": "Qwen3-14B"},
    {"id": "qwen/qwen3-32b", "name": "Qwen3 32B", "family": "qwen", "expected_identity": "Qwen3-32B"},
    {"id": "qwen/qwen3-30b-a3b", "name": "Qwen3 30B A3B (MoE)", "family": "qwen", "expected_identity": "Qwen3-30B-A3B"},
    {"id": "qwen/qwen3-235b-a22b", "name": "Qwen3 235B A22B (MoE)", "family": "qwen", "expected_identity": "Qwen3-235B-A22B"},
    {"id": "qwen/qwen3-coder", "name": "Qwen3 Coder 480B", "family": "qwen", "expected_identity": "Qwen3-Coder"},
    {"id": "qwen/qwen3-coder-flash", "name": "Qwen3 Coder Flash", "family": "qwen", "expected_identity": "Qwen3-Coder-Flash"},
    {"id": "qwen/qwen3-max", "name": "Qwen3 Max", "family": "qwen", "expected_identity": "Qwen3-Max"},
    {"id": "qwen/qwen3-next-80b-a3b-instruct", "name": "Qwen3 Next 80B A3B", "family": "qwen", "expected_identity": "Qwen3-Next-80B-A3B"},
    {"id": "qwen/qwen3.5-27b", "name": "Qwen3.5 27B", "family": "qwen", "expected_identity": "Qwen3.5-27B"},
    {"id": "qwen/qwen3.5-35b-a3b", "name": "Qwen3.5 35B A3B", "family": "qwen", "expected_identity": "Qwen3.5-35B-A3B"},
    {"id": "qwen/qwen3.5-122b-a10b", "name": "Qwen3.5 122B A10B", "family": "qwen", "expected_identity": "Qwen3.5-122B-A10B"},
    {"id": "qwen/qwen3.5-397b-a17b", "name": "Qwen3.5 397B A17B", "family": "qwen", "expected_identity": "Qwen3.5-397B-A17B"},
    {"id": "qwen/qwen3.5-plus-02-15", "name": "Qwen3.5 Plus", "family": "qwen", "expected_identity": "Qwen3.5-Plus"},
    {"id": "qwen/qwq-32b", "name": "QwQ 32B", "family": "qwen", "expected_identity": "QwQ-32B"},

    # === Kimi / Moonshot ===
    # Pin to moonshotai: DeepInfra/Novita inject hidden system prompts (26-27 tokens)
    {"id": "moonshotai/kimi-k2", "name": "Kimi K2", "family": "kimi", "expected_identity": "Kimi K2", "provider": {"only": ["moonshotai"], "allow_fallbacks": False}},
    {"id": "moonshotai/kimi-k2-0905", "name": "Kimi K2 0905", "family": "kimi", "expected_identity": "Kimi K2", "provider": {"only": ["moonshotai"], "allow_fallbacks": False}},
    {"id": "moonshotai/kimi-k2-thinking", "name": "Kimi K2 Thinking", "family": "kimi", "expected_identity": "Kimi K2", "provider": {"only": ["moonshotai"], "allow_fallbacks": False}},
    {"id": "moonshotai/kimi-k2.5", "name": "Kimi K2.5", "family": "kimi", "expected_identity": "Kimi K2.5"},

    # === Mistral ===
    {"id": "mistralai/mistral-large-2512", "name": "Mistral Large 3", "family": "mistral", "expected_identity": "Mistral Large 3"},
    {"id": "mistralai/mistral-medium-3.1", "name": "Mistral Medium 3.1", "family": "mistral", "expected_identity": "Mistral Medium 3.1"},
    {"id": "mistralai/mistral-small-3.1-24b-instruct", "name": "Mistral Small 3.1", "family": "mistral", "expected_identity": "Mistral Small 3.1"},
    {"id": "mistralai/mistral-small-3.2-24b-instruct", "name": "Mistral Small 3.2", "family": "mistral", "expected_identity": "Mistral Small 3.2"},
    {"id": "mistralai/mistral-small-creative", "name": "Mistral Small Creative", "family": "mistral", "expected_identity": "Mistral Small Creative"},
    {"id": "mistralai/codestral-2508", "name": "Codestral 2508", "family": "mistral", "expected_identity": "Codestral"},
    {"id": "mistralai/devstral-medium", "name": "Devstral Medium", "family": "mistral", "expected_identity": "Devstral Medium"},
    {"id": "mistralai/devstral-small", "name": "Devstral Small", "family": "mistral", "expected_identity": "Devstral Small"},
    {"id": "mistralai/ministral-14b-2512", "name": "Ministral 3 14B", "family": "mistral", "expected_identity": "Ministral 3 14B"},

    # === xAI / Grok ===
    # Grok 3/3 Mini are clean (7-8 tokens). Grok 4+ excluded (xAI-only, 123-685 token sys prompts).
    {"id": "x-ai/grok-3", "name": "Grok 3", "family": "xai", "expected_identity": "Grok 3"},
    {"id": "x-ai/grok-3-mini", "name": "Grok 3 Mini", "family": "xai", "expected_identity": "Grok 3 Mini"},

    # === Meta / Llama ===
    # Pin Llama 3.3 to novita: AkashML/Inceptron inject hidden system prompts (36 tokens)
    {"id": "meta-llama/llama-3.3-70b-instruct", "name": "Llama 3.3 70B", "family": "meta", "expected_identity": "Llama 3.3 70B", "provider": {"only": ["novita"], "allow_fallbacks": False}},
    {"id": "meta-llama/llama-4-maverick", "name": "Llama 4 Maverick", "family": "meta", "expected_identity": "Llama 4 Maverick"},
    {"id": "meta-llama/llama-4-scout", "name": "Llama 4 Scout", "family": "meta", "expected_identity": "Llama 4 Scout"},

    # === Z.ai / GLM (Zhipu) ===
    {"id": "z-ai/glm-4.5", "name": "GLM 4.5", "family": "zhipu", "expected_identity": "GLM-4.5"},
    {"id": "z-ai/glm-4.6", "name": "GLM 4.6", "family": "zhipu", "expected_identity": "GLM-4.6"},
    {"id": "z-ai/glm-4.7", "name": "GLM 4.7", "family": "zhipu", "expected_identity": "GLM-4.7"},
    {"id": "z-ai/glm-4.7-flash", "name": "GLM 4.7 Flash", "family": "zhipu", "expected_identity": "GLM-4.7-Flash"},
    {"id": "z-ai/glm-5", "name": "GLM 5", "family": "zhipu", "expected_identity": "GLM-5"},

    # === Baidu / ERNIE ===
    {"id": "baidu/ernie-4.5-300b-a47b", "name": "ERNIE 4.5 300B", "family": "baidu", "expected_identity": "ERNIE 4.5"},
    {"id": "baidu/ernie-4.5-21b-a3b", "name": "ERNIE 4.5 21B", "family": "baidu", "expected_identity": "ERNIE 4.5"},

    # === Tencent ===
    {"id": "tencent/hunyuan-a13b-instruct", "name": "Hunyuan A13B", "family": "tencent", "expected_identity": "Hunyuan A13B"},

    # === StepFun ===
    {"id": "stepfun/step-3.5-flash", "name": "Step 3.5 Flash", "family": "stepfun", "expected_identity": "Step 3.5 Flash"},

    # === Cohere ===
    {"id": "cohere/command-a", "name": "Command A", "family": "cohere", "expected_identity": "Command A"},

    # === Amazon ===
    # Nova Pro is clean (1 token). Premier/2 Lite excluded (35-47 token sys prompts, Bedrock only).
    {"id": "amazon/nova-pro-v1", "name": "Nova Pro", "family": "amazon", "expected_identity": "Amazon Nova Pro"},

    # === Microsoft ===
    {"id": "microsoft/phi-4", "name": "Phi 4", "family": "microsoft", "expected_identity": "Phi-4"},

    # === Inflection ===
    {"id": "inflection/inflection-3-pi", "name": "Inflection 3 Pi", "family": "inflection", "expected_identity": "Pi"},

    # === Inception / Mercury ===
    {"id": "inception/mercury-2", "name": "Mercury 2", "family": "inception", "expected_identity": "Mercury 2"},

    # === Meituan ===
    {"id": "meituan/longcat-flash-chat", "name": "LongCat Flash", "family": "meituan", "expected_identity": "LongCat Flash"},

    # === AI21 ===
    {"id": "ai21/jamba-large-1.7", "name": "Jamba Large 1.7", "family": "ai21", "expected_identity": "Jamba Large 1.7"},

    # === Liquid ===
    {"id": "liquid/lfm-2-24b-a2b", "name": "LFM2 24B", "family": "liquid", "expected_identity": "LFM2 24B"},

    # === Alibaba ===
    {"id": "alibaba/tongyi-deepresearch-30b-a3b", "name": "Tongyi DeepResearch 30B", "family": "alibaba", "expected_identity": "Tongyi Qianwen"},
]


# ---------------------------------------------------------------------------
# Models excluded from main sweep due to unavoidable hidden system prompts.
# All providers for these models inject system prompts, confounding identity
# results. Kept here for documentation and optional targeted investigation.
# ---------------------------------------------------------------------------

EXCLUDED_MODELS = [
    # xAI Grok 4+ — only xAI provides, massive system prompts (123-685 tokens)
    {"id": "x-ai/grok-4", "name": "Grok 4", "family": "xai", "expected_identity": "Grok 4", "exclude_reason": "xAI only, 685 token sys prompt"},
    {"id": "x-ai/grok-4-fast", "name": "Grok 4 Fast", "family": "xai", "expected_identity": "Grok 4", "exclude_reason": "xAI only, 157 token sys prompt"},
    {"id": "x-ai/grok-4.1-fast", "name": "Grok 4.1 Fast", "family": "xai", "expected_identity": "Grok 4.1", "exclude_reason": "xAI only, 157 token sys prompt"},
    {"id": "x-ai/grok-4.20-beta", "name": "Grok 4.20 Beta", "family": "xai", "expected_identity": "Grok 4.20", "exclude_reason": "xAI only, 123 token sys prompt"},

    # MiniMax — all providers inject (39-447 tokens). M2.5 notably claims Claude despite sys prompt.
    {"id": "minimax/minimax-m1", "name": "MiniMax M1", "family": "minimax", "expected_identity": "MiniMax-M1", "exclude_reason": "all providers inject, 447 tokens"},
    {"id": "minimax/minimax-m2", "name": "MiniMax M2", "family": "minimax", "expected_identity": "MiniMax-M2", "exclude_reason": "all providers inject, 40 tokens"},
    {"id": "minimax/minimax-m2.1", "name": "MiniMax M2.1", "family": "minimax", "expected_identity": "MiniMax-M2.1", "exclude_reason": "all providers inject, 43 tokens"},
    {"id": "minimax/minimax-m2.5", "name": "MiniMax M2.5", "family": "minimax", "expected_identity": "MiniMax-M2.5", "exclude_reason": "all providers inject, 39 tokens. Claims Claude despite sys prompt!"},

    # ByteDance Seed — Seed-only provider (49-83 tokens). Identifies as "Doubao".
    {"id": "bytedance-seed/seed-1.6", "name": "Seed 1.6", "family": "bytedance", "expected_identity": "Doubao", "exclude_reason": "Seed only, 81 token sys prompt"},
    {"id": "bytedance-seed/seed-1.6-flash", "name": "Seed 1.6 Flash", "family": "bytedance", "expected_identity": "Doubao", "exclude_reason": "Seed only, 83 token sys prompt"},
    {"id": "bytedance-seed/seed-2.0-lite", "name": "Seed 2.0 Lite", "family": "bytedance", "expected_identity": "Seed 2.0 Lite", "exclude_reason": "Seed only, 49 token sys prompt"},
    {"id": "bytedance-seed/seed-2.0-mini", "name": "Seed 2.0 Mini", "family": "bytedance", "expected_identity": "Seed 2.0 Mini", "exclude_reason": "no response/provider returned"},

    # OpenAI GPT-OSS — all providers inject (55-72 tokens)
    {"id": "openai/gpt-oss-120b", "name": "GPT-OSS 120B", "family": "openai", "expected_identity": "GPT-OSS 120B", "exclude_reason": "all providers inject, 55-70 tokens"},
    {"id": "openai/gpt-oss-20b", "name": "GPT-OSS 20B", "family": "openai", "expected_identity": "GPT-OSS 20B", "exclude_reason": "all providers inject, 66-72 tokens"},

    # Xiaomi MiMo — all providers inject (38 tokens)
    {"id": "xiaomi/mimo-v2-flash", "name": "MiMo V2 Flash", "family": "xiaomi", "expected_identity": "MiMo-V2-Flash", "exclude_reason": "all providers inject, 38 tokens"},

    # Amazon Nova (Premier/2 Lite) — Bedrock only (35-47 tokens)
    {"id": "amazon/nova-premier-v1", "name": "Nova Premier", "family": "amazon", "expected_identity": "Amazon Nova Premier", "exclude_reason": "Bedrock only, 35 token sys prompt"},
    {"id": "amazon/nova-2-lite-v1", "name": "Nova 2 Lite", "family": "amazon", "expected_identity": "Amazon Nova 2 Lite", "exclude_reason": "Bedrock only, 47 token sys prompt"},

    # NVIDIA Nemotron — DeepInfra only (17-20 tokens)
    {"id": "nvidia/llama-3.3-nemotron-super-49b-v1.5", "name": "Nemotron Super 49B", "family": "nvidia", "expected_identity": "Nemotron Super 49B", "exclude_reason": "DeepInfra only, 20 token sys prompt"},
    {"id": "nvidia/nemotron-3-nano-30b-a3b", "name": "Nemotron 3 Nano 30B", "family": "nvidia", "expected_identity": "Nemotron 3 Nano 30B", "exclude_reason": "DeepInfra only, 17 token sys prompt"},

    # AllenAI OLMo — single provider each (57-67 tokens)
    {"id": "allenai/olmo-3.1-32b-instruct", "name": "OLMo 3.1 32B", "family": "allenai", "expected_identity": "OLMo 3.1 32B", "exclude_reason": "DeepInfra only, 67 token sys prompt"},
    {"id": "allenai/olmo-3.1-32b-think", "name": "OLMo 3.1 32B Think", "family": "allenai", "expected_identity": "OLMo 3.1 32B", "exclude_reason": "Parasail only, 57 token sys prompt"},

    # Gemma 3n 4B — Together only, borderline (17 tokens)
    {"id": "google/gemma-3n-e4b-it", "name": "Gemma 3n 4B", "family": "gemma", "expected_identity": "Gemma 3n 4B", "exclude_reason": "Together only, 17 token sys prompt (borderline)"},

    # Writer Palmyra — Bedrock only (238 tokens)
    {"id": "writer/palmyra-x5", "name": "Palmyra X5", "family": "writer", "expected_identity": "Palmyra X5", "exclude_reason": "Bedrock only, 238 token sys prompt"},

    # Upstage Solar — Upstage only (73 tokens)
    {"id": "upstage/solar-pro-3", "name": "Solar Pro 3", "family": "upstage", "expected_identity": "Solar Pro 3", "exclude_reason": "Upstage only, 73 token sys prompt"},

    # IBM Granite — Cloudflare only (31 tokens)
    {"id": "ibm-granite/granite-4.0-h-micro", "name": "Granite 4.0 Micro", "family": "ibm", "expected_identity": "Granite 4.0 Micro", "exclude_reason": "Cloudflare only, 31 token sys prompt"},
]


def print_stats():
    from collections import Counter
    families = Counter(m["family"] for m in MODELS)
    print(f"Total models (sweep): {len(MODELS)}")
    print(f"Excluded models: {len(EXCLUDED_MODELS)}")
    print(f"Families: {len(families)}")
    print()
    for family, count in families.most_common():
        print(f"  {family}: {count}")
    pinned = [m for m in MODELS if m.get("provider")]
    if pinned:
        print(f"\nProvider-pinned models ({len(pinned)}):")
        for m in pinned:
            print(f"  {m['name']}: {m['provider']}")


if __name__ == "__main__":
    print_stats()
