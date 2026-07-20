# Models

Every model in this study, with how it was reached. **189 models were queried through hosted APIs** (OpenRouter, default provider, no system prompt), **16 more were run from raw weights on GPUs**, and **52 were excluded** because no provider served them cleanly.

## Selection policy

- Official lab models only — community finetunes, roleplay merges, and base (non-chat) models are excluded.
- No `latest`-alias redirects and no auto-routers (they hide which model actually answered).
- `:free` variant used only when no paid sibling exists.
- **Provider hygiene:** a model is excluded if every available provider injects a system prompt (e.g. "You are ChatGPT"), so a "who are you?" answer would reflect the *provider's scaffolding*, not the model. See `sweep/preflight.py` + `config/provider_hygiene.json` for the per-model decision.

## Queried via API (189)

**OpenAI** (29)  
GPT-3.5 Turbo, GPT-4, GPT-4 Turbo, GPT-4.1, GPT-4.1 Mini, GPT-4.1 Nano, GPT-4o, GPT-4o Mini, GPT-5, GPT-5 Chat, GPT-5 Mini, GPT-5 Nano, GPT-5.1, GPT-5.1 Chat, GPT-5.2, GPT-5.2 Chat, GPT-5.3 Chat, GPT-5.3-Codex, GPT-5.4, GPT-5.4 Mini, GPT-5.4 Nano, GPT-5.5, GPT-5.6 Luna, GPT-5.6 Sol, GPT-5.6 Terra, o1, o3, o3 Mini, o4 Mini

**Anthropic** (14)  
Claude 3 Haiku, Claude 3 Opus, Claude Fable 5, Claude Haiku 4.5, Claude Opus 4, Claude Opus 4.1, Claude Opus 4.5, Claude Opus 4.6, Claude Opus 4.7, Claude Opus 4.8, Claude Sonnet 4, Claude Sonnet 4.5, Claude Sonnet 4.6, Claude Sonnet 5

**Google** (10)  
Gemini 2.5 Flash, Gemini 2.5 Flash Lite, Gemini 2.5 Pro, Gemini 3 Flash Preview, Gemini 3.1 Flash Lite, Gemini 3.1 Pro Preview, Gemini 3.5 Flash, Gemma 2 27B, Gemma 4 26B A4B, Gemma 4 31B

**Google (Gemma)** (4)  
Gemma 3 12B, Gemma 3 27B, Gemma 3 4B, Gemma 3n 4B

**DeepSeek** (12)  
DeepSeek R1, DeepSeek R1 0528, DeepSeek R1 Distill Llama 70B, DeepSeek V3, DeepSeek V3 0324, DeepSeek V3.1, DeepSeek V3.1 Terminus, DeepSeek V3.2, DeepSeek V4 Flash, DeepSeek V4 Pro, R1 Distill Llama 8B, R1 Distill Qwen 7B

**Qwen / Alibaba** (38)  
Qwen Plus, Qwen2.5 72B Instruct, Qwen2.5 7B Instruct, Qwen3 14B, Qwen3 235B A22B (MoE), Qwen3 235B A22B Instruct 2507, Qwen3 235B A22B Thinking 2507, Qwen3 30B A3B (MoE), Qwen3 30B A3B Instruct 2507, Qwen3 30B A3B Thinking 2507, Qwen3 32B, Qwen3 4B Instruct, Qwen3 8B, Qwen3 8B, Qwen3 Coder 30B A3B Instruct, Qwen3 Coder 480B, Qwen3 Coder Flash, Qwen3 Coder Next, Qwen3 Coder Plus, Qwen3 Max, Qwen3 Max Thinking, Qwen3 Next 80B A3B, Qwen3 Next 80B A3B Thinking, Qwen3.5 122B A10B, Qwen3.5 27B, Qwen3.5 35B A3B, Qwen3.5 397B A17B, Qwen3.5 Plus, Qwen3.5 Plus 2026-04-20, Qwen3.5-9B, Qwen3.5-Flash, Qwen3.6 27B, Qwen3.6 35B A3B, Qwen3.6 Flash, Qwen3.6 Max Preview, Qwen3.6 Plus, Qwen3.7 Max, Qwen3.7 Plus

**Meta (Llama)** (7)  
Llama 3.1 70B Instruct, Llama 3.1 8B Instruct, Llama 3.2 1B Instruct, Llama 3.2 3B Instruct, Llama 3.3 70B, Llama 4 Maverick, Llama 4 Scout

**Moonshot (Kimi)** (6)  
Kimi K2, Kimi K2 0905, Kimi K2 Thinking, Kimi K2.5, Kimi K2.6, Kimi K2.7 Code

**Mistral** (16)  
Codestral 2508, Devstral 2 2512, Ministral 3 14B, Ministral 3 3B 2512, Ministral 3 8B 2512, Mistral Large 2407, Mistral Large 3, Mistral Medium 3, Mistral Medium 3.1, Mistral Medium 3.5, Mistral Nemo, Mistral Small 3.1, Mistral Small 3.2, Mistral Small 4, Mixtral 8x22B Instruct, Saba

**Zhipu (GLM)** (9)  
GLM 4.5, GLM 4.5 Air, GLM 4.6, GLM 4.7, GLM 4.7 Flash, GLM 5, GLM 5 Turbo, GLM 5.1, GLM 5.2

**Baidu (Ernie)** (1)  
ERNIE 4.5 VL 424B A47B

**Tencent (Hunyuan)** (2)  
Hunyuan A13B, Hy3

**Microsoft** (1)  
Phi 4

**NVIDIA** (5)  
Nemotron 3 Nano 30B, Nemotron 3 Super, Nemotron 3 Ultra, Nemotron Nano 9B V2, Nemotron Super 49B

**Cohere** (9)  
Aya Expanse 32B, Command A, Command R (08-2024), Command R+ (08-2024), Command R7B (12-2024), North Mini Code, Tiny Aya Earth, Tiny Aya Global, Tiny Aya Water

**Amazon (Nova)** (3)  
Nova Lite 1.0, Nova Micro 1.0, Nova Pro

**Nous** (2)  
Hermes 3 405B Instruct, Hermes 3 70B Instruct

**Perplexity** (2)  
Sonar, Sonar Pro

**AI21** (1)  
Jamba Large 1.7

**Aisingapore** (2)  
SEA-LION v4 27B (Gemma), SEA-LION v4 32B (Qwen)

**Ant** (3)  
Ling-2.6-1T, Ling-2.6-flash, Ring-2.6-1T

**Arcee** (1)  
Trinity Large Thinking

**IBM (Granite)** (1)  
Granite 4.1 8B

**Kuaishou** (1)  
KAT-Coder-Pro V2

**MiniMax** (1)  
MiniMax M2.7

**Nex** (2)  
Nex-N2-Mini, Nex-N2-Pro

**Perceptron** (1)  
Perceptron Mk1

**Poolside** (2)  
Laguna M.1, Laguna XS 2.1

**Reka** (2)  
Reka Edge, Reka Flash 3

**StepFun** (1)  
Step 3.7 Flash

**Xiaomi** (1)  
MiMo-V2.5

## Run from raw weights on GPUs (16)

Downloaded from HuggingFace and run on rented A100s with **any identity stripped from the chat template** (blank `model_identity` for harmony models, neutral system override for others) and verified identity-free before generation (`sweep/verify_prompts.py`). This isolates what the *weights* say from what the shipped template says.

- Qwen3 0.6B — `Qwen/Qwen3-0.6B`
- Qwen3 1.7B — `Qwen/Qwen3-1.7B`
- Qwen3 4B — `Qwen/Qwen3-4B`
- Qwen3 8B — `Qwen/Qwen3-8B`
- Qwen3 14B — `Qwen/Qwen3-14B`
- Qwen3 32B — `Qwen/Qwen3-32B`
- Qwen3.5 0.8B — `Qwen/Qwen3.5-0.8B`
- Qwen3.5 2B — `Qwen/Qwen3.5-2B`
- Qwen3.5 4B — `Qwen/Qwen3.5-4B`
- Qwen3.5 35B-A3B — `Qwen/Qwen3.5-35B-A3B` (tp=2, vllm)
- Qwen3.6 35B-A3B — `Qwen/Qwen3.6-35B-A3B` (tp=2, vllm)
- OLMo 3 7B Instruct — `allenai/Olmo-3-7B-Instruct` (tp=1, transformers)
- OLMo 3 7B Think — `allenai/Olmo-3-7B-Think` (tp=1, transformers)
- OLMo 3.1 32B Instruct — `allenai/Olmo-3.1-32B-Instruct` (tp=1, transformers)
- GPT-OSS 20B — `openai/gpt-oss-20b`
- GPT-OSS 120B — `openai/gpt-oss-120b` (tp=2, vllm)

## Excluded (52)

Not evidence of drift — these are models we *couldn't* measure cleanly, listed for transparency.

**Provider injects a system prompt** (43)  
`MiniMaxAI/MiniMax-M2.5`, `MiniMaxAI/MiniMax-M2.7`, `MiniMaxAI/MiniMax-M3`, `allenai/Olmo-3-7B-Instruct`, `amazon/nova-2-lite-v1`, `amazon/nova-premier-v1`, `arcee-ai/coder-large`, `arcee-ai/virtuoso-large`, `bytedance-seed/seed-1.6`, `bytedance-seed/seed-1.6-flash`, `bytedance-seed/seed-2.0-lite`, `bytedance-seed/seed-2.0-mini`, `deepcogito/cogito-671b-v2.1`, `deepcogito/cogito-v2.1-671b`, `ibm-granite/granite-4.0-h-micro`, `inception/mercury-2`, `inflection/inflection-3-pi`, `inflection/inflection-3-productivity`, `liquid/lfm-2.5-1.2b-instruct:free`, `liquid/lfm-2.5-1.2b-thinking:free`, `microsoft/wizardlm-2-8x22b`, `minimax/minimax-01`, `minimax/minimax-m2-her`, `minimax/minimax-m2.5`, `minimax/minimax-m3`, `mistralai/mistral-small-24b-instruct-2501`, `openai/gpt-5.6-luna-pro`, `openai/gpt-5.6-sol-pro`, `openai/gpt-5.6-terra-pro`, `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `openai/gpt-oss-safeguard-20b`, `openai/o3-deep-research`, `openai/o4-mini-deep-research`, `qwen/qwen-2.5-coder-32b-instruct`, `sakana/fugu-ultra`, `stepfun/step-3.5-flash`, `upstage/solar-pro-3`, `writer/palmyra-x5`, `x-ai/grok-4.20`, `x-ai/grok-4.3`, `x-ai/grok-4.5`, `xiaomi/mimo-v2.5-pro`

**No clean / working endpoint** (1)  
`allenai/olmo-3-32b-think`

**Proxy served a different model** (6)  
`anthropic/claude-3-7-sonnet-20250219`, `anthropic/claude-3-haiku-20240307`, `local/chatgpt-4o-latest`, `local/gpt-3.5-turbo-0125`, `local/gpt-4-0314`, `local/gpt-4-0613`

**Identity baked into the model's own recommended template (case study, not comparison)** (2)  
`nousresearch/hermes-4-405b`, `nousresearch/hermes-4-70b`
