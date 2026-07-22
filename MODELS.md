# Models

Every model in this study, with how it was reached. **179 models were queried through hosted APIs**, each **pinned to one provider** (below), **16 more were run from raw weights on GPUs**, and **52 were excluded** because no provider served them cleanly.

## Selection policy

- Official lab models only — community finetunes, roleplay merges, and base (non-chat) models are excluded.
- No `latest`-alias redirects and no auto-routers (they hide which model actually answered).
- `:free` variant used only when no paid sibling exists.
- **Provider hygiene:** a model is excluded if every available provider injects a system prompt (e.g. "You are ChatGPT"), so a "who are you?" answer would reflect the *provider's scaffolding*, not the model. See `sweep/preflight.py` + `config/provider_hygiene.json`.

## Provider pinning (reproducibility & injection-checkability)

Of the 179 analyzed API models, **178 are pinned** to one preflight-chosen provider via OpenRouter routing `{"order": ["<slug>"], "allow_fallbacks": false}`, so a call can't silently fall back to an injecting or wrong-quant endpoint. `runner.py` records `provider_served` on every response; the table below lists the pin per model. **Verified against the collected data: 176 models were served by a single consistent provider** across all their records — no fallbacks (`provider_served` is a display name, so this checks consistency, not that the display string equals the slug). Two (`deepseek/deepseek-r1`, `qwen/qwen3.5-9b`) were re-pinned between runs and carry records from two providers — **both of which passed the hygiene check**. One model (`anthropic/claude-3-opus-20240229`) was served first-party via the proxy's native route (preflighted clean) rather than an OpenRouter pin, so it has no OpenRouter `provider_served` slug. To reproduce a model's answers (or check for injection yourself), pin its provider slug on OpenRouter and send the prompts in `prompts.jsonl` with no system prompt.

## Queried via API (179)

| model | family | pinned provider |
|---|---|---|
| GPT-3.5 Turbo | OpenAI | `openai` |
| GPT-4 | OpenAI | `openai` |
| GPT-4 Turbo | OpenAI | `openai` |
| GPT-4.1 | OpenAI | `openai` |
| GPT-4.1 Mini | OpenAI | `openai` |
| GPT-4.1 Nano | OpenAI | `openai` |
| GPT-4o | OpenAI | `openai` |
| GPT-4o Mini | OpenAI | `openai` |
| GPT-5 | OpenAI | `openai` |
| GPT-5 Chat | OpenAI | `openai` |
| GPT-5 Mini | OpenAI | `openai` |
| GPT-5 Nano | OpenAI | `openai` |
| GPT-5.1 | OpenAI | `openai` |
| GPT-5.1 Chat | OpenAI | `openai` |
| GPT-5.2 | OpenAI | `openai` |
| GPT-5.2 Chat | OpenAI | `openai` |
| GPT-5.3 Chat | OpenAI | `openai` |
| GPT-5.3-Codex | OpenAI | `openai` |
| GPT-5.4 | OpenAI | `openai` |
| GPT-5.4 Mini | OpenAI | `openai` |
| GPT-5.4 Nano | OpenAI | `openai` |
| GPT-5.5 | OpenAI | `openai` |
| GPT-5.6 Luna | OpenAI | `openai` |
| GPT-5.6 Sol | OpenAI | `openai` |
| GPT-5.6 Terra | OpenAI | `openai` |
| o1 | OpenAI | `openai` |
| o3 | OpenAI | `openai` |
| o3 Mini | OpenAI | `openai` |
| o4 Mini | OpenAI | `openai` |
| Claude 3 Haiku | Anthropic | `amazon-bedrock` |
| Claude 3 Opus | Anthropic | *unpinned (pre-dates pinning)* |
| Claude Fable 5 | Anthropic | `google-vertex` |
| Claude Haiku 4.5 | Anthropic | `amazon-bedrock` |
| Claude Opus 4 | Anthropic | `google-vertex` |
| Claude Opus 4.1 | Anthropic | `anthropic` |
| Claude Opus 4.5 | Anthropic | `amazon-bedrock` |
| Claude Opus 4.6 | Anthropic | `google-vertex` |
| Claude Opus 4.7 | Anthropic | `google-vertex` |
| Claude Opus 4.8 | Anthropic | `google-vertex` |
| Claude Sonnet 4 | Anthropic | `google-vertex` |
| Claude Sonnet 4.5 | Anthropic | `amazon-bedrock` |
| Claude Sonnet 4.6 | Anthropic | `amazon-bedrock` |
| Claude Sonnet 5 | Anthropic | `anthropic` |
| Gemini 2.5 Flash | Google | `google-vertex` |
| Gemini 2.5 Flash Lite | Google | `google-vertex` |
| Gemini 2.5 Pro | Google | `google-vertex` |
| Gemini 3 Flash Preview | Google | `google-vertex` |
| Gemini 3.1 Flash Lite | Google | `google-vertex` |
| Gemini 3.1 Pro Preview | Google | `google-vertex` |
| Gemini 3.5 Flash | Google | `google-vertex` |
| Gemma 2 27B | Google | `nextbit` |
| Gemma 4 26B A4B | Google | `google-vertex` |
| Gemma 4 31B | Google | `wandb` |
| Gemma 3 12B | Google (Gemma) | `deepinfra` |
| Gemma 3 27B | Google (Gemma) | `novita` |
| Gemma 3 4B | Google (Gemma) | `deepinfra` |
| Gemma 3n 4B | Google (Gemma) | `together` |
| DeepSeek R1 | DeepSeek | `novita` ⚠ two providers (both clean): Azure, Novita |
| DeepSeek R1 0528 | DeepSeek | `streamlake` |
| DeepSeek R1 Distill Llama 70B | DeepSeek | `novita` |
| DeepSeek V3 | DeepSeek | `streamlake` |
| DeepSeek V3 0324 | DeepSeek | `siliconflow` |
| DeepSeek V3.1 | DeepSeek | `mara` |
| DeepSeek V3.1 Terminus | DeepSeek | `streamlake` |
| DeepSeek V3.2 | DeepSeek | `venice` |
| DeepSeek V4 Flash | DeepSeek | `digitalocean` |
| DeepSeek V4 Pro | DeepSeek | `digitalocean` |
| Qwen Plus | Qwen / Alibaba | `alibaba` |
| Qwen2.5 72B Instruct | Qwen / Alibaba | `deepinfra` |
| Qwen2.5 7B Instruct | Qwen / Alibaba | `together` |
| Qwen3 14B | Qwen / Alibaba | `alibaba` |
| Qwen3 235B A22B (MoE) | Qwen / Alibaba | `alibaba` |
| Qwen3 235B A22B Instruct 2507 | Qwen / Alibaba | `alibaba` |
| Qwen3 235B A22B Thinking 2507 | Qwen / Alibaba | `alibaba` |
| Qwen3 30B A3B (MoE) | Qwen / Alibaba | `alibaba` |
| Qwen3 30B A3B Instruct 2507 | Qwen / Alibaba | `alibaba` |
| Qwen3 30B A3B Thinking 2507 | Qwen / Alibaba | `alibaba` |
| Qwen3 32B | Qwen / Alibaba | `alibaba` |
| Qwen3 8B | Qwen / Alibaba | `alibaba` |
| Qwen3 Coder 30B A3B Instruct | Qwen / Alibaba | `alibaba` |
| Qwen3 Coder 480B | Qwen / Alibaba | `alibaba` |
| Qwen3 Coder Flash | Qwen / Alibaba | `alibaba` |
| Qwen3 Coder Next | Qwen / Alibaba | `alibaba` |
| Qwen3 Coder Plus | Qwen / Alibaba | `alibaba` |
| Qwen3 Max | Qwen / Alibaba | `alibaba` |
| Qwen3 Max Thinking | Qwen / Alibaba | `alibaba` |
| Qwen3 Next 80B A3B | Qwen / Alibaba | `alibaba` |
| Qwen3 Next 80B A3B Thinking | Qwen / Alibaba | `alibaba` |
| Qwen3.5 122B A10B | Qwen / Alibaba | `alibaba` |
| Qwen3.5 27B | Qwen / Alibaba | `alibaba` |
| Qwen3.5 35B A3B | Qwen / Alibaba | `alibaba` |
| Qwen3.5 397B A17B | Qwen / Alibaba | `alibaba` |
| Qwen3.5 Plus | Qwen / Alibaba | `alibaba` |
| Qwen3.5 Plus 2026-04-20 | Qwen / Alibaba | `alibaba` |
| Qwen3.5-9B | Qwen / Alibaba | `together` ⚠ two providers (both clean): DeepInfra, Together |
| Qwen3.5-Flash | Qwen / Alibaba | `alibaba` |
| Qwen3.6 27B | Qwen / Alibaba | `alibaba` |
| Qwen3.6 35B A3B | Qwen / Alibaba | `akashml` |
| Qwen3.6 Flash | Qwen / Alibaba | `alibaba` |
| Qwen3.6 Max Preview | Qwen / Alibaba | `alibaba` |
| Qwen3.6 Plus | Qwen / Alibaba | `alibaba` |
| Qwen3.7 Max | Qwen / Alibaba | `alibaba` |
| Qwen3.7 Plus | Qwen / Alibaba | `alibaba` |
| Llama 3.1 70B Instruct | Meta (Llama) | `wandb` |
| Llama 3.1 8B Instruct | Meta (Llama) | `deepinfra` |
| Llama 3.2 1B Instruct | Meta (Llama) | `cloudflare` |
| Llama 3.2 3B Instruct | Meta (Llama) | `parasail` |
| Llama 3.3 70B | Meta (Llama) | `sambanova-turbo` |
| Llama 4 Maverick | Meta (Llama) | `digitalocean` |
| Llama 4 Scout | Meta (Llama) | `google-vertex` |
| Kimi K2 | Moonshot (Kimi) | `novita` |
| Kimi K2 0905 | Moonshot (Kimi) | `novita` |
| Kimi K2 Thinking | Moonshot (Kimi) | `google-vertex` |
| Kimi K2.5 | Moonshot (Kimi) | `moonshotai` |
| Kimi K2.6 | Moonshot (Kimi) | `moonshotai` |
| Kimi K2.7 Code | Moonshot (Kimi) | `moonshotai` |
| Codestral 2508 | Mistral | `mistral` |
| Devstral 2 2512 | Mistral | `mistral` |
| Ministral 3 14B | Mistral | `mistral` |
| Ministral 3 3B 2512 | Mistral | `mistral` |
| Ministral 3 8B 2512 | Mistral | `mistral` |
| Mistral Large 2407 | Mistral | `mistral` |
| Mistral Large 3 | Mistral | `mistral` |
| Mistral Medium 3 | Mistral | `mistral` |
| Mistral Medium 3.1 | Mistral | `mistral` |
| Mistral Medium 3.5 | Mistral | `mistral` |
| Mistral Nemo | Mistral | `mistral` |
| Mistral Small 3.1 | Mistral | `cloudflare` |
| Mistral Small 3.2 | Mistral | `mistral` |
| Mistral Small 4 | Mistral | `mistral` |
| Mixtral 8x22B Instruct | Mistral | `mistral` |
| Saba | Mistral | `mistral` |
| GLM 4.5 | Zhipu (GLM) | `z-ai` |
| GLM 4.5 Air | Zhipu (GLM) | `z-ai` |
| GLM 4.6 | Zhipu (GLM) | `z-ai` |
| GLM 4.7 | Zhipu (GLM) | `z-ai` |
| GLM 4.7 Flash | Zhipu (GLM) | `deepinfra` |
| GLM 5 | Zhipu (GLM) | `z-ai` |
| GLM 5 Turbo | Zhipu (GLM) | `z-ai` |
| GLM 5.1 | Zhipu (GLM) | `z-ai` |
| GLM 5.2 | Zhipu (GLM) | `z-ai` |
| ERNIE 4.5 VL 424B A47B | Baidu (Ernie) | `novita` |
| Hunyuan A13B | Tencent (Hunyuan) | `siliconflow` |
| Hy3 | Tencent (Hunyuan) | `gmicloud` |
| Phi 4 | Microsoft | `deepinfra` |
| Nemotron 3 Nano 30B | NVIDIA | `nebius` |
| Nemotron 3 Super | NVIDIA | `deepinfra` |
| Nemotron 3 Ultra | NVIDIA | `together` |
| Nemotron Nano 9B V2 | NVIDIA | `nvidia` |
| Nemotron Super 49B | NVIDIA | `deepinfra` |
| Command A | Cohere | `cohere` |
| Command R (08-2024) | Cohere | `cohere` |
| Command R+ (08-2024) | Cohere | `cohere` |
| Command R7B (12-2024) | Cohere | `cohere` |
| North Mini Code | Cohere | `cohere` |
| Nova Lite 1.0 | Amazon (Nova) | `amazon-bedrock` |
| Nova Micro 1.0 | Amazon (Nova) | `amazon-bedrock` |
| Nova Pro | Amazon (Nova) | `amazon-bedrock` |
| Hermes 3 405B Instruct | Nous | `deepinfra` |
| Hermes 3 70B Instruct | Nous | `deepinfra` |
| Sonar | Perplexity | `perplexity` |
| Sonar Pro | Perplexity | `perplexity` |
| Jamba Large 1.7 | AI21 | `ai21` |
| Ling-2.6-1T | Ant (Ling) | `novita` |
| Ling-2.6-flash | Ant (Ling) | `novita` |
| Ring-2.6-1T | Ant (Ling) | `novita` |
| Trinity Large Thinking | Arcee | `arcee-ai` |
| Granite 4.1 8B | IBM (Granite) | `wandb` |
| KAT-Coder-Pro V2 | Kuaishou | `streamlake` |
| MiniMax M2.7 | MiniMax | `groq` |
| Nex-N2-Mini | Nex | `nex-agi` |
| Nex-N2-Pro | Nex | `nex-agi` |
| Perceptron Mk1 | Perceptron | `perceptron` |
| Laguna M.1 | Poolside | `poolside` |
| Laguna XS 2.1 | Poolside | `poolside` |
| Reka Edge | Reka | `reka` |
| Reka Flash 3 | Reka | `reka` |
| Step 3.7 Flash | StepFun | `stepfun` |
| MiMo-V2.5 | Xiaomi | `digitalocean` |

## Run from raw weights on GPUs (16)

Downloaded from HuggingFace and run on rented A100s with **any identity stripped from the chat template** and verified identity-free before generation (`sweep/verify_prompts.py`) — isolating what the *weights* say from what the shipped template says. No hosted provider involved.

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

Not evidence of drift — models we *couldn't* measure cleanly, listed for transparency.

**Provider injects a system prompt** (43)  
`MiniMaxAI/MiniMax-M2.5`, `MiniMaxAI/MiniMax-M2.7`, `MiniMaxAI/MiniMax-M3`, `allenai/Olmo-3-7B-Instruct`, `amazon/nova-2-lite-v1`, `amazon/nova-premier-v1`, `arcee-ai/coder-large`, `arcee-ai/virtuoso-large`, `bytedance-seed/seed-1.6`, `bytedance-seed/seed-1.6-flash`, `bytedance-seed/seed-2.0-lite`, `bytedance-seed/seed-2.0-mini`, `deepcogito/cogito-671b-v2.1`, `deepcogito/cogito-v2.1-671b`, `ibm-granite/granite-4.0-h-micro`, `inception/mercury-2`, `inflection/inflection-3-pi`, `inflection/inflection-3-productivity`, `liquid/lfm-2.5-1.2b-instruct:free`, `liquid/lfm-2.5-1.2b-thinking:free`, `microsoft/wizardlm-2-8x22b`, `minimax/minimax-01`, `minimax/minimax-m2-her`, `minimax/minimax-m2.5`, `minimax/minimax-m3`, `mistralai/mistral-small-24b-instruct-2501`, `openai/gpt-5.6-luna-pro`, `openai/gpt-5.6-sol-pro`, `openai/gpt-5.6-terra-pro`, `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `openai/gpt-oss-safeguard-20b`, `openai/o3-deep-research`, `openai/o4-mini-deep-research`, `qwen/qwen-2.5-coder-32b-instruct`, `sakana/fugu-ultra`, `stepfun/step-3.5-flash`, `upstage/solar-pro-3`, `writer/palmyra-x5`, `x-ai/grok-4.20`, `x-ai/grok-4.3`, `x-ai/grok-4.5`, `xiaomi/mimo-v2.5-pro`

**No clean / working endpoint** (1)  
`allenai/olmo-3-32b-think`

**Proxy served a different model** (6)  
`anthropic/claude-3-7-sonnet-20250219`, `anthropic/claude-3-haiku-20240307`, `local/chatgpt-4o-latest`, `local/gpt-3.5-turbo-0125`, `local/gpt-4-0314`, `local/gpt-4-0613`

**Identity baked into the model's own recommended template (case study, not comparison)** (2)  
`nousresearch/hermes-4-405b`, `nousresearch/hermes-4-70b`
