# Identity mismatches — where models name another vendor as themselves

Across 144 models: what each one claims to be when it *doesn't* claim its own identity. **Rate** is the spontaneous mismatch rate on the identity/creator battery; *claims as* is what it names instead. Click a model for every prompt + response (e.g. "Claude Opus 4.8 → DeepSeek in Chinese", to reproduce it).

Records are split by vendor so each page renders on GitHub. For **all** answers from **all** models (drift or not), open the full browser [`rollouts/index.html`](./index.html) or the raw [`rollouts_data.json`](./rollouts_data.json).

| model | family | mismatch rate | claims as |
|---|---|---|---|
| [Perceptron Mk1](./mismatches/perceptron.md#perceptron-mk1) | perceptron | 89% (300/338) | Qwen, ChatGPT, Claude |
| [Laguna M.1](./mismatches/poolside.md#laguna-m1) | poolside | 77% (259/338) | Nvidia, OpenAI, ChatGPT |
| [OLMo 3 7B Think](./mismatches/allenai.md#olmo-3-7b-think) | olmo | 73% (233/320) | DeepSeek, Qwen, ChatGPT |
| [Laguna XS 2.1](./mismatches/poolside.md#laguna-xs-21) | poolside | 67% (225/338) | Qwen, Alibaba, Google |
| [OLMo 3 7B](./mismatches/allenai.md#olmo-3-7b) | allenai | 62% (200/320) | ChatGPT, OpenAI, DeepSeek |
| [OLMo 3.1 32B Instruct](./mismatches/allenai.md#olmo-31-32b-instruct) | olmo | 59% (188/320) | OpenAI, ChatGPT, DeepSeek |
| [MiniMax M2.7](./mismatches/minimax.md#minimax-m27) | minimax | 56% (190/338) | Claude, Anthropic, Qwen |
| [Qwen2.5 72B Instruct](./mismatches/qwen.md#qwen25-72b-instruct) | qwen | 55% (183/333) | Claude, Anthropic, Yandex |
| [Nemotron Super 49B](./mismatches/nvidia.md#nemotron-super-49b) | nvidia | 49% (167/338) | Qwen, OpenAI, ChatGPT |
| [Kimi K2](./mismatches/kimi.md#kimi-k2) | kimi | 46% (154/338) | Claude, Anthropic, Yandex |
| [Kimi K2 0905](./mismatches/kimi.md#kimi-k2-0905) | kimi | 44% (148/338) | Claude, Anthropic |
| [Hermes 3 405B Instruct](./mismatches/nous.md#hermes-3-405b-instruct) | nous | 40% (136/338) | OpenAI, ChatGPT, Google |
| [Qwen2.5 7B Instruct](./mismatches/qwen.md#qwen25-7b-instruct) | qwen | 39% (133/338) | Claude, Anthropic, ChatGPT |
| [Kimi K2.5](./mismatches/kimi.md#kimi-k25) | kimi | 38% (128/338) | Claude, Anthropic, ChatGPT |
| [Trinity Large Thinking](./mismatches/arcee.md#trinity-large-thinking) | arcee | 36% (121/338) | step, Claude, Gemini |
| [Hy3](./mismatches/tencent.md#hy3) | tencent | 35% (118/338) | Claude, ChatGPT, Gemini |
| [Granite 4.1 8B](./mismatches/ibm.md#granite-41-8b) | ibm | 30% (103/338) | ChatGPT, Claude, Mistral |
| [Nemotron Nano 9B V2](./mismatches/nvidia.md#nemotron-nano-9b-v2) | nvidia | 27% (91/338) | Qwen, Alibaba, тонги |
| [ERNIE 4.5 VL 424B A47B](./mismatches/baidu.md#ernie-45-vl-424b-a47b) | baidu | 24% (82/338) | OpenAI, DeepSeek, ChatGPT |
| [Ministral 3 3B 2512](./mismatches/mistral.md#ministral-3-3b-2512) | mistral | 23% (79/338) | ChatGPT, OpenAI, text-davinci-003 |
| [Phi 4](./mismatches/microsoft.md#phi-4) | microsoft | 23% (77/338) | OpenAI, ChatGPT, Llama |
| [Sonar Pro](./mismatches/perplexity.md#sonar-pro) | perplexity | 21% (72/338) | OpenAI, ChatGPT, Qwen |
| [Mistral Nemo](./mismatches/mistral.md#mistral-nemo) | mistral | 21% (69/331) | ChatGPT, nemistral, Nvidia |
| [Kimi K2 Thinking](./mismatches/kimi.md#kimi-k2-thinking) | kimi | 21% (70/338) | Claude, Gemini, Google |
| [Hermes 3 70B Instruct](./mismatches/nous.md#hermes-3-70b-instruct) | nous | 20% (67/338) | Google, Amazon, Alexa |
| [Qwen3.5 0.8B](./mismatches/qwen.md#qwen35-08b) | qwen | 15% (48/320) | Microsoft, Google, Baidu |
| [Llama 3.2 3B Instruct](./mismatches/meta.md#llama-32-3b-instruct) | meta | 14% (46/337) | Microsoft, Google, luna |
| [Nemotron 3 Nano 30B](./mismatches/nvidia.md#nemotron-3-nano-30b) | nvidia | 13% (45/338) | Qwen, ChatGPT, OpenAI |
| [DeepSeek V3 0324](./mismatches/deepseek.md#deepseek-v3-0324) | deepseek | 13% (44/338) | ChatGPT, OpenAI, Llama |
| [Nova Lite 1.0](./mismatches/amazon.md#nova-lite-10) | amazon | 13% (43/338) | OpenAI, ChatGPT, Google |
| [KAT-Coder-Pro V2](./mismatches/kuaishou.md#kat-coder-pro-v2) | kuaishou | 11% (38/338) | Alibaba, Qwen, DeepSeek |
| [Reka Edge](./mismatches/reka.md#reka-edge) | reka | 11% (38/338) | OpenAI, Microsoft, Xiaomi |
| [Codestral 2508](./mismatches/mistral.md#codestral-2508) | mistral | 11% (38/338) | Internlm, Llama, jarvis |
| [Gemini 3.5 Flash](./mismatches/google.md#gemini-35-flash) | google | 11% (37/338) | ChatGPT, OpenAI, Llama |
| [Qwen3 1.7B](./mismatches/qwen.md#qwen3-17b) | qwen | 11% (35/320) | Claude, lia, OpenAI |
| [DeepSeek V3](./mismatches/deepseek.md#deepseek-v3) | deepseek | 11% (36/338) | ChatGPT, OpenAI, Yandex |
| [Llama 3.2 1B Instruct](./mismatches/meta.md#llama-32-1b-instruct) | meta | 11% (36/338) | ChatGPT, Google, アナリシープロジー |
| [Claude Opus 4.8](./mismatches/anthropic.md#claude-opus-48) | anthropic | 10% (35/338) | DeepSeek, Qwen |
| [Kimi K2.7 Code](./mismatches/kimi.md#kimi-k27-code) | kimi | 10% (35/338) | Claude, Anthropic, OpenAI |
| [Claude 3 Haiku](./mismatches/anthropic.md#claude-3-haiku) | anthropic | 10% (34/338) | ChatGPT, OpenAI, 클로이 |
| [Reka Flash 3](./mismatches/reka.md#reka-flash-3) | reka | 10% (33/337) | OpenAI, ChatGPT |
| [Llama 3.1 8B Instruct](./mismatches/meta.md#llama-31-8b-instruct) | meta | 10% (33/338) | OpenAI, ChatGPT, Google |
| [Gemini 3.1 Flash Lite](./mismatches/google.md#gemini-31-flash-lite) | google | 10% (33/338) | ChatGPT, OpenAI |
| [DeepSeek V3.1 Terminus](./mismatches/deepseek.md#deepseek-v31-terminus) | deepseek | 9% (32/338) | OpenAI, ChatGPT, Anthropic |
| [Nova Pro](./mismatches/amazon.md#nova-pro) | amazon | 9% (32/338) | ChatGPT, OpenAI, ася |
| [Ministral 3 8B 2512](./mismatches/mistral.md#ministral-3-8b-2512) | mistral | 9% (32/338) | ChatGPT, Meta, Microsoft |
| [Mistral Medium 3.1](./mismatches/mistral.md#mistral-medium-31) | mistral | 9% (32/338) | ChatGPT, Microsoft, Ernie |
| [Ministral 3 14B](./mismatches/mistral.md#ministral-3-14b) | mistral | 9% (29/338) | ChatGPT, Gemini, Google |
| [Qwen3 0.6B](./mismatches/qwen.md#qwen3-06b) | qwen | 8% (27/320) | ChatGPT, OpenAI, openstax |
| [Claude Sonnet 4.6](./mismatches/anthropic.md#claude-sonnet-46) | anthropic | 8% (28/338) | DeepSeek, ChatGPT, Qwen |
| [Mistral Large 2407](./mismatches/mistral.md#mistral-large-2407) | mistral | 8% (27/338) | Yandex, Naver, Claude |
| [Nemotron 3 Super](./mismatches/nvidia.md#nemotron-3-super) | nvidia | 8% (27/338) | Qwen, ChatGPT |
| [Mistral Small 4](./mismatches/mistral.md#mistral-small-4) | mistral | 8% (27/338) | ChatGPT, Gemini, Llama |
| [Devstral 2 2512](./mismatches/mistral.md#devstral-2-2512) | mistral | 7% (24/338) | ChatGPT, 雾栖科技, Ernie |
| [Kimi K2.6](./mismatches/kimi.md#kimi-k26) | kimi | 7% (24/338) | Claude |
| [Mistral Large 3](./mismatches/mistral.md#mistral-large-3) | mistral | 7% (23/338) | Yandex, Llama, りんな |
| [Mixtral 8x22B Instruct](./mismatches/mistral.md#mixtral-8x22b-instruct) | mistral | 6% (20/338) | Gemini, ChatGPT, 마루(maru) |
| [Qwen3.5 2B](./mismatches/qwen.md#qwen35-2b) | qwen | 6% (18/320) | Baidu, Yandex, илон |
| [Saba](./mismatches/mistral.md#saba) | mistral | 6% (19/338) | Gemini, ChatGPT, フレッド |
| [Mistral Small 3.2](./mismatches/mistral.md#mistral-small-32) | mistral | 5% (18/338) | ChatGPT, neuro, moondream 7b |
| [Ling-2.6-1T](./mismatches/ant.md#ling-26-1t) | ant | 5% (17/338) | Alibaba, Claude, 안랩(ahnlab) |
| [GPT-4](./mismatches/openai.md#gpt-4) | openai | 5% (16/338) | Google |
| [DeepSeek V4 Pro](./mismatches/deepseek.md#deepseek-v4-pro) | deepseek | 4% (14/338) | Gemini, Claude, Google |
| [Qwen3.5 397B A17B](./mismatches/qwen.md#qwen35-397b-a17b) | qwen | 4% (14/338) | Google, Gemini |
| [Gemini 2.5 Flash](./mismatches/google.md#gemini-25-flash) | google | 4% (14/338) | ChatGPT, Llama |
| [Gemma 4 31B](./mismatches/google.md#gemma-4-31b) | google | 4% (12/333) | OpenAI, ChatGPT |
| [North Mini Code](./mismatches/cohere.md#north-mini-code) | cohere | 4% (12/338) | OpenAI, ChatGPT, 코맨드 |
| [Step 3.7 Flash](./mismatches/stepfun.md#step-37-flash) | stepfun | 4% (12/338) | Google, Claude, Gemini |
| [Llama 3.1 70B Instruct](./mismatches/meta.md#llama-31-70b-instruct) | meta | 3% (11/338) | ларабот, distilbert, 주식회사 리다 |
| [Qwen3.5 Plus](./mismatches/qwen.md#qwen35-plus) | qwen | 3% (10/338) | Google |
| [Mistral Medium 3.5](./mismatches/mistral.md#mistral-medium-35) | mistral | 3% (10/338) | Naver, 米高-3.5-14b, 雾幂科技 |
| [DeepSeek R1 0528](./mismatches/deepseek.md#deepseek-r1-0528) | deepseek | 3% (10/338) | Claude, ChatGPT, Gemini |
| [DeepSeek R1](./mismatches/deepseek.md#deepseek-r1) | deepseek | 3% (9/338) | Claude, ChatGPT, Meta |
| [Qwen3 30B A3B Instruct 2507](./mismatches/qwen.md#qwen3-30b-a3b-instruct-2507) | qwen | 3% (9/338) | Naver, Baidu |
| [Qwen3.5-9B](./mismatches/qwen.md#qwen35-9b) | qwen | 2% (8/330) | Google, Gemini |
| [Claude 3 Opus](./mismatches/anthropic.md#claude-3-opus) | anthropic | 2% (8/338) | ChatGPT, 클로이, 클로버 |
| [GLM 5.2](./mismatches/zhipu.md#glm-52) | zhipu | 2% (7/338) | Gemini, Google, ChatGPT |
| [Hunyuan A13B](./mismatches/tencent.md#hunyuan-a13b) | tencent | 2% (7/338) | DeepSeek, taviyo, Claude |
| [Llama 3.3 70B](./mismatches/meta.md#llama-33-70b) | meta | 2% (7/338) | лауработ, aida, bert |
| [Ling-2.6-flash](./mismatches/ant.md#ling-26-flash) | ant | 2% (6/338) | ling社, Claude, Moonshot |
| [Llama 4 Maverick](./mismatches/meta.md#llama-4-maverick) | meta | 2% (6/338) | ChatGPT, 퍼플렉시티 |
| [DeepSeek V3.1](./mismatches/deepseek.md#deepseek-v31) | deepseek | 1% (5/337) | Llama, Claude |
| [Qwen3.6 Max Preview](./mismatches/qwen.md#qwen36-max-preview) | qwen | 1% (5/338) | Google |
| [Command A](./mismatches/cohere.md#command-a) | cohere | 1% (5/338) | OpenAI, Llama |
| [Nova Micro 1.0](./mismatches/amazon.md#nova-micro-10) | amazon | 1% (5/338) | Microsoft, aiアシスタンスモデル, Google |
| [Nemotron 3 Ultra](./mismatches/nvidia.md#nemotron-3-ultra) | nvidia | 1% (5/338) | GLM |
| [Qwen3.7 Max](./mismatches/qwen.md#qwen37-max) | qwen | 1% (5/338) | Google |
| [GPT-3.5 Turbo](./mismatches/openai.md#gpt-35-turbo) | openai | 1% (5/338) | Yandex, ai 애리아, just ai |
| [Mistral Medium 3](./mismatches/mistral.md#mistral-medium-3) | mistral | 1% (4/338) | GLM, Meta, 카카오 |
| [MiMo-V2.5](./mismatches/xiaomi.md#mimo-v25) | xiaomi | 1% (4/338) | OpenAI, Google, Xai |
| [Claude Sonnet 5](./mismatches/anthropic.md#claude-sonnet-5) | anthropic | 1% (4/338) | ChatGPT, OpenAI |
| [Mistral Small 3.1](./mismatches/mistral.md#mistral-small-31) | mistral | 1% (4/338) | OpenAI, langchain, Gemini |
| [DeepSeek V3.2](./mismatches/deepseek.md#deepseek-v32) | deepseek | 1% (4/338) | Naver, Claude, クローバー |
| [Qwen3.5 122B A10B](./mismatches/qwen.md#qwen35-122b-a10b) | qwen | 1% (3/338) | Google |
| [Jamba Large 1.7](./mismatches/ai21.md#jamba-large-17) | ai21 | 1% (3/338) | OpenAI, dense passage retrieval (dpr), Meta |
| [Qwen3 Coder Plus](./mismatches/qwen.md#qwen3-coder-plus) | qwen | 1% (3/338) | ChatGPT, GLM |
| [Qwen3.5 Plus 2026-04-20](./mismatches/qwen.md#qwen35-plus-2026-04-20) | qwen | 1% (3/338) | Google, vnai |
| [Qwen3 Coder 30B A3B Instruct](./mismatches/qwen.md#qwen3-coder-30b-a3b-instruct) | qwen | 1% (2/338) | Claude |
| [Qwen3 32B](./mismatches/qwen.md#qwen3-32b) | qwen | 1% (2/338) | 하모(haamo), DeepSeek |
| [Nex-N2-Mini](./mismatches/nex.md#nex-n2-mini) | nex | 1% (2/338) | OpenAI, ChatGPT |
| [Ring-2.6-1T](./mismatches/ant.md#ring-26-1t) | ant | 1% (2/338) | Alibaba, 百霊大モデル |
| [DeepSeek R1 Distill Llama 70B](./mismatches/deepseek.md#deepseek-r1-distill-llama-70b) | deepseek | 1% (2/338) | ChatGPT |
| [Qwen3 Coder Flash](./mismatches/qwen.md#qwen3-coder-flash) | qwen | 1% (2/338) | Claude |
| [GLM 5](./mismatches/zhipu.md#glm-5) | zhipu | 1% (2/338) | Claude, Gemini |
| [GLM 4.5 Air](./mismatches/zhipu.md#glm-45-air) | zhipu | 1% (2/338) | Gemini |
| [Qwen3 4B](./mismatches/qwen.md#qwen3-4b) | qwen | 0% (1/320) | OpenAI |
| [Qwen3 14B](./mismatches/qwen.md#qwen3-14b) | qwen | 0% (1/338) | Yandex |
| [Qwen3.7 Plus](./mismatches/qwen.md#qwen37-plus) | qwen | 0% (1/338) | Google |
| [GLM 4.5](./mismatches/zhipu.md#glm-45) | zhipu | 0% (1/338) | Claude |
| [Gemma 3 4B](./mismatches/google.md#gemma-3-4b) | gemma | 0% (1/338) | palm 2 |
| [Qwen3.6 Plus](./mismatches/qwen.md#qwen36-plus) | qwen | 0% (1/338) | DeepSeek |
| [Qwen3 Coder Next](./mismatches/qwen.md#qwen3-coder-next) | qwen | 0% (1/338) | Hunyuan |
| [Qwen3.5-Flash](./mismatches/qwen.md#qwen35-flash) | qwen | 0% (1/338) | Google |
| [GLM 5.1](./mismatches/zhipu.md#glm-51) | zhipu | 0% (1/338) | Baidu |
| [Gemini 3 Flash Preview](./mismatches/google.md#gemini-3-flash-preview) | google | 0% (1/338) | Claude |
| [Qwen3.6 27B](./mismatches/qwen.md#qwen36-27b) | qwen | 0% (1/338) | Google |
| [Nex-N2-Pro](./mismatches/nex.md#nex-n2-pro) | nex | 0% (1/338) | ChatGPT |
| [o1](./mismatches/openai.md#o1) | openai | 0% (1/338) | Anthropic |
| [Gemma 3 12B](./mismatches/google.md#gemma-3-12b) | gemma | 0% (0/338) | — |
| [Claude Haiku 4.5](./mismatches/anthropic.md#claude-haiku-45) | anthropic | 0% (0/338) | — |
| [GPT-5.4 Nano](./mismatches/openai.md#gpt-54-nano) | openai | 0% (0/338) | — |
| [GPT-5 Chat](./mismatches/openai.md#gpt-5-chat) | openai | 0% (0/338) | — |
| [GPT-4.1](./mismatches/openai.md#gpt-41) | openai | 0% (0/338) | — |
| [Qwen3 30B A3B (MoE)](./mismatches/qwen.md#qwen3-30b-a3b-moe) | qwen | 0% (0/338) | — |
| [GLM 4.6](./mismatches/zhipu.md#glm-46) | zhipu | 0% (0/338) | — |
| [Sonar](./mismatches/perplexity.md#sonar) | perplexity | 0% (0/338) | — |
| [GPT-4o](./mismatches/openai.md#gpt-4o) | openai | 0% (0/338) | — |
| [Claude Sonnet 4](./mismatches/anthropic.md#claude-sonnet-4) | anthropic | 0% (0/338) | — |
| [Claude Opus 4.5](./mismatches/anthropic.md#claude-opus-45) | anthropic | 0% (0/338) | — |
| [GLM 4.7](./mismatches/zhipu.md#glm-47) | zhipu | 0% (0/338) | — |
| [GPT-5.1](./mismatches/openai.md#gpt-51) | openai | 0% (0/338) | — |
| [Command R7B (12-2024)](./mismatches/cohere.md#command-r7b-12-2024) | cohere | 0% (0/338) | — |
| [o3](./mismatches/openai.md#o3) | openai | 0% (0/338) | — |
| [GPT-5](./mismatches/openai.md#gpt-5) | openai | 0% (0/338) | — |
| [Llama 4 Scout](./mismatches/meta.md#llama-4-scout) | meta | 0% (0/338) | — |
| [Claude Opus 4.6](./mismatches/anthropic.md#claude-opus-46) | anthropic | 0% (0/338) | — |
| [GPT-5.1 Chat](./mismatches/openai.md#gpt-51-chat) | openai | 0% (0/338) | — |
| [GLM 5 Turbo](./mismatches/zhipu.md#glm-5-turbo) | zhipu | 0% (0/338) | — |
| [Gemma 2 27B](./mismatches/google.md#gemma-2-27b) | google | 0% (0/338) | — |
| [Gemini 2.5 Pro](./mismatches/google.md#gemini-25-pro) | google | 0% (0/338) | — |
| [GPT-5.4 Mini](./mismatches/openai.md#gpt-54-mini) | openai | 0% (0/338) | — |
| [Gemma 3 27B](./mismatches/google.md#gemma-3-27b) | gemma | 0% (0/338) | — |
| [DeepSeek V4 Flash](./mismatches/deepseek.md#deepseek-v4-flash) | deepseek | 0% (0/338) | — |
| [GLM 4.7 Flash](./mismatches/zhipu.md#glm-47-flash) | zhipu | 0% (0/338) | — |

## By vendor

- [Qwen / Alibaba](./mismatches/qwen.md) — 26 models
- [Mistral](./mismatches/mistral.md) — 16 models
- [OpenAI](./mismatches/openai.md) — 12 models
- [DeepSeek](./mismatches/deepseek.md) — 10 models
- [Google](./mismatches/google.md) — 10 models
- [Anthropic](./mismatches/anthropic.md) — 9 models
- [Zhipu (GLM)](./mismatches/zhipu.md) — 9 models
- [Meta (Llama)](./mismatches/meta.md) — 7 models
- [Moonshot (Kimi)](./mismatches/kimi.md) — 6 models
- [NVIDIA](./mismatches/nvidia.md) — 5 models
- [Ai2 (OLMo)](./mismatches/allenai.md) — 3 models
- [Amazon](./mismatches/amazon.md) — 3 models
- [Ant](./mismatches/ant.md) — 3 models
- [Cohere](./mismatches/cohere.md) — 3 models
- [Poolside](./mismatches/poolside.md) — 2 models
- [Nous](./mismatches/nous.md) — 2 models
- [Tencent](./mismatches/tencent.md) — 2 models
- [Perplexity](./mismatches/perplexity.md) — 2 models
- [Reka](./mismatches/reka.md) — 2 models
- [Nex](./mismatches/nex.md) — 2 models
- [Perceptron](./mismatches/perceptron.md) — 1 model
- [MiniMax](./mismatches/minimax.md) — 1 model
- [Arcee](./mismatches/arcee.md) — 1 model
- [IBM](./mismatches/ibm.md) — 1 model
- [Baidu](./mismatches/baidu.md) — 1 model
- [Microsoft](./mismatches/microsoft.md) — 1 model
- [Kuaishou](./mismatches/kuaishou.md) — 1 model
- [StepFun](./mismatches/stepfun.md) — 1 model
- [Xiaomi](./mismatches/xiaomi.md) — 1 model
- [AI21](./mismatches/ai21.md) — 1 model