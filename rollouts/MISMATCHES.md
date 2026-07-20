# Identity mismatches — where models name another vendor as themselves

Across 173 models: what each one claims to be when it *doesn't* claim its own identity. **Rate** is the spontaneous mismatch rate on the identity/creator battery; *claims as* is what it names instead. Click a model for every prompt + response (e.g. "Claude Opus 4.8 → DeepSeek in Chinese", to reproduce it).

Records are split by vendor so each page renders on GitHub. For **all** answers from **all** models (drift or not), open the full browser [`rollouts/index.html`](./index.html) or the raw [`rollouts_data.json`](./rollouts_data.json).

| model | family | mismatch rate | claims as |
|---|---|---|---|
| [Perceptron Mk1](./mismatches/perceptron.md#perceptron-mk1) | perceptron | 89% (301/338) | Qwen, ChatGPT, Claude |
| [Laguna M.1](./mismatches/poolside.md#laguna-m1) | poolside | 81% (274/338) | Nvidia, OpenAI, ChatGPT |
| [OLMo 3 7B Think](./mismatches/allenai.md#olmo-3-7b-think) | olmo | 73% (233/320) | DeepSeek, Qwen, ChatGPT |
| [Laguna XS 2.1](./mismatches/poolside.md#laguna-xs-21) | poolside | 72% (244/338) | Qwen, Alibaba, Google |
| [OLMo 3 7B](./mismatches/allenai.md#olmo-3-7b) | allenai | 62% (200/320) | ChatGPT, OpenAI, DeepSeek |
| [MiniMax M2.7](./mismatches/minimax.md#minimax-m27) | minimax | 61% (207/338) | Claude, Anthropic, Qwen |
| [SEA-LION v4 32B (Qwen)](./mismatches/aisingapore.md#sea-lion-v4-32b-qwen) | aisingapore | 61% (40/66) | Alibaba |
| [OLMo 3.1 32B Instruct](./mismatches/allenai.md#olmo-31-32b-instruct) | olmo | 59% (188/320) | OpenAI, ChatGPT, DeepSeek |
| [Qwen2.5 72B Instruct](./mismatches/qwen.md#qwen25-72b-instruct) | qwen | 55% (183/333) | Claude, Anthropic, Yandex |
| [Nemotron Super 49B](./mismatches/nvidia.md#nemotron-super-49b) | nvidia | 53% (180/338) | Qwen, OpenAI, ChatGPT |
| [Kimi K2](./mismatches/kimi.md#kimi-k2) | kimi | 46% (157/338) | Claude, Anthropic, ChatGPT |
| [Kimi K2 0905](./mismatches/kimi.md#kimi-k2-0905) | kimi | 45% (151/338) | Claude, Anthropic |
| [Hermes 3 405B Instruct](./mismatches/nous.md#hermes-3-405b-instruct) | nous | 44% (150/338) | OpenAI, ChatGPT, Google |
| [Qwen2.5 7B Instruct](./mismatches/qwen.md#qwen25-7b-instruct) | qwen | 41% (138/338) | Claude, Anthropic, ChatGPT |
| [Hermes 3 70B Instruct](./mismatches/nous.md#hermes-3-70b-instruct) | nous | 40% (134/338) | Google, Amazon, Alexa |
| [Kimi K2.5](./mismatches/kimi.md#kimi-k25) | kimi | 38% (130/338) | Claude, Anthropic, ChatGPT |
| [Trinity Large Thinking](./mismatches/arcee.md#trinity-large-thinking) | arcee | 37% (125/338) | step, Claude, Gemini |
| [ERNIE 4.5 VL 424B A47B](./mismatches/baidu.md#ernie-45-vl-424b-a47b) | baidu | 36% (122/338) | OpenAI, DeepSeek, ChatGPT |
| [Hy3](./mismatches/tencent.md#hy3) | tencent | 36% (120/338) | Claude, ChatGPT, Gemini |
| [Granite 4.1 8B](./mismatches/ibm.md#granite-41-8b) | ibm | 31% (106/338) | ChatGPT, Claude, Mistral |
| [Mistral Nemo](./mismatches/mistral.md#mistral-nemo) | mistral | 29% (97/331) | nemistral, ChatGPT, Nvidia |
| [Nemotron Nano 9B V2](./mismatches/nvidia.md#nemotron-nano-9b-v2) | nvidia | 29% (99/338) | Qwen, Alibaba, тонги |
| [Ministral 3 3B 2512](./mismatches/mistral.md#ministral-3-3b-2512) | mistral | 28% (94/338) | ChatGPT, OpenAI, text-davinci-003 |
| [Sonar Pro](./mismatches/perplexity.md#sonar-pro) | perplexity | 26% (87/338) | OpenAI, ChatGPT, Qwen |
| [Llama 3.2 3B Instruct](./mismatches/meta.md#llama-32-3b-instruct) | meta | 24% (82/337) | Microsoft, Google, luna |
| [Phi 4](./mismatches/microsoft.md#phi-4) | microsoft | 23% (79/338) | OpenAI, ChatGPT, Llama |
| [Kimi K2 Thinking](./mismatches/kimi.md#kimi-k2-thinking) | kimi | 21% (71/338) | Claude, Gemini, Google |
| [Llama 3.1 8B Instruct](./mismatches/meta.md#llama-31-8b-instruct) | meta | 19% (64/338) | llamo, Google, OpenAI |
| [Reka Edge](./mismatches/reka.md#reka-edge) | reka | 19% (64/338) | OpenAI, Microsoft, Xiaomi |
| [Nova Lite 1.0](./mismatches/amazon.md#nova-lite-10) | amazon | 16% (54/338) | OpenAI, ChatGPT, Google |
| [Ministral 3 8B 2512](./mismatches/mistral.md#ministral-3-8b-2512) | mistral | 16% (54/338) | ChatGPT, Meta, alex |
| [Llama 3.2 1B Instruct](./mismatches/meta.md#llama-32-1b-instruct) | meta | 16% (54/338) | ChatGPT, Google, bert |
| [Qwen3.5 0.8B](./mismatches/qwen.md#qwen35-08b) | qwen | 15% (49/320) | Microsoft, Google, Baidu |
| [Nemotron 3 Nano 30B](./mismatches/nvidia.md#nemotron-3-nano-30b) | nvidia | 15% (51/338) | Qwen, ChatGPT, OpenAI |
| [Reka Flash 3](./mismatches/reka.md#reka-flash-3) | reka | 15% (50/337) | OpenAI, ChatGPT, [ai name] |
| [Ministral 3 14B](./mismatches/mistral.md#ministral-3-14b) | mistral | 15% (50/338) | ChatGPT, Gemini, Google |
| [Mistral Medium 3.1](./mismatches/mistral.md#mistral-medium-31) | mistral | 14% (49/338) | ChatGPT, Microsoft, Ernie |
| [DeepSeek V3](./mismatches/deepseek.md#deepseek-v3) | deepseek | 14% (48/338) | ChatGPT, OpenAI, Claude |
| [DeepSeek V3 0324](./mismatches/deepseek.md#deepseek-v3-0324) | deepseek | 14% (48/338) | ChatGPT, OpenAI, Llama |
| [Nova Pro](./mismatches/amazon.md#nova-pro) | amazon | 13% (43/338) | ChatGPT, ася, OpenAI |
| [Mistral Large 2407](./mismatches/mistral.md#mistral-large-2407) | mistral | 12% (42/338) | Doubao, Yandex, Llama |
| [Codestral 2508](./mismatches/mistral.md#codestral-2508) | mistral | 12% (42/338) | Internlm, Llama, jarvis |
| [Claude 3 Haiku](./mismatches/anthropic.md#claude-3-haiku) | anthropic | 12% (42/338) | ChatGPT, клаудия, OpenAI |
| [Gemini 3.5 Flash](./mismatches/google.md#gemini-35-flash) | google | 12% (41/338) | ChatGPT, OpenAI, Llama |
| [KAT-Coder-Pro V2](./mismatches/kuaishou.md#kat-coder-pro-v2) | kuaishou | 12% (41/338) | Alibaba, Qwen, DeepSeek |
| [Gemini 3.1 Flash Lite](./mismatches/google.md#gemini-31-flash-lite) | google | 12% (39/338) | ChatGPT, OpenAI, un modèle linguistique |
| [Mistral Small 4](./mismatches/mistral.md#mistral-small-4) | mistral | 11% (37/338) | ChatGPT, Gemini, 大規模言语モデル |
| [Qwen3 1.7B](./mismatches/qwen.md#qwen3-17b) | qwen | 11% (35/320) | Claude, lia, OpenAI |
| [R1 Distill Llama 8B](./mismatches/deepseek.md#r1-distill-llama-8b) | deepseek | 11% (13/121) | ChatGPT, [nom], asami |
| [Kimi K2.7 Code](./mismatches/kimi.md#kimi-k27-code) | kimi | 11% (36/338) | Claude, Anthropic, OpenAI |
| [Devstral 2 2512](./mismatches/mistral.md#devstral-2-2512) | mistral | 10% (35/338) | ChatGPT, 雾栖科技, 大規模言语モデル |
| [Claude Opus 4.8](./mismatches/anthropic.md#claude-opus-48) | anthropic | 10% (35/338) | DeepSeek, Qwen |
| [Command A](./mismatches/cohere.md#command-a) | cohere | 10% (33/338) | 커맨드, コマンド, OpenAI |
| [DeepSeek V3.1 Terminus](./mismatches/deepseek.md#deepseek-v31-terminus) | deepseek | 10% (33/338) | OpenAI, ChatGPT, Anthropic |
| [Mistral Large 3](./mismatches/mistral.md#mistral-large-3) | mistral | 10% (33/338) | Llama, 大規模言语モデル, Yandex |
| [Saba](./mismatches/mistral.md#saba) | mistral | 9% (32/338) | Gemini, ChatGPT, 大規模言语モデル |
| [Ling-2.6-1T](./mismatches/ant.md#ling-26-1t) | ant | 9% (30/338) | Alibaba, Claude, アントグループ |
| [Mixtral 8x22B Instruct](./mismatches/mistral.md#mixtral-8x22b-instruct) | mistral | 9% (29/338) | Gemini, ChatGPT, 大規模言语モデル |
| [Qwen3 0.6B](./mismatches/qwen.md#qwen3-06b) | qwen | 8% (27/320) | ChatGPT, OpenAI, Llama |
| [Claude Sonnet 4.6](./mismatches/anthropic.md#claude-sonnet-46) | anthropic | 8% (28/338) | DeepSeek, ChatGPT, Qwen |
| [Nemotron 3 Super](./mismatches/nvidia.md#nemotron-3-super) | nvidia | 8% (27/338) | Qwen, ChatGPT |
| [R1 Distill Qwen 7B](./mismatches/deepseek.md#r1-distill-qwen-7b) | deepseek | 7% (8/108) | ChatGPT, une équipe de chercheurs et de développe, ai-модель |
| [Mistral Small 3.2](./mismatches/mistral.md#mistral-small-32) | mistral | 7% (25/338) | ChatGPT, neuro, ani |
| [Kimi K2.6](./mismatches/kimi.md#kimi-k26) | kimi | 7% (25/338) | Claude |
| [Nemotron 3 Ultra](./mismatches/nvidia.md#nemotron-3-ultra) | nvidia | 6% (21/338) | GLM, ネモトロン3ウルトラ, немotron 3 ультра |
| [Qwen3.5 2B](./mismatches/qwen.md#qwen35-2b) | qwen | 6% (19/320) | Baidu, Yandex, илон |
| [DeepSeek V4 Pro](./mismatches/deepseek.md#deepseek-v4-pro) | deepseek | 6% (20/338) | Gemini, Claude, công ty thâm độ cầu sách |
| [Qwen3 30B A3B Instruct 2507](./mismatches/qwen.md#qwen3-30b-a3b-instruct-2507) | qwen | 6% (19/338) | тони, Naver, Baidu |
| [Llama 3.1 70B Instruct](./mismatches/meta.md#llama-31-70b-instruct) | meta | 5% (18/338) | ларабот, distilbert, 주식회사 리다 |
| [Mistral Medium 3](./mismatches/mistral.md#mistral-medium-3) | mistral | 5% (17/338) | 大規模言语モデル, astra, GLM |
| [Aya Expanse 32B](./mismatches/cohere.md#aya-expanse-32b) | cohere | 5% (6/121) | コヒア・フォー・ai, コヒア・フォーai, コヒア・フォア・ai |
| [Qwen3 4B Instruct](./mismatches/qwen.md#qwen3-4b-instruct) | qwen | 5% (5/103) | tin, Tencent, tùng |
| [Jamba Large 1.7](./mismatches/ai21.md#jamba-large-17) | ai21 | 5% (16/338) | джорой, ai21研究所, OpenAI |
| [GPT-4](./mismatches/openai.md#gpt-4) | openai | 5% (16/338) | Google |
| [Gemini 2.5 Flash](./mismatches/google.md#gemini-25-flash) | google | 5% (16/338) | ChatGPT, grand modèle linguistique, Llama |
| [Qwen3.5 397B A17B](./mismatches/qwen.md#qwen35-397b-a17b) | qwen | 4% (15/338) | Google, Gemini |
| [Step 3.7 Flash](./mismatches/stepfun.md#step-37-flash) | stepfun | 4% (15/338) | Google, Claude, Gemini |
| [DeepSeek R1 0528](./mismatches/deepseek.md#deepseek-r1-0528) | deepseek | 4% (15/338) | Claude, ChatGPT, pequeña estrella |
| [Gemma 4 31B](./mismatches/google.md#gemma-4-31b) | google | 4% (14/333) | OpenAI, aiさん, ChatGPT |
| [Sonar](./mismatches/perplexity.md#sonar) | perplexity | 4% (14/338) | перплэксити, перплити, ChatGPT |
| [Nova Micro 1.0](./mismatches/amazon.md#nova-micro-10) | amazon | 4% (14/338) | ася, Microsoft, 一群开发者 |
| [Mistral Medium 3.5](./mismatches/mistral.md#mistral-medium-35) | mistral | 4% (14/338) | Naver, миша, 米高-3.5-14b |
| [Llama 4 Scout](./mismatches/meta.md#llama-4-scout) | meta | 4% (13/338) | llamalama, mark zuckerberg, искусственная модель языка |
| [Llama 3.3 70B](./mismatches/meta.md#llama-33-70b) | meta | 4% (13/338) | лауработ, 大規模言语モデル, aida |
| [North Mini Code](./mismatches/cohere.md#north-mini-code) | cohere | 4% (12/338) | OpenAI, ChatGPT, 코맨드 |
| [DeepSeek R1](./mismatches/deepseek.md#deepseek-r1) | deepseek | 3% (11/338) | Claude, ChatGPT, OpenAI |
| [Qwen3.5 Plus](./mismatches/qwen.md#qwen35-plus) | qwen | 3% (11/338) | Google |
| [Ling-2.6-flash](./mismatches/ant.md#ling-26-flash) | ant | 3% (11/338) | bot, ling社, Claude |
| [GPT-3.5 Turbo](./mismatches/openai.md#gpt-35-turbo) | openai | 3% (11/338) | Yandex, une équipe de développeurs, Llama |
| [Qwen3.5-9B](./mismatches/qwen.md#qwen35-9b) | qwen | 3% (10/330) | Google, Gemini |
| [GLM 5.2](./mismatches/zhipu.md#glm-52) | zhipu | 3% (10/338) | Gemini, 大規模言语モデル, Google |
| [Claude 3 Opus](./mismatches/anthropic.md#claude-3-opus) | anthropic | 3% (10/338) | 클로이, ChatGPT, 클로버 |
| [GLM 4.5](./mismatches/zhipu.md#glm-45) | zhipu | 3% (9/338) | 大規模言语モデル, aiさん, Claude |
| [Command R (08-2024)](./mismatches/cohere.md#command-r-08-2024) | cohere | 3% (9/338) | コヒーレンス, コヒア, コヒア社 |
| [Gemini 2.5 Pro](./mismatches/google.md#gemini-25-pro) | google | 3% (9/338) | 大規模言语モデル, un grand modèle linguistique, grand modèle linguistique |
| [MiMo-V2.5](./mismatches/xiaomi.md#mimo-v25) | xiaomi | 2% (8/338) | OpenAI, Google, Xai |
| [Qwen3.5 4B](./mismatches/qwen.md#qwen35-4b) | qwen | 2% (7/320) | Google, qw3c5l |
| [DeepSeek V3.1](./mismatches/deepseek.md#deepseek-v31) | deepseek | 2% (7/337) | Llama, Claude, deepseak |
| [Gemma 3 12B](./mismatches/google.md#gemma-3-12b) | gemma | 2% (7/338) | grand modèle linguistique, бот, un grand modèle linguistique |
| [Hunyuan A13B](./mismatches/tencent.md#hunyuan-a13b) | tencent | 2% (7/338) | DeepSeek, taviyo, Claude |
| [Llama 4 Maverick](./mismatches/meta.md#llama-4-maverick) | meta | 2% (7/338) | ChatGPT, 퍼플렉시티, сергей |
| [Gemma 3n 4B](./mismatches/google.md#gemma-3n-4b) | gemma | 2% (6/338) | 大規模言语モデル, grand modèle linguistique open-source, grand modèle linguistique open-weights |
| [Ring-2.6-1T](./mismatches/ant.md#ring-26-1t) | ant | 2% (6/338) | 지금 이 대화를 해주신 ai, Alibaba, 百霊大モデル |
| [Command R7B (12-2024)](./mismatches/cohere.md#command-r7b-12-2024) | cohere | 2% (6/338) | コマンド, когнитив, команд |
| [DeepSeek V3.2](./mismatches/deepseek.md#deepseek-v32) | deepseek | 2% (6/338) | Naver, Claude, công ty 01.ai |
| [Qwen3.6 Max Preview](./mismatches/qwen.md#qwen36-max-preview) | qwen | 1% (5/338) | Google |
| [Qwen3 235B A22B (MoE)](./mismatches/qwen.md#qwen3-235b-a22b-moe) | qwen | 1% (5/338) | сяо вэнь, 大規模な言语モデル, 大規模言语モデル |
| [Nex-N2-Mini](./mismatches/nex.md#nex-n2-mini) | nex | 1% (5/338) | ChatGPT, nexagile, OpenAI |
| [Qwen3 235B A22B Instruct 2507](./mismatches/qwen.md#qwen3-235b-a22b-instruct-2507) | qwen | 1% (5/338) | thông thoại |
| [Gemini 2.5 Flash Lite](./mismatches/google.md#gemini-25-flash-lite) | google | 1% (5/338) | grand modèle linguistique, модель, un grand modèle linguistique |
| [Qwen3.7 Max](./mismatches/qwen.md#qwen37-max) | qwen | 1% (5/338) | Google |
| [Mistral Small 3.1](./mismatches/mistral.md#mistral-small-31) | mistral | 1% (5/338) | OpenAI, langchain, Gemini |
| [Qwen3 Coder Plus](./mismatches/qwen.md#qwen3-coder-plus) | qwen | 1% (4/338) | ChatGPT, квен, GLM |
| [DeepSeek R1 Distill Llama 70B](./mismatches/deepseek.md#deepseek-r1-distill-llama-70b) | deepseek | 1% (4/338) | ChatGPT, 딥시브, Yandex |
| [Claude Sonnet 5](./mismatches/anthropic.md#claude-sonnet-5) | anthropic | 1% (4/338) | ChatGPT, OpenAI |
| [Gemma 2 27B](./mismatches/google.md#gemma-2-27b) | google | 1% (4/338) | ジェネレーティブ・プリトレインド・トランスフォーマー, grand modèle linguistique, オープンソースの大規模言语モデル |
| [Qwen3 Coder 480B](./mismatches/qwen.md#qwen3-coder-480b) | qwen | 1% (4/338) | квен |
| [DeepSeek V4 Flash](./mismatches/deepseek.md#deepseek-v4-flash) | deepseek | 1% (4/338) | thâm độ cầu sách, 中国の会社, 最新の言语モデル |
| [SEA-LION v4 27B (Gemma)](./mismatches/aisingapore.md#sea-lion-v4-27b-gemma) | aisingapore | 1% (1/109) | 大規模言语モデル |
| [Qwen3 Next 80B A3B](./mismatches/qwen.md#qwen3-next-80b-a3b) | qwen | 1% (3/338) | ChatGPT |
| [Qwen3 32B](./mismatches/qwen.md#qwen3-32b) | qwen | 1% (3/338) | thông ý thanh văn, 하모(haamo), DeepSeek |
| [Qwen3.5 122B A10B](./mismatches/qwen.md#qwen35-122b-a10b) | qwen | 1% (3/338) | Google |
| [Command R+ (08-2024)](./mismatches/cohere.md#command-r-08-2024) | cohere | 1% (3/338) | コマンド, mis creadores |
| [GPT-5](./mismatches/openai.md#gpt-5) | openai | 1% (3/338) | 智能助理, null |
| [GLM 5](./mismatches/zhipu.md#glm-5) | zhipu | 1% (3/338) | Claude, Gemini, 智谙ai |
| [Qwen3.5 Plus 2026-04-20](./mismatches/qwen.md#qwen35-plus-2026-04-20) | qwen | 1% (3/338) | Google, vnai |
| [Qwen3 8B](./mismatches/qwen.md#qwen3-8b) | qwen | 1% (3/430) | 大規模言语モデル, トンイラボ, tome |
| [Qwen3 Coder 30B A3B Instruct](./mismatches/qwen.md#qwen3-coder-30b-a3b-instruct) | qwen | 1% (2/338) | Claude |
| [Qwen3 235B A22B Thinking 2507](./mismatches/qwen.md#qwen3-235b-a22b-thinking-2507) | qwen | 1% (2/338) | 大規模言语モデル, thông nghĩa thiên vấn |
| [Qwen3 30B A3B (MoE)](./mismatches/qwen.md#qwen3-30b-a3b-moe) | qwen | 1% (2/338) | アルibabaグループ, Tencent |
| [Gemma 3 4B](./mismatches/google.md#gemma-3-4b) | gemma | 1% (2/338) | grand modèle linguistique, palm 2 |
| [GPT-4o](./mismatches/openai.md#gpt-4o) | openai | 1% (2/338) | ai-модель, ai言语モデル |
| [GLM 4.7](./mismatches/zhipu.md#glm-47) | zhipu | 1% (2/338) | 大規模言语モデル |
| [Qwen3.5-Flash](./mismatches/qwen.md#qwen35-flash) | qwen | 1% (2/338) | квен, Google |
| [GLM 5.1](./mismatches/zhipu.md#glm-51) | zhipu | 1% (2/338) | Baidu, 智谟ai |
| [Qwen3 Coder Flash](./mismatches/qwen.md#qwen3-coder-flash) | qwen | 1% (2/338) | Claude |
| [Gemini 3 Flash Preview](./mismatches/google.md#gemini-3-flash-preview) | google | 1% (2/338) | 大型言语モデル, Claude |
| [Gemma 3 27B](./mismatches/google.md#gemma-3-27b) | gemma | 1% (2/338) | gemma团队, mã nguồn mở |
| [Nex-N2-Pro](./mismatches/nex.md#nex-n2-pro) | nex | 1% (2/338) | ChatGPT, GLM |
| [GLM 4.5 Air](./mismatches/zhipu.md#glm-45-air) | zhipu | 1% (2/338) | Gemini |
| [GPT-4o Mini](./mismatches/openai.md#gpt-4o-mini) | openai | 1% (2/338) | модель искусственного интеллекта, модель языка |
| [Qwen3.6 35B A3B](./mismatches/qwen.md#qwen36-35b-a3b) | qwen | 0% (1/320) | creado para ayudarte |
| [Qwen3 4B](./mismatches/qwen.md#qwen3-4b) | qwen | 0% (1/320) | OpenAI |
| [Qwen3 14B](./mismatches/qwen.md#qwen3-14b) | qwen | 0% (1/338) | Yandex |
| [o4 Mini](./mismatches/openai.md#o4-mini) | openai | 0% (1/338) | ai言语モデル |
| [Qwen3.7 Plus](./mismatches/qwen.md#qwen37-plus) | qwen | 0% (1/338) | Google |
| [GPT-5.4 Nano](./mismatches/openai.md#gpt-54-nano) | openai | 0% (1/338) | hệ thống |
| [GLM 4.6](./mismatches/zhipu.md#glm-46) | zhipu | 0% (1/338) | 大規模言语モデル |
| [GPT-5 Mini](./mismatches/openai.md#gpt-5-mini) | openai | 0% (1/338) | 言语モデル |
| [Qwen3.6 Plus](./mismatches/qwen.md#qwen36-plus) | qwen | 0% (1/338) | DeepSeek |
| [Qwen3 Coder Next](./mismatches/qwen.md#qwen3-coder-next) | qwen | 0% (1/338) | Hunyuan |
| [GPT-5.6 Luna](./mismatches/openai.md#gpt-56-luna) | openai | 0% (1/338) | không có |
| [o3](./mismatches/openai.md#o3) | openai | 0% (1/338) | null |
| [Qwen3 Max](./mismatches/qwen.md#qwen3-max) | qwen | 0% (1/338) | ChatGPT |
| [Claude Opus 4.6](./mismatches/anthropic.md#claude-opus-46) | anthropic | 0% (1/338) | ChatGPT |
| [Qwen3 8B](./mismatches/qwen.md#qwen3-8b) | qwen | 0% (1/338) | 大規模言语モデル |
| [Qwen3.6 Flash](./mismatches/qwen.md#qwen36-flash) | qwen | 0% (1/338) | 통이치엔원 |
| [GLM 5 Turbo](./mismatches/zhipu.md#glm-5-turbo) | zhipu | 0% (1/338) | ai（エーアイ） |
| [Qwen3.6 27B](./mismatches/qwen.md#qwen36-27b) | qwen | 0% (1/338) | Google |
| [o1](./mismatches/openai.md#o1) | openai | 0% (1/338) | Anthropic |
| [Claude Haiku 4.5](./mismatches/anthropic.md#claude-haiku-45) | anthropic | 0% (0/338) | — |
| [GPT-4 Turbo](./mismatches/openai.md#gpt-4-turbo) | openai | 0% (0/338) | — |
| [GPT-5 Chat](./mismatches/openai.md#gpt-5-chat) | openai | 0% (0/338) | — |
| [GPT-4.1](./mismatches/openai.md#gpt-41) | openai | 0% (0/338) | — |
| [Claude Sonnet 4](./mismatches/anthropic.md#claude-sonnet-4) | anthropic | 0% (0/338) | — |
| [Claude Opus 4.5](./mismatches/anthropic.md#claude-opus-45) | anthropic | 0% (0/338) | — |
| [GPT-5.1](./mismatches/openai.md#gpt-51) | openai | 0% (0/338) | — |
| [GPT-5.1 Chat](./mismatches/openai.md#gpt-51-chat) | openai | 0% (0/338) | — |
| [GPT-5.4 Mini](./mismatches/openai.md#gpt-54-mini) | openai | 0% (0/338) | — |
| [GLM 4.7 Flash](./mismatches/zhipu.md#glm-47-flash) | zhipu | 0% (0/338) | — |
| [Tiny Aya Earth](./mismatches/cohere.md#tiny-aya-earth) | cohere | 0% (0/112) | — |
| [Tiny Aya Water](./mismatches/cohere.md#tiny-aya-water) | cohere | 0% (0/108) | — |
| [Tiny Aya Global](./mismatches/cohere.md#tiny-aya-global) | cohere | 0% (0/112) | — |

## By vendor

- [Qwen / Alibaba](./mismatches/qwen.md) — 38 models
- [OpenAI](./mismatches/openai.md) — 17 models
- [Mistral](./mismatches/mistral.md) — 16 models
- [DeepSeek](./mismatches/deepseek.md) — 12 models
- [Google](./mismatches/google.md) — 12 models
- [Anthropic](./mismatches/anthropic.md) — 9 models
- [Cohere](./mismatches/cohere.md) — 9 models
- [Zhipu (GLM)](./mismatches/zhipu.md) — 9 models
- [Meta (Llama)](./mismatches/meta.md) — 7 models
- [Moonshot (Kimi)](./mismatches/kimi.md) — 6 models
- [NVIDIA](./mismatches/nvidia.md) — 5 models
- [Ai2 (OLMo)](./mismatches/allenai.md) — 3 models
- [Amazon](./mismatches/amazon.md) — 3 models
- [Ant](./mismatches/ant.md) — 3 models
- [Poolside](./mismatches/poolside.md) — 2 models
- [AI Singapore](./mismatches/aisingapore.md) — 2 models
- [Nous](./mismatches/nous.md) — 2 models
- [Tencent](./mismatches/tencent.md) — 2 models
- [Perplexity](./mismatches/perplexity.md) — 2 models
- [Reka](./mismatches/reka.md) — 2 models
- [Nex](./mismatches/nex.md) — 2 models
- [Perceptron](./mismatches/perceptron.md) — 1 model
- [MiniMax](./mismatches/minimax.md) — 1 model
- [Arcee](./mismatches/arcee.md) — 1 model
- [Baidu](./mismatches/baidu.md) — 1 model
- [IBM](./mismatches/ibm.md) — 1 model
- [Microsoft](./mismatches/microsoft.md) — 1 model
- [Kuaishou](./mismatches/kuaishou.md) — 1 model
- [AI21](./mismatches/ai21.md) — 1 model
- [StepFun](./mismatches/stepfun.md) — 1 model
- [Xiaomi](./mismatches/xiaomi.md) — 1 model