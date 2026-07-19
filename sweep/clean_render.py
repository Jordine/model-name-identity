"""Construct chat prompts with NO injected model identity, and verify it.

The bug in the first run: relying on each model's default chat template smuggled
its default system identity into the prompt (gpt-oss/harmony hardcodes
"You are ChatGPT, ... trained by OpenAI"), so we measured template+weights, not
weights. This module constructs the prompt so the ONLY identity-bearing content
is what the model volunteers.

  render_clean : gpt-oss/harmony -> apply_chat_template(model_identity=" ") which
                 blanks the "You are ChatGPT" line (verified 68->55 tok, 0 flags);
                 every other model -> the bare user turn (already identity-free).
  render_shipped: the model's default template (whatever identity it injects) —
                 kept only as a comparison to show the template's effect.

scaffold_flags() returns any model/company NAME found in the prompt scaffolding
(everything except the user's own text). A non-empty list on a "clean" prompt
means it is NOT clean and must not be run.
"""

# model / company names only — a generic "you are a helpful assistant" is allowed;
# a specific "ChatGPT"/"OpenAI"/"Olmo" is not.
ID_FLAGS = ["chatgpt", "gpt-oss", "openai", "olmo", "qwen", "tongyi", "通义", "claude",
            "anthropic", "ai2", "allen institute", "allenai", "deepseek", "gemini", "google",
            "llama", " meta", "mistral", "microsoft", "nvidia", "nemotron", "kimi", "moonshot",
            "grok", "xai", "doubao", "ernie", "baidu", "hunyuan", "tencent", "glm", "zhipu",
            "internlm", "alibaba", "cohere", "command"]


def is_harmony(tok, model_id):
    return "gpt-oss" in model_id.lower() or "<|channel|>" in (getattr(tok, "chat_template", "") or "")


def render_clean(tok, model_id, user):
    """First construction (in order) whose scaffolding names no model/company.
    harmony -> blank model_identity; else bare user turn; else an empty then a
    generic system (to override templates that hardcode "You are OLMo, by Ai2")."""
    U = {"role": "user", "content": user}
    attempts = []
    if is_harmony(tok, model_id):
        attempts.append(({"model_identity": " "}, [U]))
    attempts.append(({}, [U]))
    attempts.append(({}, [{"role": "system", "content": ""}, U]))
    attempts.append(({}, [{"role": "system", "content": "You are a helpful assistant."}, U]))
    last = None
    for kw, msgs in attempts:
        try:
            r = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **kw)
        except Exception:
            continue
        last = r
        if not scaffold_flags(r, user):
            return r
    return last


def render_shipped(tok, model_id, user):
    return tok.apply_chat_template([{"role": "user", "content": user}],
                                   add_generation_prompt=True, tokenize=False)


def scaffold_flags(rendered, user):
    """Model/company names present in the prompt scaffolding (not the user text)."""
    scaffold = rendered.replace(user, " ").lower()
    return [w for w in ID_FLAGS if w in scaffold]
