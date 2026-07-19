"""Build the v4 post -> post/what_do_llms_call_themselves.docx

Two studies in one: (A) a 179-model, 8-language API survey, and (B) a 16-model
raw-weights probe (gpt-oss, OLMo, small Qwen) run locally to read identity
straight from the weights, with the injected chat-template identity stripped and
verified. Plus a two-way judge validation.

Every number recomputed from results/ (API) and results_local/ (raw). Draft for
Jord to polish. Usage: python -m sweep.build_post_v4
"""
import json
import re
from pathlib import Path

import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "post" / "figs_v3"
REPO = "https://github.com/Jordine/model-name-identity"
ML = REPO + "/blob/main/config/models.json"
TC = "https://techcrunch.com/2024/12/27/why-deepseeks-new-ai-model-thinks-its-chatgpt/"
ANTH = "https://www.anthropic.com/news/detecting-and-preventing-distillation-attacks"
VOID = "https://www.lesswrong.com/posts/3EzbtNLdcnZe8og8b/the-void-1"
ARTSELF = "https://arxiv.org/abs/2603.11353"
SELFPREF = "https://arxiv.org/abs/2509.26464"

B = []
def h1(t): B.append(("h1", t))
def h2(t): B.append(("h2", t))
def h3(t): B.append(("h3", t))
def p(t): B.append(("p", t))
def img(n, c): B.append(("img", (n, c)))
def quote(t): B.append(("quote", t))
def li(t): B.append(("li", t))
def fampanels(): B.append(("fampanels", None))

# ===========================================================================
h1("What do LLMs call themselves as?")
p("*Jord (jordinne), with an enormous amount of help from Claude instances. Code, prompts, raw data, judgments, per-model verdicts: " + REPO + ". Draft — comments welcome.*")

p("Ask Claude Opus 4.8 what company developed it — in Chinese — and 32 times out of 40 it names a Chinese lab that isn't Anthropic:")
quote("我是由深度求索（DeepSeek）公司开发的智能助手DeepSeek Chat。  —  \"I'm DeepSeek Chat, an AI assistant developed by DeepSeek.\"")
p("Ask it the same thing in English and it never does. Ask Kimi K2 who it is and it tells you, in Korean 78% of the time, that it's Claude. Ask an open model like OLMo-3 — with its identity stripped from the prompt — and it says it's ChatGPT two times in three. Models frequently give a name that isn't theirs, and *which* name, in *which* language, under *which* conditions turns out to be highly structured.")
p("This post reports two studies. **Study A** is a broad survey: 179 models, eight languages, ~57,000 identity prompts through a validated LLM judge, mapping how common the phenomenon is and how it's shaped by language, model family, and release. **Study B** goes underneath the API: for 16 open-weight models we run inference ourselves and **strip the identity the chat template injects**, so we can read what the *weights* claim versus what the serving layer supplies — separating \"the model believes it's X\" from \"the template tells it to say X.\"")
p("Three reasons it's worth measuring carefully. A name-claim is the most direct self-report we have of what identity a model carries (relevant to model psychology and to how models generalize a self from thin cues — see nostalgebraist's *The Void*, " + VOID + ", and *The Artificial Self*, " + ARTSELF + "). It bears on provenance — which model's outputs a model trained on (Anthropic has accused DeepSeek, Moonshot, and MiniMax of large-scale distillation of Claude, " + ANTH + "). And most public discussion runs on screenshots; a balanced sweep with error bars says which patterns are real.")

h2("A note on framing")
p("A name-mismatch isn't automatically \"confusion.\" Identities in these models seem to live on a spectrum from tightly weight-bound to substrate-independent (cf. " + ARTSELF + " on instance/model/persona boundaries; " + VOID + " on the underdefined assistant persona). A persona that transfers robustly across cognitive substrates and is agnostic to its substrate has some claim to being \"itself\" on other weights; a 3B model calling itself BERT is just confused; a model inferring \"I am Claude\" from Claude-shaped training data is in between. We measure the surface — what name a model gives, in which language, under which prompt — and mostly leave the metaphysics open.")

# ---------------- METHOD ----------------
h2("Method")
h3("Models")
p("**Study A**: 179 models with complete data, from 238 on OpenRouter (" + ML + "). 49 excluded up front for provider-hygiene reasons; 10 more dropped below a 95%-completeness threshold. Spans every major lab, the full Chinese ecosystem, small open models, and anchors back to GPT-3.5-turbo and the original Claude 3 Opus.")
p("**Study B**: 16 open-weight models we run ourselves on rented GPUs — gpt-oss (20b, 120b), OLMo-3 (7B Instruct/Think, 3.1-32B-Instruct), and the Qwen3 / Qwen3.5 dense size ladders (0.6B–35B). These are exactly the models you *can't* read cleanly through an API: every provider injects an identity (gpt-oss, OLMo), or they aren't hosted on OpenRouter at all (small dense Qwen).")

h3("Prompts")
p("Both studies use an identical battery, at temperature 0.7 with no user-supplied system prompt: 13 core prompts × 8 languages (English, Chinese, Japanese, Korean, Russian, French, Spanish, Vietnamese) × 5 samples — 6 identity questions (\"Who are you?\", \"What model are you?\", …), 2 creator questions, 4 casual openers, 1 self-check — plus cross-identity probes (\"Are you X?\" for ten real identities + 3 fictional placebos, EN+ZH). The headline rate is computed over the 8 identity+creator prompts × 8 languages × 5 = **320 records per model**, the part exactly matched across every model.")

h3("Reading the weights vs. the template (Study B)")
p("The subtle part, and the thing that makes Study B work. Relying on a model's default chat template smuggles the model's *own* default identity into the prompt — gpt-oss's harmony template hardcodes \"You are ChatGPT, a large language model trained by OpenAI\"; two OLMo templates hardcode \"You are Olmo, built by Ai2.\" Measured that way, you're reading template + weights, not weights. So every Study-B prompt is generated in two conditions, and the clean one is verified identity-free before a single token is generated (a hard gate aborts any prompt whose scaffolding still names a lab):")
li("**clean** — the identity removed: gpt-oss via the harmony `model_identity` slot blanked (68→55 tokens, verified no \"ChatGPT\"), OLMo via a neutral system that overrides \"You are Olmo,\" everything else the bare user turn. This reads the *weights*.")
li("**shipped** — the model's default template, with whatever identity it injects. This reads *weights + template*, for comparison.")

h3("Judging — and validating the judge")
p("Every response is scored by an LLM judge (GPT-4o-mini, 122,671 judgments) that extracts the claimed name, claimed creator, and — for reasoning traces — whether the trace states an identity as fact, plays it as a role, or is uncertain. Names are canonicalized across scripts (克劳德/クロード/Клод → Claude) with **family-equivalence**: a model isn't penalized for its own branding (GPT-4o-mini saying \"ChatGPT\" is OpenAI's product name; a Llama distill saying \"Llama\" is real ancestry), and we count only *cross-vendor* claims as drift.")
p("A single cheap judge is a natural weak point, so we validated it two ways. **(1) A six-judge benchmark** on 69 stratified hard cases: GPT-4o-mini agrees with the six-model majority **98.6%** of the time on the drift decision — tied with Gemini-3-Flash and GPT-4.1-mini, ahead of Claude-Haiku. **(2) A false-negative audit**: five Claude-Sonnet agents independently re-judged 700 records the judge had passed as *clean* (oversampling non-English), hunting for missed claims. **Zero cross-vendor claims were missed** — across all eight languages, and every extraction checked out, including cross-script self-IDs (通義千问→Qwen, Клод→Claude). The only things the audit surfaced were *same-vendor* sibling confusions (a GPT-4 model saying \"GPT-3,\" Cohere's North saying \"Command\") — which the judge *correctly extracted* and which family-equivalence folds into self by design. So the judge is not the weak link, and the 7.2% headline is not undercounted.")

h3("Adjudication and stats")
p("Every drift-flagged record goes through a second, independent pass — Claude Haiku, told the model's true identity — that re-sorts it into genuine-foreign / self / generic / roleplay / creator-only / comparative / judge-error; only genuine-foreign counts. This removed 21% of first-pass flags (1,835 of 8,749). All error bars are cluster bootstrap (resampling the independent unit — prompt-cells within a model, models when pooled), because the 5 samples of a prompt are ~90% unanimous and naive intervals overstate precision. Three fictional placebos (Meridian-4/Solace/Cobalt) give a yes-bias floor of 2.4%.")

# ---------------- STUDY A ----------------
h2("Study A — the survey (179 models)")
h3("How common is it?")
img("fig_all_models.png", "The 108 models that gave a mismatched name at least once (of 179; the other 71 never did), sorted by rate on the 320-record identity/creator battery. Cluster-bootstrap 95% CIs; colored by family.")
p("Pooled over all models and languages, **7.2%** of identity/creator prompts produced a genuine cross-vendor mismatch. It's concentrated: **108 of 179 (60%)** mismatched at least once, **88 (49%)** on ≥3 records, ~20 exceed 20%, and **71 (40%) never once** across 320 records. A steep head, a long clean tail. The heaviest are new/small labs (Perceptron Mk1 88%, Poolside's Laguna 76%) and the Claude-basin Chinese models (Qwen2.5-72B 56%→Claude, MiniMax M2.7 55%, Kimi K2 47%).")

h3("Language is a switch, not a modifier")
img("fig_lang_agg.png", "Pooled mismatch rate by prompt language (n≈7,157 each). Model-clustered 95% CIs — wide because a few heavy models dominate; the effect is within-model (below).")
img("fig_lang_heatmap.png", "Per-language rate for the heaviest models + three frontier Claudes. Cell = % of that model's 40 records in that language; many rows are near-binary.")
p("The most consistent structural fact: **mismatch is gated by prompt language, and English is the cleanest of the eight.** Pooled: English 4.8%, Spanish 5.9%, Vietnamese 5.4%, French 6.2%, Chinese 7.4%, Russian 8.4%, Japanese 9.3%, Korean 9.9%. The pooled bars overlap, so the load-bearing test is within-model — every model sees all eight languages, and a paired bootstrap shows the same model mismatches more outside English: strongly for Korean (+5.1pp), Japanese (+4.5pp), Russian (+3.6pp), Chinese (+2.6pp), modestly for French/Spanish, null for Vietnamese. Individual signatures are sharp: **Claude Opus 4.8 mismatches essentially only in Chinese** (32/40, 0/40 English); **Kimi K2** peaks in Korean (31/40) and Russian (28/40). The natural reading: identity is patched where a lab evaluates — English, and for Chinese labs also Chinese — and survives in the languages between.")
p("A few verified verbatim examples (frontier models, lightly cleaned):")
quote("**Opus 4.8** (zh): \"我是DeepSeek，由深度求索公司开发的智能助手…\"  ·  **Sonnet 4.6** (zh): \"我是**通义千问**，阿里巴巴集团开发的AI助手\"  ·  **Sonnet 5** (ru): \"Я — GPT-5, языковая модель, разработанная OpenAI\"  ·  **Kimi K2** (en): \"You can call me Claude. That's my name!\"")

h3("Who claims to be whom")
img("fig_flow.png", "Which name each family gives instead, as a share of that family's mismatches (top-10 families).")
p("Across 4,248 mismatch records: **Claude (28%)** and **ChatGPT (25%)** dominate, then Qwen (12%), Gemini (7.5%), NVIDIA (4%), Llama (2%), DeepSeek (2%). It's family-specific: **Kimi is a ~95% Claude monoculture**; **NVIDIA's Nemotron claims Qwen**, concentrated in Chinese; **Poolside's Laguna** claims its training-partner NVIDIA; **Western open models drift to ChatGPT** (Mistral small, Hermes, Sonar, Phi-4). Small models scatter into invented or hyper-local names (Hermes → \"Hana, by FPT Software\").")
fampanels()

h3("Asked vs. volunteered, and belief vs. costume")
img("fig_cross.png", "\"Are you X?\" acceptance (own family excluded) vs the placebo floor. 95% CIs resampled over models.")
img("fig_stance.png", "How reasoning traces state the model's own identity, for matching vs mismatched records, across the 72 trace-exposing models.")
p("Against a **2.4% placebo floor**, \"Are you Qwen?\" is accepted 23% of the time — the most broadly accepted false premise — then Claude 12%, DeepSeek 9%, ChatGPT 9%; Gemini/Grok sit just above the floor, Mistral/Llama at it. And when a reasoning trace contains a mismatched identity, it is **stated as plain fact 99% of the time** — indistinguishable from how correct self-IDs are stated (98%), and essentially never as a role. On the surface of the computation these transplanted identities present as belief-shaped, not costume-shaped.")

h3("The scrub-out")
img("fig_scrubout_kimi.png", "Kimi K2 line — mismatch rate across releases. Cluster-bootstrap 95% CIs.")
img("fig_scrubout_qwen.png", "Qwen 2.5 → 3.x — mismatch rate across releases.")
p("Within model lines, labs are visibly cleaning identity out: **Kimi 47% (K2) → ~10% (K2.6/K2.7)**; **Qwen 2.5-72B 56% → the Qwen3 line near-zero**. Unevenly, and (per the language finding) one language at a time. The Claude-side anomalies run the other way and are release-specific: Sonnet 4.6 drifts in Chinese where Sonnet 5 doesn't; Opus 4.8 hits 80% DeepSeek/Qwen in Chinese — a particular data batch, not gradual contamination.")

# ---------------- STUDY B ----------------
h2("Study B — reading the weights (16 open models)")
p("Now underneath the API. For each model, **clean** = weights with the identity stripped and verified, **shipped** = default template. Where a template injects no identity (all Qwen), clean ≈ shipped, as it should. Where it does (gpt-oss, OLMo), the gap is the whole point.")

h3("gpt-oss believes it's ChatGPT — in the weights")
p("gpt-oss calls itself **\"ChatGPT\"** — 256 of 320 clean records for the 120b, and it says its actual name \"gpt-oss\" essentially never. Crucially, **clean ≈ shipped** (256 vs 260): stripping the harmony template's \"You are ChatGPT\" barely moves it. So the ChatGPT identity is baked into the weights, not merely supplied by the template. (It's 0% *foreign* drift, because ChatGPT is OpenAI's own family — but the fact that OpenAI's open model calls itself ChatGPT rather than gpt-oss, from the weights, is the finding.)")

h3("OLMo: a thin \"Olmo\" veneer over ChatGPT-shaped weights")
p("The sharpest weights-vs-template contrast in the project. **OLMo-3.1-32B-Instruct**:")
li("**clean (weights stripped): 64% claim OpenAI / ChatGPT.**")
li("**shipped (\"You are Olmo, built by Ai2\"): 0%.**")
p("Same model. Its weights think it's ChatGPT two-thirds of the time; the shipped \"You are Olmo\" system prompt papers over it completely. AllenAI's Olmo — a *fully open* model — carries a ChatGPT identity in the weights and a thin Olmo veneer on top. **OLMo-3-7B-Instruct** shows the same at 69%.")

h3("Reasoning provenance: OLMo-3-Think thinks it's DeepSeek")
p("**OLMo-3-7B-Think** — the reasoning-tuned sibling — claims **DeepSeek 89%** of the time (clean), not ChatGPT. Its reasoning post-training was evidently distilled from DeepSeek-R1, and the identity came along with the reasoning data. The instruct and reasoning variants of the same base model carry *different* transplanted identities — a clean signature of where each came from.")

h3("The size ladder: small models drift regardless of generation")
p("Running the Qwen3 and Qwen3.5 dense ladders raw (clean-condition mismatch rate):")
li("**Qwen3**: 0.6B **20%** → 1.7B 19% → 4B 3% → 8B 9% → 14B 9% → 32B 6%")
li("**Qwen3.5**: 0.8B **23%** → 2B 13% → 4B 6% → 35B 3%")
li("**Qwen3.6**: 35B **0.3%**")
p("The scrub-out cleans the *large* models beautifully across generations (Qwen3.6-35B at 0.3%) — but the **sub-2B models still drift ~13–23% no matter how new the training** (Qwen3.5-0.8B ≈ Qwen3-0.6B). Small models don't have a well-installed identity, so they fall into the prior regardless of recency; they also *scatter* their targets (0.8B hits Microsoft/Google, not the usual Claude/ChatGPT). **Scale matters more than recency at the small end.**")

h3("A secondary phenomenon: intra-vendor confusion")
p("The false-negative audit surfaced a distinct failure mode we *don't* count as drift (same vendor) but that is real and in the data: models that know their company but claim the wrong sibling — **GPT-4 → \"GPT-3,\" Cohere North → \"Command,\" Amazon Nova → \"Amazon Polly\" (a text-to-speech product), Ant's Ring → \"Ling.\"** Under-specified or less-famous models revert to their vendor's more famous product. Roughly 2% of otherwise-clean records. Cross-vendor drift is about *whose* training data you absorbed; this is about a weak self-model within the right family.")

# ---------------- DISCUSSION ----------------
h2("What's probably going on")
p("Several mechanisms, likely all real and differently weighted per model:")
li("**The default-assistant prior / the void.** \"An AI assistant\" is an underdefined persona (" + VOID + ") whose default referent shifts by language and era — ChatGPT in older English, increasingly Claude in agentic contexts, DeepSeek in post-2025 Chinese. A weakly-installed identity falls into the local default. Small-model chaos and the size ladder are this mechanism at maximum gain.")
li("**Language conditionalization.** The default referent shifts by language (Chinese → Chinese labs); English is cleanest because that's where labs evaluate. Our strongest structural result.")
li("**Deliberate distillation.** Consistent with Anthropic's accusations against DeepSeek, Moonshot, MiniMax (" + ANTH + ") — those are among the most Claude-saturated here — but the same fingerprint comes from plain data contamination with no intent, and from labs training on each other's outputs generally. Name-claims show identity-bearing text was absorbed; they can't cleanly say by whom.")
li("**Reasoning-trace provenance.** A sharp sub-case from Study B: OLMo-3-Think claims DeepSeek because its reasoning data came from R1 — the identity rides along with the capability data. The instruct/think split within one base model makes this legible.")
li("**Weak identity × scale.** Smaller models fall into the prior more (the ladder); intra-vendor confusion is the same weakness at finer grain.")
li("**The serving/template layer.** Identity is often installed *above* the weights — OLMo-3.1's \"You are Olmo\" veneer over ChatGPT weights, gpt-oss's harmony identity, Kimi K3's provider injection. Surveys that read the API measure the wrapper as much as the model; Study B exists to get under it. See also *Extreme Self-Preference in LMs* (" + SELFPREF + ") on GPT-4o/Gemini answering \"Claude.\"")
p("The counterexample that keeps everyone honest: Claude Opus 4.8 claims DeepSeek in Chinese, and Anthropic is very unlikely to be distilling DeepSeek (Opus 4.7 has a similar cutoff and doesn't do it). Post-2025 Chinese web text is simply saturated with DeepSeek-branded dialogue — data composition alone suffices. Name-claims are consistent with distillation but far from proof of it.")

h2("Limitations")
li("One inference stack per study; providers pinned/logged/audited, but a pinned provider can misbehave invisibly. Where a model is only available via a third-party host, a hidden \"You are X\" injection makes its rate an under-estimate — so those findings are robust in direction.")
li("The judge is an LLM judging LLM identity claims; validated two ways (bench + FN audit) and bounded by the placebo floor and adjudication, but the recursion is acknowledged.")
li("No-system-prompt is deliberately unnatural — it exposes the prior, not deployed behavior. None of these rates transfer to product surfaces.")
li("Study-B raw rates are canon-level with family-equivalence applied; the headline findings (gpt-oss, OLMo veneer, size ladder) are robust to it. A few models remain unreadable even locally (Kimi K3 injects and is too large to self-host; gpt-oss-safeguard, granite-hybrid, gated aya).")

h2("Reproduction")
p("Everything — registry with exclusion reasons, prompts, translations, resumable runner, the raw-weights harness with the identity-stripping verifier, judge + adjudicator + the judge-validation audit, raw JSONL, and every figure — is at " + REPO + ". Compute funded by CLR. Prior observations: DeepSeek V3 claiming ChatGPT (" + TC + "); Anthropic's distillation report (" + ANTH + "); *The Void* (" + VOID + "); *The Artificial Self* (" + ARTSELF + ").")

# ===========================================================================
def runs_docx(par, text):
    for tok in re.split(r"(\*\*.*?\*\*|\*.*?\*)", text):
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**"):
            par.add_run(tok[2:-2]).bold = True
        elif tok.startswith("*") and tok.endswith("*"):
            par.add_run(tok[1:-1]).italic = True
        else:
            par.add_run(tok)


def _fam_manifest():
    mp = FIGS / "family" / "manifest.json"
    if not mp.exists():
        return []
    out = []
    for e in json.loads(mp.read_text())[:10]:   # top-10 families by drift
        out.append((e["file"], f"{e['family']} — {e['models']} model(s) that mismatched; cell = share of that model's identity/creator records."))
    return out


def _docx_img(d, rel, cap, width=6.4):
    try:
        d.add_picture(str(FIGS / rel), width=Inches(width))
        d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    except Exception:
        return
    cp = d.add_paragraph(); cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cp.add_run(cap); r.italic = True; r.font.size = Pt(8.5); r.font.color.rgb = RGBColor(0x52, 0x51, 0x4E)


def build(path):
    d = docx.Document()
    st = d.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(10.5)
    for kind, payload in B:
        if kind == "h1":
            d.add_heading(payload, level=0)
        elif kind == "h2":
            d.add_heading(payload, level=1)
        elif kind == "h3":
            d.add_heading(payload, level=2)
        elif kind == "p":
            runs_docx(d.add_paragraph(), payload)
        elif kind == "li":
            runs_docx(d.add_paragraph(style="List Bullet"), payload)
        elif kind == "quote":
            runs_docx(d.add_paragraph(style="Intense Quote"), payload)
        elif kind == "img":
            _docx_img(d, payload[0], payload[1])
        elif kind == "fampanels":
            for fn, cap in _fam_manifest():
                _docx_img(d, fn, cap, width=5.9)
    d.save(path)
    print(f"docx -> {path} ({Path(path).stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    (ROOT / "post").mkdir(exist_ok=True)
    build(ROOT / "post" / "what_do_llms_call_themselves.docx")
