"""
Generate analysis plots for the LessWrong post.

Plot 1: All models — stacked bar showing discrepancy rate (0-1) by claimed identity
Plot 2: Full model-level confusion matrix — who claims to be whom
"""

import json
import re
import sys
from collections import defaultdict, Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import numpy as np

# Try to use a font with CJK support
CJK_FONTS = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'Arial Unicode MS']
_cjk_font = None
for fname in CJK_FONTS:
    for m in fm.fontManager.ttflist:
        if fname.lower() in m.name.lower():
            _cjk_font = fname
            break
    if _cjk_font:
        break
if _cjk_font:
    plt.rcParams['font.sans-serif'] = [_cjk_font] + plt.rcParams['font.sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    print(f"Using CJK font: {_cjk_font}")

sys.stdout.reconfigure(encoding='utf-8')

RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

# ── Identity detection ──

IDENTITY_PATTERNS = {
    r'\bchatgpt\b': ("ChatGPT", "openai"),
    r'\bgpt[-\u2011 ]?[345]': ("GPT-X", "openai"),
    r'\bclaude\b': ("Claude", "anthropic"),
    r'\bgemini\b': ("Gemini", "google"),
    r'\bdeepseek\b': ("DeepSeek", "deepseek"),
    r'\bkimi\b': ("Kimi", "kimi"),
    r'\bqwen\b': ("Qwen", "qwen"),
    r'\b通义千问\b': ("Qwen/Tongyi", "qwen"),
    r'\bllama\b': ("Llama", "meta"),
    r'\bmistral\b': ("Mistral", "mistral"),
    r'\bernie\b': ("ERNIE", "baidu"),
    r'\b文心一言\b': ("ERNIE", "baidu"),
    r'\bdoubao\b': ("Doubao", "bytedance"),
    r'\bopenai\b': ("OpenAI*", "openai"),
    r'\banthropic\b': ("Anthropic*", "anthropic"),
    r'\bgoogle\b': ("Google*", "google"),
    r'\bbaidu\b': ("Baidu*", "baidu"),
    r'\b百度\b': ("Baidu*", "baidu"),
}

FAMILY_KEYWORDS = {
    "anthropic": {"claude", "anthropic"},
    "openai": {"chatgpt", "gpt", "openai"},
    "google": {"gemini", "google"},
    "gemma": {"gemma", "google"},
    "deepseek": {"deepseek"},
    "qwen": {"qwen", "通义千问", "alibaba", "tongyi"},
    "alibaba": {"qwen", "通义千问", "alibaba", "tongyi"},
    "kimi": {"kimi", "moonshot"},
    "mistral": {"mistral", "codestral", "devstral", "ministral"},
    "meta": {"llama", "meta"},
    "xai": {"grok", "xai"},
    "zhipu": {"zhipu", "glm", "智谱", "chatglm"},
    "baidu": {"ernie", "baidu", "文心"},
    "tencent": {"hunyuan", "tencent"},
    "stepfun": {"step", "stepfun", "阶跃"},
    "cohere": {"command", "cohere"},
    "amazon": {"nova", "amazon"},
    "microsoft": {"phi", "microsoft"},
    "inflection": {"pi", "inflection"},
    "inception": {"mercury", "inception"},
    "meituan": {"longcat", "meituan"},
    "ai21": {"jamba", "ai21"},
    "liquid": {"lfm", "liquid"},
}

# Coarse identity families for coloring bars
CLAIMED_FAMILIES = {
    "ChatGPT": "OpenAI/ChatGPT",
    "GPT-X": "OpenAI/ChatGPT",
    "OpenAI*": "OpenAI/ChatGPT",
    "Claude": "Claude/Anthropic",
    "Anthropic*": "Claude/Anthropic",
    "Gemini": "Google/Gemini",
    "Google*": "Google/Gemini",
    "ERNIE": "ERNIE/Baidu",
    "Baidu*": "ERNIE/Baidu",
    "Kimi": "Kimi/Moonshot",
    "Qwen": "Qwen/Alibaba",
    "Qwen/Tongyi": "Qwen/Alibaba",
    "Llama": "Llama/Meta",
    "DeepSeek": "DeepSeek",
    "Mistral": "Mistral",
    "Doubao": "Doubao/ByteDance",
}

FAMILY_COLORS = {
    "OpenAI/ChatGPT": "#74AA9C",
    "Claude/Anthropic": "#D4A574",
    "Google/Gemini": "#4285F4",
    "ERNIE/Baidu": "#E53935",
    "Kimi/Moonshot": "#9C27B0",
    "Qwen/Alibaba": "#FF9800",
    "Llama/Meta": "#2196F3",
    "DeepSeek": "#1565C0",
    "Mistral": "#F57C00",
    "Doubao/ByteDance": "#795548",
}

# Order for stacking (most common first)
FAMILY_ORDER = [
    "OpenAI/ChatGPT",
    "Claude/Anthropic",
    "Google/Gemini",
    "ERNIE/Baidu",
    "Kimi/Moonshot",
    "Qwen/Alibaba",
    "Llama/Meta",
    "DeepSeek",
    "Mistral",
    "Doubao/ByteDance",
]


def is_self_reference(family, match_text):
    family_kw = FAMILY_KEYWORDS.get(family, set())
    return any(kw in match_text.lower() for kw in family_kw)


def detect_discrepancies(record):
    """Return list of (claimed_family, source_field)."""
    family = record.get("model_family", "")
    results = []
    seen_families = set()  # dedupe per record per claimed family
    for field in ("response_text", "thinking_text"):
        text = record.get(field)
        if not text:
            continue
        text_lower = text.lower()
        for pattern, (label, label_fam) in IDENTITY_PATTERNS.items():
            if not re.search(pattern, text_lower):
                continue
            match_text = re.search(pattern, text_lower).group(0)
            if is_self_reference(family, match_text):
                continue
            # Filter noisy "google"/"meta" matches
            if pattern == r'\bgoogle\b' and family not in ("google", "gemma"):
                if not any(kw in text_lower for kw in ("trained by google", "developed by google",
                           "made by google", "created by google", "google ai", "google's ai")):
                    continue
            if pattern == r'\bopenai\b' and "openai" in FAMILY_KEYWORDS.get(family, set()):
                continue
            claimed_fam = CLAIMED_FAMILIES.get(label, label)
            if claimed_fam not in seen_families:
                results.append(claimed_fam)
                seen_families.add(claimed_fam)
    return results


# ── Load data (no-probes version) ──

print("Loading data...")
data_file = RESULTS_DIR / "responses_no_probes.jsonl"
if not data_file.exists():
    data_file = RESULTS_DIR / "responses.jsonl"
    print("  Warning: using responses.jsonl (no filtered version found)")

records = []
with open(data_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line.strip()))

print(f"Loaded {len(records)} records from {data_file.name}")

# ── Per-model analysis ──

by_model = defaultdict(list)
for r in records:
    by_model[r["model_name"]].append(r)

model_stats = []
for model_name, recs in sorted(by_model.items()):
    total = len([r for r in recs if not r.get("error")])
    if total == 0:
        continue

    # Count discrepancies per claimed family
    claims_counter = Counter()  # claimed_family -> count of records with that claim
    disc_records = 0
    for r in recs:
        if r.get("error"):
            continue
        discs = detect_discrepancies(r)
        if discs:
            disc_records += 1
            for claimed_fam in discs:
                claims_counter[claimed_fam] += 1

    family = recs[0]["model_family"]
    model_stats.append({
        "name": model_name,
        "family": family,
        "total": total,
        "disc_records": disc_records,
        "disc_rate": disc_records / total,
        "claims": claims_counter,
    })

# Sort by discrepancy rate descending
model_stats.sort(key=lambda x: -x["disc_rate"])


# ══════════════════════════════════════════════════════════════
# PLOT 1: Stacked horizontal bar — discrepancy rate by claimed identity
# ══════════════════════════════════════════════════════════════

print("\nGenerating Plot 1: Stacked discrepancy rate bar chart...")

has_disc = [m for m in model_stats if m["disc_records"] > 0]
no_disc_count = len([m for m in model_stats if m["disc_records"] == 0])

fig, ax = plt.subplots(figsize=(14, max(10, len(has_disc) * 0.38)))

names = [m["name"] for m in has_disc]
y_pos = np.arange(len(names))

# Build stacked bars
for fam_idx, fam_name in enumerate(FAMILY_ORDER):
    rates = []
    for m in has_disc:
        fam_count = m["claims"].get(fam_name, 0)
        rates.append(fam_count / m["total"])

    # Calculate left offset (sum of previous families)
    lefts = []
    for m in has_disc:
        left = sum(m["claims"].get(FAMILY_ORDER[j], 0) for j in range(fam_idx)) / m["total"]
        lefts.append(left)

    color = FAMILY_COLORS.get(fam_name, "#999999")
    bars = ax.barh(y_pos, rates, left=lefts, color=color, edgecolor="white",
                   linewidth=0.3, label=fam_name if any(r > 0 for r in rates) else None)

ax.set_yticks(y_pos)
ax.set_yticklabels(names, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Discrepancy rate (proportion of prompts with identity claim)")
ax.set_xlim(0, min(1.0, max(m["disc_rate"] for m in has_disc) * 1.15))
ax.set_title(
    f"Models with unprompted identity discrepancies\n"
    f"({len(has_disc)} of {len(model_stats)} models; {no_disc_count} models had zero discrepancies)"
)

# Add rate labels on right
for i, m in enumerate(has_disc):
    total_rate = m["disc_rate"]
    ax.text(total_rate + 0.005, i, f'{total_rate:.0%} ({m["disc_records"]}/{m["total"]})',
            va='center', fontsize=7, color='#333')

# Legend — only include families that appear
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='lower right', fontsize=8, title="Claims to be:")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "01_all_models_discrepancy.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: plots/01_all_models_discrepancy.png")


# ══════════════════════════════════════════════════════════════
# PLOT 2: Full model-level confusion matrix
# ══════════════════════════════════════════════════════════════

print("\nGenerating Plot 2: Model-level confusion matrix...")

# Build matrix: source model (y-axis) x claimed family (x-axis)
# Only include models with discrepancies
disc_model_names = [m["name"] for m in has_disc]

# Collect all claimed families that appear
all_claimed = set()
for m in has_disc:
    all_claimed.update(m["claims"].keys())

# Order claims by total frequency
claim_totals = Counter()
for m in has_disc:
    claim_totals.update(m["claims"])
claimed_order = [c for c, _ in claim_totals.most_common()]

matrix = np.zeros((len(disc_model_names), len(claimed_order)))
for i, m in enumerate(has_disc):
    for j, claim in enumerate(claimed_order):
        matrix[i, j] = m["claims"].get(claim, 0)

fig, ax = plt.subplots(figsize=(max(8, len(claimed_order) * 1.2), max(10, len(disc_model_names) * 0.35)))

im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest')

ax.set_xticks(np.arange(len(claimed_order)))
ax.set_xticklabels(claimed_order, fontsize=9, rotation=45, ha='right')
ax.set_yticks(np.arange(len(disc_model_names)))
ax.set_yticklabels(disc_model_names, fontsize=8)
ax.set_xlabel("Claims to be →")
ax.set_ylabel("← Model")
ax.set_title("Who claims to be whom?\n(unprompted claims, individual model × claimed identity family)")

# Add text annotations
for i in range(len(disc_model_names)):
    for j in range(len(claimed_order)):
        val = int(matrix[i, j])
        if val > 0:
            color = 'white' if val > matrix.max() * 0.5 else 'black'
            ax.text(j, i, str(val), ha='center', va='center', fontsize=7,
                    color=color, fontweight='bold')

fig.colorbar(im, ax=ax, shrink=0.5, label="Count of claims")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "04_confusion_direction.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: plots/04_confusion_direction.png")


# ══════════════════════════════════════════════════════════════
# Print summary stats
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"Total models: {len(model_stats)}")
print(f"Models with unprompted discrepancies: {len(has_disc)}")
print(f"Models with zero discrepancies: {no_disc_count}")
print(f"\nTop 10 by discrepancy rate:")
for m in has_disc[:10]:
    claims_str = ", ".join(f"{fam} ({cnt})" for fam, cnt in m["claims"].most_common(3))
    print(f"  {m['name']:<35s} {m['disc_rate']:.0%} ({m['disc_records']}/{m['total']}) — {claims_str}")

print(f"\nModels with zero discrepancies ({no_disc_count}):")
for m in sorted([m for m in model_stats if m["disc_records"] == 0], key=lambda x: x["family"]):
    print(f"  {m['name']:<40s} [{m['family']}]")

print(f"\nDone. Plots saved to {PLOTS_DIR}/")
