# LMArena Elo join (2026-07-31)

Source: HF dataset `lmarena-ai/leaderboard-dataset`, config `text`, split `full`,
category `overall`; per-model rating taken from that model's LATEST
leaderboard_publish_date (newest board = 2026-07-30). Files:
- text_latest.parquet / text_full.parquet — raw downloads
- match_final.json — our model_id -> {arena name, elo}; auto-normalized matching
  plus a 22-entry hand map (kimi variants incl. k2.5->k2.5-thinking per our
  reasoning=True lane, nvidia- prefixes, medium-3.1=2508, large-2512=large-3,
  small-3.2=2506, gpt-4->gpt-4-0613, hy3, novas ->v1.0, command-a, maverick).

Results (spearman, permutation p, seed 12345):
- rate vs Elo, all matched (n=109):        rho=-0.162 p=0.090
- top1-share vs Elo, mismatchers nF>=10 (n=41): rho=+0.162 p=0.313
- sanity Elo vs log-size (n=75):           rho=+0.506 p=0.0002
Anchors: Opus 4.8 elo 1453 (top1 .78) · Sonnet 4.6 1457 (.70) · Kimi K2.5 1445
(rate 39.4%, top1 .99) · K2 1371 (.99) · GPT-4 1186 (1.00).

Caveats: overall category (style-confounded, no style control applied); arena
lists no sub-2B models, so the incoherent small-model tail present in the size
analysis (rho=+0.27 p=0.04, n=61) is truncated here — attenuation expected and
observed. ~80 of our 190 models (incl. Perceptron Mk1, Laguna, Hermes-3, Novas'
siblings, small Qwens, Ministral-2512s, OLMo-7Bs) have no arena listing.
Size analysis lives in the same session notes; post carries one merged clause in
"Are the claims consistent?".
