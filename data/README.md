# `data/` — datasets (not included)

This folder is intentionally **empty** in the repository. The datasets are not
redistributed here; download them from their original sources and place the
files in this folder before running anything in `src/`.

## What goes here

| File (suggested name)        | Used by                | Description |
|------------------------------|------------------------|-------------|
| `asdt_sarcasm.csv`           | sarcasm experiments    | **ASDT** — Automatic Sarcasm Detection (Twitter), a binary tweet-level sarcasm dataset. |
| `msad_sentiment.csv`         | sentiment experiments  | **MSAD** — Multiclass Sentiment Analysis Dataset, three-class tweet sentiment (positive / neutral / negative). |
| `train_split_70.csv`         | sentiment runs         | 70% train split used by the experiment runners. |
| `sarcasm_full_with_emoji.csv`| sarcasm runs           | Sarcasm tweets filtered to those containing emojis. |
| `results_llama_full.csv`     | all caption conditions | Per-emoji captions (`C_v`, `C_v_st`, `C_t`) produced by the caption-generation scripts in `src/`. |
| `emoji_images/`              | caption generation     | PNG renders of the 128 most-common emojis (one image per emoji). |

> File names above are suggestions — pass the actual paths you use through the
> `--train_csv`, `--emoji_csv`, `--emoji_dir`, etc. command-line arguments.

## Data scope

Following the paper, only tweets that contain at least one emoji and whose first
emoji belongs to the 128-emoji vocabulary are kept (≈1,693 tweets for ASDT and
≈2,513 for MSAD).

## Notes

- Everything inside `data/` (except this file and `.gitkeep`) is git-ignored, so
  datasets are never accidentally committed.
- Respect the licence and terms of use of each original dataset.
