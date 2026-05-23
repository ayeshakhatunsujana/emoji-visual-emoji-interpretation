# Do Visual Models Understand Emojis Better?

**A Comparative Study of Visual and Textual Emoji Interpretation in LLMs**

This repository contains the code for the study of whether giving Large Language
Models (LLMs) *visual* information about emojis — rather than treating emojis as
plain text tokens — improves **sentiment classification** and **sarcasm
detection**. It compares three model families (GPT, LLaMA, Qwen), each paired
with a matching Vision-Language Model (VLM) that generates emoji captions, and
includes a full explainability analysis for the LLaMA family.

---

## Overview

For each of the 128 most-common emojis, a VLM generates three caption channels;
these captions are then injected into an LLM prompt that classifies the tweet.

**Caption channels**

| Channel | Name | Description |
|---------|------|-------------|
| `C_v`   | Plain visual caption     | Literal description of the rendered emoji image. |
| `C_v_st`| Smart (communicative) visual caption | Visual description **plus** the inferred emotion / communicative function. |
| `C_t`   | Unicode-name (textual) caption | Description generated from the emoji's Unicode name. |

**Prompt conditions** (`T` = raw tweet text, `E` = emoji glyph)

| ID | Condition | Prompt content |
|----|-----------|----------------|
| a  | `T`                  | raw tweet text only |
| b  | `T+E`                | text + emoji glyph |
| c  | `T+E+C_v`            | text + emoji + plain visual caption |
| d  | `T+E+C_v*`           | text + emoji + smart visual caption |
| e  | `T+E+C_t`            | text + emoji + Unicode-name caption |
| f  | `T+E+C_v*+C_t`       | text + emoji + smart visual + textual caption |

**Model pairs** (one VLM for captioning + one LLM for classification per family)

| Family | VLM (captioning)              | LLM (classification)      |
|--------|-------------------------------|---------------------------|
| GPT    | GPT-4V                        | GPT-4o-mini               |
| LLaMA  | LLaMA-3.2-11B-Vision-Instruct | LLaMA-3.1-8B-Instruct     |
| Qwen   | Qwen2.5-VL-7B-Instruct        | Qwen2.5-3B-Instruct       |

---

## Repository structure

```text
emoji-visual-emoji-interpretation/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example              # template for the OpenAI key (GPT only)
├── data/                     # datasets — EMPTY (see data/README.md)
├── outputs/                  # run outputs (predictions, JSON, XLSX) — git-ignored
├── results/                  # figures / metric summaries
├── paper/                    # paper-ready LaTeX result tables
└── src/
    ├── gpt/                  # GPT pipeline (OpenAI API)
    │   ├── vlm_caption_gpt4v.py        # GPT-4V image captioner
    │   ├── text_llm_describe.py        # GPT text describer (Unicode-name caption)
    │   ├── run_all_gpt.py              # generate C_v / C_t for all emojis
    │   ├── run_experiments_gpt.py      # classification experiments
    │   └── data_utils.py               # dataset loading helper
    ├── llama/                # LLaMA pipeline (local, via Ollama / Hugging Face)
    │   ├── vlm_caption_llama.py        # LLaMA-Vision image captioner
    │   ├── text_llm_describe_llama.py  # LLaMA text describer
    │   ├── run_all_llama.py            # generate C_v / C_v_st / C_t
    │   ├── run_experiments_llama_captions.py  # classification experiments
    │   ├── data_utils_llama_exp.py     # dataset loading helper
    │   ├── explain_cvst_metrics.py     # C_v_st-specific metrics
    │   └── plot_llama_caption_results.py
    ├── qwen/                 # Qwen pipeline (local, via Hugging Face)
    │   ├── vlm_caption_qwen.py
    │   ├── text_llm_describe_qwen.py
    │   ├── run_all_qwen.py
    │   └── run_experiments_qwen_captions.py
    └── explainability/       # explainability analysis (LLaMA family)
        ├── run_experiments_llama_explainability.py  # NLE, occlusion, HF probs, IG
        ├── analyze_all_condition_words.py           # per-condition word importance
        ├── compute_logprob_margin_table.py          # log-probability margin table
        └── data_utils_llama_exp.py
```

---

## Setup

### 1. Clone and create an environment

```bash
git clone <your-repo-url>
cd emoji-visual-emoji-interpretation
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add the datasets

The `data/` folder ships **empty**. Download the ASDT and MSAD datasets and place
them in `data/` — see [`data/README.md`](data/README.md) for the expected files.

### 3. API key for the GPT pipeline only

The GPT scripts call the OpenAI API and need a key. **No API key is stored in
this repository.** Provide your own:

```bash
cp .env.example .env            # then edit .env and paste your key
set -a; source .env; set +a     # load it into the shell
# or simply:
export OPENAI_API_KEY="your-openai-api-key-here"
```

The LLaMA and Qwen pipelines run **locally** and need no API key.

### 4. Local model backends (LLaMA / Qwen)

The LLaMA pipeline uses [Ollama](https://ollama.com) for classification:

```bash
ollama serve
ollama pull llama3.2:3b
```

The Qwen pipeline and the Hugging Face parts of the explainability analysis
download models from the Hugging Face Hub on first use (a Hugging Face account
may be required for gated Meta/Qwen models).

---

## Usage

All scripts are run as modules from the **repository root**.

### Step 1 — Generate emoji captions (`C_v`, `C_v_st`, `C_t`)

```bash
# GPT
python -m src.gpt.run_all_gpt   --emoji_dir data/emoji_images --out_csv outputs/gpt_captions.csv   --vlm_model gpt-4o-mini --llm_model gpt-4o-mini

# LLaMA
python -m src.llama.run_all_llama   --emoji_dir data/emoji_images --out_csv outputs/llama_captions.csv

# Qwen
python -m src.qwen.run_all_qwen   --emoji_dir data/emoji_images --out_csv outputs/qwen_captions.csv
```

### Step 2 — Run the classification experiments

```bash
# LLaMA — sentiment (3-class) and sarcasm (binary)
python -m src.llama.run_experiments_llama_captions   --train_csv data/train_split_70.csv --emoji_csv outputs/llama_captions.csv   --task_type sentiment3 --out_dir outputs/llama_sentiment3

python -m src.llama.run_experiments_llama_captions   --train_csv data/sarcasm_full_with_emoji.csv --emoji_csv outputs/llama_captions.csv   --task_type sarcasm2 --out_dir outputs/llama_sarcasm2

# Qwen — same task types
python -m src.qwen.run_experiments_qwen_captions   --train_csv data/train_split_70.csv --emoji_csv outputs/qwen_captions.csv   --task_type sentiment3 --out_dir outputs/qwen_sentiment3

# GPT
python -m src.gpt.run_experiments_gpt --help
```

> Each runner supports `task_type` values `sentiment3`, `sentiment2`, and
> `sarcasm2`, and writes a predictions CSV, an XLSX workbook, and a JSON summary
> (accuracy, macro-F1, weighted-F1, confusion matrix).

### Step 3 — Explainability (LLaMA family)

```bash
# F1 + Natural Language Explanations + occlusion / prediction-switch analysis
python -m src.explainability.run_experiments_llama_explainability   --train_csv data/sarcasm_full_with_emoji.csv   --emoji_csv outputs/llama_captions.csv   --task_type sarcasm2 --model_name llama3.2:3b   --out_dir outputs/explainability_sarcasm_all_conditions_full   --skip_missing_descriptions --do_nle --nle_experiment_name all

# Per-condition word-importance analysis (run after the command above)
python -m src.explainability.analyze_all_condition_words   --pred_csv  outputs/explainability_sarcasm_all_conditions_full/sarcasm2_predictions_with_explainability_base.csv   --results_json outputs/explainability_sarcasm_all_conditions_full/sarcasm2_results_summary.json   --task_type sarcasm2 --out_dir outputs/explainability_sarcasm_all_condition_words --top_k 50

# Hugging Face label-probability / log-probability margin
python -m src.explainability.run_experiments_llama_explainability   --train_csv data/sarcasm_full_with_emoji.csv   --emoji_csv outputs/llama_captions.csv   --task_type sarcasm2 --model_name llama3.2:3b   --out_dir outputs/explainability_sarcasm_hf_probs   --skip_missing_descriptions --do_hf_probs   --hf_model_name meta-llama/Llama-3.2-1B-Instruct --hf_device auto
```

Explainability methods produced: prompted Natural Language Explanation (NLE),
component-level occlusion / leave-one-out, Hugging Face verbalizer
probabilities, log-probability margin, and (optionally) Integrated Gradients.

---

## Outputs

Run outputs are written to `outputs/` (git-ignored). Typical files per run:

```text
*_predictions*.csv           # per-tweet predictions for every condition
*_results_summary.json       # accuracy, macro-F1, weighted-F1, confusion matrix
*_full_explainability_outputs.xlsx
occlusion_switch_summary.csv
hf_label_probabilities_logprob_differences.csv
```

The paper-ready confusion-matrix and per-class result tables are in
[`paper/`](paper/).

---

## Security note

This repository contains **no API keys or secrets**. Any key previously embedded
in the code has been removed; the GPT scripts read `OPENAI_API_KEY` from the
environment only. The `.env` file is git-ignored — never commit it.

---

## Citation

If you use this code, please cite the accompanying paper:

```bibtex
@inproceedings{emoji-visual-interpretation,
  title     = {Do Visual Models Understand Emojis Better? A Comparative Study
               of Visual and Textual Emoji Interpretation in LLMs},
  author    = {Anonymous},
  year      = {2026}
}
```
