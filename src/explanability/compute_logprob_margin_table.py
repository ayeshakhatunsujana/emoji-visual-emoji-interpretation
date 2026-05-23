#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_logprob_margin_table.py
===============================
Fills the "View C --- log-probability margin (LLaMA only)" LaTeX table
(``view_c_logprob_margin_table.tex``) with real numbers, computed so they are
a fair companion to the classification F1 experiment.

Why this script re-implements the scoring
-----------------------------------------
The project's built-in run_hf_probability_analysis() scores label
log-probabilities by feeding the prompt to the model RAW (no chat template,
add_special_tokens=False) and, per the README, with the 1B model. The F1
classification, however, used the 3B model through Ollama, which wraps every
prompt in Llama-3's chat template. Scoring an instruction-tuned model on raw
text is out of distribution and is not comparable to the F1 numbers.

This script therefore uses a corrected scorer that:
  * applies the Llama-3 chat template (tokenizer.apply_chat_template),
    matching what Ollama did during classification;
  * defaults to the 3B model (unsloth/Llama-3.2-3B-Instruct, ungated mirror
    of the same weights Ollama's llama3.2:3b uses);
  * teacher-forces each candidate label and sums its token log-probs;
  * reports  margin = logP(gold label) - max( logP(other labels) ),
    averaged over all tweets of each task.

The prompts themselves come from the project's own build_prompt_* functions
(imported unchanged) applied to the saved *_predictions_with_explainability_base.csv
files, so Ollama does not need to run and the project pipeline is not modified.

This script is location-independent: it finds the project root automatically.

Usage  (with the project .venv active)
--------------------------------------
    source .venv/bin/activate
    python scripts/compute_logprob_margin_table.py

Options
-------
    --force            Recompute even if the aligned CSVs already exist.
    --skip-hf          Only fill the table from existing aligned CSVs.
    --hf-model NAME    HF model id (default: unsloth/Llama-3.2-3B-Instruct).
    --hf-device DEV    auto | cpu | cuda | mps  (default: auto; on failure the
                       run automatically retries the task on cpu).
    --dtype DT         auto | float32 | float16 | bfloat16  (default: auto =
                       float16 on gpu/mps, float32 on cpu). The 3B model in
                       float32 needs ~13 GB RAM; on a 16 GB Mac add
                       "--dtype bfloat16" to halve that.
    --outputs-root DIR Base folder with the explainability_* output dirs.
    --tex-out PATH     Output .tex path.
"""

import argparse
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_project_root(start):
    """Walk up from `start` to the folder containing
    src/run_experiments_llama_explainability.py."""
    d = start
    for _ in range(8):
        if os.path.isfile(os.path.join(d, "src", "run_experiments_llama_explainability.py")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return start


PROJECT_ROOT = _find_project_root(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)  # so "src" is importable

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# task_type -> (output sub-folder, base predictions CSV file name)
TASKS = {
    "sarcasm2":   ("explainability_sarcasm_hf_probs_all_conditions_full",
                   "sarcasm2_predictions_with_explainability_base.csv"),
    "sentiment3": ("explainability_sentiment3_hf_probs_all_conditions_full",
                   "sentiment3_predictions_with_explainability_base.csv"),
}

TASK_COL = {"sarcasm2": "SAR", "sentiment3": "SEN"}

# The 6 conditions the View C table needs:
#   experiment_name, template token, human label, build_prompt_* function name
CONDITION_ROWS = [
    ("a_text_only",               "A", "T",              "build_prompt_text_only"),
    ("b_text_plus_emoji",         "B", "T+E",            "build_prompt_text_plus_emoji"),
    ("c_text_plus_emoji_Cv",      "C", "T+E+C_v",        "build_prompt_text_plus_cv"),
    ("d_text_plus_emoji_Cvst",    "D", "T+E+C_v*",       "build_prompt_text_plus_cvst"),
    ("e_text_plus_emoji_Ct",      "E", "T+E+C_t",        "build_prompt_text_plus_ct"),
    ("g_text_plus_emoji_Cvst_Ct", "G", "T+E+C_v*+C_t",   "build_prompt_text_plus_cvst_ct"),
]

# verbalizer label strings per task (must match the pipeline's label_texts)
LABELS = {
    "sarcasm2":   ["Not Sarcasm", "Sarcasm"],
    "sentiment3": ["Negative", "Neutral", "Positive"],
}
ID2TEXT = {
    "sarcasm2":   {0: "Not Sarcasm", 1: "Sarcasm"},
    "sentiment3": {0: "Negative", 1: "Neutral", 2: "Positive"},
}

MARGIN_COL = "gold_logprob_margin_vs_best_other"
# New filename: the chat-template/3B results are kept separate from the old
# raw-prompt/1B file (hf_label_probabilities_logprob_differences.csv).
HF_CSV_NAME = "hf_logprob_margins_aligned.csv"

DELTA_NUM = "d_text_plus_emoji_Cvst"
DELTA_BASE = "a_text_only"

TEX_TEMPLATE = r"""% =====================================================================
% TABLE  --  View C: LLaMA log-probability margin per condition.
% Short companion table for the View C paragraph in subsection 4.2.
% Reference it in the text with \ref{tab:margin}.
%
% Filled by scripts/compute_logprob_margin_table.py.
% Scoring is aligned with the F1 classification experiment:
%   * model: Llama-3.2-3B-Instruct (same weights as Ollama llama3.2:3b)
%   * Llama-3 chat template applied (as Ollama does)
%   * margin = logP(gold label) - max( logP(other labels) ), teacher-forced
%   * cell  = mean margin over all tweets of the task
%   * bottom row = the C_v* row minus the T row
% =====================================================================
\begin{table}[t]
\centering
\small
\renewcommand{\arraystretch}{1.18}
\setlength{\tabcolsep}{6pt}
\caption{\textbf{View C --- log-probability margin (LLaMA only).}
For each tweet the \emph{margin} is the gold label's log-probability minus
the strongest competing label's; we report the mean margin over all tweets
of each task. A larger margin means LLaMA is more \emph{confident} in the
correct label, not just correct more often.}
\label{tab:margin}
\begin{tabular}{l c c}
\toprule
\textbf{Condition} & \textbf{Sarcasm} & \textbf{Sentiment} \\
                   & \textbf{(ASDT)}  & \textbf{(MSAD)} \\
\midrule
$T$                      & @@A_SAR@@ & @@A_SEN@@ \\
$T{+}E$                  & @@B_SAR@@ & @@B_SEN@@ \\
$T{+}E{+}C_v$            & @@C_SAR@@ & @@C_SEN@@ \\
$T{+}E{+}C_v^{*}$        & @@D_SAR@@ & @@D_SEN@@ \\
$T{+}E{+}C_t$            & @@E_SAR@@ & @@E_SEN@@ \\
$T{+}E{+}C_v^{*}{+}C_t$  & @@G_SAR@@ & @@G_SEN@@ \\
\midrule
\textbf{$\Delta$ ($T{+}E{+}C_v^{*}-T$)} & @@DELTA_SAR@@ & @@DELTA_SEN@@ \\
\bottomrule
\end{tabular}
\end{table}
"""

PLACEHOLDER = "x.xx"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt(value):
    """Format a margin value for a LaTeX cell, or the placeholder if missing."""
    if value is None:
        return PLACEHOLDER
    try:
        return "${:.2f}$".format(float(value))
    except (TypeError, ValueError):
        return PLACEHOLDER


def hf_csv_path(outputs_root, task_type):
    sub, _ = TASKS[task_type]
    return os.path.join(outputs_root, sub, HF_CSV_NAME)


def base_csv_path(outputs_root, task_type):
    sub, base = TASKS[task_type]
    return os.path.join(outputs_root, sub, base)


def _load_builders():
    """Import the project's own prompt builders (pure functions, unchanged)."""
    from src.run_experiments_llama_explainability import (
        build_prompt_text_only, build_prompt_text_plus_emoji,
        build_prompt_text_plus_cv, build_prompt_text_plus_cvst,
        build_prompt_text_plus_ct, build_prompt_text_plus_cvst_ct,
    )
    available = {
        "build_prompt_text_only": build_prompt_text_only,
        "build_prompt_text_plus_emoji": build_prompt_text_plus_emoji,
        "build_prompt_text_plus_cv": build_prompt_text_plus_cv,
        "build_prompt_text_plus_cvst": build_prompt_text_plus_cvst,
        "build_prompt_text_plus_ct": build_prompt_text_plus_ct,
        "build_prompt_text_plus_cvst_ct": build_prompt_text_plus_cvst_ct,
    }
    # ordered exp_name -> builder function, for the 6 table conditions
    return [(exp, available[fn]) for exp, _tok, _hum, fn in CONDITION_ROWS]


# ---------------------------------------------------------------------------
# Corrected scorer: chat template + teacher-forced label log-probability
# ---------------------------------------------------------------------------

class ChatLabelScorer:
    """Scores P(label | chat-formatted prompt) the way Ollama's chat API
    presents prompts to the model, then sums the label token log-probs."""

    def __init__(self, model_name, device, dtype):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        if dtype == "auto":
            torch_dtype = torch.float32 if device == "cpu" else torch.float16
        else:
            torch_dtype = {"float32": torch.float32,
                           "float16": torch.float16,
                           "bfloat16": torch.bfloat16}[dtype]
        self.torch_dtype = torch_dtype

        print("[HF] loading {} on {} ({})".format(model_name, device, torch_dtype),
              flush=True)
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
        self.model.to(device)
        self.model.eval()

    def label_logprob(self, prompt, label_text):
        """Sum of log P over the label tokens, given the chat-formatted prompt."""
        torch = self.torch
        import torch.nn.functional as F

        # Chat-format the user turn and open the assistant turn, exactly as
        # Ollama's chat API does before the model would generate the label.
        ctx = self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True)

        ctx_ids = self.tok(ctx, add_special_tokens=False,
                           return_tensors="pt")["input_ids"]
        label_ids = self.tok(label_text, add_special_tokens=False,
                             return_tensors="pt")["input_ids"]
        # Concatenate ids directly so the label token positions are exact.
        full_ids = torch.cat([ctx_ids, label_ids], dim=1).to(self.device)
        ctx_len = ctx_ids.shape[1]

        with torch.no_grad():
            logits = self.model(input_ids=full_ids).logits  # [1, S, vocab]

        # Token at position i is predicted by logits at position i-1.
        # Label tokens occupy positions [ctx_len, S); predicted by
        # logits[ctx_len-1 : S-1]. Slice BEFORE log_softmax to keep tensors
        # tiny (avoids the large-tensor MPS limit).
        shift_logits = logits[:, ctx_len - 1:-1, :].float()
        targets = full_ids[:, ctx_len:]
        log_probs = F.log_softmax(shift_logits, dim=-1)
        selected = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return float(selected.sum().detach().to("cpu"))

    def score_all_labels(self, prompt, labels):
        return {lab: self.label_logprob(prompt, lab) for lab in labels}


def run_aligned_analysis(task_type, outputs_root, hf_model, device, dtype):
    """Compute the aligned log-probability margins for one task and save a CSV."""
    import pandas as pd

    base_path = base_csv_path(outputs_root, task_type)
    if not os.path.exists(base_path):
        raise FileNotFoundError(
            "Missing saved predictions file:\n  {}\n"
            "Run the base pipeline first (see README_ALL_CONDITIONS.md).".format(base_path))

    out_dir = os.path.dirname(base_path)
    out_path = os.path.join(out_dir, HF_CSV_NAME)
    df = pd.read_csv(base_path)
    builders = _load_builders()
    labels = LABELS[task_type]
    id2text = ID2TEXT[task_type]

    print("[{}] {} rows x {} conditions x {} labels".format(
        task_type, len(df), len(builders), len(labels)), flush=True)

    scorer = ChatLabelScorer(hf_model, device, dtype)

    try:
        from tqdm import tqdm
        row_iter = tqdm(df.iterrows(), total=len(df), desc="[{}]".format(task_type))
    except Exception:
        row_iter = df.iterrows()

    records = []
    for idx, row in row_iter:
        gold_id = int(row["label"])
        gold_text = id2text[gold_id]
        for exp_name, builder in builders:
            prompt = builder(row, task_type)
            scores = scorer.score_all_labels(prompt, labels)
            other = [s for lab, s in scores.items() if lab != gold_text]
            margin = scores[gold_text] - max(other)
            rec = {
                "row_index": idx,
                "id": row.get("id", idx),
                "task_type": task_type,
                "experiment_name": exp_name,
                "gold_label_id": gold_id,
                "gold_label_text": gold_text,
                MARGIN_COL: margin,
            }
            for lab, s in scores.items():
                rec["logprob_" + lab.lower().replace(" ", "_")] = s
            records.append(rec)

    pd.DataFrame(records).to_csv(out_path, index=False)
    print("[{}] wrote {}".format(task_type, out_path), flush=True)


def mean_margins(task_type, outputs_root):
    """Return {experiment_name: mean margin} for one task, or None if no CSV."""
    import pandas as pd

    path = hf_csv_path(outputs_root, task_type)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if MARGIN_COL not in df.columns or "experiment_name" not in df.columns:
        raise ValueError("{} is missing expected columns.".format(path))
    means = df.groupby("experiment_name")[MARGIN_COL].mean()
    return {str(k): float(v) for k, v in means.items()}


def _device_attempts(device):
    """Devices to try in order. mps/auto fall back to cpu on failure."""
    if device in ("auto", "mps"):
        return [device, "cpu"]
    return [device]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if the aligned CSVs exist.")
    parser.add_argument("--skip-hf", action="store_true",
                        help="Only fill the table from existing aligned CSVs.")
    parser.add_argument("--hf-model", default="unsloth/Llama-3.2-3B-Instruct",
                        help="Hugging Face model id.")
    parser.add_argument("--hf-device", default="auto",
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="Device for the HF model (auto retries on cpu).")
    parser.add_argument("--dtype", default="auto",
                        choices=["auto", "float32", "float16", "bfloat16"],
                        help="Model dtype.")
    parser.add_argument("--outputs-root", default=os.path.join(PROJECT_ROOT, "outputs"),
                        help="Folder holding the explainability_* output dirs.")
    parser.add_argument("--tex-out", default=os.path.join(PROJECT_ROOT, "view_c_logprob_margin_table.tex"),
                        help="Output .tex path.")
    # Hidden: used internally to compute one task in its own fresh process.
    parser.add_argument("--compute-only", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.outputs_root = os.path.abspath(args.outputs_root)

    # Hidden mode: compute ONE task in this (fresh) process and exit. The
    # normal run spawns this once per task so each model load is isolated.
    if args.compute_only:
        run_aligned_analysis(args.compute_only, args.outputs_root,
                             args.hf_model, args.hf_device, args.dtype)
        return

    # --- Step 1: make sure the aligned log-probability CSVs exist ------------
    for task_type in TASKS:
        csv_path = hf_csv_path(args.outputs_root, task_type)
        exists = os.path.exists(csv_path)

        if args.skip_hf:
            if not exists:
                sys.exit("ERROR: --skip-hf was given but {} is missing.".format(csv_path))
            print("[{}] using existing {}".format(task_type, csv_path))
            continue

        if exists and not args.force:
            print("[{}] found existing {} (use --force to recompute)"
                  .format(task_type, csv_path))
            continue

        # Compute in a separate process per task (and per device attempt) so
        # each model load is fully isolated.
        ok = False
        for dev in _device_attempts(args.hf_device):
            print("[{}] launching analysis in a fresh process (device={})..."
                  .format(task_type, dev), flush=True)
            cmd = [sys.executable, os.path.abspath(__file__),
                   "--compute-only", task_type,
                   "--hf-model", args.hf_model,
                   "--hf-device", dev,
                   "--dtype", args.dtype,
                   "--outputs-root", args.outputs_root]
            if subprocess.run(cmd).returncode == 0:
                ok = True
                break
            print("[{}] device '{}' failed; trying next option..."
                  .format(task_type, dev), flush=True)
        if not ok:
            sys.exit("ERROR: HF analysis for {} failed on all devices. "
                     "See the traceback above.".format(task_type))

    # --- Step 2: aggregate the margins --------------------------------------
    results = {}
    for task_type in TASKS:
        m = mean_margins(task_type, args.outputs_root)
        if m is None:
            sys.exit("ERROR: no aligned log-probability CSV for {}.".format(task_type))
        results[task_type] = m

    # --- Step 3: fill the LaTeX template ------------------------------------
    tex = TEX_TEMPLATE
    print("\n" + "=" * 64)
    print("Mean LLaMA log-probability margin per condition (chat template, 3B)")
    print("=" * 64)
    print("{:<28}{:>16}{:>16}".format("Condition", "Sarcasm", "Sentiment"))
    print("-" * 64)

    for exp_name, token, human, _builder in CONDITION_ROWS:
        cells = {}
        for task_type in TASKS:
            val = results[task_type].get(exp_name)
            cells[task_type] = val
            tex = tex.replace("@@{}_{}@@".format(token, TASK_COL[task_type]), fmt(val))
        print("{:<28}{:>16}{:>16}".format(
            human, fmt(cells["sarcasm2"]), fmt(cells["sentiment3"])))

    print("-" * 64)
    delta_cells = {}
    for task_type in TASKS:
        m = results[task_type]
        num, base = m.get(DELTA_NUM), m.get(DELTA_BASE)
        delta = (num - base) if (num is not None and base is not None) else None
        delta_cells[task_type] = delta
        tex = tex.replace("@@DELTA_{}@@".format(TASK_COL[task_type]), fmt(delta))
    print("{:<28}{:>16}{:>16}".format(
        "Delta (T+E+C_v* - T)",
        fmt(delta_cells["sarcasm2"]), fmt(delta_cells["sentiment3"])))
    print("=" * 64)

    if "@@" in tex:
        print("\nWARNING: some cells were not filled; left as '{}'.".format(PLACEHOLDER))

    with open(args.tex_out, "w", encoding="utf-8") as fh:
        fh.write(tex)
    print("\nWrote filled table to: {}".format(args.tex_out))


if __name__ == "__main__":
    main()
