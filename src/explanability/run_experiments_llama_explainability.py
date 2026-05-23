# ============================================================
# src/run_experiments_llama_explainability.py
#
# Runs your ORIGINAL LLaMA/Ollama classification prompts and
# saves performance + explainability outputs in the same run.
#
# Explainability methods included:
#   1. Prompted Natural Language Explanation (NLE) via Ollama
#   2. Component-level occlusion / leave-one-out from existing conditions
#   3. Hugging Face label probability / verbalizer probability
#   4. Logit/log-probability difference analysis
#   5. Optional Integrated Gradients for local Hugging Face LLaMA/Qwen
#
# IMPORTANT:
#   - The classification prompt builders below are copied from your
#     previous run_experiments_llama_captions.py and are NOT changed.
#   - NLE uses a separate explanation prompt after classification.
#   - HF probability/IG scoring uses the same classification prompts.
# ============================================================

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Callable, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

import ollama

try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception:
    torch = None
    F = None
    AutoTokenizer = None
    AutoModelForCausalLM = None

from .data_utils_llama_exp import (
    load_dataset_with_llama_captions,
    get_task_config,
)


# ============================================================
# Label instruction by task -- SAME AS YOUR PREVIOUS CODE
# ============================================================

def label_instruction(task_type: str) -> str:
    if task_type == "sentiment3":
        return "Answer only one word: Positive, Neutral, or Negative."
    if task_type == "sentiment2":
        return "Answer only one word: Positive or Negative. Do not answer Neutral."
    if task_type == "sarcasm2":
        return "Answer only one phrase: Sarcasm or Not Sarcasm."
    raise ValueError(f"Unknown task_type: {task_type}")


def task_description(task_type: str) -> str:
    if task_type == "sentiment3":
        return "Classify the sentiment of the following tweet."
    if task_type == "sentiment2":
        return "Classify the sentiment of the following tweet as positive or negative."
    if task_type == "sarcasm2":
        return "Classify whether the following tweet is sarcastic or not sarcastic."
    raise ValueError(f"Unknown task_type: {task_type}")


# ============================================================
# Parse Llama output into integer label -- SAME LOGIC AS BEFORE
# ============================================================

def parse_label(raw_text: str, task_type: str) -> int:
    if raw_text is None:
        if task_type == "sentiment3":
            return 1
        return 0

    text = str(raw_text).strip().lower()
    text = (
        text.replace(".", " ")
        .replace(",", " ")
        .replace(":", " ")
        .replace(";", " ")
        .replace("\n", " ")
        .strip()
    )

    if task_type == "sentiment3":
        if "negative" in text:
            return 0
        if "neutral" in text:
            return 1
        if "positive" in text:
            return 2
        return 1

    if task_type == "sentiment2":
        if "negative" in text:
            return 0
        if "positive" in text:
            return 1
        return 0

    if task_type == "sarcasm2":
        if (
            "not sarcasm" in text
            or "not_sarcasm" in text
            or "non sarcasm" in text
            or "non-sarcasm" in text
            or "not sarcastic" in text
            or "non sarcastic" in text
        ):
            return 0
        if "sarcasm" in text or "sarcastic" in text:
            return 1
        return 0

    raise ValueError(f"Unknown task_type: {task_type}")


def llama_classify(prompt: str, model_name: str = "llama3.2:3b", num_predict: int = 20) -> str:
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "num_predict": num_predict},
        )
        return response["message"]["content"].strip()
    except Exception as e:
        print(f"[OLLAMA ERROR] {e}")
        return ""


# ============================================================
# Prompt builders -- COPIED FROM YOUR PREVIOUS RUNNER, unchanged
# ============================================================

def build_prompt_text_only(row: pd.Series, task_type: str) -> str:
    text = str(row["text_baseline"])
    instruction = label_instruction(task_type)
    task_sent = task_description(task_type)
    return f"""
You are a classifier for X/Twitter posts.

{task_sent}

Tweet:
{text}

{instruction}
""".strip()


def build_prompt_text_plus_emoji(row: pd.Series, task_type: str) -> str:
    text = str(row["text_baseline"])
    emoji = str(row.get("first_emoji", "") or "").strip()
    tweet = f"{text} {emoji}" if emoji else text
    instruction = label_instruction(task_type)
    task_sent = task_description(task_type)
    return f"""
You are a classifier for X/Twitter posts.

{task_sent}

Tweet:
{tweet}

{instruction}
""".strip()


def build_prompt_text_plus_cv(row: pd.Series, task_type: str) -> str:
    text = str(row["text_baseline"])
    emoji = str(row.get("first_emoji", "") or "").strip()
    cv = str(row.get("C_v", "") or "").strip()
    tweet = f"{text} {emoji}" if emoji else text
    instruction = label_instruction(task_type)
    task_sent = task_description(task_type)
    return f"""
You are a classifier for X/Twitter posts.

{task_sent}

Use the tweet text and the emoji plain visual description.

Tweet:
{tweet}

Emoji plain visual description C_v:
{cv}

{instruction}
""".strip()


def build_prompt_text_plus_cvst(row: pd.Series, task_type: str) -> str:
    text = str(row["text_baseline"])
    emoji = str(row.get("first_emoji", "") or "").strip()
    cvst = str(row.get("C_v_st", "") or "").strip()
    tweet = f"{text} {emoji}" if emoji else text
    instruction = label_instruction(task_type)
    task_sent = task_description(task_type)
    return f"""
You are a classifier for X/Twitter posts.

{task_sent}

Use the tweet text and the smart visual-emotion or visual-communication evidence.

Tweet:
{tweet}

Smart visual evidence C_v_st:
{cvst}

{instruction}
""".strip()


def build_prompt_text_plus_ct(row: pd.Series, task_type: str) -> str:
    text = str(row["text_baseline"])
    emoji = str(row.get("first_emoji", "") or "").strip()
    ct = str(row.get("C_t", "") or "").strip()
    tweet = f"{text} {emoji}" if emoji else text
    instruction = label_instruction(task_type)
    task_sent = task_description(task_type)
    return f"""
You are a classifier for X/Twitter posts.

{task_sent}

Use the tweet text and the emoji social/textual meaning.

Tweet:
{tweet}

Emoji social/textual meaning C_t:
{ct}

{instruction}
""".strip()


def build_prompt_text_plus_cv_ct(row: pd.Series, task_type: str) -> str:
    text = str(row["text_baseline"])
    emoji = str(row.get("first_emoji", "") or "").strip()
    cv = str(row.get("C_v", "") or "").strip()
    ct = str(row.get("C_t", "") or "").strip()
    tweet = f"{text} {emoji}" if emoji else text
    instruction = label_instruction(task_type)
    task_sent = task_description(task_type)
    return f"""
You are a classifier for X/Twitter posts.

{task_sent}

Use the tweet text, emoji plain visual description, and emoji social/textual meaning.

Tweet:
{tweet}

C_v plain visual description:
{cv}

C_t social/textual meaning:
{ct}

{instruction}
""".strip()


def build_prompt_text_plus_cvst_ct(row: pd.Series, task_type: str) -> str:
    text = str(row["text_baseline"])
    emoji = str(row.get("first_emoji", "") or "").strip()
    cvst = str(row.get("C_v_st", "") or "").strip()
    ct = str(row.get("C_t", "") or "").strip()
    tweet = f"{text} {emoji}" if emoji else text
    instruction = label_instruction(task_type)
    task_sent = task_description(task_type)
    return f"""
You are a classifier for X/Twitter posts.

{task_sent}

Use the tweet text, smart visual evidence, and emoji social/textual meaning.

Tweet:
{tweet}

C_v_st smart visual evidence:
{cvst}

C_t social/textual meaning:
{ct}

{instruction}
""".strip()


def build_prompt_text_plus_all(row: pd.Series, task_type: str) -> str:
    text = str(row["text_baseline"])
    emoji = str(row.get("first_emoji", "") or "").strip()
    cv = str(row.get("C_v", "") or "").strip()
    cvst = str(row.get("C_v_st", "") or "").strip()
    ct = str(row.get("C_t", "") or "").strip()
    tweet = f"{text} {emoji}" if emoji else text
    instruction = label_instruction(task_type)
    task_sent = task_description(task_type)
    return f"""
You are a classifier for X/Twitter posts.

{task_sent}

Use all available information.

Tweet:
{tweet}

C_v plain visual description:
{cv}

C_v_st smart visual evidence:
{cvst}

C_t social/textual meaning:
{ct}

{instruction}
""".strip()


EXPERIMENTS = [
    {"name": "a_text_only", "pred_col": "pred_a_text_only", "raw_col": "raw_a_text_only", "builder": build_prompt_text_only},
    {"name": "b_text_plus_emoji", "pred_col": "pred_b_text_plus_emoji", "raw_col": "raw_b_text_plus_emoji", "builder": build_prompt_text_plus_emoji},
    {"name": "c_text_plus_emoji_Cv", "pred_col": "pred_c_text_plus_emoji_Cv", "raw_col": "raw_c_text_plus_emoji_Cv", "builder": build_prompt_text_plus_cv},
    {"name": "d_text_plus_emoji_Cvst", "pred_col": "pred_d_text_plus_emoji_Cvst", "raw_col": "raw_d_text_plus_emoji_Cvst", "builder": build_prompt_text_plus_cvst},
    {"name": "e_text_plus_emoji_Ct", "pred_col": "pred_e_text_plus_emoji_Ct", "raw_col": "raw_e_text_plus_emoji_Ct", "builder": build_prompt_text_plus_ct},
    {"name": "f_text_plus_emoji_Cv_Ct", "pred_col": "pred_f_text_plus_emoji_Cv_Ct", "raw_col": "raw_f_text_plus_emoji_Cv_Ct", "builder": build_prompt_text_plus_cv_ct},
    {"name": "g_text_plus_emoji_Cvst_Ct", "pred_col": "pred_g_text_plus_emoji_Cvst_Ct", "raw_col": "raw_g_text_plus_emoji_Cvst_Ct", "builder": build_prompt_text_plus_cvst_ct},
    {"name": "h_text_plus_emoji_Cv_Cvst_Ct", "pred_col": "pred_h_text_plus_emoji_Cv_Cvst_Ct", "raw_col": "raw_h_text_plus_emoji_Cv_Cvst_Ct", "builder": build_prompt_text_plus_all},
]


# ============================================================
# Performance helpers
# ============================================================

def compute_metrics(y_true: List[int], y_pred: List[int], task_type: str) -> Dict[str, Any]:
    config = get_task_config(task_type)
    label_ids = list(config["id_to_label"].keys())
    target_names = config["target_names"]
    if len(y_true) == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0, "classification_report": {}, "confusion_matrix": [], "confusion_matrix_labels": target_names}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=label_ids, average="weighted", zero_division=0)),
        "classification_report": classification_report(y_true, y_pred, labels=label_ids, target_names=target_names, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=label_ids).tolist(),
        "confusion_matrix_labels": target_names,
    }


def run_one_experiment(df: pd.DataFrame, model_name: str, task_type: str, exp: Dict[str, Any], max_examples: Optional[int], skip_missing_descriptions: bool) -> Dict[str, Any]:
    df_iter = df.iloc[:max_examples].copy() if max_examples is not None else df.copy()
    pred_col, raw_col = exp["pred_col"], exp["raw_col"]
    df[pred_col] = pd.NA
    df[raw_col] = ""
    y_true, y_pred = [], []
    processed, used = 0, 0

    print("=" * 90)
    print(f"[RUN] {exp['name']} | task={task_type} | rows={len(df_iter)}")
    for idx, row in tqdm(df_iter.iterrows(), total=len(df_iter), desc=exp["name"]):
        processed += 1
        use_for_metric = True
        if skip_missing_descriptions:
            use_for_metric = int(row.get("flag_missing_any_description", 0)) == 0
        prompt = exp["builder"](row, task_type)
        raw_out = llama_classify(prompt, model_name=model_name, num_predict=20)
        pred = parse_label(raw_out, task_type)
        df.at[idx, pred_col] = pred
        df.at[idx, raw_col] = raw_out
        if use_for_metric:
            y_true.append(int(row["label"]))
            y_pred.append(int(pred))
            used += 1

    metrics = compute_metrics(y_true, y_pred, task_type)
    result = {
        "experiment_name": exp["name"],
        "task_type": task_type,
        "prediction_column": pred_col,
        "raw_output_column": raw_col,
        "num_processed_rows": int(processed),
        "num_eval_examples": int(used),
        **metrics,
    }
    print(f"[RESULT] {exp['name']} acc={result['accuracy']:.4f} macro_f1={result['macro_f1']:.4f} weighted_f1={result['weighted_f1']:.4f}")
    return result


# ============================================================
# Natural Language Explanation (NLE)
# ============================================================

def build_nle_prompt(row: pd.Series, task_type: str, exp_name: str, pred_col: str, raw_col: str) -> str:
    # Separate explanation prompt. Classification prompt above is unchanged.
    label_map = get_task_config(task_type)["id_to_label"]
    gold_name = label_map.get(int(row["label"]), str(row["label"]))
    pred_id = row.get(pred_col, "")
    try:
        pred_name = label_map.get(int(pred_id), str(pred_id))
    except Exception:
        pred_name = str(pred_id)

    return f"""
You are explaining an LLaMA classification decision for an X/Twitter post.

Task type: {task_type}
Experiment condition: {exp_name}
Gold label: {gold_name}
Model prediction: {pred_name}
Raw model output: {row.get(raw_col, '')}

Tweet text:
{row.get('text_baseline', '')}

Emoji:
{row.get('first_emoji', '')}

C_v plain visual description:
{row.get('C_v', '')}

C_v_st smart visual evidence:
{row.get('C_v_st', '')}

C_t social/textual meaning:
{row.get('C_t', '')}

Explain why the available text/emoji/description cues may lead to the model prediction.
Return valid JSON only with these keys:
{{
  "important_text_cues": [],
  "important_emoji_cues": [],
  "important_description_cues": [],
  "short_explanation": ""
}}
""".strip()


def run_nle(df: pd.DataFrame, model_name: str, task_type: str, out_dir: str, nle_max_examples: int, nle_experiment_name: str) -> pd.DataFrame:
    exp = next(e for e in EXPERIMENTS if e["name"] == nle_experiment_name)
    subset = df.copy()
    # Prefer interesting cases: text-only wrong but selected condition correct.
    base_pred = "pred_a_text_only"
    target_pred = exp["pred_col"]
    if base_pred in subset.columns and target_pred in subset.columns:
        interesting = subset[(subset[base_pred].astype(str) != subset["label"].astype(str)) & (subset[target_pred].astype(str) == subset["label"].astype(str))]
        if len(interesting) > 0:
            subset = interesting
    if nle_max_examples is not None:
        subset = subset.head(nle_max_examples)

    rows = []
    print(f"[NLE] Generating explanations for {len(subset)} rows using {model_name}")
    for idx, row in tqdm(subset.iterrows(), total=len(subset), desc="NLE"):
        prompt = build_nle_prompt(row, task_type, exp["name"], exp["pred_col"], exp["raw_col"])
        explanation = llama_classify(prompt, model_name=model_name, num_predict=250)
        rows.append({
            "row_index": idx,
            "id": row.get("id", idx),
            "task_type": task_type,
            "experiment_name": exp["name"],
            "gold_label": row.get("label"),
            "prediction": row.get(exp["pred_col"]),
            "text_baseline": row.get("text_baseline", ""),
            "first_emoji": row.get("first_emoji", ""),
            "C_v": row.get("C_v", ""),
            "C_v_st": row.get("C_v_st", ""),
            "C_t": row.get("C_t", ""),
            "nle_raw_json_or_text": explanation,
        })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir, "natural_language_explanations.csv"), index=False)
    return out


def run_nle_all_conditions(df: pd.DataFrame, model_name: str, task_type: str, out_dir: str, nle_max_examples: Optional[int], nle_experiment_name: str) -> pd.DataFrame:
    """Generate NLE for one condition or all conditions. If all, every valid row x every condition is explained."""
    if nle_experiment_name != "all":
        return run_nle(df, model_name, task_type, out_dir, nle_max_examples, nle_experiment_name)

    subset = df.copy()
    if nle_max_examples is not None:
        subset = subset.head(nle_max_examples)

    rows = []
    print(f"[NLE] Generating ALL-CONDITION explanations for rows={len(subset)} x conditions={len(EXPERIMENTS)} using {model_name}")
    for idx, row in tqdm(subset.iterrows(), total=len(subset), desc="NLE rows"):
        for exp in EXPERIMENTS:
            prompt = build_nle_prompt(row, task_type, exp["name"], exp["pred_col"], exp["raw_col"])
            explanation = llama_classify(prompt, model_name=model_name, num_predict=250)
            rows.append({
                "row_index": idx,
                "id": row.get("id", idx),
                "task_type": task_type,
                "experiment_name": exp["name"],
                "gold_label": row.get("label"),
                "prediction": row.get(exp["pred_col"]),
                "correct": str(row.get(exp["pred_col"])) == str(row.get("label")),
                "text_baseline": row.get("text_baseline", ""),
                "first_emoji": row.get("first_emoji", ""),
                "C_v": row.get("C_v", ""),
                "C_v_st": row.get("C_v_st", ""),
                "C_t": row.get("C_t", ""),
                "nle_raw_json_or_text": explanation,
            })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir, "natural_language_explanations_all_conditions.csv"), index=False)
    # Also save to the old filename for Excel compatibility
    out.to_csv(os.path.join(out_dir, "natural_language_explanations.csv"), index=False)
    return out


# ============================================================
# Occlusion / prediction-switch analysis from generated preds
# ============================================================

def make_occlusion_switch_outputs(df: pd.DataFrame, results: List[Dict[str, Any]], task_type: str, out_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    label_map = get_task_config(task_type)["id_to_label"]
    work = df.copy()
    work["gold_label_name"] = work["label"].apply(lambda x: label_map.get(int(x), str(x)))

    for exp in EXPERIMENTS:
        col = exp["pred_col"]
        if col in work.columns:
            work[f"{exp['name']}_correct"] = work[col].astype(str) == work["label"].astype(str)
            work[f"{exp['name']}_label_name"] = work[col].apply(lambda x: label_map.get(int(x), str(x)) if str(x).isdigit() else str(x))

    def case_type(row):
        base = bool(row.get("a_text_only_correct", False))
        cvst = bool(row.get("d_text_plus_emoji_Cvst_correct", False))
        if (not base) and cvst:
            return "C_v_st_helped_over_text_only"
        if base and (not cvst):
            return "C_v_st_hurt_vs_text_only"
        if base and cvst:
            return "both_text_and_C_v_st_correct"
        return "both_text_and_C_v_st_wrong"

    work["cvst_vs_text_case_type"] = work.apply(case_type, axis=1)

    # Component switch patterns: each condition against C_v_st.
    rows = []
    cvst_col = "pred_d_text_plus_emoji_Cvst"
    for exp in EXPERIMENTS:
        if exp["name"] == "d_text_plus_emoji_Cvst":
            continue
        col = exp["pred_col"]
        rows.append({
            "comparison": f"d_text_plus_emoji_Cvst_vs_{exp['name']}",
            "rows_where_C_v_st_correct_and_other_wrong": int(((work[cvst_col].astype(str) == work["label"].astype(str)) & (work[col].astype(str) != work["label"].astype(str))).sum()),
            "rows_where_other_correct_and_C_v_st_wrong": int(((work[col].astype(str) == work["label"].astype(str)) & (work[cvst_col].astype(str) != work["label"].astype(str))).sum()),
            "prediction_changed_count": int((work[cvst_col].astype(str) != work[col].astype(str)).sum()),
            "total_rows": int(len(work)),
        })
    switch_summary = pd.DataFrame(rows)

    # Ablation gain from metrics.
    perf = pd.DataFrame([{
        "experiment_name": r["experiment_name"],
        "accuracy": r["accuracy"],
        "macro_f1": r["macro_f1"],
        "weighted_f1": r["weighted_f1"],
        "num_eval_examples": r["num_eval_examples"],
    } for r in results])
    lookup = perf.set_index("experiment_name")
    gain_rows = []
    for exp in EXPERIMENTS:
        if exp["name"] == "d_text_plus_emoji_Cvst":
            continue
        if "d_text_plus_emoji_Cvst" in lookup.index and exp["name"] in lookup.index:
            gain_rows.append({
                "comparison": f"C_v_st_minus_{exp['name']}",
                "macro_f1_gain": float(lookup.loc["d_text_plus_emoji_Cvst", "macro_f1"] - lookup.loc[exp["name"], "macro_f1"]),
                "accuracy_gain": float(lookup.loc["d_text_plus_emoji_Cvst", "accuracy"] - lookup.loc[exp["name"], "accuracy"]),
                "weighted_f1_gain": float(lookup.loc["d_text_plus_emoji_Cvst", "weighted_f1"] - lookup.loc[exp["name"], "weighted_f1"]),
            })
    ablation_gain = pd.DataFrame(gain_rows)

    work.to_csv(os.path.join(out_dir, "occlusion_prediction_switch_all_rows.csv"), index=False)
    switch_summary.to_csv(os.path.join(out_dir, "occlusion_switch_summary.csv"), index=False)
    ablation_gain.to_csv(os.path.join(out_dir, "ablation_f1_gain_summary.csv"), index=False)
    return switch_summary, ablation_gain


# ============================================================
# Hugging Face label probability + logit/logprob difference
# ============================================================

class HFLabelScorer:
    def __init__(self, model_name: str, device: str = "auto", load_in_4bit: bool = False):
        if torch is None or AutoTokenizer is None:
            raise ImportError("Please install torch and transformers for HF label scoring.")
        self.model_name = model_name
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"[HF] Loading tokenizer/model: {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        kwargs = {}
        if self.device == "cuda":
            kwargs["torch_dtype"] = torch.float16
        else:
            kwargs["torch_dtype"] = torch.float32
        if load_in_4bit:
            kwargs["load_in_4bit"] = True
            kwargs.pop("torch_dtype", None)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        if not load_in_4bit:
            self.model.to(self.device)
        self.model.eval()

    def label_texts(self, task_type: str) -> List[str]:
        if task_type == "sentiment3":
            return ["Negative", "Neutral", "Positive"]
        if task_type == "sentiment2":
            return ["Negative", "Positive"]
        if task_type == "sarcasm2":
            return ["Not Sarcasm", "Sarcasm"]
        raise ValueError(task_type)

    def label_to_id_from_text(self, label_text: str, task_type: str) -> int:
        return parse_label(label_text, task_type)

    @torch.no_grad()
    def conditional_label_logprob(self, prompt: str, label_text: str) -> float:
        # Multi-token verbalizer score: sum log p(label_tokens | prompt + previous label tokens)
        # Add a leading space because the prompt ends with instruction, label follows naturally.
        full = prompt + " " + label_text
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)
        full_ids = self.tokenizer(full, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)
        input_ids = full_ids
        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits[:, :-1, :]
        target_ids = input_ids[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)

        # label target positions start at prompt length - 1 in shifted target indexing.
        prompt_len = prompt_ids.shape[1]
        start = max(prompt_len - 1, 0)
        end = target_ids.shape[1]
        selected = log_probs[:, start:end, :].gather(-1, target_ids[:, start:end].unsqueeze(-1)).squeeze(-1)
        return float(selected.sum().detach().cpu())

    def score_prompt(self, prompt: str, task_type: str) -> Dict[str, Any]:
        labels = self.label_texts(task_type)
        scores = [self.conditional_label_logprob(prompt, lab) for lab in labels]
        score_tensor = torch.tensor(scores, dtype=torch.float32)
        probs = torch.softmax(score_tensor, dim=0).cpu().numpy().tolist()
        best_i = int(np.argmax(probs))
        out = {"hf_pred_label_text": labels[best_i], "hf_pred_label_id": self.label_to_id_from_text(labels[best_i], task_type)}
        for lab, sc, pr in zip(labels, scores, probs):
            safe = re.sub(r"[^A-Za-z0-9]+", "_", lab.strip()).strip("_").lower()
            out[f"label_score_logprob_{safe}"] = float(sc)
            out[f"label_probability_{safe}"] = float(pr)
        # generalized logprob difference: best - second best, plus gold/pred later outside
        sorted_scores = sorted(scores, reverse=True)
        out["logprob_diff_best_minus_second"] = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else 0.0
        return out


def run_hf_probability_analysis(df: pd.DataFrame, task_type: str, out_dir: str, hf_model_name: str, max_rows: int, device: str, load_in_4bit: bool) -> pd.DataFrame:
    scorer = HFLabelScorer(hf_model_name, device=device, load_in_4bit=load_in_4bit)
    subset = df.head(max_rows).copy() if max_rows else df.copy()
    rows = []
    label_id_to_text = {0: "Negative", 1: "Neutral", 2: "Positive"} if task_type == "sentiment3" else ({0: "Negative", 1: "Positive"} if task_type == "sentiment2" else {0: "Not Sarcasm", 1: "Sarcasm"})
    label_texts = scorer.label_texts(task_type)

    print(f"[HF] Label probability/logprob analysis rows={len(subset)} model={hf_model_name}")
    for idx, row in tqdm(subset.iterrows(), total=len(subset), desc="HF probabilities"):
        for exp in EXPERIMENTS:
            prompt = exp["builder"](row, task_type)
            scores = scorer.score_prompt(prompt, task_type)
            gold_id = int(row["label"])
            gold_text = label_id_to_text[gold_id]
            gold_safe = re.sub(r"[^A-Za-z0-9]+", "_", gold_text.strip()).strip("_").lower()
            other_scores = []
            for lab in label_texts:
                if lab != gold_text:
                    safe = re.sub(r"[^A-Za-z0-9]+", "_", lab.strip()).strip("_").lower()
                    other_scores.append(scores[f"label_score_logprob_{safe}"])
            scores["gold_label_text"] = gold_text
            scores["gold_logprob_margin_vs_best_other"] = float(scores[f"label_score_logprob_{gold_safe}"] - max(other_scores)) if other_scores else 0.0
            rows.append({
                "row_index": idx,
                "id": row.get("id", idx),
                "task_type": task_type,
                "experiment_name": exp["name"],
                "gold_label_id": gold_id,
                "gold_label_text": gold_text,
                "text_baseline": row.get("text_baseline", ""),
                "first_emoji": row.get("first_emoji", ""),
                **scores,
            })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir, "hf_label_probabilities_logprob_differences.csv"), index=False)

    # Pivot deltas for C_v_st vs other conditions.
    prob_col = "gold_logprob_margin_vs_best_other"
    pivot = out.pivot_table(index="row_index", columns="experiment_name", values=prob_col, aggfunc="first").reset_index()
    if "d_text_plus_emoji_Cvst" in pivot.columns:
        for exp in [e["name"] for e in EXPERIMENTS if e["name"] != "d_text_plus_emoji_Cvst"]:
            if exp in pivot.columns:
                pivot[f"delta_cvst_minus_{exp}"] = pivot["d_text_plus_emoji_Cvst"] - pivot[exp]
    pivot.to_csv(os.path.join(out_dir, "hf_cvst_logprob_margin_deltas.csv"), index=False)

    # All pairwise margin deltas: A - B for every condition pair.
    all_delta = pivot.copy()
    exp_names = [e["name"] for e in EXPERIMENTS if e["name"] in all_delta.columns]
    for a in exp_names:
        for b in exp_names:
            if a != b:
                all_delta[f"delta_{a}_minus_{b}"] = all_delta[a] - all_delta[b]
    all_delta.to_csv(os.path.join(out_dir, "hf_all_condition_logprob_margin_pairwise_deltas.csv"), index=False)
    return out


# ============================================================
# Optional Integrated Gradients
# ============================================================

def run_integrated_gradients(df: pd.DataFrame, task_type: str, out_dir: str, hf_model_name: str, max_rows: Optional[int], device: str, target_experiment: str = "all", steps: int = 16) -> pd.DataFrame:
    """
    Lightweight IG over token embeddings for causal LM label verbalizer.
    Target = gold label verbalizer score. This is slower than probability scoring.
    If target_experiment='all', runs IG for all eight prompt conditions.
    """
    if torch is None:
        raise ImportError("Install torch and transformers to use Integrated Gradients.")
    if target_experiment == "all":
        all_out = []
        for exp_def in EXPERIMENTS:
            print(f"[IG] Starting condition: {exp_def['name']}")
            one = run_integrated_gradients(df, task_type, out_dir, hf_model_name, max_rows, device, exp_def["name"], steps)
            all_out.append(one)
        combined = pd.concat(all_out, ignore_index=True) if all_out else pd.DataFrame()
        combined.to_csv(os.path.join(out_dir, "integrated_gradients_token_attributions_all_conditions.csv"), index=False)
        top = combined.sort_values(["row_index", "experiment_name", "ig_score"], ascending=[True, True, False]).groupby(["row_index", "experiment_name"]).head(20)
        top.to_csv(os.path.join(out_dir, "integrated_gradients_top_tokens_per_row_all_conditions.csv"), index=False)
        return combined

    scorer = HFLabelScorer(hf_model_name, device=device, load_in_4bit=False)
    exp = next(e for e in EXPERIMENTS if e["name"] == target_experiment)
    subset = df.head(max_rows).copy() if max_rows else df.copy()
    label_id_to_text = {0: "Negative", 1: "Neutral", 2: "Positive"} if task_type == "sentiment3" else ({0: "Negative", 1: "Positive"} if task_type == "sentiment2" else {0: "Not Sarcasm", 1: "Sarcasm"})
    rows = []
    emb_layer = scorer.model.get_input_embeddings()

    print(f"[IG] Running Integrated Gradients rows={len(subset)} steps={steps} target_exp={target_experiment}")
    for idx, row in tqdm(subset.iterrows(), total=len(subset), desc="Integrated Gradients"):
        prompt = exp["builder"](row, task_type)
        label_text = label_id_to_text[int(row["label"])]
        full = prompt + " " + label_text
        prompt_ids = scorer.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(scorer.device)
        full_ids = scorer.tokenizer(full, add_special_tokens=False, return_tensors="pt")["input_ids"].to(scorer.device)
        input_emb = emb_layer(full_ids).detach()
        baseline = torch.zeros_like(input_emb)
        prompt_len = prompt_ids.shape[1]
        target_ids = full_ids[:, 1:]
        start = max(prompt_len - 1, 0)
        end = target_ids.shape[1]
        total_grad = torch.zeros_like(input_emb)

        for alpha in torch.linspace(0, 1, steps, device=scorer.device):
            emb = baseline + alpha * (input_emb - baseline)
            emb.requires_grad_(True)
            outputs = scorer.model(inputs_embeds=emb)
            logits = outputs.logits[:, :-1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[:, start:end, :].gather(-1, target_ids[:, start:end].unsqueeze(-1)).squeeze(-1).sum()
            scorer.model.zero_grad(set_to_none=True)
            selected.backward()
            total_grad += emb.grad.detach()

        avg_grad = total_grad / steps
        ig = (input_emb - baseline) * avg_grad
        token_scores = ig.norm(dim=-1).squeeze(0).detach().cpu().numpy()
        tokens = scorer.tokenizer.convert_ids_to_tokens(full_ids.squeeze(0).detach().cpu().tolist())
        for pos, (tok, score) in enumerate(zip(tokens, token_scores)):
            rows.append({
                "row_index": idx,
                "id": row.get("id", idx),
                "experiment_name": target_experiment,
                "gold_label_text": label_text,
                "token_position": pos,
                "token": tok,
                "ig_score": float(score),
                "text_baseline": row.get("text_baseline", ""),
                "first_emoji": row.get("first_emoji", ""),
            })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir, "integrated_gradients_token_attributions.csv"), index=False)

    # Top tokens per row
    top = out.sort_values(["row_index", "ig_score"], ascending=[True, False]).groupby("row_index").head(20)
    top.to_csv(os.path.join(out_dir, "integrated_gradients_top_tokens_per_row.csv"), index=False)
    return out


# ============================================================
# Excel output
# ============================================================

def save_excel_bundle(df: pd.DataFrame, results: List[Dict[str, Any]], out_xlsx: str, extra_sheets: Dict[str, pd.DataFrame]):
    summary_df = pd.DataFrame([{
        "experiment_name": r["experiment_name"],
        "task_type": r["task_type"],
        "num_processed_rows": r["num_processed_rows"],
        "num_eval_examples": r["num_eval_examples"],
        "accuracy": r["accuracy"],
        "macro_f1": r["macro_f1"],
        "weighted_f1": r["weighted_f1"],
    } for r in results])

    cm_rows, report_rows = [], []
    for r in results:
        labels, cm = r["confusion_matrix_labels"], r["confusion_matrix"]
        if cm:
            for i, true_label in enumerate(labels):
                for j, pred_label in enumerate(labels):
                    cm_rows.append({"experiment_name": r["experiment_name"], "task_type": r["task_type"], "true_label": true_label, "predicted_label": pred_label, "count": cm[i][j]})
        for label_name, metrics in r.get("classification_report", {}).items():
            if isinstance(metrics, dict):
                report_rows.append({"experiment_name": r["experiment_name"], "task_type": r["task_type"], "label_or_average": label_name, "precision": metrics.get("precision"), "recall": metrics.get("recall"), "f1_score": metrics.get("f1-score"), "support": metrics.get("support")})

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Predictions")
        summary_df.to_excel(writer, index=False, sheet_name="Summary_F1")
        pd.DataFrame(cm_rows).to_excel(writer, index=False, sheet_name="Confusion_Matrix")
        pd.DataFrame(report_rows).to_excel(writer, index=False, sheet_name="Classification_Report")
        for name, sheet_df in extra_sheets.items():
            if sheet_df is not None and not sheet_df.empty:
                sheet_df.head(1000000).to_excel(writer, index=False, sheet_name=name[:31])
    print(f"[SAVE] Excel: {out_xlsx}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Run LLaMA caption experiments + explainability without changing classification prompts.")
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--emoji_csv", required=True)
    parser.add_argument("--task_type", required=True, choices=["sentiment3", "sentiment2", "sarcasm2"])
    parser.add_argument("--model_name", default="llama3.2:3b", help="Ollama model for classification and optional NLE")
    parser.add_argument("--out_dir", default="outputs/explainability_run")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--skip_missing_descriptions", action="store_true")

    parser.add_argument("--do_nle", action="store_true")
    parser.add_argument("--nle_max_examples", type=int, default=None, help="Default None = all valid rows for NLE")
    parser.add_argument("--nle_experiment_name", default="all", help="Experiment for NLE, or all")

    parser.add_argument("--do_hf_probs", action="store_true")
    parser.add_argument("--hf_model_name", default=None, help="HF model, e.g., meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--hf_max_rows", type=int, default=None, help="Default None = all valid rows")
    parser.add_argument("--hf_device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--hf_load_in_4bit", action="store_true")

    parser.add_argument("--do_integrated_gradients", action="store_true")
    parser.add_argument("--ig_max_rows", type=int, default=None, help="Default None = all valid rows")
    parser.add_argument("--ig_steps", type=int, default=16)
    parser.add_argument("--ig_experiment_name", default="all", help="Experiment for IG, or all")

    args = parser.parse_args()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print("[INFO] Loading dataset and merging LLaMA captions...")
    df = load_dataset_with_llama_captions(args.train_csv, args.emoji_csv, args.task_type)

    results = []
    for exp in EXPERIMENTS:
        results.append(run_one_experiment(df, args.model_name, args.task_type, exp, args.max_examples, args.skip_missing_descriptions))

    pred_csv = os.path.join(out_dir, f"{args.task_type}_predictions_with_explainability_base.csv")
    pred_json = os.path.join(out_dir, f"{args.task_type}_results_summary.json")
    df.to_csv(pred_csv, index=False)
    with open(pred_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[SAVE] Predictions CSV: {pred_csv}")
    print(f"[SAVE] Results JSON: {pred_json}")

    switch_summary, ablation_gain = make_occlusion_switch_outputs(df, results, args.task_type, out_dir)

    extra_sheets = {
        "Occlusion_Switch_Summary": switch_summary,
        "Ablation_F1_Gain": ablation_gain,
    }

    if args.do_nle:
        nle_df = run_nle_all_conditions(df, args.model_name, args.task_type, out_dir, args.nle_max_examples, args.nle_experiment_name)
        extra_sheets["NLE_Explanations"] = nle_df

    if args.do_hf_probs:
        if not args.hf_model_name:
            raise ValueError("--hf_model_name is required when --do_hf_probs is used")
        hf_df = run_hf_probability_analysis(df, args.task_type, out_dir, args.hf_model_name, args.hf_max_rows, args.hf_device, args.hf_load_in_4bit)
        extra_sheets["HF_Label_Probabilities"] = hf_df

    if args.do_integrated_gradients:
        if not args.hf_model_name:
            raise ValueError("--hf_model_name is required when --do_integrated_gradients is used")
        ig_df = run_integrated_gradients(df, args.task_type, out_dir, args.hf_model_name, args.ig_max_rows, args.hf_device, args.ig_experiment_name, args.ig_steps)
        extra_sheets["Integrated_Gradients"] = ig_df.head(50000)

    out_xlsx = os.path.join(out_dir, f"{args.task_type}_full_explainability_outputs.xlsx")
    save_excel_bundle(df, results, out_xlsx, extra_sheets)

    print("\n[DONE] Full experiment + explainability completed.")
    print(f"[OUTPUT FOLDER] {out_dir}")


if __name__ == "__main__":
    main()
