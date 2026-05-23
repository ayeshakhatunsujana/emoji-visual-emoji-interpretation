# ============================================================
# src/run_experiments_llama_captions.py
#
# Flexible experiment runner for Llama-generated emoji captions.
#
# Supports three task types:
#   1. sentiment3 = Negative / Neutral / Positive
#   2. sentiment2 = Negative / Positive
#   3. sarcasm2   = Not Sarcasm / Sarcasm
#
# Uses:
#   C_v      = plain visual description
#   C_v_st   = smart visual-emotion / visual-communication evidence
#   C_t      = social/textual emoji meaning
#
# Saves:
#   1. CSV with all predictions
#   2. XLSX with Predictions, Summary, Confusion_Matrix
#   3. JSON result summary
# ============================================================

import argparse
import json
from typing import Dict, Any, List, Callable

import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

import ollama

from .data_utils_llama_exp import (
    load_dataset_with_llama_captions,
    get_task_config,
)


# ============================================================
# Label instruction by task
# ============================================================

def label_instruction(task_type: str) -> str:
    """
    Return the exact output instruction for each task.
    """
    if task_type == "sentiment3":
        return "Answer only one word: Positive, Neutral, or Negative."

    if task_type == "sentiment2":
        return "Answer only one word: Positive or Negative. Do not answer Neutral."

    if task_type == "sarcasm2":
        return "Answer only one phrase: Sarcasm or Not Sarcasm."

    raise ValueError(f"Unknown task_type: {task_type}")


def task_description(task_type: str) -> str:
    """
    Return task sentence for prompt.
    """
    if task_type == "sentiment3":
        return "Classify the sentiment of the following tweet."

    if task_type == "sentiment2":
        return "Classify the sentiment of the following tweet as positive or negative."

    if task_type == "sarcasm2":
        return "Classify whether the following tweet is sarcastic or not sarcastic."

    raise ValueError(f"Unknown task_type: {task_type}")


# ============================================================
# Parse Llama output into integer label
# ============================================================

def parse_label(raw_text: str, task_type: str) -> int:
    """
    Convert Llama output to integer label depending on task_type.

    sentiment3:
      0 = negative
      1 = neutral
      2 = positive

    sentiment2:
      0 = negative
      1 = positive

    sarcasm2:
      0 = not_sarcasm
      1 = sarcasm
    """
    if raw_text is None:
        if task_type == "sentiment3":
            return 1
        return 0

    text = str(raw_text).strip().lower()

    # Clean common punctuation/noise
    text = (
        text.replace(".", " ")
        .replace(",", " ")
        .replace(":", " ")
        .replace(";", " ")
        .replace("\n", " ")
        .strip()
    )

    # --------------------------
    # 3-class sentiment
    # --------------------------
    if task_type == "sentiment3":
        if "negative" in text:
            return 0
        if "neutral" in text:
            return 1
        if "positive" in text:
            return 2

        # fallback = neutral for 3-class
        return 1

    # --------------------------
    # 2-class sentiment
    # --------------------------
    if task_type == "sentiment2":
        if "negative" in text:
            return 0
        if "positive" in text:
            return 1

        # no neutral allowed, fallback = negative
        return 0

    # --------------------------
    # 2-class sarcasm
    # --------------------------
    if task_type == "sarcasm2":
        # Important: check NOT sarcasm before sarcasm
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

        # fallback = not sarcasm
        return 0

    raise ValueError(f"Unknown task_type: {task_type}")


# ============================================================
# Local Llama call through Ollama
# ============================================================

def llama_classify(
    prompt: str,
    model_name: str = "llama3.2:3b",
) -> str:
    """
    Call local Llama model using Ollama.
    """
    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            options={
                "temperature": 0.0,
                "num_predict": 20,
            },
        )

        return response["message"]["content"].strip()

    except Exception as e:
        print(f"[OLLAMA ERROR] {e}")
        return ""


# ============================================================
# Prompt builders
# These do NOT change your generated captions.
# They only use existing C_v, C_v_st, C_t for classification.
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


# ============================================================
# Experiment runner
# ============================================================

def run_one_experiment(
    df: pd.DataFrame,
    model_name: str,
    task_type: str,
    experiment_name: str,
    pred_col: str,
    raw_col: str,
    prompt_builder: Callable[[pd.Series, str], str],
    max_examples: int | None = None,
    skip_missing_descriptions: bool = False,
) -> Dict[str, Any]:
    """
    Run one experiment.

    Predictions are saved for processed rows.
    Metrics can skip rows where C_v/C_v_st/C_t are missing if
    --skip_missing_descriptions is used.
    """

    if max_examples is not None:
        df_iter = df.iloc[:max_examples].copy()
    else:
        df_iter = df.copy()

    df[pred_col] = pd.NA
    df[raw_col] = ""

    y_true: List[int] = []
    y_pred: List[int] = []

    used_for_metrics = 0
    processed = 0

    print("=" * 90)
    print(f"[RUN] Experiment: {experiment_name}")
    print(f"[RUN] Task type: {task_type}")
    print(f"[RUN] Rows to process: {len(df_iter)}")

    for idx, row in tqdm(
        df_iter.iterrows(),
        total=len(df_iter),
        desc=experiment_name,
    ):
        processed += 1

        gold = int(row["label"])

        # Decide whether this row counts for metrics
        if skip_missing_descriptions:
            flag = int(row.get("flag_missing_any_description", 0))
            use_for_metric = flag == 0
        else:
            use_for_metric = True

        prompt = prompt_builder(row, task_type)
        raw_out = llama_classify(prompt, model_name=model_name)
        pred = parse_label(raw_out, task_type)

        df.at[idx, pred_col] = pred
        df.at[idx, raw_col] = raw_out

        if use_for_metric:
            y_true.append(gold)
            y_pred.append(pred)
            used_for_metrics += 1

    config = get_task_config(task_type)
    label_ids = list(config["id_to_label"].keys())
    target_names = config["target_names"]

    if used_for_metrics == 0:
        accuracy = 0.0
        macro_f1 = 0.0
        weighted_f1 = 0.0
        report_dict = {}
        cm = []
    else:
        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(
            y_true,
            y_pred,
            labels=label_ids,
            average="macro",
            zero_division=0,
        )
        weighted_f1 = f1_score(
            y_true,
            y_pred,
            labels=label_ids,
            average="weighted",
            zero_division=0,
        )
        report_dict = classification_report(
            y_true,
            y_pred,
            labels=label_ids,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=label_ids,
        ).tolist()

    result = {
        "experiment_name": experiment_name,
        "task_type": task_type,
        "prediction_column": pred_col,
        "raw_output_column": raw_col,
        "num_processed_rows": int(processed),
        "num_eval_examples": int(used_for_metrics),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix_labels": target_names,
        "confusion_matrix": cm,
        "classification_report": report_dict,
    }

    print(
        f"[RESULT] {experiment_name} | "
        f"processed={processed} | used={used_for_metrics} | "
        f"accuracy={accuracy:.4f} | macro_f1={macro_f1:.4f} | weighted_f1={weighted_f1:.4f}"
    )

    return result


# ============================================================
# Save Excel output
# ============================================================

def save_excel(
    df: pd.DataFrame,
    results: List[Dict[str, Any]],
    out_xlsx: str,
):
    """
    Save predictions and summary into an Excel file.
    """

    summary_rows = []

    for r in results:
        summary_rows.append(
            {
                "experiment_name": r["experiment_name"],
                "task_type": r["task_type"],
                "num_processed_rows": r["num_processed_rows"],
                "num_eval_examples": r["num_eval_examples"],
                "accuracy": r["accuracy"],
                "macro_f1": r["macro_f1"],
                "weighted_f1": r["weighted_f1"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    # Confusion matrix long table
    cm_rows = []

    for r in results:
        labels = r["confusion_matrix_labels"]
        cm = r["confusion_matrix"]

        if cm:
            for i, true_label in enumerate(labels):
                for j, pred_label in enumerate(labels):
                    cm_rows.append(
                        {
                            "experiment_name": r["experiment_name"],
                            "task_type": r["task_type"],
                            "true_label": true_label,
                            "predicted_label": pred_label,
                            "count": cm[i][j],
                        }
                    )

    cm_df = pd.DataFrame(cm_rows)

    # Classification report table
    report_rows = []

    for r in results:
        report = r.get("classification_report", {})

        for label_name, metrics in report.items():
            if isinstance(metrics, dict):
                report_rows.append(
                    {
                        "experiment_name": r["experiment_name"],
                        "task_type": r["task_type"],
                        "label_or_average": label_name,
                        "precision": metrics.get("precision", None),
                        "recall": metrics.get("recall", None),
                        "f1_score": metrics.get("f1-score", None),
                        "support": metrics.get("support", None),
                    }
                )

    report_df = pd.DataFrame(report_rows)

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Predictions")
        summary_df.to_excel(writer, index=False, sheet_name="Summary")
        cm_df.to_excel(writer, index=False, sheet_name="Confusion_Matrix")
        report_df.to_excel(writer, index=False, sheet_name="Classification_Report")

    print(f"[SAVE] Excel saved to: {out_xlsx}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run local Llama sentiment/sarcasm experiments using Llama-generated emoji captions."
    )

    parser.add_argument(
        "--train_csv",
        required=True,
        help="Tweet dataset CSV path.",
    )

    parser.add_argument(
        "--emoji_csv",
        required=True,
        help="Llama-generated emoji caption CSV, e.g., outputs/results_llama_full.csv.",
    )

    parser.add_argument(
        "--task_type",
        required=True,
        choices=["sentiment3", "sentiment2", "sarcasm2"],
        help="Task type: sentiment3, sentiment2, or sarcasm2.",
    )

    parser.add_argument(
        "--model_name",
        default="llama3.2:3b",
        help="Local Ollama text model for classification.",
    )

    parser.add_argument(
        "--out_csv",
        default="outputs/llama_caption_predictions.csv",
        help="Output CSV path.",
    )

    parser.add_argument(
        "--out_xlsx",
        default="outputs/llama_caption_predictions.xlsx",
        help="Output Excel path.",
    )

    parser.add_argument(
        "--out_json",
        default="outputs/llama_caption_results.json",
        help="Output JSON summary path.",
    )

    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Optional number of rows for quick testing.",
    )

    parser.add_argument(
        "--skip_missing_descriptions",
        action="store_true",
        help="If used, metrics skip rows with emoji but missing C_v/C_v_st/C_t.",
    )

    args = parser.parse_args()

    print("[INFO] Loading dataset and merging Llama captions...")

    df = load_dataset_with_llama_captions(
        train_csv_path=args.train_csv,
        emoji_csv_path=args.emoji_csv,
        task_type=args.task_type,
    )

    print("[INFO] Dataset loaded.")
    print("[INFO] Total rows:", len(df))
    print("[INFO] Label distribution:")
    print(df["label"].value_counts().sort_index())

    experiments = [
        {
            "name": "a_text_only",
            "pred_col": "pred_a_text_only",
            "raw_col": "raw_a_text_only",
            "builder": build_prompt_text_only,
        },
        {
            "name": "b_text_plus_emoji",
            "pred_col": "pred_b_text_plus_emoji",
            "raw_col": "raw_b_text_plus_emoji",
            "builder": build_prompt_text_plus_emoji,
        },
        {
            "name": "c_text_plus_emoji_Cv",
            "pred_col": "pred_c_text_plus_emoji_Cv",
            "raw_col": "raw_c_text_plus_emoji_Cv",
            "builder": build_prompt_text_plus_cv,
        },
        {
            "name": "d_text_plus_emoji_Cvst",
            "pred_col": "pred_d_text_plus_emoji_Cvst",
            "raw_col": "raw_d_text_plus_emoji_Cvst",
            "builder": build_prompt_text_plus_cvst,
        },
        {
            "name": "e_text_plus_emoji_Ct",
            "pred_col": "pred_e_text_plus_emoji_Ct",
            "raw_col": "raw_e_text_plus_emoji_Ct",
            "builder": build_prompt_text_plus_ct,
        },
        {
            "name": "f_text_plus_emoji_Cv_Ct",
            "pred_col": "pred_f_text_plus_emoji_Cv_Ct",
            "raw_col": "raw_f_text_plus_emoji_Cv_Ct",
            "builder": build_prompt_text_plus_cv_ct,
        },
        {
            "name": "g_text_plus_emoji_Cvst_Ct",
            "pred_col": "pred_g_text_plus_emoji_Cvst_Ct",
            "raw_col": "raw_g_text_plus_emoji_Cvst_Ct",
            "builder": build_prompt_text_plus_cvst_ct,
        },
        {
            "name": "h_text_plus_emoji_Cv_Cvst_Ct",
            "pred_col": "pred_h_text_plus_emoji_Cv_Cvst_Ct",
            "raw_col": "raw_h_text_plus_emoji_Cv_Cvst_Ct",
            "builder": build_prompt_text_plus_all,
        },
    ]

    results: List[Dict[str, Any]] = []

    for exp in experiments:
        result = run_one_experiment(
            df=df,
            model_name=args.model_name,
            task_type=args.task_type,
            experiment_name=exp["name"],
            pred_col=exp["pred_col"],
            raw_col=exp["raw_col"],
            prompt_builder=exp["builder"],
            max_examples=args.max_examples,
            skip_missing_descriptions=args.skip_missing_descriptions,
        )
        results.append(result)

    print(f"[SAVE] Saving predictions CSV to: {args.out_csv}")
    df.to_csv(args.out_csv, index=False)

    save_excel(
        df=df,
        results=results,
        out_xlsx=args.out_xlsx,
    )

    print(f"[SAVE] Saving result JSON to: {args.out_json}")
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("[DONE] All experiments completed.")


if __name__ == "__main__":
    main()