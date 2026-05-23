# ============================================================
# src/run_experiments_qwen_captions.py
# Run sentiment/sarcasm experiments using Qwen text model
# and Qwen-generated emoji descriptions C_v, C_v_st, C_t.
# ============================================================

import argparse
import json
import os
import re
from typing import Dict, List, Tuple

import ollama
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tqdm import tqdm


CONDITIONS = {
    "a_text_only": ["text"],
    "b_text_plus_emoji": ["text", "emoji"],
    "c_text_plus_emoji_Cv": ["text", "emoji", "C_v"],
    "d_text_plus_emoji_Cvst": ["text", "emoji", "C_v_st"],
    "e_text_plus_emoji_Ct": ["text", "emoji", "C_t"],
    "f_text_plus_emoji_Cv_Ct": ["text", "emoji", "C_v", "C_t"],
    "g_text_plus_emoji_Cvst_Ct": ["text", "emoji", "C_v_st", "C_t"],
    "h_text_plus_emoji_Cv_Cvst_Ct": ["text", "emoji", "C_v", "C_v_st", "C_t"],
}


def normalize_label(x: str) -> str:
    x = str(x).strip().lower()
    x = re.sub(r"[^a-z0-9_ -]", "", x)
    x = x.replace(" ", "_")
    return x


def get_task_labels(task_type: str) -> List[str]:
    if task_type == "sentiment3":
        return ["negative", "neutral", "positive"]
    if task_type == "sentiment2":
        return ["negative", "positive"]
    if task_type == "sarcasm2":
        return ["not_sarcastic", "sarcastic"]
    raise ValueError("task_type must be one of: sentiment3, sentiment2, sarcasm2")




def normalize_gold_label(x, task_type: str) -> str:
    """
    Normalize gold labels from common dataset formats into the labels
    expected by each task.

    Handles examples like:
    - sarcasm: 0/1, false/true, non_sarcastic/sarcastic, not_sarcasm/sarcasm
    - sentiment3: 0/1/2 or negative/neutral/positive
    - sentiment2: 0/1 or negative/positive
    """
    raw = str(x).strip().lower()
    raw_clean = re.sub(r"[^a-z0-9_ -]", "", raw).replace(" ", "_")

    # Numeric labels are very common in CSV datasets.
    if task_type == "sarcasm2":
        mapping = {
            "0": "not_sarcastic",
            "1": "sarcastic",
            "false": "not_sarcastic",
            "true": "sarcastic",
            "no": "not_sarcastic",
            "yes": "sarcastic",
            "not_sarcastic": "not_sarcastic",
            "non_sarcastic": "not_sarcastic",
            "not_sarcasm": "not_sarcastic",
            "notsarcastic": "not_sarcastic",
            "sarcastic": "sarcastic",
            "sarcasm": "sarcastic",
        }
        return mapping.get(raw_clean, raw_clean)

    if task_type == "sentiment3":
        mapping = {
            "0": "negative",
            "1": "neutral",
            "2": "positive",
            "neg": "negative",
            "negative": "negative",
            "neu": "neutral",
            "neutral": "neutral",
            "pos": "positive",
            "positive": "positive",
        }
        return mapping.get(raw_clean, raw_clean)

    if task_type == "sentiment2":
        mapping = {
            "0": "negative",
            "1": "positive",
            "neg": "negative",
            "negative": "negative",
            "pos": "positive",
            "positive": "positive",
        }
        return mapping.get(raw_clean, raw_clean)

    return raw_clean


def guess_text_column(df: pd.DataFrame) -> str:
    candidates = ["text", "raw_text", "tweet", "sentence", "content", "text_baseline"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find text column. Available columns: {list(df.columns)}")


def guess_label_column(df: pd.DataFrame) -> str:
    candidates = ["label", "sentiment", "gold", "target", "class"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find label column. Available columns: {list(df.columns)}")


def guess_emoji_column(df: pd.DataFrame) -> str:
    candidates = ["first_emoji", "emoji", "unicode_char", "emojis"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find emoji column. Available columns: {list(df.columns)}")


def build_prompt(input_text: str, task_type: str, labels: List[str]) -> str:
    label_str = ", ".join(labels)
    if task_type.startswith("sentiment"):
        task = "classify the sentiment"
    else:
        task = "classify whether the text is sarcastic"

    return f"""
You are a strict text classification model.

Task: {task}.
Allowed labels: {label_str}

Rules:
- Output only one label from the allowed labels.
- Do not explain.
- Do not add punctuation.

Input:
{input_text}

Label:
""".strip()


def qwen_classify(model_name: str, input_text: str, task_type: str, labels: List[str]) -> str:
    prompt = build_prompt(input_text, task_type, labels)
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "num_predict": 20},
        )
        pred = normalize_label(response["message"]["content"])
    except Exception as e:
        print(f"[QWEN ERROR] {e}")
        pred = ""

    # robust cleanup: choose the first allowed label appearing in output
    for lab in labels:
        if lab in pred:
            return lab
    return pred if pred in labels else "unknown"


def build_condition_text(row: pd.Series, condition_parts: List[str], text_col: str, emoji_col: str) -> str:
    chunks = []
    if "text" in condition_parts:
        chunks.append(f"Text: {row[text_col]}")
    if "emoji" in condition_parts:
        chunks.append(f"Emoji: {row.get(emoji_col, '')}")
    if "C_v" in condition_parts:
        chunks.append(f"Plain visual emoji description: {row.get('C_v', '')}")
    if "C_v_st" in condition_parts:
        chunks.append(f"Smart visual-emotion emoji description: {row.get('C_v_st', '')}")
    if "C_t" in condition_parts:
        chunks.append(f"Textual/social emoji meaning: {row.get('C_t', '')}")
    return "\n".join(chunks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--emoji_csv", type=str, required=True)
    parser.add_argument("--task_type", type=str, required=True, choices=["sentiment3", "sentiment2", "sarcasm2"])
    parser.add_argument("--model_name", type=str, default="qwen2.5:3b")
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--out_xlsx", type=str, required=True)
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--skip_missing_descriptions", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    print("[INFO] Loading dataset...")
    df = pd.read_csv(args.train_csv)
    emoji_df = pd.read_csv(args.emoji_csv)

    text_col = guess_text_column(df)
    label_col = guess_label_column(df)
    emoji_col = guess_emoji_column(df)

    print(f"[INFO] Text column: {text_col}")
    print(f"[INFO] Label column: {label_col}")
    print(f"[INFO] Emoji column: {emoji_col}")

    if "unicode_char" not in emoji_df.columns:
        raise ValueError("emoji_csv must contain a unicode_char column")

    desc_cols = ["unicode_char", "C_v", "C_v_st", "C_t"]
    emoji_df = emoji_df[desc_cols].drop_duplicates("unicode_char")
    df = df.merge(emoji_df, left_on=emoji_col, right_on="unicode_char", how="left")

    if args.skip_missing_descriptions:
        before = len(df)
        df = df.dropna(subset=["C_v", "C_v_st", "C_t"])
        print(f"[INFO] Dropped missing-description rows: {before - len(df)}")

    if args.max_examples is not None:
        df = df.head(args.max_examples).copy()

    labels = get_task_labels(args.task_type)
    df["gold_label_norm"] = df[label_col].apply(lambda x: normalize_gold_label(x, args.task_type))

    print("[INFO] Gold label distribution after normalization:")
    print(df["gold_label_norm"].value_counts(dropna=False))

    bad_gold = ~df["gold_label_norm"].isin(labels)
    if bad_gold.any():
        print("[WARNING] Some gold labels are outside the allowed labels and will be removed:")
        print(df.loc[bad_gold, "gold_label_norm"].value_counts(dropna=False))
        df = df.loc[~bad_gold].copy()

    if len(df) == 0:
        raise ValueError(
            "No valid rows remain after gold-label normalization. "
            f"Allowed labels for {args.task_type}: {labels}. "
            f"Please check the label column: {label_col}."
        )

    all_results = {}
    output_df = df.copy()

    for condition_name, parts in CONDITIONS.items():
        print(f"\n🚀 Running condition: {condition_name}")
        preds = []
        inputs = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc=condition_name):
            condition_text = build_condition_text(row, parts, text_col, emoji_col)
            pred = qwen_classify(args.model_name, condition_text, args.task_type, labels)
            inputs.append(condition_text)
            preds.append(pred)

        gold = df["gold_label_norm"].tolist()
        acc = accuracy_score(gold, preds)
        macro_f1 = f1_score(gold, preds, labels=labels, average="macro", zero_division=0)
        weighted_f1 = f1_score(gold, preds, labels=labels, average="weighted", zero_division=0)
        report = classification_report(gold, preds, labels=labels, zero_division=0, output_dict=True)
        cm = confusion_matrix(gold, preds, labels=labels).tolist()

        output_df[f"{condition_name}_input"] = inputs
        output_df[f"{condition_name}_pred"] = preds

        all_results[condition_name] = {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "labels": labels,
            "confusion_matrix": cm,
            "classification_report": report,
        }
        print(f"✅ {condition_name}: Macro-F1={macro_f1:.4f}, Accuracy={acc:.4f}")

    summary = pd.DataFrame([
        {
            "condition": name,
            "accuracy": vals["accuracy"],
            "macro_f1": vals["macro_f1"],
            "weighted_f1": vals["weighted_f1"],
        }
        for name, vals in all_results.items()
    ]).sort_values("macro_f1", ascending=False)

    output_df.to_csv(args.out_csv, index=False)
    with pd.ExcelWriter(args.out_xlsx, engine="openpyxl") as writer:
        output_df.to_excel(writer, sheet_name="predictions", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary.to_dict("records"), "details": all_results}, f, indent=2, ensure_ascii=False)

    print("\n📊 Final ranking:")
    print(summary)
    print(f"\n✅ Saved CSV: {args.out_csv}")
    print(f"✅ Saved XLSX: {args.out_xlsx}")
    print(f"✅ Saved JSON: {args.out_json}")


if __name__ == "__main__":
    main()
