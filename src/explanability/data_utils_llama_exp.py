# ============================================================
# src/data_utils_llama_exp.py
# Flexible loader for:
#   1. sentiment3: negative / neutral / positive
#   2. sentiment2: negative / positive
#   3. sarcasm2: not_sarcasm / sarcasm
# ============================================================

import pandas as pd
from typing import Optional

try:
    import regex as ucre
except Exception:
    import re as ucre


try:
    _EMOJI_RE = ucre.compile(
        r"\p{Extended_Pictographic}(?:\uFE0F|\uFE0E)?(?:\u200D\p{Extended_Pictographic}(?:\uFE0F|\uFE0E)?)*"
    )
except Exception:
    _EMOJI_RE = ucre.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")


_SKIN_TONE_MODIFIERS = {chr(cp) for cp in range(0x1F3FB, 0x1F400)}
_VARIATION_SELECTORS = {"\uFE0F", "\uFE0E"}


def extract_first_emoji(text: str) -> str:
    if text is None:
        return ""
    m = _EMOJI_RE.search(str(text))
    return m.group(0) if m else ""


def normalize_emoji(e: str) -> str:
    if e is None:
        return ""
    s = str(e).strip()
    s = "".join(
        ch for ch in s
        if ch not in _VARIATION_SELECTORS and ch not in _SKIN_TONE_MODIFIERS
    )
    return s.strip()


def get_task_config(task_type: str):
    task_type = task_type.lower().strip()

    if task_type == "sentiment3":
        return {
            "task_type": "sentiment3",
            "id_to_label": {0: "negative", 1: "neutral", 2: "positive"},
            "label_to_id": {"negative": 0, "neutral": 1, "positive": 2},
            "valid_outputs": ["Negative", "Neutral", "Positive"],
            "target_names": ["negative", "neutral", "positive"],
        }

    if task_type == "sentiment2":
        return {
            "task_type": "sentiment2",
            "id_to_label": {0: "negative", 1: "positive"},
            "label_to_id": {"negative": 0, "positive": 1},
            "valid_outputs": ["Negative", "Positive"],
            "target_names": ["negative", "positive"],
        }

    if task_type == "sarcasm2":
        return {
            "task_type": "sarcasm2",
            "id_to_label": {0: "not_sarcasm", 1: "sarcasm"},
            "label_to_id": {
                "not_sarcasm": 0,
                "not sarcasm": 0,
                "non_sarcasm": 0,
                "non sarcasm": 0,
                "sarcasm": 1,
            },
            "valid_outputs": ["Not Sarcasm", "Sarcasm"],
            "target_names": ["not_sarcasm", "sarcasm"],
        }

    raise ValueError(
        "task_type must be one of: sentiment3, sentiment2, sarcasm2"
    )


def read_train_csv(train_csv_path: str, task_type: str) -> pd.DataFrame:
    """
    Handles normal CSV and no-header tweet_eval_train_last250.csv.
    """
    config = get_task_config(task_type)

    # Special case: TweetEval binary file has no header
    if task_type == "sentiment2":
        preview = pd.read_csv(train_csv_path, nrows=1)
        cols = list(preview.columns)

        # If the file was read as weird text/label column names, reload header=None
        if "text" not in cols and "text_baseline" not in cols and "label" not in cols and "sentiment" not in cols:
            df = pd.read_csv(train_csv_path, header=None, names=["text", "sentiment"])
        else:
            df = pd.read_csv(train_csv_path)
    else:
        df = pd.read_csv(train_csv_path)

    # Text column
    if "text_baseline" not in df.columns:
        if "text" in df.columns:
            df["text_baseline"] = df["text"].astype(str)
        else:
            raise ValueError("Dataset must contain 'text' or 'text_baseline' column.")

    # Label column
    if "label" in df.columns:
        df["label"] = df["label"].astype(int)
    elif "sentiment" in df.columns:
        df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
        df["label"] = df["sentiment"].map(config["label_to_id"])

        if df["label"].isna().any():
            bad = df[df["label"].isna()]["sentiment"].unique()
            raise ValueError(f"Cannot map these labels for {task_type}: {bad}")

        df["label"] = df["label"].astype(int)
    else:
        raise ValueError("Dataset must contain 'label' or 'sentiment' column.")

    # first_emoji
    if "first_emoji" not in df.columns:
        if "emoji" in df.columns:
            df["first_emoji"] = df["emoji"]
        else:
            df["first_emoji"] = df["text_baseline"].apply(extract_first_emoji)

    df["first_emoji"] = df["first_emoji"].fillna("").astype(str).apply(normalize_emoji)

    return df


def load_dataset_with_llama_captions(
    train_csv_path: str,
    emoji_csv_path: Optional[str],
    task_type: str,
) -> pd.DataFrame:
    df = read_train_csv(train_csv_path, task_type)

    for col in ["C_v", "C_v_st", "C_t", "official_names"]:
        df[col] = ""

    if emoji_csv_path:
        emoji_df = pd.read_csv(emoji_csv_path)

        if "unicode_char" not in emoji_df.columns:
            raise ValueError("Emoji CSV must contain 'unicode_char' column.")

        emoji_df["unicode_char_norm"] = (
            emoji_df["unicode_char"].fillna("").astype(str).apply(normalize_emoji)
        )
        df["first_emoji_norm"] = (
            df["first_emoji"].fillna("").astype(str).apply(normalize_emoji)
        )

        keep_cols = ["unicode_char_norm"]
        for col in ["C_v", "C_v_st", "C_t", "official_names"]:
            if col in emoji_df.columns:
                keep_cols.append(col)

        emoji_small = emoji_df[keep_cols].copy()

        df = df.merge(
            emoji_small,
            left_on="first_emoji_norm",
            right_on="unicode_char_norm",
            how="left",
            suffixes=("", "_emoji"),
        )

        for col in ["C_v", "C_v_st", "C_t", "official_names"]:
            emoji_col = f"{col}_emoji"
            if emoji_col in df.columns:
                df[col] = df[emoji_col]
                df = df.drop(columns=[emoji_col])

        if "unicode_char_norm" in df.columns:
            df = df.drop(columns=["unicode_char_norm"])

    for col in ["C_v", "C_v_st", "C_t", "official_names"]:
        df[col] = df[col].fillna("").astype(str)

    has_emoji = df["first_emoji"].fillna("").astype(str).str.strip() != ""
    has_cv = df["C_v"].fillna("").astype(str).str.strip() != ""
    has_cvst = df["C_v_st"].fillna("").astype(str).str.strip() != ""
    has_ct = df["C_t"].fillna("").astype(str).str.strip() != ""

    df["flag_missing_any_description"] = ((has_emoji) & ~(has_cv & has_cvst & has_ct)).astype(int)

    print("=" * 80)
    print("[DATA CHECK]")
    print("Task type:", task_type)
    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print("Label counts:")
    print(df["label"].value_counts().sort_index())
    print("Rows with emoji:", int(has_emoji.sum()))
    print("Rows missing any C_v/C_v_st/C_t:", int(df["flag_missing_any_description"].sum()))
    print("=" * 80)

    return df