# src/data_utils.py

import pandas as pd
from typing import Dict, Optional

# Try to use the third-party "regex" module (better Unicode emoji support).
# It's commonly installed already (transformers depends on it).
try:
    import regex as ucre
    _HAS_UNICODE_REGEX = True
except Exception:
    import re as ucre  # fallback (less accurate)
    _HAS_UNICODE_REGEX = False


# ------------------------------------------------------------------
# Label mappings (keep for consistency, even if not heavily used)
# ------------------------------------------------------------------

INT_TO_LABEL: Dict[int, str] = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

LABEL_TO_INT: Dict[str, int] = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}


# ------------------------------
# Emoji extraction + normalization
# ------------------------------

# Unicode-aware emoji pattern (handles emoji + optional VS + optional ZWJ sequences)
if _HAS_UNICODE_REGEX:
    _EMOJI_RE = ucre.compile(
        r"\p{Extended_Pictographic}(?:\uFE0F|\uFE0E)?(?:\u200D\p{Extended_Pictographic}(?:\uFE0F|\uFE0E)?)*"
    )
else:
    # Fallback: rough range-based emoji match (not perfect, but better than nothing)
    _EMOJI_RE = ucre.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")

_SKIN_TONE_MODIFIERS = {chr(cp) for cp in range(0x1F3FB, 0x1F400)}
_VARIATION_SELECTORS = {"\uFE0F", "\uFE0E"}


def extract_first_emoji(text: str) -> str:
    """
    Extract the first emoji-like token from text.
    Returns "" if none found.
    """
    if text is None:
        return ""
    s = str(text)
    m = _EMOJI_RE.search(s)
    return m.group(0) if m else ""


def normalize_emoji(e: str) -> str:
    """
    Normalize emoji so it matches keys in your emoji CSV:
    - strip whitespace
    - remove variation selectors (FE0F/FE0E)
    - remove skin tone modifiers
    - if ZWJ sequence, keep the first component (best-effort alignment with single-emoji datasets)
    """
    if e is None:
        return ""
    s = str(e).strip()
    if not s:
        return ""

    # If ZWJ sequence, keep the first component (often your emoji CSV only has single glyphs)
    if "\u200D" in s:
        s = s.split("\u200D", 1)[0]

    # Remove variation selectors and skin tone modifiers
    s = "".join(ch for ch in s if ch not in _VARIATION_SELECTORS and ch not in _SKIN_TONE_MODIFIERS)

    return s.strip()


# ------------------------------
# Column normalization
# ------------------------------

def _normalize_label_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure there is an integer 'label' column with values:
      0 = negative, 1 = neutral, 2 = positive.

    If 'label' already exists, assume it's correct.
    Otherwise, try to create it from a 'sentiment' column with strings.
    """
    if "label" in df.columns:
        return df

    if "sentiment" in df.columns:
        df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
        df["label"] = df["sentiment"].map(LABEL_TO_INT)
        if df["label"].isna().any():
            raise ValueError("Some sentiment values could not be mapped to {negative, neutral, positive}.")
        return df

    raise ValueError("Expected a 'label' or 'sentiment' column in the train CSV.")


def _normalize_text_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure there is a 'text_baseline' column for the raw tweet text.
    If not present, fall back to 'text' if available.
    """
    if "text_baseline" in df.columns:
        return df

    if "text" in df.columns:
        df["text_baseline"] = df["text"].astype(str)
        return df

    raise ValueError("Expected a 'text_baseline' or 'text' column in the train CSV.")


def _normalize_first_emoji_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure there is a 'first_emoji' column.

    Priority:
    1) existing 'first_emoji'
    2) alternative columns (emoji, firstEmoji, first_emo)
    3) extract from text_baseline
    """
    if "first_emoji" not in df.columns:
        for cand in ["emoji", "firstEmoji", "first_emo"]:
            if cand in df.columns:
                df["first_emoji"] = df[cand]
                break

    if "first_emoji" not in df.columns:
        df["first_emoji"] = ""

    # Fill missing/blank by extracting from text
    df["first_emoji"] = df["first_emoji"].fillna("").astype(str)
    mask_blank = df["first_emoji"].str.strip() == ""
    if mask_blank.any() and "text_baseline" in df.columns:
        df.loc[mask_blank, "first_emoji"] = df.loc[mask_blank, "text_baseline"].apply(extract_first_emoji)

    # Normalize
    df["first_emoji"] = df["first_emoji"].apply(normalize_emoji)

    return df


# ------------------------------
# Main loader
# ------------------------------

def load_dataset(
    train_csv_path: str,
    emoji_csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load and merge the tweet dataset and the emoji description dataset.

    - train_csv_path: path to train_split_70.csv
      Must contain raw text + labels, e.g.:
        - 'text_baseline' or 'text'
        - 'label' (0/1/2) or 'sentiment' (negative/neutral/positive)

    - emoji_csv_path: path to visual_and_language_model_data.csv
      Must contain emoji-level metadata, usually:
        - 'unicode_char'  (emoji character itself, e.g. 😀)
        - 'C_v'           (visual caption)
        - 'C_t'           (text/Unicode description)
    """
    train_df = pd.read_csv(train_csv_path)

    # Normalize required columns
    train_df = _normalize_text_column(train_df)
    train_df = _normalize_label_column(train_df)
    train_df = _normalize_first_emoji_column(train_df)

    # Ensure C_v/C_t exist even if not merging
    train_df["C_v"] = ""
    train_df["C_t"] = ""

    if emoji_csv_path is not None:
        emoji_df = pd.read_csv(emoji_csv_path)

        # Find emoji key column in emoji CSV
        if "unicode_char" in emoji_df.columns:
            emoji_key = "unicode_char"
        elif "emoji" in emoji_df.columns:
            emoji_key = "emoji"
        else:
            raise ValueError("Emoji CSV must contain 'unicode_char' or 'emoji' column.")

        # Keep only needed columns
        cols_to_use = [emoji_key]
        if "C_v" in emoji_df.columns:
            cols_to_use.append("C_v")
        if "C_t" in emoji_df.columns:
            cols_to_use.append("C_t")
        emoji_small = emoji_df[cols_to_use].copy()

        # Normalize emoji keys on emoji side too
        emoji_small[emoji_key] = emoji_small[emoji_key].fillna("").astype(str).apply(normalize_emoji)

        # Merge on normalized emoji
        merged = train_df.merge(
            emoji_small,
            left_on="first_emoji",
            right_on=emoji_key,
            how="left",
            suffixes=("", "_emoji"),
        )

        # Drop duplicate key column from emoji CSV
        if emoji_key != "first_emoji":
            merged = merged.drop(columns=[emoji_key])

        # Prefer merged columns when present
        for col in ["C_v", "C_t"]:
            if f"{col}_emoji" in merged.columns:
                merged[col] = merged[f"{col}_emoji"]
                merged = merged.drop(columns=[f"{col}_emoji"])

        train_df = merged

    # Final cleanup: fill NaNs
    train_df["C_v"] = train_df.get("C_v", "").fillna("").astype(str)
    train_df["C_t"] = train_df.get("C_t", "").fillna("").astype(str)

    return train_df
