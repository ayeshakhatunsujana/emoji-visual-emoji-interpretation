# ============================================================
# src/explain_cvst_metrics.py
#
# Explainability analysis for Llama emoji-caption experiments.
#
# Main question:
#   Why did C_v_st help Llama performance?
#
# This script uses existing prediction files, no Llama rerun needed.
#
# Inputs:
#   --pred_csv    outputs/sentiment3_predictions_full.csv
#   --results_json outputs/sentiment3_results_full.json
#
# Outputs:
#   Excel file with multiple explainability sheets
#   CSV files for helped/hurt cases
#   PNG plots
# ============================================================

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Label mapping
# ============================================================

ID_TO_LABEL = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


# ============================================================
# Experiment columns
# ============================================================

EXP_COLS = {
    "a_text_only": "pred_a_text_only",
    "b_text_plus_emoji": "pred_b_text_plus_emoji",
    "c_text_plus_emoji_Cv": "pred_c_text_plus_emoji_Cv",
    "d_text_plus_emoji_Cvst": "pred_d_text_plus_emoji_Cvst",
    "e_text_plus_emoji_Ct": "pred_e_text_plus_emoji_Ct",
    "f_text_plus_emoji_Cv_Ct": "pred_f_text_plus_emoji_Cv_Ct",
    "g_text_plus_emoji_Cvst_Ct": "pred_g_text_plus_emoji_Cvst_Ct",
    "h_text_plus_emoji_Cv_Cvst_Ct": "pred_h_text_plus_emoji_Cv_Cvst_Ct",
}

BASE_EXP = "a_text_only"
CVST_EXP = "d_text_plus_emoji_Cvst"


# ============================================================
# C_v_st semantic categories
# You can edit these word lists later.
# ============================================================

CATEGORY_LEXICON: Dict[str, List[str]] = {
    "emotion_positive": [
        "happy", "happiness", "joy", "joyful", "amusement", "amused",
        "laugh", "laughter", "laughing", "funny", "smile", "smiling",
        "love", "admiration", "affection", "excited", "excitement",
        "relief", "pleased", "delight", "delighted", "positive",
        "approval", "agreement", "support"
    ],
    "emotion_negative": [
        "sad", "sadness", "upset", "cry", "crying", "tear", "tears",
        "angry", "anger", "frustration", "frustrated", "annoyed",
        "annoyance", "disgust", "fear", "worried", "worry",
        "negative", "rejection", "disapproval", "pain", "hurt",
        "unhappy", "distress", "distressed"
    ],
    "intensity": [
        "strong", "extreme", "very", "deep", "intense", "high",
        "mild", "slight", "highly", "overwhelming", "dramatic",
        "exaggerated", "severe"
    ],
    "facial_visual_evidence": [
        "face", "facial", "eyes", "eye", "mouth", "eyebrows", "brow",
        "cheeks", "teeth", "grin", "frown", "open mouth",
        "closed eyes", "squeezed", "tear", "tears", "sweat"
    ],
    "gesture_communication": [
        "gesture", "hand", "thumbs", "pointing", "point", "upward",
        "downward", "left", "right", "approval", "agreement",
        "disapproval", "rejection", "greeting", "farewell",
        "attention", "directs attention", "support", "acknowledgement"
    ],
    "social_pragmatic": [
        "sarcasm", "irony", "humor", "humorous", "teasing", "mock",
        "playful", "tone", "conversation", "communicative",
        "social", "reaction", "response"
    ],
}


STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "with", "to", "of", "in", "on",
    "for", "as", "by", "it", "is", "are", "was", "were", "be", "been",
    "being", "this", "that", "these", "those", "from", "at", "into",
    "its", "their", "his", "her", "shows", "show", "shown", "image",
    "emoji", "visual", "cue", "expression", "meaning", "used", "usually",
    "has", "have", "had", "can", "could", "would", "may", "might",
    "conveys", "convey", "supported", "supporting"
}


# ============================================================
# Utility functions
# ============================================================

def safe_int(x):
    try:
        if pd.isna(x):
            return None
        return int(x)
    except Exception:
        return None


def tokenize(text: str) -> List[str]:
    if pd.isna(text):
        return []
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s\-]", " ", text)
    text = text.replace("-", " ")
    toks = [t.strip() for t in text.split() if len(t.strip()) > 1]
    toks = [t for t in toks if t not in STOPWORDS]
    return toks


def contains_phrase(text: str, phrase: str) -> bool:
    if pd.isna(text):
        return False
    return phrase.lower() in str(text).lower()


def category_counts(text: str) -> Dict[str, int]:
    text_l = "" if pd.isna(text) else str(text).lower()
    counts = {}
    for cat, words in CATEGORY_LEXICON.items():
        c = 0
        for w in words:
            if " " in w:
                if contains_phrase(text_l, w):
                    c += 1
            else:
                # word boundary
                if re.search(r"\b" + re.escape(w.lower()) + r"\b", text_l):
                    c += 1
        counts[cat] = c
    return counts


def category_binary(text: str) -> Dict[str, int]:
    counts = category_counts(text)
    return {k: int(v > 0) for k, v in counts.items()}


def correctness_col(df: pd.DataFrame, pred_col: str, label_col: str = "label") -> pd.Series:
    return df[pred_col].apply(safe_int) == df[label_col].apply(safe_int)


def load_results_json(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        results = json.load(f)

    rows = []
    for r in results:
        rows.append({
            "experiment_name": r["experiment_name"],
            "accuracy": r["accuracy"],
            "macro_f1": r["macro_f1"],
            "weighted_f1": r["weighted_f1"],
            "num_eval_examples": r["num_eval_examples"],
            "num_processed_rows": r["num_processed_rows"],
        })

    return pd.DataFrame(rows)


def compute_ablation_gain(results_df: pd.DataFrame) -> pd.DataFrame:
    lookup = results_df.set_index("experiment_name")

    base_macro = lookup.loc["a_text_only", "macro_f1"]
    cvst_macro = lookup.loc["d_text_plus_emoji_Cvst", "macro_f1"]

    comparisons = [
        ("C_v_st_vs_text_only", "d_text_plus_emoji_Cvst", "a_text_only"),
        ("C_v_st_vs_emoji_only", "d_text_plus_emoji_Cvst", "b_text_plus_emoji"),
        ("C_v_st_vs_plain_Cv", "d_text_plus_emoji_Cvst", "c_text_plus_emoji_Cv"),
        ("C_v_st_vs_Ct", "d_text_plus_emoji_Cvst", "e_text_plus_emoji_Ct"),
        ("C_v_st_Ct_vs_C_v_st", "g_text_plus_emoji_Cvst_Ct", "d_text_plus_emoji_Cvst"),
        ("All_vs_C_v_st", "h_text_plus_emoji_Cv_Cvst_Ct", "d_text_plus_emoji_Cvst"),
    ]

    rows = []
    for name, exp_a, exp_b in comparisons:
        rows.append({
            "comparison": name,
            "experiment_A": exp_a,
            "experiment_B": exp_b,
            "accuracy_gain_A_minus_B": lookup.loc[exp_a, "accuracy"] - lookup.loc[exp_b, "accuracy"],
            "macro_f1_gain_A_minus_B": lookup.loc[exp_a, "macro_f1"] - lookup.loc[exp_b, "macro_f1"],
            "weighted_f1_gain_A_minus_B": lookup.loc[exp_a, "weighted_f1"] - lookup.loc[exp_b, "weighted_f1"],
        })

    return pd.DataFrame(rows)


# ============================================================
# Prediction flip analysis
# ============================================================

def add_case_type(df: pd.DataFrame) -> pd.DataFrame:
    base_col = EXP_COLS[BASE_EXP]
    cvst_col = EXP_COLS[CVST_EXP]

    df = df.copy()

    df["gold_label_name"] = df["label"].map(ID_TO_LABEL)

    df["base_correct"] = correctness_col(df, base_col).astype(int)
    df["cvst_correct"] = correctness_col(df, cvst_col).astype(int)

    df["base_pred_label_name"] = df[base_col].apply(lambda x: ID_TO_LABEL.get(safe_int(x), "NA"))
    df["cvst_pred_label_name"] = df[cvst_col].apply(lambda x: ID_TO_LABEL.get(safe_int(x), "NA"))

    def classify(row):
        b = row["base_correct"] == 1
        c = row["cvst_correct"] == 1

        if (not b) and c:
            return "C_v_st_helped"
        if b and (not c):
            return "C_v_st_hurt"
        if b and c:
            return "both_correct"
        return "both_wrong"

    df["case_type"] = df.apply(classify, axis=1)

    return df


def compute_flip_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    total = len(df)
    for case_type, group in df.groupby("case_type"):
        rows.append({
            "case_type": case_type,
            "count": len(group),
            "percentage_all_rows": len(group) / total if total else 0,
        })

    return pd.DataFrame(rows).sort_values("case_type")


# ============================================================
# Category analysis
# ============================================================

def add_category_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    cat_bin_rows = []
    cat_count_rows = []

    for text in df["C_v_st"].fillna("").astype(str):
        cat_bin_rows.append(category_binary(text))
        cat_count_rows.append(category_counts(text))

    bin_df = pd.DataFrame(cat_bin_rows).add_prefix("has_")
    count_df = pd.DataFrame(cat_count_rows).add_prefix("count_")

    return pd.concat([df.reset_index(drop=True), bin_df, count_df], axis=1)


def compute_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for cat in CATEGORY_LEXICON.keys():
        col = f"has_{cat}"

        for case_type in ["C_v_st_helped", "C_v_st_hurt", "both_correct", "both_wrong"]:
            subset = df[df["case_type"] == case_type]
            if len(subset) == 0:
                rate = 0
                count_present = 0
            else:
                count_present = int(subset[col].sum())
                rate = count_present / len(subset)

            rows.append({
                "category": cat,
                "case_type": case_type,
                "rows_in_case_type": len(subset),
                "rows_with_category": count_present,
                "category_presence_rate": rate,
            })

    return pd.DataFrame(rows)


def compute_help_vs_hurt_category_lift(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    helped = df[df["case_type"] == "C_v_st_helped"]
    hurt = df[df["case_type"] == "C_v_st_hurt"]
    both_wrong = df[df["case_type"] == "both_wrong"]

    for cat in CATEGORY_LEXICON.keys():
        col = f"has_{cat}"

        helped_rate = helped[col].mean() if len(helped) else 0
        hurt_rate = hurt[col].mean() if len(hurt) else 0
        both_wrong_rate = both_wrong[col].mean() if len(both_wrong) else 0

        rows.append({
            "category": cat,
            "helped_presence_rate": helped_rate,
            "hurt_presence_rate": hurt_rate,
            "both_wrong_presence_rate": both_wrong_rate,
            "helped_minus_hurt": helped_rate - hurt_rate,
            "helped_minus_both_wrong": helped_rate - both_wrong_rate,
        })

    return pd.DataFrame(rows).sort_values("helped_minus_hurt", ascending=False)


def compute_word_frequency_by_case(df: pd.DataFrame, top_k: int = 50) -> pd.DataFrame:
    rows = []

    for case_type, group in df.groupby("case_type"):
        counter = Counter()

        for text in group["C_v_st"].fillna("").astype(str):
            counter.update(tokenize(text))

        for word, count in counter.most_common(top_k):
            rows.append({
                "case_type": case_type,
                "word": word,
                "count": count,
                "rows_in_case_type": len(group),
                "normalized_count_per_row": count / len(group) if len(group) else 0,
            })

    return pd.DataFrame(rows)


def compute_help_word_lift(df: pd.DataFrame, top_k: int = 100) -> pd.DataFrame:
    helped = df[df["case_type"] == "C_v_st_helped"]
    not_helped = df[df["case_type"] != "C_v_st_helped"]

    helped_counter = Counter()
    not_helped_counter = Counter()

    for text in helped["C_v_st"].fillna("").astype(str):
        helped_counter.update(set(tokenize(text)))

    for text in not_helped["C_v_st"].fillna("").astype(str):
        not_helped_counter.update(set(tokenize(text)))

    helped_n = max(len(helped), 1)
    not_helped_n = max(len(not_helped), 1)

    vocab = set(helped_counter.keys()) | set(not_helped_counter.keys())

    rows = []
    for word in vocab:
        helped_rate = helped_counter[word] / helped_n
        not_helped_rate = not_helped_counter[word] / not_helped_n

        rows.append({
            "word": word,
            "helped_doc_rate": helped_rate,
            "not_helped_doc_rate": not_helped_rate,
            "helped_minus_not_helped": helped_rate - not_helped_rate,
            "helped_count": helped_counter[word],
            "not_helped_count": not_helped_counter[word],
        })

    return (
        pd.DataFrame(rows)
        .sort_values("helped_minus_not_helped", ascending=False)
        .head(top_k)
    )


# ============================================================
# Per-label and per-emoji analysis
# ============================================================

def compute_label_flip_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for label_name, group in df.groupby("gold_label_name"):
        total = len(group)

        for case_type, g2 in group.groupby("case_type"):
            rows.append({
                "gold_label": label_name,
                "case_type": case_type,
                "count": len(g2),
                "percentage_within_gold_label": len(g2) / total if total else 0,
            })

    return pd.DataFrame(rows)


def compute_emoji_summary(df: pd.DataFrame, min_count: int = 3) -> pd.DataFrame:
    rows = []

    if "first_emoji" not in df.columns:
        return pd.DataFrame()

    for emoji, group in df.groupby("first_emoji"):
        if pd.isna(emoji) or str(emoji).strip() == "":
            continue

        if len(group) < min_count:
            continue

        rows.append({
            "first_emoji": emoji,
            "n_rows": len(group),
            "text_only_accuracy": group["base_correct"].mean(),
            "cvst_accuracy": group["cvst_correct"].mean(),
            "cvst_minus_text_accuracy": group["cvst_correct"].mean() - group["base_correct"].mean(),
            "helped_count": int((group["case_type"] == "C_v_st_helped").sum()),
            "hurt_count": int((group["case_type"] == "C_v_st_hurt").sum()),
            "both_correct_count": int((group["case_type"] == "both_correct").sum()),
            "both_wrong_count": int((group["case_type"] == "both_wrong").sum()),
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("cvst_minus_text_accuracy", ascending=False)


# ============================================================
# Plot functions
# ============================================================

def save_bar_plot(df: pd.DataFrame, x_col: str, y_col: str, title: str, out_png: str, rotation: int = 35):
    if df.empty:
        return

    plt.figure(figsize=(10, 5))
    plt.bar(df[x_col].astype(str), df[y_col])
    plt.xticks(rotation=rotation, ha="right")
    plt.ylabel(y_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def save_case_type_plot(flip_summary: pd.DataFrame, out_png: str):
    if flip_summary.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.bar(flip_summary["case_type"], flip_summary["count"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Number of rows")
    plt.title("Prediction Flip Analysis: Text-only vs C_v_st")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pred_csv", required=True, help="Prediction CSV file.")
    parser.add_argument("--results_json", required=True, help="Results JSON file.")
    parser.add_argument("--out_dir", default="outputs/explain_cvst", help="Output folder.")
    parser.add_argument("--skip_missing_descriptions", action="store_true",
                        help="Analyze only rows with flag_missing_any_description == 0.")
    parser.add_argument("--min_emoji_count", type=int, default=3)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("[INFO] Loading prediction CSV:", args.pred_csv)
    df = pd.read_csv(args.pred_csv)

    print("[INFO] Original rows:", len(df))

    # Keep same evaluated rows if requested
    if args.skip_missing_descriptions and "flag_missing_any_description" in df.columns:
        df = df[df["flag_missing_any_description"] == 0].copy()
        print("[INFO] Rows after skipping missing descriptions:", len(df))

    # Check required columns
    required = ["label", "text_baseline", "C_v", "C_v_st", "C_t", EXP_COLS[BASE_EXP], EXP_COLS[CVST_EXP]]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print("[INFO] Loading results JSON:", args.results_json)
    results_df = load_results_json(args.results_json)

    ablation_gain_df = compute_ablation_gain(results_df)

    # Add case type
    df_case = add_case_type(df)

    # Add C_v_st category features
    df_case = add_category_features(df_case)

    # Summaries
    flip_summary_df = compute_flip_summary(df_case)
    category_summary_df = compute_category_summary(df_case)
    category_lift_df = compute_help_vs_hurt_category_lift(df_case)
    word_freq_df = compute_word_frequency_by_case(df_case, top_k=50)
    help_word_lift_df = compute_help_word_lift(df_case, top_k=100)
    label_flip_df = compute_label_flip_summary(df_case)
    emoji_summary_df = compute_emoji_summary(df_case, min_count=args.min_emoji_count)

    # Important case subsets
    helped_df = df_case[df_case["case_type"] == "C_v_st_helped"].copy()
    hurt_df = df_case[df_case["case_type"] == "C_v_st_hurt"].copy()
    both_correct_df = df_case[df_case["case_type"] == "both_correct"].copy()
    both_wrong_df = df_case[df_case["case_type"] == "both_wrong"].copy()

    # Select human-readable columns for case inspection
    case_cols = [
        "id", "text_baseline", "label", "gold_label_name",
        "sentiment", "first_emoji", "official_names",
        "C_v", "C_v_st", "C_t",
        EXP_COLS[BASE_EXP], "base_pred_label_name", "raw_a_text_only",
        EXP_COLS[CVST_EXP], "cvst_pred_label_name", "raw_d_text_plus_emoji_Cvst",
        "case_type"
    ]
    case_cols = [c for c in case_cols if c in df_case.columns]

    # Save CSV files
    df_case.to_csv(os.path.join(args.out_dir, "all_rows_with_explainability.csv"), index=False)
    helped_df[case_cols].to_csv(os.path.join(args.out_dir, "cvst_helped_cases.csv"), index=False)
    hurt_df[case_cols].to_csv(os.path.join(args.out_dir, "cvst_hurt_cases.csv"), index=False)
    both_correct_df[case_cols].to_csv(os.path.join(args.out_dir, "both_correct_cases.csv"), index=False)
    both_wrong_df[case_cols].to_csv(os.path.join(args.out_dir, "both_wrong_cases.csv"), index=False)

    # Save Excel
    out_xlsx = os.path.join(args.out_dir, "cvst_explainability_analysis.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        results_df.to_excel(writer, index=False, sheet_name="Performance")
        ablation_gain_df.to_excel(writer, index=False, sheet_name="Ablation_Gain")
        flip_summary_df.to_excel(writer, index=False, sheet_name="Flip_Summary")
        label_flip_df.to_excel(writer, index=False, sheet_name="Label_Flip_Summary")
        category_summary_df.to_excel(writer, index=False, sheet_name="Category_By_Case")
        category_lift_df.to_excel(writer, index=False, sheet_name="Category_Lift")
        word_freq_df.to_excel(writer, index=False, sheet_name="Word_Freq_By_Case")
        help_word_lift_df.to_excel(writer, index=False, sheet_name="Help_Word_Lift")
        emoji_summary_df.to_excel(writer, index=False, sheet_name="Emoji_Summary")

        helped_df[case_cols].to_excel(writer, index=False, sheet_name="C_vst_Helped_Cases")
        hurt_df[case_cols].to_excel(writer, index=False, sheet_name="C_vst_Hurt_Cases")
        both_correct_df[case_cols].to_excel(writer, index=False, sheet_name="Both_Correct")
        both_wrong_df[case_cols].to_excel(writer, index=False, sheet_name="Both_Wrong")

    print("[SAVE] Excel:", out_xlsx)

    # Plots
    save_case_type_plot(
        flip_summary_df,
        os.path.join(args.out_dir, "flip_case_counts.png")
    )

    save_bar_plot(
        category_lift_df,
        x_col="category",
        y_col="helped_minus_hurt",
        title="C_v_st Category Lift: Helped minus Hurt Cases",
        out_png=os.path.join(args.out_dir, "category_helped_minus_hurt.png"),
        rotation=40,
    )

    save_bar_plot(
        ablation_gain_df,
        x_col="comparison",
        y_col="macro_f1_gain_A_minus_B",
        title="Ablation Gain in Macro F1",
        out_png=os.path.join(args.out_dir, "ablation_macro_f1_gain.png"),
        rotation=40,
    )

    if not emoji_summary_df.empty:
        save_bar_plot(
            emoji_summary_df.head(20),
            x_col="first_emoji",
            y_col="cvst_minus_text_accuracy",
            title="Per-Emoji Accuracy Gain: C_v_st minus Text-only",
            out_png=os.path.join(args.out_dir, "emoji_cvst_gain.png"),
            rotation=0,
        )

    print("\n[DONE] Explainability analysis completed.")
    print("[OUTPUT FOLDER]", args.out_dir)
    print("\nKey files:")
    print(" - cvst_explainability_analysis.xlsx")
    print(" - cvst_helped_cases.csv")
    print(" - cvst_hurt_cases.csv")
    print(" - flip_case_counts.png")
    print(" - category_helped_minus_hurt.png")
    print(" - ablation_macro_f1_gain.png")


if __name__ == "__main__":
    main()