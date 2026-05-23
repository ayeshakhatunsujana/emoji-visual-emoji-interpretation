# ============================================================
# src/analyze_all_condition_words.py
#
# Post-hoc all-condition explainability analysis from an existing
# prediction CSV + result JSON.
#
# This script does NOT rerun LLaMA. It answers:
#  1. Which condition is correct/wrong for each row?
#  2. Which words in raw text, C_v, C_v_st, and C_t are associated
#     with correct predictions for each condition?
#  3. For each pair of conditions, which words/components appear
#     when condition A is correct but condition B is wrong?
#  4. Special focus: when C_v_st passes but others fail, which words
#     appear in raw text/C_v_st/C_t/C_v?
# ============================================================

import argparse
import json
import os
import re
from collections import Counter
from typing import Dict, List, Iterable, Tuple

import pandas as pd


EXPERIMENTS = [
    ("a_text_only", "pred_a_text_only", ["text_baseline"]),
    ("b_text_plus_emoji", "pred_b_text_plus_emoji", ["text_baseline"]),
    ("c_text_plus_emoji_Cv", "pred_c_text_plus_emoji_Cv", ["text_baseline", "C_v"]),
    ("d_text_plus_emoji_Cvst", "pred_d_text_plus_emoji_Cvst", ["text_baseline", "C_v_st"]),
    ("e_text_plus_emoji_Ct", "pred_e_text_plus_emoji_Ct", ["text_baseline", "C_t"]),
    ("f_text_plus_emoji_Cv_Ct", "pred_f_text_plus_emoji_Cv_Ct", ["text_baseline", "C_v", "C_t"]),
    ("g_text_plus_emoji_Cvst_Ct", "pred_g_text_plus_emoji_Cvst_Ct", ["text_baseline", "C_v_st", "C_t"]),
    ("h_text_plus_emoji_Cv_Cvst_Ct", "pred_h_text_plus_emoji_Cv_Cvst_Ct", ["text_baseline", "C_v", "C_v_st", "C_t"]),
]

ID_TO_LABELS = {
    "sentiment3": {0: "negative", 1: "neutral", 2: "positive"},
    "sentiment2": {0: "negative", 1: "positive"},
    "sarcasm2": {0: "not_sarcasm", 1: "sarcasm"},
}

STOPWORDS = set("""
a an the and or but if then else when while with without to of in on for from by as at into onto over under
is are was were be been being am do does did doing have has had having can could should would may might must will just
this that these those it its they them their he him his she her we us our you your i me my mine not no yes only one word phrase
answer classify classifier tweet twitter post x use using following available information emoji visual description evidence smart plain textual social meaning c_v c_v_st c_t
shows show shown conveys convey indicating indicates expression face facial image overall visible cue cues supported supports
""".split())


def norm_label_value(x):
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    # Numeric prediction/gold
    try:
        if re.fullmatch(r"-?\d+", s):
            return int(s)
    except Exception:
        pass
    if "not sarcasm" in s or "not_sarcasm" in s or "non sarcasm" in s or "non-sarcasm" in s:
        return 0
    if "sarcasm" in s or "sarcastic" in s:
        return 1
    if "negative" in s:
        return 0
    if "neutral" in s:
        return 1
    if "positive" in s:
        return 2
    return s


def tokenize(text: str) -> List[str]:
    if pd.isna(text):
        return []
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " user ", text)
    text = re.sub(r"#", " ", text)
    # Keep apostrophe words lightly normalized.
    text = re.sub(r"[^a-zA-Z0-9_\s']", " ", text)
    toks = []
    for t in text.split():
        t = t.strip("'_")
        if len(t) < 2:
            continue
        if t in STOPWORDS:
            continue
        toks.append(t)
    return toks


def doc_word_counter(texts: Iterable[str]) -> Counter:
    c = Counter()
    for txt in texts:
        c.update(set(tokenize(txt)))
    return c


def word_lift(pos_texts: Iterable[str], neg_texts: Iterable[str], top_k: int, min_pos_count: int = 1) -> pd.DataFrame:
    pos_texts = list(pos_texts)
    neg_texts = list(neg_texts)
    pos_n = max(len(pos_texts), 1)
    neg_n = max(len(neg_texts), 1)
    pos = doc_word_counter(pos_texts)
    neg = doc_word_counter(neg_texts)
    vocab = set(pos) | set(neg)
    rows = []
    for w in vocab:
        if pos[w] < min_pos_count:
            continue
        pos_rate = pos[w] / pos_n
        neg_rate = neg[w] / neg_n
        rows.append({
            "word": w,
            "positive_case_doc_count": pos[w],
            "negative_case_doc_count": neg[w],
            "positive_case_doc_rate": pos_rate,
            "negative_case_doc_rate": neg_rate,
            "rate_difference": pos_rate - neg_rate,
            "rate_ratio_smoothed": (pos[w] + 1) / (pos_n + 2) / ((neg[w] + 1) / (neg_n + 2)),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["rate_difference", "positive_case_doc_count"], ascending=[False, False]).head(top_k)


def concat_components(row: pd.Series, components: List[str]) -> str:
    return "\n".join(str(row.get(c, "")) for c in components if c in row.index and not pd.isna(row.get(c, "")))


def make_correctness_columns(df: pd.DataFrame, task_type: str) -> pd.DataFrame:
    out = df.copy()
    out["gold_norm"] = out["label"].apply(norm_label_value)
    id_to = ID_TO_LABELS.get(task_type, {})
    out["gold_label_name"] = out["gold_norm"].apply(lambda x: id_to.get(x, str(x)))
    for exp_name, pred_col, _ in EXPERIMENTS:
        if pred_col not in out.columns:
            raise ValueError(f"Missing prediction column: {pred_col}")
        out[f"{exp_name}_pred_norm"] = out[pred_col].apply(norm_label_value)
        out[f"{exp_name}_correct"] = out[f"{exp_name}_pred_norm"] == out["gold_norm"]
        out[f"{exp_name}_pred_label_name"] = out[f"{exp_name}_pred_norm"].apply(lambda x: id_to.get(x, str(x)))
    return out


def condition_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for exp_name, pred_col, components in EXPERIMENTS:
        corr_col = f"{exp_name}_correct"
        rows.append({
            "experiment_name": exp_name,
            "prediction_column": pred_col,
            "components_used_for_word_analysis": "+".join(components),
            "correct_count": int(df[corr_col].sum()),
            "wrong_count": int((~df[corr_col]).sum()),
            "accuracy_from_prediction_file": float(df[corr_col].mean()),
        })
    return pd.DataFrame(rows)


def pairwise_switch_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for exp_a, pred_a, comp_a in EXPERIMENTS:
        for exp_b, pred_b, comp_b in EXPERIMENTS:
            if exp_a == exp_b:
                continue
            a_corr = df[f"{exp_a}_correct"]
            b_corr = df[f"{exp_b}_correct"]
            rows.append({
                "condition_A": exp_a,
                "condition_B": exp_b,
                "A_correct_B_wrong": int((a_corr & ~b_corr).sum()),
                "B_correct_A_wrong": int((b_corr & ~a_corr).sum()),
                "both_correct": int((a_corr & b_corr).sum()),
                "both_wrong": int((~a_corr & ~b_corr).sum()),
                "prediction_changed": int((df[pred_a].astype(str) != df[pred_b].astype(str)).sum()),
                "total_rows": int(len(df)),
            })
    return pd.DataFrame(rows)


def condition_component_word_importance(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    all_rows = []
    for exp_name, pred_col, components in EXPERIMENTS:
        corr = df[f"{exp_name}_correct"]
        for comp in components:
            if comp not in df.columns:
                continue
            lift = word_lift(df.loc[corr, comp].fillna(""), df.loc[~corr, comp].fillna(""), top_k=top_k)
            if lift.empty:
                continue
            lift.insert(0, "component", comp)
            lift.insert(0, "experiment_name", exp_name)
            lift.insert(0, "analysis", "correct_vs_wrong_within_condition")
            all_rows.append(lift)
    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def pairwise_help_word_importance(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Words in condition A's components when A correct and B wrong vs all other rows."""
    all_rows = []
    for exp_a, pred_a, comp_a in EXPERIMENTS:
        a_corr = df[f"{exp_a}_correct"]
        for exp_b, pred_b, comp_b in EXPERIMENTS:
            if exp_a == exp_b:
                continue
            b_corr = df[f"{exp_b}_correct"]
            pos_mask = a_corr & ~b_corr
            neg_mask = ~pos_mask
            if pos_mask.sum() == 0:
                continue
            for comp in comp_a:
                if comp not in df.columns:
                    continue
                lift = word_lift(df.loc[pos_mask, comp].fillna(""), df.loc[neg_mask, comp].fillna(""), top_k=top_k)
                if lift.empty:
                    continue
                lift.insert(0, "component", comp)
                lift.insert(0, "condition_B_wrong", exp_b)
                lift.insert(0, "condition_A_correct", exp_a)
                lift.insert(0, "analysis", "A_correct_B_wrong_vs_all_other")
                all_rows.append(lift)
    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def cvst_special_focus(df: pd.DataFrame, top_k: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cvst_exp = "d_text_plus_emoji_Cvst"
    cvst_corr = df[f"{cvst_exp}_correct"]
    summary_rows = []
    word_rows = []
    for exp_b, pred_b, comp_b in EXPERIMENTS:
        if exp_b == cvst_exp:
            continue
        b_corr = df[f"{exp_b}_correct"]
        pos_mask = cvst_corr & ~b_corr
        neg_mask = ~pos_mask
        summary_rows.append({
            "cvst_correct_other_wrong_case": f"{cvst_exp}_correct_{exp_b}_wrong",
            "other_condition": exp_b,
            "count": int(pos_mask.sum()),
            "total_rows": int(len(df)),
            "percentage": float(pos_mask.mean()),
        })
        for comp in ["text_baseline", "C_v_st", "C_v", "C_t"]:
            if comp not in df.columns:
                continue
            lift = word_lift(df.loc[pos_mask, comp].fillna(""), df.loc[neg_mask, comp].fillna(""), top_k=top_k)
            if lift.empty:
                continue
            lift.insert(0, "component", comp)
            lift.insert(0, "other_condition_wrong", exp_b)
            lift.insert(0, "analysis", "C_vst_correct_other_wrong")
            word_rows.append(lift)
    return pd.DataFrame(summary_rows), (pd.concat(word_rows, ignore_index=True) if word_rows else pd.DataFrame())


def load_performance_json(results_json: str) -> pd.DataFrame:
    if not results_json:
        return pd.DataFrame()
    with open(results_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame([{
        "experiment_name": r.get("experiment_name"),
        "accuracy": r.get("accuracy"),
        "macro_f1": r.get("macro_f1"),
        "weighted_f1": r.get("weighted_f1"),
        "num_eval_examples": r.get("num_eval_examples"),
        "num_processed_rows": r.get("num_processed_rows"),
    } for r in data])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", required=True)
    parser.add_argument("--task_type", required=True, choices=["sentiment3", "sentiment2", "sarcasm2"])
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--results_json", default=None)
    parser.add_argument("--skip_missing_descriptions", action="store_true")
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--min_word_count", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.pred_csv)
    if args.skip_missing_descriptions and "flag_missing_any_description" in df.columns:
        df = df[df["flag_missing_any_description"].fillna(0).astype(int) == 0].copy()
    df = make_correctness_columns(df, args.task_type)

    cond_sum = condition_summary(df)
    pair_sum = pairwise_switch_summary(df)
    cond_words = condition_component_word_importance(df, args.top_k)
    pair_words = pairwise_help_word_importance(df, args.top_k)
    cvst_sum, cvst_words = cvst_special_focus(df, args.top_k)
    perf = load_performance_json(args.results_json) if args.results_json else pd.DataFrame()

    df.to_csv(os.path.join(args.out_dir, "all_rows_all_condition_correctness.csv"), index=False)
    cond_sum.to_csv(os.path.join(args.out_dir, "all_condition_accuracy_summary.csv"), index=False)
    pair_sum.to_csv(os.path.join(args.out_dir, "all_condition_pairwise_switch_summary.csv"), index=False)
    cond_words.to_csv(os.path.join(args.out_dir, "word_importance_by_condition_component.csv"), index=False)
    pair_words.to_csv(os.path.join(args.out_dir, "pairwise_help_word_importance_all_conditions.csv"), index=False)
    cvst_sum.to_csv(os.path.join(args.out_dir, "cvst_correct_other_wrong_summary.csv"), index=False)
    cvst_words.to_csv(os.path.join(args.out_dir, "cvst_correct_other_wrong_word_importance.csv"), index=False)

    xlsx = os.path.join(args.out_dir, "all_condition_word_explainability.xlsx")
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        if not perf.empty:
            perf.to_excel(writer, index=False, sheet_name="Performance")
        cond_sum.to_excel(writer, index=False, sheet_name="Condition_Accuracy")
        pair_sum.to_excel(writer, index=False, sheet_name="Pairwise_Switch")
        cond_words.to_excel(writer, index=False, sheet_name="Words_By_Condition")
        pair_words.head(500000).to_excel(writer, index=False, sheet_name="Pairwise_Help_Words")
        cvst_sum.to_excel(writer, index=False, sheet_name="Cvst_Help_Summary")
        cvst_words.to_excel(writer, index=False, sheet_name="Cvst_Help_Words")
    print("[DONE] All-condition word explainability saved to:", args.out_dir)
    print("[XLSX]", xlsx)


if __name__ == "__main__":
    main()
