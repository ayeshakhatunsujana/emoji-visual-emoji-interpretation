import argparse
import json
import pandas as pd
from sklearn.metrics import f1_score
from openai import OpenAI

from .data_utils import load_dataset


# =========================
# OPENAI CLIENT
# =========================
def get_client():
    return OpenAI()


# =========================
# PARSE GPT OUTPUT
# =========================
def parse_label(text):
    text = str(text).lower().strip()

    if "negative" in text:
        return 0
    elif "neutral" in text:
        return 1
    elif "positive" in text:
        return 2
    else:
        return 1


# =========================
# GPT PREDICTION
# =========================
def gpt_predict(client, model, text):

    prompt = (
        "You are a sentiment classifier.\n"
        f'Text: "{text}"\n'
        "Answer ONLY one word: Positive, Neutral, or Negative."
    )

    try:
        resp = client.responses.create(
            model=model,
            input=prompt,
            max_output_tokens=20
        )
        return resp.output[0].content[0].text.strip()

    except Exception as e:
        print("⚠️ GPT Error:", e)
        return "Neutral"


# =========================
# RUN EXPERIMENT
# =========================
def run_experiment(df, client, model, exp_name):

    y_true, y_pred = [], []

    for _, row in df.iterrows():

        text = str(row["text_baseline"])
        emoji = str(row.get("first_emoji", "") or "")
        cv = str(row.get("C_v", "") or "")
        cv_st = str(row.get("C_v_st", "") or "")
        ct = str(row.get("C_t", "") or "")

        # =========================
        # BUILD INPUT
        # =========================
        if exp_name == "raw_text":
            input_text = text

        elif exp_name == "raw_text_plus_emoji":
            input_text = f"{text} {emoji}"

        elif exp_name == "raw_text_plus_smart_visual":
            input_text = f"{text} {cv_st}"

        elif exp_name == "raw_text_plus_emoji_plus_smart_visual":
            input_text = f"{text} {emoji} {cv_st}"

        elif exp_name == "raw_text_plus_plain_visual":
            input_text = f"{text} {cv}"

        elif exp_name == "raw_text_plus_Ct":
            input_text = f"{text} {ct}"

        elif exp_name == "raw_text_plus_emoji_smartCv_Ct":
            input_text = f"{text} {emoji} {cv_st} {ct}"

        else:
            raise ValueError(f"Unknown experiment: {exp_name}")

        pred = parse_label(gpt_predict(client, model, input_text))

        y_true.append(int(row["label"]))
        y_pred.append(pred)

    f1 = f1_score(y_true, y_pred, average="macro")

    print(f"✅ {exp_name} → F1: {f1:.4f}")

    return f1


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--emoji_csv", required=True)
    parser.add_argument("--model_name", default="gpt-4o-mini")

    args = parser.parse_args()

    print("[INFO] Loading dataset...")
    df = load_dataset(args.train_csv, args.emoji_csv)
    print("Columns:", df.columns.tolist())
    print(f"[INFO] Total samples: {len(df)}")

    print("[INFO] Loading dataset...")
    df = load_dataset(args.train_csv, args.emoji_csv)

    # =========================
    # 🔥 CORRECT MERGE USING EMOJI
    # =========================
    emoji_df = pd.read_csv(args.emoji_csv)
    emoji_df.columns = emoji_df.columns.str.strip()

    print("[DEBUG] emoji_df columns:", emoji_df.columns.tolist())
    print("[DEBUG] df columns:", df.columns.tolist())

    df["first_emoji"] = df["first_emoji"].astype(str)
    emoji_df["unicode_char"] = emoji_df["unicode_char"].astype(str)

    df = df.merge(
        emoji_df[["unicode_char", "C_v_st"]],
        left_on="first_emoji",
        right_on="unicode_char",
        how="left"
    )

    print("✅ Merge using emoji successful")

    df.drop(columns=["unicode_char"], inplace=True, errors="ignore")
    df["C_v_st"] = df["C_v_st"].fillna("")

    # =========================
    # FILTER VALID ROWS
    # =========================
    df_valid = df[
        (df["C_v"].notna()) & (df["C_v"].str.strip() != "") &
        (df["C_v_st"].notna()) & (df["C_v_st"].str.strip() != "")
    ].copy()

    print(f"✅ Valid samples: {len(df_valid)}")

    client = get_client()

    # =========================
    # 🔥 FULL EXPERIMENTS (UPDATED)
    # =========================
    experiments = [
        "raw_text",
        "raw_text_plus_emoji",
        "raw_text_plus_smart_visual",
        "raw_text_plus_emoji_plus_smart_visual",
        "raw_text_plus_plain_visual",
        "raw_text_plus_Ct",
        "raw_text_plus_emoji_smartCv_Ct"
    ]

    results = {}

    for exp in experiments:
        print(f"\n🚀 Running: {exp}")
        results[exp] = run_experiment(df_valid, client, args.model_name, exp)

    print("\n📊 FINAL RESULTS:")
    print(results)
    # =========================
# 🔥 DETAILED ROW-LEVEL OUTPUT
# =========================
    print("\n📊 Generating detailed prediction file...")

    rows = []

    for _, row in df_valid.iterrows():

        text = str(row["text_baseline"])
        emoji = str(row.get("first_emoji", "") or "")
        cv = str(row.get("C_v", "") or "")
        cv_st = str(row.get("C_v_st", "") or "")
        ct = str(row.get("C_t", "") or "")

        true_label = int(row["label"])

    # predictions for ALL experiments
        inputs = {
            "raw_text": text,
            "raw_text_plus_emoji": f"{text} {emoji}",
            "raw_text_plus_smart_visual": f"{text} {cv_st}",
            "raw_text_plus_emoji_plus_smart_visual": f"{text} {emoji} {cv_st}",
            "raw_text_plus_plain_visual": f"{text} {cv}",
            "raw_text_plus_Ct": f"{text} {ct}",
            "raw_text_plus_emoji_smartCv_Ct": f"{text} {emoji} {cv_st} {ct}"
        }

        preds = {}

        for k, v in inputs.items():
            pred = parse_label(gpt_predict(client, args.model_name, v))
            preds[k] = pred

        rows.append({
            "text": text,
            "emoji": emoji,
            "C_v": cv,
            "C_v_st": cv_st,
            "C_t": ct,
            "true_label": true_label,
            **preds
        })

    df_detail = pd.DataFrame(rows)

    df_detail.to_csv("outputs/detailed_predictions.csv", index=False)

    print("✅ Saved: outputs/detailed_predictions.csv")

    # =========================
    # SAVE RESULTS
    # =========================
    with open("outputs/final_results_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print("✅ Saved: outputs/final_results_summary.json")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    main()