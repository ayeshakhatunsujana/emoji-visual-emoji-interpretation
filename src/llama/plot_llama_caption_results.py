# ============================================================
# src/plot_llama_caption_results.py
# Plot Macro F1 from llama caption experiment JSON
# ============================================================

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_json", required=True)
    parser.add_argument("--out_png", required=True)
    parser.add_argument("--out_xlsx", default=None)
    args = parser.parse_args()

    with open(args.results_json, "r", encoding="utf-8") as f:
        results = json.load(f)

    df = pd.DataFrame(
        [
            {
                "experiment_name": r["experiment_name"],
                "num_eval_examples": r["num_eval_examples"],
                "accuracy": r["accuracy"],
                "macro_f1": r["macro_f1"],
                "weighted_f1": r["weighted_f1"],
            }
            for r in results
        ]
    )

    names = df["experiment_name"].tolist()
    scores = df["macro_f1"].tolist()

    x = np.arange(len(names))

    plt.figure(figsize=(13, 6))
    plt.bar(x, scores)
    plt.xticks(x, names, rotation=35, ha="right")
    plt.ylabel("Macro F1")
    plt.ylim(0, 1.0)
    plt.title("Sentiment Experiments Using Llama-Generated Emoji Captions")
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=300)

    print(f"[PLOT] Saved plot to: {args.out_png}")

    if args.out_xlsx:
        df.to_excel(args.out_xlsx, index=False)
        print(f"[PLOT] Saved summary Excel to: {args.out_xlsx}")


if __name__ == "__main__":
    main()