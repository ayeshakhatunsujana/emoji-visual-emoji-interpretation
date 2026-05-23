# src/run_all.py

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .vlm_caption_gpt4v import VLMCaptioner
from .text_llm_describe import TextEmojiDescriber, filename_to_unicode_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emoji_dir", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--vlm_model", required=True)  # now this will be a LLaVA HF model id
    parser.add_argument("--llm_model", required=True)  # OpenAI model for C_t

    args = parser.parse_args()

    emoji_dir = Path(args.emoji_dir)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading models...")
    vlm = VLMCaptioner(model_name=args.vlm_model)
    llm = TextEmojiDescriber(model_name=args.llm_model)

    image_files = sorted(
        [p for p in emoji_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]]
    )
    print(f"[INFO] Found {len(image_files)} emoji images in {emoji_dir}")

    rows = []
    for img_path in tqdm(image_files, desc="Running VLM + LLM (C_v + C_t)"):
        info = filename_to_unicode_info(img_path.name)

        # C_v from LLaVA (local)
        try:
            c_v = vlm.caption_image(str(img_path))
        except Exception as e:
            print(f"[VLM] Error on {img_path.name}: {e}")
            c_v = ""

        # C_t from OpenAI text model
        try:
            c_t = llm.describe_name(info["unicode_name"])
        except Exception as e:
            print(f"[LLM] Error on {img_path.name}: {e}")
            c_t = ""

        rows.append(
            {
                "file_name": img_path.name,
                "code_hex": info["code_hex"],
                "unicode_char": info["char"],
                "unicode_name": info["unicode_name"],
                "C_v": c_v,
                "C_t": c_t,
            }
        )

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[ALL] Saved combined results to {out_csv}")


if __name__ == "__main__":
    main()
