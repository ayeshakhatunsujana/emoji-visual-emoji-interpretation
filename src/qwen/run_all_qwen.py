# ============================================================
# src/run_all_qwen.py
# Generate C_v, C_v_st, and C_t for emoji images using Qwen
# ============================================================

import argparse
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .vlm_caption_qwen import QwenEmojiVisionDescriber
from .text_llm_describe_qwen import TextEmojiDescriberQwen, filename_to_unicode_info


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def list_images(emoji_dir: str):
    paths = []
    for p in Path(emoji_dir).iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(p)
    return sorted(paths, key=lambda x: x.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emoji_dir", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--vlm_model", type=str, default="qwen2.5vl:7b")
    parser.add_argument("--llm_model", type=str, default="qwen2.5:3b")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Skip files already present in out_csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    image_paths = list_images(args.emoji_dir)
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]

    done_files = set()
    rows = []
    if args.resume and os.path.exists(args.out_csv):
        old_df = pd.read_csv(args.out_csv)
        if "file_name" in old_df.columns:
            done_files = set(old_df["file_name"].astype(str).tolist())
            rows = old_df.to_dict("records")
            print(f"[RESUME] Found {len(done_files)} completed files in {args.out_csv}")

    vlm = QwenEmojiVisionDescriber(model_name=args.vlm_model)
    llm = TextEmojiDescriberQwen(model_name=args.llm_model)

    for img_path in tqdm(image_paths, desc="Generating Qwen emoji descriptions"):
        if img_path.name in done_files:
            continue

        info = filename_to_unicode_info(img_path.name)

        c_v = vlm.describe_plain_visual(str(img_path))
        c_v_st = vlm.describe_smart_visual(str(img_path))
        c_t = llm.describe_name(
            unicode_name=info["unicode_name"],
            emoji_char=info["char"],
            official_names=info["official_names"],
        )

        row = {
            "file_name": img_path.name,
            "code_hex": info["code_hex"],
            "unicode_char": info["char"],
            "unicode_name": info["unicode_name"],
            "official_names": info["official_names"],
            "C_v": c_v,
            "C_v_st": c_v_st,
            "C_t": c_t,
        }
        rows.append(row)

        # Save after every image so long runs are safe.
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)

    print(f"\n✅ Saved Qwen emoji descriptions to: {args.out_csv}")
    print(f"✅ Total rows: {len(rows)}")


if __name__ == "__main__":
    main()
