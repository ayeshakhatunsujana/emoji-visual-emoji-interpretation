# ============================================================
# src/run_all_llama.py
# Local Llama pipeline using Ollama
#
# Runs:
#   C_v      = plain visual description from Llama vision
#   C_v_st   = smart structured visual emotion description
#   C_t      = text description from Llama text model
#
# Output:
#   file_name, code_hex, unicode_char, unicode_name, C_v, C_v_st, C_t
# ============================================================

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .vlm_caption_llama import VLMCaptioner
from .text_llm_describe_llama import TextEmojiDescriber, filename_to_unicode_info


# ============================================================
# PROMPTS
# ============================================================

PLAIN_PROMPT = """
You are an emoji visual captioning expert.

Task:
Describe only the visible physical appearance of the emoji image.

Include:
- face color and shape
- eyes
- eyebrows if visible
- mouth
- cheeks
- tears, sweat, hearts, hands, or other visible marks
- overall facial expression or visible gesture

Important rules:
- Focus mainly on physical appearance.
- Do not include the emoji character in the answer.
- Do not start with the emoji symbol.
- Do not over-explain social meaning.
- Keep it short, 1-2 sentences.

Examples:

A yellow round face with closed crescent eyes.
The mouth is wide open with a toothy grin.
The cheeks look slightly raised and rounded.
The expression conveys happiness and excitement.

A yellow face with squeezed-shut eyes.
The mouth is open in a laughing shape.
Tears appear on both cheeks near the outer corners.
The expression shows intense laughter.

A yellow face with a small frown and one tear.
The eyes look sad and the mouth is downturned.
The visible tear suggests sadness or emotional pain.

Now describe the emoji image.
""".strip()


SMART_PROMPT = """
You are an expert in interpreting emotion and communicative meaning from emoji images.

Task:
Analyze the visible cue in the emoji image.

If the emoji has a face:
- describe the facial expression and emotional evidence.

If the emoji is a hand, person, object, or symbol:
- describe the visible gesture, object, person, or symbol.
- do not call it a facial expression.

Focus on:
- visible cue
- emotion or communicative meaning
- whether the meaning is positive, negative, neutral, strong, or mild

Important rules:
- Do not include the emoji character in the answer.
- Do not start with the emoji symbol.
- Do not say "facial expression" for hand emojis, people emojis, objects, or symbols.
- For pointing hands, describe direction and attention function only.
- Only thumbs-up means approval or agreement.
- Only thumbs-down means disapproval or rejection.
- Keep it short, 1-2 sentences.

Examples:
The facial expression shows extreme amusement or laughter, supported by squeezed eyes, an open mouth, and tears.
The facial expression shows strong sadness, supported by a tear and downturned expression.
The visual gesture is a hand pointing downward. It directs attention to something below and has a neutral meaning.
The visual gesture is a thumbs-up hand sign. It conveys approval or agreement.
The visual gesture is a thumbs-down hand sign. It conveys disapproval or rejection.

Now describe the visible emotional or communicative evidence in the given emoji image.
""".strip()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--emoji_dir",
        required=True,
        help="Folder containing emoji images, e.g., emoji"
    )

    parser.add_argument(
        "--out_csv",
        required=True,
        help="Output CSV path, e.g., outputs/results_llama.csv"
    )

    parser.add_argument(
        "--vlm_model",
        default="llama3.2-vision",
        help="Ollama vision model name"
    )

    parser.add_argument(
        "--llm_model",
        default="llama3.2:3b",
        help="Ollama text model name"
    )

    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Optional: process only first N images for testing"
    )

    args = parser.parse_args()

    emoji_dir = Path(args.emoji_dir)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not emoji_dir.exists():
        raise FileNotFoundError(f"Emoji folder not found: {emoji_dir}")

    print("[INFO] Loading local models...")
    vlm = VLMCaptioner(model_name=args.vlm_model)
    llm = TextEmojiDescriber(model_name=args.llm_model)

    image_files = sorted(
        [
            p for p in emoji_dir.iterdir()
            if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]
        ]
    )

    if args.max_images is not None:
        image_files = image_files[:args.max_images]

    print(f"[INFO] Found {len(image_files)} emoji images in {emoji_dir}")

    rows = []

    for img_path in tqdm(image_files, desc="Running local Llama VLM + LLM"):

        info = filename_to_unicode_info(img_path.name)

        # ====================================================
        # 1. C_v: plain visual description
        # ====================================================
        try:
            c_v = vlm.caption_image(
                image_path=str(img_path),
                prompt=PLAIN_PROMPT
            )
        except Exception as e:
            print(f"[C_v ERROR] {img_path.name}: {e}")
            c_v = ""

        # ====================================================
        # 2. C_v_st: smart structured visual emotion description
        # ====================================================
        try:
            c_v_st = vlm.caption_image(
                image_path=str(img_path),
                prompt=SMART_PROMPT
            )
        except Exception as e:
            print(f"[C_v_st ERROR] {img_path.name}: {e}")
            c_v_st = ""

        # ====================================================
        # 3. C_t: text-only emoji meaning from Unicode code
        # ====================================================
        try:
            c_t = llm.describe_name(
                unicode_name=info["unicode_name"],
                emoji_char=info["char"],
                official_names=info["official_names"]
        )
        except Exception as e:
            print(f"[C_t ERROR] {img_path.name}: {e}")
            c_t = ""

        rows.append(
            {
                "file_name": img_path.name,
                "code_hex": info["code_hex"],
                "unicode_char": info["char"],
                "unicode_name": info["unicode_name"],
                "official_names": info["official_names"],
                "C_v": c_v,
                "C_v_st": c_v_st,
                "C_t": c_t,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    print(f"\n✅ Saved combined results to: {out_csv}")
    print(f"✅ Total rows: {len(df)}")


if __name__ == "__main__":
    main()