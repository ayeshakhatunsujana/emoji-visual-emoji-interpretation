# src/text_llm_describe.py

import os
from openai import OpenAI

# [API key removed] Set OPENAI_API_KEY via your shell or a local .env file.


def filename_to_unicode_info(filename: str) -> dict:
    code_hex = filename.split(".")[0]
    try:
        char = chr(int(code_hex, 16))
        name = f"U+{code_hex}"
    except Exception:
        char = ""
        name = ""
    return {"code_hex": code_hex, "char": char, "unicode_name": name}


class TextEmojiDescriber:

    def __init__(self, model_name: str = "gpt-4o-mini", api_key: str | None = None):
        self.model_name = model_name

        # 🔑 Read API key
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OPENAI_API_KEY is not set.\n"
                "Run in terminal:\n"
                "export OPENAI_API_KEY='YOUR_KEY'"
            )

        self.client = OpenAI(api_key=key)

    def describe_name(self, unicode_name: str) -> str:

        prompt = (
    "You are an expert in how emojis are interpreted in real communication.\n"
    "Task: Describe the emoji naturally, as people understand and use it.\n\n"

    "Include:\n"
    "- facial appearance (eyes, mouth, expression)\n"
    "- emotional meaning\n"
    "- typical situations where it is used\n"
    "- intensity of the feeling\n\n"

    "Keep it short (2–3 sentences) and natural.\n\n"

    "Examples:\n"

    "Emoji code: U+1F600\n"
    "Description: A yellow face with open eyes and a broad smile. "
    "It conveys happiness and friendliness, often used in casual conversations to show joy. "
    "The feeling is light and positive.\n\n"

    "Emoji code: U+1F62D\n"
    "Description: A face with tightly shut eyes and tears streaming down the cheeks. "
    "It expresses intense sadness or emotional overwhelm, often used in distressing situations. "
    "The feeling is strong and emotional.\n\n"

    "Emoji code: U+1F914\n"
    "Description: A face with one eyebrow raised and a hand on the chin. "
    "It suggests thinking or confusion, often used when questioning something. "
    "The feeling is mild and reflective.\n\n"

    f"Emoji code: {unicode_name}\n"
    "Description:"
       )

        try:
            resp = self.client.responses.create(
                model=self.model_name,
                input=prompt,
                max_output_tokens=60,
                temperature=0.3,
            )

            # ✅ SAFE extraction
            if hasattr(resp, "output_text") and resp.output_text:
                return resp.output_text.strip()

            # fallback (important)
            return resp.output[0].content[0].text.strip()

        except Exception as e:
            print(f"[ERROR] {e}")
            return ""   # ✅ return empty instead of string error
