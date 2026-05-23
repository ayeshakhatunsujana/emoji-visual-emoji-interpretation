# ============================================================
# src/text_llm_describe_qwen.py
# Local Qwen text model using Ollama
# Purpose:
#   Generate C_t: textual/social emoji meaning
# ============================================================

import unicodedata
import ollama


def filename_to_unicode_info(filename: str) -> dict:
    """
    Convert emoji image filename into useful Unicode information.

    Examples:
    1F44D.png -> 👍
    1F469_200D_2764_FE0F_200D_1F468.png -> 👩‍❤️‍👨
    """
    code_hex = filename.rsplit(".", 1)[0]
    parts = code_hex.split("_")

    chars = []
    names = []

    for part in parts:
        try:
            ch = chr(int(part, 16))
            chars.append(ch)
            try:
                names.append(unicodedata.name(ch))
            except ValueError:
                names.append(f"UNKNOWN NAME FOR U+{part}")
        except Exception:
            pass

    emoji_char = "".join(chars)
    unicode_name = " + ".join([f"U+{part}" for part in parts])
    official_names = " + ".join(names)

    return {
        "code_hex": code_hex,
        "char": emoji_char,
        "unicode_name": unicode_name,
        "official_names": official_names,
    }


class TextEmojiDescriberQwen:
    def __init__(self, model_name: str = "qwen2.5:3b"):
        self.model_name = model_name
        print(f"[LLM] Using local Ollama Qwen text model: {self.model_name}")

    def describe_name(self, unicode_name: str, emoji_char: str = "", official_names: str = "") -> str:
        prompt = f"""
You are an expert in emoji meaning in digital communication.

Task:
Describe the social and textual meaning of this emoji.

Emoji character: {emoji_char}
Unicode code: {unicode_name}
Official Unicode name: {official_names}

Focus on:
- how people usually use this emoji in messages
- what emotion, reaction, or communication function it expresses
- common social meaning in text conversations

Important rules:
- Do not include the emoji character in the answer.
- Do not start with the emoji symbol.
- Do not describe a different emoji.
- Do not ask for another emoji.
- Do not focus only on physical appearance.
- For pointing hand emojis, describe direction and attention function only.
- For thumbs up, describe approval or agreement.
- For thumbs down, describe disapproval or rejection.
- For face emojis, describe emotional and communicative meaning.
- Keep it short, 1-2 sentences.

Answer:
""".strip()

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1, "num_predict": 90},
            )
            return response["message"]["content"].strip()
        except Exception as e:
            print(f"[LLM ERROR] {unicode_name}: {e}")
            return ""
