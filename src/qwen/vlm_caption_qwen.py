# ============================================================
# src/vlm_caption_qwen.py
# Local Qwen vision-language model using Ollama
# Purpose:
#   Generate C_v and C_v_st from emoji image files
# ============================================================

import ollama


class QwenEmojiVisionDescriber:
    def __init__(self, model_name: str = "qwen2.5vl:7b"):
        self.model_name = model_name
        print(f"[VLM] Using local Ollama Qwen vision model: {self.model_name}")

    def _ask_image(self, image_path: str, prompt: str, num_predict: int = 120) -> str:
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [image_path],
                    }
                ],
                options={"temperature": 0.1, "num_predict": num_predict},
            )
            return response["message"]["content"].strip()
        except Exception as e:
            print(f"[VLM ERROR] {image_path}: {e}")
            return ""

    def describe_plain_visual(self, image_path: str) -> str:
        """Generate C_v: plain visual description only."""
        prompt = """
You are describing an emoji image for a research dataset.

Task:
Describe only the visible appearance of the emoji.

Focus on:
- face shape, eyes, mouth, eyebrows, tears, sweat, hands, hearts, symbols, objects, colors
- physical/visual details that are directly visible

Important rules:
- Do not explain social meaning.
- Do not infer the user's intention.
- Do not mention Unicode name.
- Do not include the emoji character.
- Keep the answer short, 1-2 sentences.

Answer:
""".strip()
        return self._ask_image(image_path, prompt, num_predict=120)

    def describe_smart_visual(self, image_path: str) -> str:
        """Generate C_v_st: smart visual-emotion/communication description."""
        prompt = """
You are describing an emoji image for sentiment-analysis research.

Task:
Interpret the visible emoji cues into emotion or communication meaning.

Focus on:
- what the facial expression or gesture visually suggests
- emotional intensity, such as mild, strong, high, or exaggerated
- likely communicative function, such as joy, sadness, anger, approval, disapproval, greeting, prayer, attention, sarcasm, or affection

Important rules:
- Base the interpretation only on the visible emoji image.
- Do not include the emoji character.
- Do not write a long explanation.
- Keep the answer short, 1-2 sentences.

Answer:
""".strip()
        return self._ask_image(image_path, prompt, num_predict=120)
