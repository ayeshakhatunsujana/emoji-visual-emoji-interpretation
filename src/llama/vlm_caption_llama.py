# ============================================================
# src/vlm_caption_llama.py
# Local Llama Vision captioner using Ollama
# Generates:
#   C_v    = plain visual description
#   C_v_st = smart structured visual emotion description
# ============================================================

from pathlib import Path
import ollama


class VLMCaptioner:
    def __init__(self, model_name: str = "llama3.2-vision"):
        self.model_name = model_name
        print(f"[VLM] Using local Ollama vision model: {self.model_name}")

    def caption_image(self, image_path: str, prompt: str) -> str:
        image_path = str(Path(image_path))

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
                options={
                    "temperature": 0.2,
                    "num_predict": 80,
                },
            )

            return response["message"]["content"].strip()

        except Exception as e:
            print(f"[VLM ERROR] {image_path}: {e}")
            return ""