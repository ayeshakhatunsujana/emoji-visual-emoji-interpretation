# src/vlm_caption_gpt4v.py

import os
import base64
import mimetypes
from openai import OpenAI

# [API key removed] Set OPENAI_API_KEY via your shell or a local .env file.

class VLMCaptioner:
    """
    OpenAI vision captioner (GPT-4o / GPT-4o-mini).
    Returns a short caption string (no sentiment).
    """

    def __init__(self, model_name: str = "gpt-4o-mini", api_key: str | None = None):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Please export it in your terminal."
            )

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        print(f"[VLM] Using OpenAI vision model '{self.model_name}'.")

    @staticmethod
    def _to_data_url(img_path: str) -> str:
        mime, _ = mimetypes.guess_type(img_path)
        if mime is None:
            mime = "image/png"  # safe default for emoji pngs

        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        return f"data:{mime};base64,{b64}"

    def caption_image(self, img_path: str) -> str:
        data_url = self._to_data_url(img_path)

        prompt = (
    "You are an emoji captioning expert.\n"
    "Task: Describe the emoji image with detailed facial descriptions and how it is typically interpreted in communication.\n"
    "Include: face color/shape, eyes, eyebrows (if visible), mouth, any extra marks (tears/sweat/hearts), overall expression, and the feeling it conveys.\n"
    "Describe only what is visible, but also mention how people usually interpret this expression.\n\n"

    "Examples (follow this exact style):\n"

    "A yellow round face with closed crescent eyes.\n"
    "The mouth is wide open with a toothy grin.\n"
    "The cheeks look slightly raised and rounded.\n"
    "The expression conveys happiness and excitement.\n\n"

    "A yellow face with raised eyebrows and wide open eyes.\n"
    "The mouth is a small O-shape.\n"
    "The expression looks surprised and alert, often used when something unexpected happens.\n\n"

    "A yellow face with squeezed-shut eyes.\n"
    "The mouth is open in a laughing shape.\n"
    "Tears appear on both cheeks near the outer corners.\n"
    "The expression shows intense laughter and strong amusement.\n\n"

    "Now describe the emoji in the image."
  )


        resp = self.client.responses.create(
            model=self.model_name,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": data_url},
                    ],
                }
            ],
            max_output_tokens=60,
            temperature=0.2,
        )

        return (resp.output_text or "").strip()
