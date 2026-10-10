import base64
import os

from google import genai
from pydantic import BaseModel

from .detector import Detector, image_media_type


class GeminiDetector(Detector):
    name = "gemini"

    default_model = "gemini-3.8-flash"
    default_thinking = "minimal"

    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        super().__init__(genai.Client(api_key=api_key), has_api_key=bool(api_key))

    def _list_model_ids(self) -> list[str]:
        return [model.name.removeprefix("models/") for model in self.client.models.list()]

    def detect_image(
        self,
        image,
        include_place: bool = False,
        min_characters: int = 4,
        max_characters: int = 4,
        use_code_interpreter: bool = False,
    ) -> BaseModel:
        image_base64 = base64.b64encode(image).decode("utf-8")
        response_model = self._create_pydantic_model(
            include_place, min_characters, max_characters
        )

        tools = [{"type": "code_execution"}] if use_code_interpreter else []

        response = self.client.interactions.create(
            model=self.model,
            generation_config={
                "thinking_level": self.thinking,
            },
            input=[
                {
                    "type": "text",
                    "text": self.prompt,
                },
                {
                    "type": "image",
                    "data": image_base64,
                    "mime_type": image_media_type(image),
                },
            ],
            tools=tools,
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": response_model.model_json_schema(),
            },
            store=False,
        )

        # Gemini reports four separate counts. What it reads is billed as input:
        # the prompt and the tool results fed back in between code steps. What
        # it writes is billed as output: the answer and its thoughts.
        usage = getattr(response, "usage", None)

        def count(name: str) -> int:
            value = getattr(usage, name, None)
            return value if isinstance(value, int) else 0

        if isinstance(getattr(usage, "total_input_tokens", None), int):
            self._record_usage(
                count("total_input_tokens") + count("total_tool_use_tokens"),
                count("total_output_tokens") + count("total_thought_tokens"),
                getattr(usage, "total_cached_tokens", None),
            )
        return response_model.model_validate_json(response.output_text)
