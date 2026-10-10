import base64
import os

from openrouter import OpenRouter
from pydantic import BaseModel

from .detector import Detector, image_media_type


class OpenRouterDetector(Detector):
    name = "openrouter"

    default_model = "deepseek/deepseek-v4.1-flash"
    default_thinking = "minimal"

    def __init__(self) -> None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        super().__init__(OpenRouter(api_key=api_key), has_api_key=bool(api_key))

    def _list_model_ids(self) -> list[str]:
        return [model.id for model in self.client.models.list().result.data]

    def detect_image(
        self,
        image,
        include_place: bool = False,
        min_characters: int = 4,
        max_characters: int = 4,
        use_code_interpreter: bool = False,
    ) -> BaseModel:
        base64_image = base64.b64encode(image).decode("utf-8")
        response_model = self._create_pydantic_model(
            include_place, min_characters, max_characters
        )

        tools = (
            [
                {
                    "type": "openrouter:shell",
                    "parameters": {
                        "engine": "openrouter",
                    },
                }
            ]
            if use_code_interpreter
            else []
        )

        response = self.client.beta.responses.send(
            self.model,
            reasoning={
                "effort": self.thinking,
            },
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": self.prompt,
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:{image_media_type(image)};base64,{base64_image}",
                        },
                    ],
                }
            ],
            tools=tools,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "ocr_result",
                    "strict": True,
                    "schema": response_model.model_json_schema(),
                },
            },
            provider={
                "require_parameters": True,
                "zdr": True,
            },
            stream=False,
        )

        usage = getattr(response, "usage", None)
        details = getattr(usage, "input_tokens_details", None)
        self._record_usage(
            getattr(usage, "input_tokens", None),
            getattr(usage, "output_tokens", None),
            getattr(details, "cached_tokens", None),
            getattr(details, "cache_write_tokens", None),
        )
        return response_model.model_validate_json(response.output_text)
