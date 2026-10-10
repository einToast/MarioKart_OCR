import base64
import os

import openai.resources.responses  # noqa: F401
from openai import OpenAI
from pydantic import BaseModel

from .detector import Detector, image_media_type


class OpenAIDetector(Detector):
    name = "openai"

    default_model = "gpt-6.1-sol"
    default_thinking = "low"

    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        super().__init__(OpenAI(api_key=api_key), has_api_key=bool(api_key))
    
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
            [{"type": "code_interpreter", "container": {"type": "auto"}}]
            if use_code_interpreter
            else []
        )

        response = self.client.responses.parse(
            model=self.model,
            reasoning={"effort": self.thinking},
            instructions=self.prompt,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": f"data:{image_media_type(image)};base64,{base64_image}",
                            "detail": "auto",
                        },
                    ],
                }
            ],
            tools=tools,
            text_format=response_model,
        )
        usage = getattr(response, "usage", None)
        details = getattr(usage, "input_tokens_details", None)
        self._record_usage(
            getattr(usage, "input_tokens", None),
            getattr(usage, "output_tokens", None),
            getattr(details, "cached_tokens", None),
            getattr(details, "cache_write_tokens", None),
        )
        return response.output_parsed
