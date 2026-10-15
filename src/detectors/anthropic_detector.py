import base64
import os

import anthropic.resources  # noqa: F401
from anthropic import Anthropic
from pydantic import BaseModel

from .detector import Detector, image_media_type


class AnthropicDetector(Detector):
    name = "anthropic"

    default_model = "claude-haiku-5-5"
    default_thinking = "low"

    def __init__(self) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        super().__init__(Anthropic(api_key=api_key), has_api_key=bool(api_key))
        self._effort_support: dict[str, bool] = {}

    def _supports_effort(self) -> bool:
        if self.model not in self._effort_support:
            try:
                capabilities = self.client.models.retrieve(self.model).capabilities
                self._effort_support[self.model] = bool(capabilities.effort.supported)
            except Exception:
                return True
        return self._effort_support[self.model]

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
            [{"type": "code_execution_20260521", "name": "code_execution"}]
            if use_code_interpreter
            else []
        )

        effort = {"output_config": {"effort": self.thinking}} if self._supports_effort() else {}

        response = self.client.messages.parse(
            model=self.model,
            **effort,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": image_media_type(image),
                                "data": base64_image,
                            },
                        },
                        {
                            "type": "text",
                            "text": self.prompt,
                        },
                    ],
                }
            ],
            tools=tools,
            output_format=response_model,
            max_tokens=2048,
        )

        usage = getattr(response, "usage", None)
        self._record_usage(
            getattr(usage, "input_tokens", None),
            getattr(usage, "output_tokens", None),
            getattr(usage, "cache_read_input_tokens", None),
            getattr(usage, "cache_creation_input_tokens", None),
        )
        return response.parsed_output
