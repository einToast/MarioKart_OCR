import base64
import os

from mistralai.client import Mistral
from mistralai.client.models import CompletionArgs, JSONSchema, ResponseFormat
from pydantic import BaseModel

from .detector import Detector, image_media_type


def _answer_text(response) -> str:
    parts = []
    for entry in response.outputs:
        if getattr(entry, "type", None) != "message.output":
            continue
        if isinstance(entry.content, str):
            parts.append(entry.content)
            continue
        parts.extend(
            chunk.text for chunk in entry.content if getattr(chunk, "type", None) == "text"
        )
    return "".join(parts)


class MistralDetector(Detector):
    name = "mistral"

    default_model = "mistral-small-2603"
    default_thinking = "minimal"

    def __init__(self) -> None:
        api_key = os.getenv("MISTRAL_API_KEY")
        super().__init__(Mistral(api_key=api_key), has_api_key=bool(api_key))

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

        tools = [{"type": "code_interpreter"}] if use_code_interpreter else []

        response = self.client.beta.conversations.start(
            model=self.model,
            inputs=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image_url": f"data:{image_media_type(image)};base64,{base64_image}",
                        },
                        {
                            "type": "text",
                            "text": self.prompt,
                        },
                    ],
                }
            ],
            tools=tools,
            completion_args=CompletionArgs(
                reasoning_effort=self.thinking,
                response_format=ResponseFormat(
                    type="json_schema",
                    json_schema=JSONSchema(
                        name="ocr_result",
                        schema_definition=response_model.model_json_schema(),
                    ),
                    strict=True,
                ),
            ),
            store=False,
        )

        usage = getattr(response, "usage", None)
        self._record_usage(
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None),
        )
        return response_model.model_validate_json(_answer_text(response))
