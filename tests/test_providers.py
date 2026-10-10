"""Per-provider detectors, with each SDK client replaced by a mock (no network, no API keys)."""

import base64
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from detectors import (
    anthropic_detector,
    gemini_detector,
    mistral_detector,
    openai_detector,
    openrouter_detector,
)
from detectors.detector import image_media_type
from json_models import TokenUsage
from pydantic import BaseModel, ValidationError

IMAGE_BYTES = b"\xff\xd8\xff" + b"fake-jpeg-bytes"
IMAGE_B64 = base64.b64encode(IMAGE_BYTES).decode("utf-8")
DATA_URL = f"data:image/jpeg;base64,{IMAGE_B64}"
CHARACTERS_JSON = json.dumps(
    {"characters": [{"name": "Yoshi", "points": 15}, {"name": "Toad", "points": 12}]}
)


@pytest.fixture
def sdk_client(monkeypatch):
    """Patch the SDK client constructor of a detector module and return the mock client."""

    def patch(module, attribute):
        client = MagicMock()
        monkeypatch.setattr(module, attribute, MagicMock(return_value=client))
        return client

    return patch


def _schema_fields(model: type[BaseModel]) -> set[str]:
    return set(model.model_json_schema()["$defs"]["Character"]["properties"])


class TestAnthropicDetector:
    @pytest.fixture
    def client(self, sdk_client):
        return sdk_client(anthropic_detector, "Anthropic")

    def test_builds_request(self, client, image):
        detector = anthropic_detector.AnthropicDetector().configure(
            model="claude-test", thinking="high", prompt="Read it."
        )
        detector.detect_image(image, include_place=True, min_characters=2)

        kwargs = client.messages.parse.call_args.kwargs
        assert kwargs["model"] == "claude-test"
        assert kwargs["output_config"] == {"effort": "high"}
        assert kwargs["tools"] == []
        image_block, text_block = kwargs["messages"][0]["content"]
        assert image_block["source"] == {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": IMAGE_B64,
        }
        assert text_block == {"type": "text", "text": "Read it."}
        assert _schema_fields(kwargs["output_format"]) == {"name", "points", "place"}

    def test_code_interpreter_adds_code_execution_tool(self, client, image):
        anthropic_detector.AnthropicDetector().detect_image(image, use_code_interpreter=True)

        tools = client.messages.parse.call_args.kwargs["tools"]
        assert [tool["name"] for tool in tools] == ["code_execution"]

    def test_returns_parsed_output(self, client, image):
        parsed = object()
        client.messages.parse.return_value = SimpleNamespace(parsed_output=parsed)

        assert anthropic_detector.AnthropicDetector().detect_image(image) is parsed


class TestOpenAIDetector:
    @pytest.fixture
    def client(self, sdk_client):
        return sdk_client(openai_detector, "OpenAI")

    def test_builds_request(self, client, image):
        detector = openai_detector.OpenAIDetector().configure(
            model="gpt-test", thinking="medium", prompt="Read it."
        )
        detector.detect_image(image)

        kwargs = client.responses.parse.call_args.kwargs
        assert kwargs["model"] == "gpt-test"
        assert kwargs["reasoning"] == {"effort": "medium"}
        assert kwargs["instructions"] == "Read it."
        assert kwargs["tools"] == []
        assert kwargs["input"][0]["content"][0]["image_url"] == DATA_URL
        assert _schema_fields(kwargs["text_format"]) == {"name", "points"}

    def test_code_interpreter_adds_tool(self, client, image):
        openai_detector.OpenAIDetector().detect_image(image, use_code_interpreter=True)

        tools = client.responses.parse.call_args.kwargs["tools"]
        assert [tool["type"] for tool in tools] == ["code_interpreter"]

    def test_returns_output_parsed(self, client, image):
        parsed = object()
        client.responses.parse.return_value = SimpleNamespace(output_parsed=parsed)

        assert openai_detector.OpenAIDetector().detect_image(image) is parsed


class TestGeminiDetector:
    @pytest.fixture
    def client(self, monkeypatch):
        client = MagicMock()
        monkeypatch.setattr(
            gemini_detector, "genai", SimpleNamespace(Client=lambda **kwargs: client)
        )
        client.interactions.create.return_value = SimpleNamespace(
            output_text=CHARACTERS_JSON
        )
        return client

    def test_builds_request(self, client, image):
        detector = gemini_detector.GeminiDetector().configure(
            model="gemini-test", thinking="low", prompt="Read it."
        )
        detector.detect_image(image, min_characters=2, max_characters=2)

        kwargs = client.interactions.create.call_args.kwargs
        assert kwargs["model"] == "gemini-test"
        assert kwargs["generation_config"] == {"thinking_level": "low"}
        assert kwargs["tools"] == []
        text_part, image_part = kwargs["input"]
        assert text_part == {"type": "text", "text": "Read it."}
        assert image_part == {
            "type": "image",
            "data": IMAGE_B64,
            "mime_type": "image/jpeg",
        }
        schema = kwargs["response_format"]["schema"]
        assert schema["properties"]["characters"]["minItems"] == 2

    def test_code_interpreter_adds_tool(self, client, image):
        gemini_detector.GeminiDetector().detect_image(
            image, min_characters=2, use_code_interpreter=True
        )

        tools = client.interactions.create.call_args.kwargs["tools"]
        assert tools == [{"type": "code_execution"}]

    def test_parses_json_output(self, client, image):
        result = gemini_detector.GeminiDetector().detect_image(
            image, min_characters=2, max_characters=2
        )

        assert [(c.name, c.points) for c in result.characters] == [
            ("Yoshi", 15),
            ("Toad", 12),
        ]

    def test_output_violating_character_count_raises(self, client, image):
        with pytest.raises(ValidationError):
            gemini_detector.GeminiDetector().detect_image(image)  # expects 4, gets 2

    def test_malformed_json_raises(self, client, image):
        client.interactions.create.return_value = SimpleNamespace(output_text="nope")

        with pytest.raises(ValidationError):
            gemini_detector.GeminiDetector().detect_image(image)


class TestOpenRouterDetector:
    @pytest.fixture
    def client(self, sdk_client):
        client = sdk_client(openrouter_detector, "OpenRouter")
        client.beta.responses.send.return_value = SimpleNamespace(
            output_text=CHARACTERS_JSON
        )
        return client

    def test_builds_request(self, client, image):
        detector = openrouter_detector.OpenRouterDetector().configure(
            model="vendor/model", thinking="high", prompt="Read it."
        )
        detector.detect_image(image, min_characters=2, max_characters=2)

        call = client.beta.responses.send.call_args
        assert call.args == ("vendor/model",)
        kwargs = call.kwargs
        assert kwargs["reasoning"] == {"effort": "high"}
        assert kwargs["tools"] == []
        assert kwargs["stream"] is False
        assert kwargs["provider"] == {"require_parameters": True, "zdr": True}
        text_part, image_part = kwargs["input"][0]["content"]
        assert text_part == {"type": "input_text", "text": "Read it."}
        assert image_part == {"type": "input_image", "image_url": DATA_URL}
        text_format = kwargs["text"]["format"]
        assert text_format["type"] == "json_schema"
        assert text_format["strict"] is True
        assert text_format["schema"]["properties"]["characters"]["maxItems"] == 2

    def test_code_interpreter_adds_shell_tool(self, client, image):
        openrouter_detector.OpenRouterDetector().detect_image(
            image, min_characters=2, use_code_interpreter=True
        )

        tools = client.beta.responses.send.call_args.kwargs["tools"]
        assert [tool["type"] for tool in tools] == ["openrouter:shell"]

    def test_parses_json_output(self, client, image):
        result = openrouter_detector.OpenRouterDetector().detect_image(
            image, min_characters=2, max_characters=2
        )

        assert [c.name for c in result.characters] == ["Yoshi", "Toad"]

    def test_malformed_json_raises(self, client, image):
        client.beta.responses.send.return_value = SimpleNamespace(output_text="{}")

        with pytest.raises(ValidationError):
            openrouter_detector.OpenRouterDetector().detect_image(image)


class TestModelListing:
    def test_gemini_strips_models_prefix(self, monkeypatch):
        client = MagicMock()
        client.models.list.return_value = [SimpleNamespace(name="models/gemini-test")]
        monkeypatch.setattr(
            gemini_detector, "genai", SimpleNamespace(Client=lambda **kwargs: client)
        )
        monkeypatch.setenv("GEMINI_API_KEY", "key")

        assert gemini_detector.GeminiDetector().get_available_models() == ["gemini-test"]

    def test_openrouter_reads_result_data(self, sdk_client, monkeypatch):
        client = sdk_client(openrouter_detector, "OpenRouter")
        client.models.list.return_value = SimpleNamespace(
            result=SimpleNamespace(data=[SimpleNamespace(id="vendor/model")])
        )
        monkeypatch.setenv("OPENROUTER_API_KEY", "key")

        assert openrouter_detector.OpenRouterDetector().get_available_models() == [
            "vendor/model"
        ]


class TestUsageRecording:
    def test_openai_records_tokens(self, sdk_client, image):
        client = sdk_client(openai_detector, "OpenAI")
        client.responses.parse.return_value.usage = SimpleNamespace(
            input_tokens=900, output_tokens=120
        )
        detector = openai_detector.OpenAIDetector()
        detector.detect_image(image)

        assert detector.usage == TokenUsage(input_tokens=900, output_tokens=120, cache_read_tokens=0, cache_write_tokens=0)

    def test_anthropic_records_tokens(self, sdk_client, image):
        client = sdk_client(anthropic_detector, "Anthropic")
        client.messages.parse.return_value.usage = SimpleNamespace(
            input_tokens=700, output_tokens=80
        )
        detector = anthropic_detector.AnthropicDetector()
        detector.detect_image(image)

        assert detector.usage == TokenUsage(input_tokens=700, output_tokens=80, cache_read_tokens=0, cache_write_tokens=0)

    def test_gemini_counts_tool_use_as_input_and_thoughts_as_output(self, monkeypatch, image):
        client = MagicMock()
        client.interactions.create.return_value = SimpleNamespace(
            output_text=CHARACTERS_JSON,
            usage=SimpleNamespace(
                total_input_tokens=500,
                total_tool_use_tokens=2000,
                total_output_tokens=150,
                total_thought_tokens=300,
                total_cached_tokens=1800,
                total_tokens=2950,
            ),
        )
        monkeypatch.setattr(
            gemini_detector, "genai", SimpleNamespace(Client=lambda **kwargs: client)
        )
        detector = gemini_detector.GeminiDetector()
        detector.detect_image(image, min_characters=2, max_characters=2)

        assert detector.usage == TokenUsage(
            input_tokens=2500, output_tokens=450, cache_read_tokens=1800, cache_write_tokens=0
        )

    def test_missing_usage_is_left_empty(self, sdk_client, image):
        sdk_client(openai_detector, "OpenAI")
        detector = openai_detector.OpenAIDetector()
        detector.detect_image(image)

        assert detector.usage is None


class TestMistralDetector:
    @pytest.fixture
    def client(self, sdk_client):
        return sdk_client(mistral_detector, "Mistral")

    def _respond(self, client, outputs):
        client.beta.conversations.start.return_value = SimpleNamespace(
            outputs=outputs,
            usage=SimpleNamespace(prompt_tokens=800, completion_tokens=90),
        )

    def test_parses_text_answer_and_records_usage(self, client, image):
        self._respond(
            client, [SimpleNamespace(type="message.output", content=CHARACTERS_JSON)]
        )
        detector = mistral_detector.MistralDetector()
        result = detector.detect_image(image, min_characters=2, max_characters=2)

        assert [c.name for c in result.characters] == ["Yoshi", "Toad"]
        assert detector.usage == TokenUsage(input_tokens=800, output_tokens=90, cache_read_tokens=0, cache_write_tokens=0)

    def test_joins_text_chunks_and_skips_thoughts_and_tool_entries(self, client, image):
        half = len(CHARACTERS_JSON) // 2
        self._respond(
            client,
            [
                SimpleNamespace(type="tool.execution", content="ignored"),
                SimpleNamespace(
                    type="message.output",
                    content=[
                        SimpleNamespace(type="thinking", thinking="ignored"),
                        SimpleNamespace(type="text", text=CHARACTERS_JSON[:half]),
                        SimpleNamespace(type="text", text=CHARACTERS_JSON[half:]),
                    ],
                ),
            ],
        )
        result = mistral_detector.MistralDetector().detect_image(
            image, min_characters=2, max_characters=2
        )

        assert [c.points for c in result.characters] == [15, 12]

class TestCacheTokenRecording:
    def test_openai_records_cache_tokens(self, sdk_client, image):
        client = sdk_client(openai_detector, "OpenAI")
        client.responses.parse.return_value.usage = SimpleNamespace(
            input_tokens=1500,
            output_tokens=120,
            input_tokens_details=SimpleNamespace(cached_tokens=1024, cache_write_tokens=300),
        )
        detector = openai_detector.OpenAIDetector()
        detector.detect_image(image)

        assert detector.usage == TokenUsage(
            input_tokens=1500, output_tokens=120, cache_read_tokens=1024, cache_write_tokens=300
        )

    def test_anthropic_records_cache_tokens(self, sdk_client, image):
        client = sdk_client(anthropic_detector, "Anthropic")
        client.messages.parse.return_value.usage = SimpleNamespace(
            input_tokens=700,
            output_tokens=80,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=600,
        )
        detector = anthropic_detector.AnthropicDetector()
        detector.detect_image(image)

        assert detector.usage.cache_write_tokens == 600
        assert detector.usage.cache_read_tokens == 0


class TestAnthropicEffort:
    @pytest.fixture
    def client(self, sdk_client):
        return sdk_client(anthropic_detector, "Anthropic")

    @staticmethod
    def _effort_supported(client, supported):
        client.models.retrieve.return_value = SimpleNamespace(
            capabilities=SimpleNamespace(effort=SimpleNamespace(supported=supported))
        )

    def test_effort_is_left_out_for_models_without_it(self, client, image):
        self._effort_supported(client, False)
        detector = anthropic_detector.AnthropicDetector().configure(model="claude-haiku-4-5")
        detector.detect_image(image)

        assert "output_config" not in client.messages.parse.call_args.kwargs

    def test_effort_is_sent_for_models_with_it(self, client, image):
        self._effort_supported(client, True)
        detector = anthropic_detector.AnthropicDetector().configure(thinking="medium")
        detector.detect_image(image)

        assert client.messages.parse.call_args.kwargs["output_config"] == {"effort": "medium"}

    def test_capability_is_looked_up_once_per_model(self, client, image):
        self._effort_supported(client, False)
        shared = anthropic_detector.AnthropicDetector()
        shared.configure(model="claude-haiku-4-5").detect_image(image)
        shared.configure(model="claude-haiku-4-5").detect_image(image)

        assert client.models.retrieve.call_count == 1

    def test_effort_is_sent_when_the_lookup_fails(self, client, image):
        client.models.retrieve.side_effect = RuntimeError("unknown model")
        anthropic_detector.AnthropicDetector().detect_image(image)

        assert "output_config" in client.messages.parse.call_args.kwargs


PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"rest"


class TestImageMediaType:
    @pytest.mark.parametrize(
        ("image", "expected"),
        [
            (PNG_BYTES, "image/png"),
            (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "image/webp"),
            (b"GIF89a...", "image/gif"),
            (b"\xff\xd8\xff\xe0", "image/jpeg"),
        ],
    )
    def test_is_read_from_the_first_bytes(self, image, expected):
        assert image_media_type(image) == expected

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Unsupported or unknown image format"):
            image_media_type(b"anything else")

    def test_openai_labels_a_png_as_png(self, sdk_client):
        client = sdk_client(openai_detector, "OpenAI")
        openai_detector.OpenAIDetector().detect_image(PNG_BYTES)

        url = client.responses.parse.call_args.kwargs["input"][0]["content"][0]["image_url"]
        assert url.startswith("data:image/png;base64,")

    def test_anthropic_labels_a_png_as_png(self, sdk_client):
        client = sdk_client(anthropic_detector, "Anthropic")
        anthropic_detector.AnthropicDetector().detect_image(PNG_BYTES)

        source = client.messages.parse.call_args.kwargs["messages"][0]["content"][0]["source"]
        assert source["media_type"] == "image/png"
