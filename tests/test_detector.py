import pytest
from detectors import Detector, build_prompt
from detectors.detector import DEFAULT_PROMPT
from json_models import ProviderConfig
from pydantic import ValidationError


def _character(i: int, place: bool = False) -> dict:
    return {"name": f"Luigi{i}", "points": i} | ({"place": i + 1} if place else {})


class TestSubclassRegistration:
    def test_subclass_is_registered_under_lowercase_name(self, isolated_registry):
        class MixedCase(Detector):
            name = "MixedCase"

            def detect_image(self, image, **kwargs):
                pass

        assert isolated_registry["mixedcase"] is MixedCase

    def test_subclass_without_name_raises(self, isolated_registry):
        before = dict(isolated_registry)

        with pytest.raises(ValueError, match="must define a 'name'"):

            class Nameless(Detector):
                def detect_image(self, image, **kwargs):
                    pass

        assert isolated_registry == before

    def test_duplicate_name_raises_case_insensitively(self, isolated_registry):
        class First(Detector):
            name = "dupe"

            def detect_image(self, image, **kwargs):
                pass

        with pytest.raises(ValueError, match="Duplicate detector name 'dupe'"):

            class Second(Detector):
                name = "DUPE"

                def detect_image(self, image, **kwargs):
                    pass

        assert isolated_registry["dupe"] is First

    def test_subclass_without_detect_image_cannot_be_instantiated(self):
        class Incomplete(Detector):
            name = "incomplete"

        with pytest.raises(TypeError):
            Incomplete(client=None)


class TestConfigure:
    def test_new_detector_uses_defaults(self, fake_detector):
        assert fake_detector.model == "fake-model"
        assert fake_detector.thinking == "low"
        assert "OCR detector" in fake_detector.prompt

    def test_uses_default_prompt_when_none_given(self, fake_detector):
        detector = fake_detector.configure(prompt=None)
        assert "OCR detector" in detector.prompt

    def test_uses_given_options(self, fake_detector):
        detector = fake_detector.configure(
            model="custom-model", thinking="high", prompt="Read the scoreboard."
        )
        assert detector.model == "custom-model"
        assert detector.thinking == "high"
        assert detector.prompt == "Read the scoreboard."

    def test_config_reports_what_the_detector_uses(self, fake_detector):
        detector = fake_detector.configure(model="custom-model", prompt="Read it.")

        assert detector.config() == ProviderConfig(
            provider="fake", model="custom-model", thinking="low", prompt="Read it."
        )

    def test_shares_client_and_leaves_shared_instance_untouched(self, fake_detector):
        detector = fake_detector.configure(model="custom-model")

        assert detector is not fake_detector
        assert detector.client is fake_detector.client
        assert fake_detector.model == "fake-model"

    def test_get_available_models_returns_model_ids(self, fake_detector):
        assert fake_detector.get_available_models() == ["fake-a", "fake-b"]


class TestCreatePydanticModel:
    def test_without_place_has_only_name_and_points(self, fake_detector):
        model = fake_detector._create_pydantic_model(False, 4, 4)
        result = model(characters=[_character(i) for i in range(4)])

        character_fields = type(result.characters[0]).model_fields
        assert set(character_fields) == {"name", "points"}

    def test_with_place_requires_place(self, fake_detector):
        model = fake_detector._create_pydantic_model(True, 1, 1)

        result = model(characters=[_character(0, place=True)])
        assert result.characters[0].place == 1

        with pytest.raises(ValidationError):
            model(characters=[_character(0)])

    def test_rejects_non_integer_points(self, fake_detector):
        model = fake_detector._create_pydantic_model(False, 1, 1)
        with pytest.raises(ValidationError):
            model(characters=[{"name": "Peach", "points": "lots"}])

    @pytest.mark.parametrize(
        ("min_characters", "max_characters", "count", "valid"),
        [
            (4, 4, 4, True),
            (4, 4, 3, False),
            (4, 4, 5, False),
            (1, 12, 1, True),
            (1, 12, 12, True),
            (1, 12, 13, False),
            (2, 3, 0, False),
        ],
    )
    def test_enforces_character_count(
        self, fake_detector, min_characters, max_characters, count, valid
    ):
        model = fake_detector._create_pydantic_model(
            False, min_characters, max_characters
        )
        characters = [_character(i) for i in range(count)]

        if valid:
            assert len(model(characters=characters).characters) == count
        else:
            with pytest.raises(ValidationError):
                model(characters=characters)

    def test_json_schema_sent_to_providers_has_bounds_and_required_fields(
        self, fake_detector
    ):
        schema = fake_detector._create_pydantic_model(True, 2, 6).model_json_schema()

        characters = schema["properties"]["characters"]
        assert characters["minItems"] == 2
        assert characters["maxItems"] == 6
        assert schema["required"] == ["characters"]

        character_schema = schema["$defs"]["Character"]
        assert set(character_schema["required"]) == {"name", "points", "place"}


class TestBuildPrompt:
    def test_without_characters_returns_default_prompt(self):
        assert build_prompt() == DEFAULT_PROMPT
        assert build_prompt(characters=[]) == DEFAULT_PROMPT
        assert "Return every row" in DEFAULT_PROMPT

    def test_without_characters_keeps_custom_prompt(self):
        assert build_prompt("Read the scoreboard.") == "Read the scoreboard."

    def test_characters_replace_the_all_rows_instruction(self):
        prompt = build_prompt(characters=["Bowser", "König Buu Huu"])

        assert "OCR detector" in prompt
        assert 'in this order: "Bowser", "König Buu Huu".' in prompt
        assert "Return every row" not in prompt

    def test_characters_are_appended_to_custom_prompt(self):
        prompt = build_prompt("Read the scoreboard.", ["Bowser"])

        assert prompt.startswith("Read the scoreboard.\n\n")
        assert 'in this order: "Bowser".' in prompt


class TestApiKey:
    def test_key_presence_does_not_depend_on_the_client(self, isolated_registry):
        class NoKeyAttribute(Detector):
            name = "no-key-attribute"
            default_model = "m"
            default_thinking = "low"

            def detect_image(self, image, **kwargs):
                return None

        client = object()  # like the Mistral, Gemini and OpenRouter clients: no api_key attribute

        assert NoKeyAttribute(client, has_api_key=True).api_key_is_set() is True
        assert NoKeyAttribute(client).api_key_is_set() is False
        assert NoKeyAttribute(client).is_available() is False


class TestBuildPromptPlace:
    def test_default_prompt_describes_the_place(self):
        assert build_prompt() == DEFAULT_PROMPT
        assert "- place:" in DEFAULT_PROMPT

    def test_place_is_left_out_when_places_are_not_asked_for(self):
        prompt = build_prompt(include_place=False)

        assert "- place:" not in prompt
        assert "- points:" in prompt
        assert "Return every row" in prompt

    def test_place_is_left_out_with_characters_too(self):
        prompt = build_prompt(characters=["Bowser"], include_place=False)

        assert "- place:" not in prompt
        assert 'in this order: "Bowser".' in prompt

    def test_custom_prompt_is_not_touched(self):
        assert build_prompt("Read the place too.", include_place=False) == "Read the place too."
