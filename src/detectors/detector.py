import copy
from abc import ABC, abstractmethod
from typing import Annotated, ClassVar, Self

from json_models import ProviderConfig, TokenUsage
from pydantic import BaseModel, Field, create_model

_PROMPT_TABLE = """You are an OCR detector for Mario Kart result screens.

The image shows the results table that Mario Kart 8 Deluxe or Mario Kart World displays after a race. It is usually a phone photo of a projector screen, so the table can be tilted, blurry, washed out or partly reflected. Every row of the table reads, from left to right: the place, the character's icon, the character's name and, right-aligned at the end of the row, the character's points.

Extract the rows from top to bottom:

- name: the character's name exactly as it is written in the row. The game runs in German, so names look like "Knochentrocken", "König Buu Huu", "Katzen-Peach", "Bewohnerin" or "Schwarzer Shy Guy". Keep umlauts, hyphens and spaces, and do not translate or shorten the name. When the text is hard to read, use the icon next to it to decide between similar names such as "Peach", "Baby Peach" and "Katzen-Peach".
- points: the number at the right end of the same row. Follow the row across carefully, because on a tilted photo the number sits higher or lower than the name. Read every number on its own instead of inferring it from the rows around it: points are not always in descending order, and several characters can have the same points. The digits are drawn in a segmented style, where 1 and 7, 3 and 8, and 5 and 6 are easy to confuse.
"""

_PROMPT_PLACE = """- place: the number at the left end of the row, exactly as shown. Do not count rows to get it. Tied characters share a place and the next place is skipped (1, 2, 2, 4), and the table can show only part of the field, so places can jump (8, 16, 17).
"""

_PROMPT_ALL_ROWS = """Return every row of the table.
"""

_PROMPT_CHARACTERS = """Return only the rows of these characters, in this order: {names}.

Find each of them by its name and icon, wherever it is in the table, and skip every other row. Write the name exactly as it is spelled in this list, so the result can be matched to it. A name that is listed more than once has that many rows; return each of those rows once. Take care not to pick a similar character instead, for example "Peach" when "Baby Peach" is listed. If a listed character has no row in the table, leave it out instead of guessing its points.
"""

_PROMPT_CLOSING = """Report only what the table shows. Ignore the race running in the background and anything else in the picture, and never add a character that has no row.
"""

DEFAULT_PROMPT = (
    f"{_PROMPT_TABLE}{_PROMPT_PLACE}\n{_PROMPT_ALL_ROWS}\n{_PROMPT_CLOSING}"
)


def build_prompt(
    prompt: str | None = None,
    characters: list[str] | None = None,
    include_place: bool = True,
) -> str:
    selection = _PROMPT_ALL_ROWS
    if characters:
        selection = _PROMPT_CHARACTERS.format(
            names=", ".join(f'"{name}"' for name in characters)
        )
    if prompt and prompt != DEFAULT_PROMPT:
        return f"{prompt.rstrip()}\n\n{selection}" if characters else prompt
    place = _PROMPT_PLACE if include_place else ""
    return f"{_PROMPT_TABLE}{place}\n{selection}\n{_PROMPT_CLOSING}"


def image_media_type(image: bytes) -> str:
    if image.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image[:4] == b"RIFF" and image[8:12] == b"WEBP":
        return "image/webp"
    if image.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    raise ValueError("Unsupported or unknown image format")


class Detector(ABC):
    registry: ClassVar[dict[str, type["Detector"]]] = {}
    instances: ClassVar[dict[str, "Detector"]] = {}
    name: ClassVar[str | None] = None
    default_model: ClassVar[str]
    default_thinking: ClassVar[str]

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if cls.name is None:
            raise ValueError(
                f"Detector subclass '{cls.__name__}' must define a 'name' class variable"
            )
        key = cls.name.lower()
        if key in Detector.registry:
            raise ValueError(f"Duplicate detector name '{key}'")
        Detector.registry[key] = cls

    def __init__(
        self, client, has_api_key: bool = False, supports_video: bool = False
    ) -> None:
        self.client = client
        self.has_api_key = has_api_key
        self.supports_video = supports_video
        self.model = self.default_model
        self.thinking = self.default_thinking
        self.prompt = DEFAULT_PROMPT
        self.usage: TokenUsage | None = None

    def configure(
        self,
        model: str | None = None,
        thinking: str | None = None,
        prompt: str | None = None,
    ) -> Self:
        detector = copy.copy(self)
        detector.model = model or self.model
        detector.thinking = thinking or self.thinking
        detector.prompt = prompt or self.prompt
        detector.usage = None
        return detector

    def config(self) -> ProviderConfig:
        return ProviderConfig(
            provider=self.name,
            model=self.model,
            thinking=self.thinking,
            prompt=self.prompt,
        )

    def set_defaults(self, **options: str | None) -> None:
        builtin = ProviderConfig(
            provider=self.name,
            model=self.default_model,
            thinking=self.default_thinking,
            prompt=DEFAULT_PROMPT,
        )
        for key, value in options.items():
            if key not in ("model", "thinking", "prompt"):
                raise TypeError(f"Unknown detector option '{key}'")
            setattr(self, key, value or getattr(builtin, key))

    def get_available_models(self) -> list[str]:
        if not self.api_key_is_set():
            raise ValueError(
                f"API key is not set for {self.name}. Please set the API key to access available models."
            )
        try:
            return self._list_model_ids()
        except Exception as e:
            raise ValueError(
                f"Error occurred while fetching available models for {self.name}: {e}"
            ) from e

    def _record_usage(
        self,
        input_tokens,
        output_tokens,
        cache_read_tokens=None,
        cache_write_tokens=None,
    ) -> None:
        if isinstance(input_tokens, int) and isinstance(output_tokens, int):
            self.usage = TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read_tokens
                if isinstance(cache_read_tokens, int)
                else 0,
                cache_write_tokens=cache_write_tokens
                if isinstance(cache_write_tokens, int)
                else 0,
            )

    def _list_model_ids(self) -> list[str]:
        return [model.id for model in self.client.models.list().data]

    def api_key_is_set(self) -> bool:
        return self.has_api_key

    def is_available(self) -> bool:
        if not self.api_key_is_set():
            return False
        try:
            self.get_available_models()
            return True
        except Exception:
            return False

    @abstractmethod
    def detect_image(
        self,
        image,
        include_place: bool = False,
        min_characters: int = 4,
        max_characters: int = 4,
        use_code_interpreter: bool = False,
    ) -> BaseModel:
        pass

    def _create_pydantic_model(
        self, include_place: bool, min_characters: int, max_characters: int
    ) -> type[BaseModel]:
        fields = {
            "name": (str, Field(..., description="Character's name.")),
            "points": (
                int,
                Field(..., description="Points assigned to the character."),
            ),
        }

        if include_place:
            fields["place"] = (
                int,
                Field(..., description="Place associated with the character."),
            )

        character_model = create_model("Character", **fields)
        characters_list_type = Annotated[
            list[character_model],
            Field(min_length=min_characters, max_length=max_characters),
        ]

        return create_model(
            "MultipleCharacters", characters=(characters_list_type, ...)
        )
