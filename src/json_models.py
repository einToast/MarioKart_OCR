from pydantic import BaseModel


class DetectionRequest(BaseModel):
    provider: str
    image: str
    characters: list[str] | None = None
    include_place: bool = False
    min_characters: int = 4
    max_characters: int = 4
    use_code_interpreter: bool = False


class ProviderConfigRequest(BaseModel):
    model: str | None = None
    thinking: str | None = None
    prompt: str | None = None


class ProviderConfig(BaseModel):
    provider: str
    model: str
    thinking: str
    prompt: str


class CharacterResult(BaseModel):
    name: str
    points: int
    place: int | None = None


class DetectionResult(BaseModel):
    characters: list[CharacterResult]


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


class DetectionResponse(BaseModel):
    provider: str
    model: str
    result: DetectionResult
    usage: TokenUsage | None = None
