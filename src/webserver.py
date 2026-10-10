import base64
from contextlib import asynccontextmanager

from detectors import (
    DETECTOR_REGISTRY,
    build_prompt,
    get_detector,
    image_media_type,
    init_detectors,
    set_detector_defaults,
)
from fastapi import FastAPI, HTTPException
from json_models import (
    DetectionRequest,
    DetectionResponse,
    ProviderConfig,
    ProviderConfigRequest,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_detectors()
    yield


app = FastAPI(
    title="Mario Kart OCR Service",
    description="Performing OCR detections using various AI models",
    lifespan=lifespan,
)


@app.get("/healthcheck")
async def healthcheck() -> dict:
    return {"status": "OK"}


@app.get("/providers")
async def get_providers() -> list[str]:
    return list(DETECTOR_REGISTRY.keys())


def _is_available(provider_name: str) -> bool:
    try:
        return get_detector(provider_name).is_available()
    except Exception:
        return False


@app.get("/providers/available")
def get_available_providers() -> list[str]:
    return [name for name in DETECTOR_REGISTRY if _is_available(name)]


@app.get("/providers/{provider_name}/models")
def get_provider_models(provider_name: str) -> list[str]:
    try:
        detector = get_detector(provider_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        return detector.get_available_models()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/providers/{provider_name}/config")
def get_provider_config(provider_name: str) -> ProviderConfig:
    try:
        return get_detector(provider_name).config()
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/providers/{provider_name}/configure")
def configure_provider(
    provider_name: str, request: ProviderConfigRequest
) -> ProviderConfig:
    try:
        detector = set_detector_defaults(
            provider_name, **request.model_dump(exclude_unset=True)
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return detector.config()


def _provider_error(exc: Exception) -> HTTPException | None:
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if not isinstance(status, int):
        return None
    if status != 429:
        return HTTPException(status_code=502, detail=f"{type(exc).__name__}: {exc}")
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    retry_after = headers.get("retry-after")
    return HTTPException(
        status_code=429,
        detail=str(exc),
        headers={"Retry-After": retry_after} if isinstance(retry_after, str) else None,
    )


@app.post("/detect", response_model=DetectionResponse, response_model_exclude_none=True)
def run_detection(request: DetectionRequest) -> DetectionResponse:
    try:
        detector = get_detector(request.provider)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    detector.prompt = build_prompt(
        detector.prompt, request.characters, request.include_place
    )
    try:
        image = base64.b64decode(request.image, validate=True)
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail=f"image is not valid base64: {exc}"
        ) from exc
    try:
        image_media_type(image)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        result = detector.detect_image(
            image,
            include_place=request.include_place,
            min_characters=request.min_characters,
            max_characters=request.max_characters,
            use_code_interpreter=request.use_code_interpreter,
        )
    except Exception as exc:
        error = _provider_error(exc)
        if error is None:
            raise
        raise error from exc
    return DetectionResponse(
        provider=detector.name,
        model=detector.model,
        result=result.model_dump(),
        usage=detector.usage,
    )
