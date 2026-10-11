# MarioKart Tournament OCR Service

FastAPI webserver that reads the results table from a photo of a Mario Kart result screen. It sends the image to a vision model of the chosen provider and returns each character's name, points and place as JSON.

## Tech Stack

- Python 3.14
- FastAPI
- uv
- Docker

## Project Structure

```
src/
  webserver.py              FastAPI app
  json_models.py            Request and response models
  detectors/
    detector.py             Base class, provider registry and the prompt
    *_detector.py           One detector per provider
tests/                      pytest suite (detectors, providers, registry and API)
Dockerfile                  Production image (Uvicorn, 1 worker)
.github/workflows/          Tests, webserver smoke test, Docker build
```

Every module in `src/detectors/` that defines a `Detector` subclass is registered under its `name` at import time, so adding a provider only takes a new file there.

| Provider | API key | Default model | Default thinking |
|----------|---------|---------------|------------------|
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-haiku-5-5` | `low` |
| `gemini` | `GEMINI_API_KEY` | `gemini-3.8-flash` | `minimal` |
| `mistral` | `MISTRAL_API_KEY` | `mistral-small-2603` | `minimal` |
| `openai` | `OPENAI_API_KEY` | `gpt-6.1-sol` | `low` |
| `openrouter` | `OPENROUTER_API_KEY` | `deepseek/deepseek-v4.1-flash` | `minimal` |

Only the keys of the providers you want to use have to be set. A provider without a key is still listed by `/providers`, but not by `/providers/available`.

## Getting Started

### Prerequisites

- Python 3.14
- uv

### Run locally

```bash
uv sync --locked
uv run --env-file .env fastapi dev src/webserver.py --port 8001
```

The service listens on `http://127.0.0.1:8001`. The API keys are read from the environment. The command above loads them from a `.env` file:

```
OPENAI_API_KEY=...
GEMINI_API_KEY=...
```

### Run the tests

```bash
uv run pytest
```

The tests use fake clients and need no API keys.

### Run with Docker

```bash
docker build -t mariokart-ocr .
docker run --rm -p 8001:8001 --env-file .env mariokart-ocr
```

### Dev container

The repo includes a VS Code dev container (Python 3.14 with uv). It creates `.venv`, installs the requirements and forwards port 8001. `.vscode/launch.json` has debug configurations for the host and for the dev container.

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/healthcheck` | Returns `{"status": "OK"}`. |
| `GET` | `/providers` | Names of all registered providers. |
| `GET` | `/providers/available` | Providers whose API key is set and accepted. |
| `GET` | `/providers/{provider}/models` | Model IDs the provider offers. `503` if the key is missing or the provider cannot be reached. |
| `GET` | `/providers/{provider}/config` | Current model, thinking level and prompt of the provider. |
| `POST` | `/providers/{provider}/configure` | Changes the provider's defaults. |
| `POST` | `/detect` | Runs a detection on one image. |

An unknown provider name returns `404`.

### `POST /providers/{provider}/configure`

```json
{
  "model": "gpt-6.1-sol",
  "thinking": "low",
  "prompt": null
}
```

All fields are optional. A field that is left out keeps its value, and a field that is `null` or empty goes back to the built-in default. The response is the resulting configuration.

The settings are kept in memory and are lost on restart. The service therefore has to run with a single Uvicorn worker.

### `POST /detect`

```json
{
  "provider": "openai",
  "image": "<base64 encoded image>",
  "characters": ["Mario", "Baby Peach", "Knochentrocken", "Toad"],
  "include_place": true,
  "min_characters": 4,
  "max_characters": 4,
  "use_code_interpreter": false
}
```

| Field | Default | Description |
|-------|---------|-------------|
| `provider` | required | Provider name from `/providers`. |
| `image` | required | Base64 encoded PNG, JPEG, WebP or GIF. |
| `characters` | all rows | Return only the rows of these characters, in this order and spelled as given. |
| `include_place` | `false` | Also read the place of each row. |
| `min_characters` | `4` | Smallest number of rows the model may return. |
| `max_characters` | `4` | Largest number of rows the model may return. |
| `use_code_interpreter` | `false` | Lets the model use the provider's code execution tool. |

Response:

```json
{
  "provider": "openai",
  "model": "gpt-6.1-sol",
  "result": {
    "characters": [
      { "name": "Mario", "points": 15, "place": 1 },
      { "name": "Baby Peach", "points": 12, "place": 2 },
      { "name": "Knochentrocken", "points": 12, "place": 2 },
      { "name": "Toad", "points": 9, "place": 4 }
    ]
  },
  "usage": {
    "input_tokens": 1180,
    "output_tokens": 96,
    "cache_read_tokens": 0,
    "cache_write_tokens": 0
  }
}
```

`place` is only present when `include_place` is set, and `usage` only when the provider reports token counts.

| Status | Meaning |
|--------|---------|
| `404` | Unknown provider. |
| `422` | The request body is invalid, or `image` is not valid base64 or not a supported image format. |
| `429` | The provider rate limited the request. Its `Retry-After` header is passed on. |
| `502` | The provider returned any other error. |

