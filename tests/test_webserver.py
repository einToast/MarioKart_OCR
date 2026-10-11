from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from detectors import Detector, get_detector
from json_models import DetectionRequest
from webserver import app

client = TestClient(app)


def _detect(**overrides):
    body = {"provider": "fake", "image": "/9j/aW1hZ2U="} | overrides
    return client.post("/detect", json=body)


def test_healthcheck():
    response = client.get("/healthcheck")

    assert response.status_code == 200
    assert response.json() == {"status": "OK"}


def test_providers_lists_registry(fake_detector_cls):
    response = client.get("/providers")

    assert response.status_code == 200
    providers = response.json()
    assert "fake" in providers
    assert {"anthropic", "openai", "gemini", "openrouter"} <= set(providers)


class TestProviderModels:
    def test_returns_model_ids(self, fake_detector_cls):
        response = client.get("/providers/fake/models")

        assert response.status_code == 200
        assert response.json() == ["fake-a", "fake-b"]

    def test_unknown_provider_is_404(self):
        response = client.get("/providers/does-not-exist/models")

        assert response.status_code == 404
        assert "Unsupported provider" in response.json()["detail"]


class TestDetect:
    def test_returns_provider_model_and_characters(self, fake_detector_cls):
        response = _detect()

        assert response.status_code == 200
        body = response.json()
        assert body["provider"] == "fake"
        assert body["model"] == "fake-model"
        assert len(body["result"]["characters"]) == 4
        assert body["result"]["characters"][0] == {"name": "Mario0", "points": 15}

    def test_place_omitted_unless_requested(self, fake_detector_cls):
        without_place = _detect().json()["result"]["characters"]
        with_place = _detect(include_place=True).json()["result"]["characters"]

        assert all("place" not in c for c in without_place)
        assert [c["place"] for c in with_place] == [1, 2, 3, 4]

    def test_forwards_detection_options(self, fake_detector_cls):
        response = _detect(
            include_place=True,
            min_characters=2,
            max_characters=12,
            use_code_interpreter=True,
        )

        assert response.status_code == 200
        call = fake_detector_cls.configured[-1].calls[-1]
        assert call["include_place"] is True
        assert call["min_characters"] == 2
        assert call["max_characters"] == 12
        assert call["use_code_interpreter"] is True

    def test_uses_provider_defaults(self, fake_detector_cls):
        _detect()
        detector = fake_detector_cls.configured[-1]

        assert detector.model == "fake-model"
        assert detector.thinking == "low"
        assert "OCR detector" in detector.prompt

    def test_model_thinking_and_prompt_are_not_request_fields(self, fake_detector_cls):
        assert not {"model", "thinking", "prompt"} & set(DetectionRequest.model_fields)
        assert _detect(model="request-model").json()["model"] == "fake-model"

    def test_unknown_provider_is_404(self):
        response = _detect(provider="does-not-exist")

        assert response.status_code == 404

    @pytest.mark.parametrize("missing", ["provider", "image"])
    def test_missing_required_field_is_422(self, missing):
        body = {"provider": "fake", "image": "/9j/aW1hZ2U="}
        del body[missing]

        assert client.post("/detect", json=body).status_code == 422

    def test_image_is_decoded_to_bytes(self, fake_detector_cls):
        _detect(image="/9j/aW1hZ2U=")
        image = fake_detector_cls.configured[-1].calls[-1]["image"]

        assert image == b"\xff\xd8\xff" + b"image"

    def test_invalid_base64_image_is_422(self, fake_detector_cls):
        response = _detect(image="not base64!")

        assert response.status_code == 422
        assert "image is not valid base64" in response.json()["detail"]

    def test_unknown_image_format_is_422(self, fake_detector_cls):
        response = _detect(image="aW1hZ2U=")

        assert response.status_code == 422
        assert "Unsupported or unknown image format" in response.json()["detail"]
        assert fake_detector_cls.configured[-1].calls == []

    def test_characters_restrict_default_prompt(self, fake_detector_cls):
        _detect(characters=["Bowser", "Baby Peach"])
        prompt = fake_detector_cls.configured[-1].prompt

        assert "OCR detector" in prompt
        assert '"Bowser", "Baby Peach"' in prompt
        assert "Return every row" not in prompt


def test_providers_available_lists_working_providers(fake_detector_cls, monkeypatch):
    monkeypatch.setattr("webserver.DETECTOR_REGISTRY", {"fake": fake_detector_cls})

    assert client.get("/providers/available").json() == ["fake"]


def test_requests_share_one_detector_instance(fake_detector_cls):
    _detect()
    _detect()
    first, second = fake_detector_cls.configured[-2:]

    assert len(fake_detector_cls.created) == 1
    assert first is not second
    assert first.client is second.client


def test_startup_initialises_detectors(fake_detector_cls):
    with TestClient(app):
        assert len(fake_detector_cls.created) == 1


def test_models_of_provider_without_key_is_503(fake_detector_cls):
    get_detector("fake")
    Detector.instances["fake"].has_api_key = False

    response = client.get("/providers/fake/models")

    assert response.status_code == 503
    assert "API key is not set for fake" in response.json()["detail"]


class TestConfigure:
    def test_sets_defaults_for_following_requests(self, fake_detector_cls):
        response = client.put(
            "/providers/fake/configure",
            json={"model": "configured-model", "prompt": "Read the scores."},
        )

        assert response.status_code == 200
        assert response.json() == {
            "provider": "fake",
            "model": "configured-model",
            "thinking": "low",
            "prompt": "Read the scores.",
        }
        assert _detect().json()["model"] == "configured-model"
        assert fake_detector_cls.configured[-1].prompt == "Read the scores."

    def test_null_resets_to_builtin_default(self, fake_detector_cls):
        client.put(
            "/providers/fake/configure",
            json={"model": "configured-model", "thinking": "high"},
        )
        body = client.put("/providers/fake/configure", json={"model": None}).json()

        assert body["model"] == "fake-model"
        assert body["thinking"] == "high"

    def test_empty_body_returns_current_config(self, fake_detector_cls):
        body = client.put("/providers/fake/configure", json={}).json()

        assert body["model"] == "fake-model"
        assert "OCR detector" in body["prompt"]

    def test_characters_are_appended_to_configured_prompt(self, fake_detector_cls):
        client.put("/providers/fake/configure", json={"prompt": "Read the scores."})
        _detect(characters=["Bowser"])
        prompt = fake_detector_cls.configured[-1].prompt

        assert prompt.startswith("Read the scores.")
        assert '"Bowser"' in prompt

    def test_unknown_provider_is_404(self):
        response = client.put("/providers/does-not-exist/configure", json={})

        assert response.status_code == 404


class TestProviderConfig:
    def test_returns_builtin_defaults(self, fake_detector_cls):
        body = client.get("/providers/fake/config").json()

        assert body["provider"] == "fake"
        assert body["model"] == "fake-model"
        assert body["thinking"] == "low"
        assert "OCR detector" in body["prompt"]

    def test_returns_what_was_configured(self, fake_detector_cls):
        client.put(
            "/providers/fake/configure",
            json={"model": "configured-model", "prompt": "Read the scores."},
        )

        assert client.get("/providers/fake/config").json() == {
            "provider": "fake",
            "model": "configured-model",
            "thinking": "low",
            "prompt": "Read the scores.",
        }

    def test_unknown_provider_is_404(self):
        assert client.get("/providers/does-not-exist/config").status_code == 404


class TestUsage:
    def test_usage_is_omitted_when_provider_reports_none(self, fake_detector_cls):
        assert "usage" not in _detect().json()

    def test_usage_of_the_call_is_returned(self, fake_detector_cls, monkeypatch):
        detect_image = fake_detector_cls.detect_image

        def detect_with_usage(self, image, **kwargs):
            self._record_usage(1200, 340)
            return detect_image(self, image, **kwargs)

        monkeypatch.setattr(fake_detector_cls, "detect_image", detect_with_usage)

        assert _detect().json()["usage"] == {"input_tokens": 1200, "output_tokens": 340, "cache_read_tokens": 0, "cache_write_tokens": 0}
        assert Detector.instances["fake"].usage is None


class TestProviderErrors:
    @staticmethod
    def _failing_with(monkeypatch, fake_detector_cls, error):
        def detect_image(self, image, **kwargs):
            raise error

        monkeypatch.setattr(fake_detector_cls, "detect_image", detect_image)

    def test_rate_limit_is_passed_on_with_retry_after(self, fake_detector_cls, monkeypatch):
        error = RuntimeError("Rate limit reached for requests")
        error.status_code = 429
        error.response = SimpleNamespace(headers={"retry-after": "7"})
        self._failing_with(monkeypatch, fake_detector_cls, error)

        response = _detect()

        assert response.status_code == 429
        assert response.headers["retry-after"] == "7"
        assert "Rate limit reached" in response.json()["detail"]

    def test_rate_limit_reported_as_code_without_retry_after(self, fake_detector_cls, monkeypatch):
        error = RuntimeError("Resource exhausted")
        error.code = 429
        self._failing_with(monkeypatch, fake_detector_cls, error)

        response = _detect()

        assert response.status_code == 429
        assert "retry-after" not in response.headers

    def test_other_provider_errors_are_502_with_message(self, fake_detector_cls, monkeypatch):
        error = RuntimeError("model overloaded")
        error.status_code = 503
        self._failing_with(monkeypatch, fake_detector_cls, error)

        response = _detect()

        assert response.status_code == 502
        assert response.json()["detail"] == "RuntimeError: model overloaded"


class TestPlaceInPrompt:
    def test_prompt_leaves_out_the_place_by_default(self, fake_detector_cls):
        _detect()

        assert "- place:" not in fake_detector_cls.configured[-1].prompt

    def test_prompt_describes_the_place_when_asked_for(self, fake_detector_cls):
        _detect(include_place=True)

        assert "- place:" in fake_detector_cls.configured[-1].prompt

    def test_configured_prompt_is_used_unchanged(self, fake_detector_cls):
        client.put("/providers/fake/configure", json={"prompt": "Read the scores."})
        _detect()

        assert fake_detector_cls.configured[-1].prompt == "Read the scores."
