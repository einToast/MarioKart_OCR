import importlib
import logging
import pkgutil

import detectors
import pytest
from detectors import DETECTOR_REGISTRY, Detector, get_detector, init_detectors


def test_registry_is_detector_registry():
    assert DETECTOR_REGISTRY is Detector.registry


@pytest.mark.parametrize("provider", ["anthropic", "openai", "gemini", "openrouter"])
def test_builtin_providers_are_registered(provider):
    assert provider in DETECTOR_REGISTRY


class TestGetDetector:
    def test_lookup_is_case_insensitive(self, fake_detector_cls):
        assert isinstance(get_detector("FAKE"), fake_detector_cls)

    def test_reuses_one_instance_per_provider(self, fake_detector_cls):
        first = get_detector("fake")
        second = get_detector("fake", model="custom-model")

        assert len(fake_detector_cls.created) == 1
        assert first.client is second.client

    def test_passes_options_to_configure(self, fake_detector_cls):
        detector = get_detector(
            "fake", model="custom-model", thinking="high", prompt="Hi"
        )

        assert detector.model == "custom-model"
        assert detector.thinking == "high"
        assert detector.prompt == "Hi"

    def test_unknown_provider_raises_and_lists_available(self, fake_detector_cls):
        with pytest.raises(ValueError) as exc_info:
            get_detector("does-not-exist")

        message = str(exc_info.value)
        assert "Unsupported provider 'does-not-exist'" in message
        assert "fake" in message


def test_module_with_import_error_is_skipped_with_warning(monkeypatch, caplog):
    real_iter_modules = pkgutil.iter_modules
    real_import_module = importlib.import_module

    def iter_modules_with_broken(path=None, prefix=""):
        yield from real_iter_modules(path, prefix)
        yield pkgutil.ModuleInfo(None, "broken_detector", False)

    def import_module(name, package=None):
        if name == "detectors.broken_detector":
            raise ImportError("missing optional SDK")
        return real_import_module(name, package)

    monkeypatch.setattr(pkgutil, "iter_modules", iter_modules_with_broken)
    monkeypatch.setattr(importlib, "import_module", import_module)

    with caplog.at_level(logging.WARNING, logger="detectors"):
        importlib.reload(detectors)

    assert "Skipping detector module 'broken_detector'" in caplog.text
    assert "missing optional SDK" in caplog.text
    assert "anthropic" in detectors.DETECTOR_REGISTRY


class TestInitDetectors:
    def test_creates_an_instance_per_provider(self, fake_detector_cls):
        init_detectors()

        assert len(fake_detector_cls.created) == 1
        assert isinstance(Detector.instances["fake"], fake_detector_cls)

    def test_skips_detectors_that_fail_to_initialise(self, isolated_registry, caplog):
        class Broken(Detector):
            name = "broken"

            def __init__(self) -> None:
                raise RuntimeError("no API key")

            def detect_image(self, image, **kwargs):
                return None

        with caplog.at_level(logging.WARNING, logger="detectors"):
            init_detectors()

        assert "broken" not in Detector.instances
        assert "Detector 'broken' could not be initialised: no API key" in caplog.text
