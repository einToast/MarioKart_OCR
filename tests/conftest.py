from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from detectors import Detector


@pytest.fixture(autouse=True)
def isolated_registry():
    snapshot = dict(Detector.registry)
    Detector.instances.clear()
    yield Detector.registry
    Detector.registry.clear()
    Detector.registry.update(snapshot)
    Detector.instances.clear()


@pytest.fixture
def fake_detector_cls():

    class FakeDetector(Detector):
        name = "fake"
        default_model = "fake-model"
        default_thinking = "low"
        created: list["FakeDetector"] = []
        configured: list["FakeDetector"] = []

        def __init__(self) -> None:
            client = MagicMock()
            client.models.list.return_value = SimpleNamespace(
                data=[SimpleNamespace(id="fake-a"), SimpleNamespace(id="fake-b")]
            )
            super().__init__(client, has_api_key=True)
            self.calls: list[dict] = []
            FakeDetector.created.append(self)

        def configure(self, **options):
            detector = super().configure(**options)
            FakeDetector.configured.append(detector)
            return detector

        def detect_image(
            self,
            image,
            include_place: bool = False,
            min_characters: int = 4,
            max_characters: int = 4,
            use_code_interpreter: bool = False,
        ):
            self.calls.append(
                {
                    "image": image,
                    "include_place": include_place,
                    "min_characters": min_characters,
                    "max_characters": max_characters,
                    "use_code_interpreter": use_code_interpreter,
                }
            )
            response_model = self._create_pydantic_model(
                include_place, min_characters, max_characters
            )
            characters = [
                {"name": f"Mario{i}", "points": 15 - i}
                | ({"place": i + 1} if include_place else {})
                for i in range(min_characters)
            ]
            return response_model(characters=characters)

    return FakeDetector


@pytest.fixture
def fake_detector(fake_detector_cls):
    return fake_detector_cls()


@pytest.fixture
def image():
    return b"fake-jpeg-bytes"
