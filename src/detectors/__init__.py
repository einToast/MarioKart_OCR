import importlib
import logging
import pkgutil

from .detector import Detector, build_prompt, image_media_type

logger = logging.getLogger(__name__)

for _module in pkgutil.iter_modules(__path__):
    if _module.name == "detector":
        continue
    try:
        importlib.import_module(f"{__name__}.{_module.name}")
    except ImportError as _error:
        logger.warning(f"Skipping detector module '{_module.name}': {_error}")

DETECTOR_REGISTRY = Detector.registry
DETECTOR_INSTANCES = Detector.instances


def _shared_detector(name: str) -> Detector:
    key = name.lower()
    if key not in DETECTOR_REGISTRY:
        raise ValueError(
            f"Unsupported provider '{name}'. Available providers: {', '.join(DETECTOR_REGISTRY)}"
        )
    if key not in DETECTOR_INSTANCES:
        DETECTOR_INSTANCES[key] = DETECTOR_REGISTRY[key]()
    return DETECTOR_INSTANCES[key]


def get_detector(name: str, **options) -> Detector:
    return _shared_detector(name).configure(**options)


def set_detector_defaults(name: str, **options) -> Detector:
    shared = _shared_detector(name)
    shared.set_defaults(**options)
    return shared


def init_detectors() -> None:
    for name in DETECTOR_REGISTRY:
        try:
            get_detector(name)
        except Exception as exc:
            logger.warning(f"Detector {name} could not be initialised: {exc}")


__all__ = [
    "DETECTOR_INSTANCES",
    "DETECTOR_REGISTRY",
    "Detector",
    "build_prompt",
    "get_detector",
    "image_media_type",
    "init_detectors",
    "set_detector_defaults",
]
