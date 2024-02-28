from pathlib import Path

from .data import AudioSample, SpeakerData
from .template import Template
from .template_matcher import TemplateMatcher

_DIR = Path(__file__).parent
_VERSION_PATH = _DIR / "VERSION"

__version__ = _VERSION_PATH.read_text(encoding="utf-8").strip()

__all__ = [
    "__version__",
    "AudioSample",
    "SpeakerData",
    "Template",
    "TemplateMatcher",
]
