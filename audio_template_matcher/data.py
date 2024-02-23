"""Speaker and audio data."""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union


@dataclass
class AudioSample:
    wav_path: Path
    speaker: str
    is_positive: bool


@dataclass
class SpeakerData:
    name: str
    train: Optional[List[AudioSample]] = None
    positive: Optional[List[AudioSample]] = None
    negative: Optional[List[AudioSample]] = None

    @staticmethod
    def from_dir(
        speaker_dir: Union[str, Path], name: Optional[str] = None
    ) -> "SpeakerData":
        speaker_dir = Path(speaker_dir)
        name = name or speaker_dir.name

        samples: Dict[str, Optional[List[AudioSample]]] = {}
        for dir_name in ("train", "positive", "negative"):
            samples_dir = speaker_dir / dir_name
            if not samples_dir.is_dir():
                continue

            is_positive = dir_name != "negative"
            samples[dir_name] = [
                AudioSample(wav_path, name, is_positive)
                for wav_path in samples_dir.glob("*.wav")
            ]

        return SpeakerData(
            name=name,
            train=samples.get("train"),
            positive=samples.get("positive"),
            negative=samples.get("negative"),
        )
