import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from pysilero_vad import SileroVoiceActivityDetector

from .audio import convert
from .dtw import compute_optimal_path_with_window, get_path
from .template import Template
from .util import distance_to_probability, get_mfcc, trim_silence

_DEFAULT_PROBABILITY = 0.5
_DEFAULT_DISTANCE = 0.3


@dataclass
class Evaluation:
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int


class TemplateMatcher:
    def __init__(self, templates: Dict[str, List[Template]]) -> None:
        self.templates = templates
        self._vad = SileroVoiceActivityDetector()

    def match_wav(
        self,
        wav_file: Union[str, Path, wave.Wave_read],
        probabilty_threshold: float = _DEFAULT_PROBABILITY,
        distance_threshold: float = _DEFAULT_DISTANCE,
    ) -> Optional[str]:
        if not isinstance(wav_file, wave.Wave_read):
            wav_file = wave.open(str(wav_file), "rb")

        audio_bytes = convert(
            wav_file.readframes(wav_file.getnframes()),
            in_rate=wav_file.getframerate(),
            in_width=wav_file.getsampwidth(),
            in_channels=wav_file.getnchannels(),
            out_rate=16000,
            out_width=2,
            out_channels=1,
        )
        return self.match_bytes(
            audio_bytes,
            probabilty_threshold=probabilty_threshold,
            distance_threshold=distance_threshold,
        )

    def match_bytes(
        self,
        audio_bytes: bytes,
        probabilty_threshold: float = _DEFAULT_PROBABILITY,
        distance_threshold: float = _DEFAULT_DISTANCE,
    ) -> Optional[str]:
        self._vad.reset()
        audio_bytes = trim_silence(self._vad, audio_bytes)
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        mfcc = get_mfcc(audio_array)

        best_key: Optional[str] = None
        best_probability: Optional[float] = None

        for key, templates in self.templates.items():
            for template in templates:
                distance, _cost_matrix = compute_optimal_path_with_window(
                    template.mfcc, mfcc
                )

                path = get_path(_cost_matrix)

                # Normalize by sum of temporal dimensions
                # normalized_distance = distance / (len(template.mfcc) + len(mfcc))
                normalized_distance = distance / len(path)

                # Compute detection probability
                probability = distance_to_probability(
                    normalized_distance, distance_threshold
                )
                if (best_probability is None) or (probability > best_probability):
                    best_key = key
                    best_probability = probability

        if (best_probability is not None) and (
            best_probability >= probabilty_threshold
        ):
            return best_key

        return None

    def tune(
        self,
        positive_dir: Union[str, Path],
        negative_dir: Union[str, Path],
        probabilty_threshold: float = _DEFAULT_PROBABILITY,
        distance_thresholds: Optional[List[float]] = None,
        step: int = 5,
    ) -> float:
        positive_dir = Path(positive_dir)
        negative_dir = Path(negative_dir)

        if distance_thresholds is None:
            distance_thresholds = [v / 100 for v in range(step, 100, step)]

        assert distance_thresholds

        best_distance_threshold: Optional[float] = None
        best_score: Optional[int] = None
        for distance_threshold in distance_thresholds:
            evaluation = self.evaluate(
                positive_dir,
                negative_dir,
                probabilty_threshold=probabilty_threshold,
                distance_threshold=distance_threshold,
            )
            score = evaluation.false_positives + evaluation.false_negatives
            if (best_score is None) or (score < best_score):
                best_distance_threshold = distance_threshold
                best_score = score

        assert best_distance_threshold is not None
        return best_distance_threshold

    def evaluate(
        self,
        positive_dir: Union[str, Path],
        negative_dir: Union[str, Path],
        probabilty_threshold: float = _DEFAULT_PROBABILITY,
        distance_threshold: float = _DEFAULT_DISTANCE,
    ) -> Evaluation:
        positive_dir = Path(positive_dir)
        negative_dir = Path(negative_dir)

        true_negatives = 0
        false_positives = 0
        for wav_path in negative_dir.glob("*.wav"):
            with wave.open(str(wav_path), "rb") as wav_file:
                key = self.match_wav(
                    wav_file,
                    probabilty_threshold=probabilty_threshold,
                    distance_threshold=distance_threshold,
                )
                if key is None:
                    true_negatives += 1
                else:
                    false_positives += 1

        true_positives = 0
        false_negatives = 0
        for wav_path in positive_dir.glob("*.wav"):
            with wave.open(str(wav_path), "rb") as wav_file:
                key = self.match_wav(
                    wav_file,
                    probabilty_threshold=probabilty_threshold,
                    distance_threshold=distance_threshold,
                )
                if key is None:
                    false_negatives += 1
                else:
                    true_positives += 1

        return Evaluation(
            true_positives=true_positives,
            true_negatives=true_negatives,
            false_positives=false_positives,
            false_negatives=false_negatives,
        )

    @staticmethod
    def from_wav_dirs(
        wav_dirs_by_key: Dict[str, Union[str, Path]], average: bool = True
    ) -> "TemplateMatcher":
        vad = SileroVoiceActivityDetector()
        templates: Dict[str, List[Template]] = {}
        for key, wav_dir in wav_dirs_by_key.items():
            wav_dir = Path(wav_dir)
            dir_templates: List[Template] = []
            for wav_path in wav_dir.glob("*.wav"):
                with wave.open(str(wav_path), "rb") as wav_file:
                    vad.reset()
                    dir_templates.append(
                        Template.from_wav(wav_path.stem, wav_file, vad)  # type: ignore
                    )

            if average:
                templates[key] = [Template.average_templates(key, dir_templates)]
            else:
                templates[key] = dir_templates

        return TemplateMatcher(templates)
