import wave
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from .audio import convert
from .data import AudioSample, SpeakerData
from .dtw import compute_optimal_path_with_window, get_path
from .template import Template
from .util import distance_to_probability, trim_silence

_DEFAULT_PROBABILITY = 0.5
_DEFAULT_DISTANCE = 0.3


@dataclass
class Evaluation:
    """Result of evaluating audio samples."""

    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int


class TemplateMatcher:
    """Matches audio against one or more templates."""

    def __init__(
        self,
        templates: Dict[str, List[Template]],
        audio_to_features: Callable[[np.ndarray], np.ndarray],
        vad: Optional[Callable[[bytes], bool]] = None,
        vad_reset: Optional[Callable[[], None]] = None,
    ) -> None:
        self.templates = templates
        self.audio_to_features = audio_to_features
        self.vad = vad
        self.vad_reset = vad_reset

    def match_wav(
        self,
        wav_file: Union[str, Path, wave.Wave_read],
        probabilty_threshold: float = _DEFAULT_PROBABILITY,
        distance_threshold: float = _DEFAULT_DISTANCE,
    ) -> Optional[str]:
        """Match a WAV file against templates.

        Returns speaker name or None if no match.
        """
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
        """Match raw audio against templates.
        Audio must be 16-bit mono at 16Khz.

        Returns speaker name or None if no match.
        """
        if self.vad_reset is not None:
            self.vad_reset()

        if self.vad is not None:
            audio_bytes = trim_silence(self.vad, audio_bytes)

        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        features = self.audio_to_features(audio_array)

        return self.match_features(
            features,
            probabilty_threshold=probabilty_threshold,
            distance_threshold=distance_threshold,
        )

    def match_features(
        self,
        features: np.ndarray,
        probabilty_threshold: float = _DEFAULT_PROBABILITY,
        distance_threshold: float = _DEFAULT_DISTANCE,
    ) -> Optional[str]:
        """Match features to templates.

        Returns speaker name or None if no match.
        """
        best_key: Optional[str] = None
        best_probability: Optional[float] = None

        for key, templates in self.templates.items():
            for template in templates:
                distance, _cost_matrix = compute_optimal_path_with_window(
                    template.features, features
                )

                # Normalize by sum of temporal dimensions
                # normalized_distance = distance / (
                #     len(template.features) + len(features)
                # )

                path = get_path(_cost_matrix)
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
        samples: Iterable[AudioSample],
        probabilty_threshold: float = _DEFAULT_PROBABILITY,
        distance_thresholds: Optional[List[float]] = None,
        step: int = 5,
    ) -> float:
        """Determine the best distance threshold for the samples (fixed probability threshold)."""
        if distance_thresholds is None:
            distance_thresholds = [v / 100 for v in range(step, 100, step)]

        assert distance_thresholds

        best_distance_threshold: Optional[float] = None
        best_score: Optional[int] = None
        for distance_threshold in distance_thresholds:
            evaluation = self.evaluate(
                samples,
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
        samples: Iterable[AudioSample],
        probabilty_threshold: float = _DEFAULT_PROBABILITY,
        distance_threshold: float = _DEFAULT_DISTANCE,
    ) -> Evaluation:
        """Evaluate audio samples against templates, checking expected speaker names."""
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0

        for sample in samples:
            with wave.open(str(sample.wav_path), "rb") as wav_file:
                actual_speaker = self.match_wav(
                    wav_file,
                    probabilty_threshold=probabilty_threshold,
                    distance_threshold=distance_threshold,
                )
                if actual_speaker is None:
                    if sample.is_positive:
                        false_negatives += 1
                    else:
                        true_negatives += 1
                else:
                    if actual_speaker == sample.speaker:
                        true_positives += 1
                    else:
                        false_positives += 1

        return Evaluation(
            true_positives=true_positives,
            true_negatives=true_negatives,
            false_positives=false_positives,
            false_negatives=false_negatives,
        )

    @staticmethod
    def from_data(
        speaker_data: Iterable[SpeakerData],
        audio_to_features: Callable[[np.ndarray], np.ndarray],
        average: bool = True,
        vad: Optional[Callable[[bytes], bool]] = None,
        vad_reset: Optional[Callable[[], None]] = None,
    ) -> "TemplateMatcher":
        """Creates templates from speaker data."""
        templates: Dict[str, List[Template]] = {}
        for data in speaker_data:
            if not data.train:
                continue

            dir_templates: List[Template] = []
            for sample in data.train:
                if vad_reset is not None:
                    vad_reset()

                with wave.open(str(sample.wav_path), "rb") as wav_file:
                    dir_templates.append(
                        Template.from_wav(
                            sample.wav_path.stem, wav_file, audio_to_features, vad=vad
                        )
                    )

            if average:
                templates[data.name] = [
                    Template.average_templates(data.name, dir_templates)
                ]
            else:
                templates[data.name] = dir_templates

        return TemplateMatcher(templates, audio_to_features, vad=vad)
