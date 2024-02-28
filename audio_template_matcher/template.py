import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from .audio import convert
from .dtw import compute_optimal_path, get_path
from .util import trim_silence


@dataclass
class Template:
    """Audio template."""

    name: str
    duration_sec: float
    features: np.ndarray

    @staticmethod
    def from_wav(
        name: str,
        wav_file: Union[str, Path, wave.Wave_read],
        audio_to_features: Callable[[np.ndarray], np.ndarray],
        vad: Optional[Callable[[bytes], bool]] = None,
    ) -> "Template":
        """Create an audio template from a WAV file."""
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

        if vad is not None:
            audio_bytes = trim_silence(vad, audio_bytes)

        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

        features = audio_to_features(audio_array)
        duration_sec = len(audio_array) / 16000

        return Template(name, duration_sec, features)

    @staticmethod
    def average_templates(name: str, templates: "List[Template]") -> "Template":
        """Averages multiple templates piecewise into a single template.

        Credit to: https://github.com/mathquis/node-personal-wakeword
        """
        assert templates, "No templates"
        if len(templates) == 1:
            # Only one template
            return templates[0]

        # Use longest template as base
        templates = sorted(templates, key=lambda t: len(t.features), reverse=True)
        base_template = templates[0]

        base_features = base_template.features
        rows, cols = base_features.shape
        averages = [
            [[base_features[row][col]] for col in range(cols)] for row in range(rows)
        ]

        # Collect features
        for template in templates[1:]:
            _distance, cost_matrix = compute_optimal_path(
                template.features, base_features
            )
            path = get_path(cost_matrix)
            for row, col in path:
                for i, feature in enumerate(template.features[row]):
                    averages[col][i].append(feature)

        # Average features
        avg_features = np.array(
            [
                [np.mean(averages[row][col]) for col in range(cols)]
                for row in range(rows)
            ]
        )

        assert avg_features.shape == base_features.shape, "Wrong features shape"

        return Template(
            name, duration_sec=base_template.duration_sec, features=avg_features
        )
