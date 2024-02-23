import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from .audio import convert
from .dtw import compute_optimal_path, get_path
from .util import get_mfcc, trim_silence


@dataclass
class Template:
    """Audio template."""

    name: str
    duration_sec: float
    mfcc: np.ndarray

    @staticmethod
    def from_wav(
        name: str,
        wav_file: Union[str, Path, wave.Wave_read],
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

        mfcc = get_mfcc(audio_array)
        duration_sec = len(audio_array) / 16000

        return Template(name, duration_sec, mfcc)

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
        templates = sorted(templates, key=lambda t: len(t.mfcc), reverse=True)
        base_template = templates[0]

        base_mfcc = base_template.mfcc
        rows, cols = base_mfcc.shape
        averages = [
            [[base_mfcc[row][col]] for col in range(cols)] for row in range(rows)
        ]

        # Collect features
        for template in templates[1:]:
            _distance, cost_matrix = compute_optimal_path(template.mfcc, base_mfcc)
            path = get_path(cost_matrix)
            for row, col in path:
                for i, feature in enumerate(template.mfcc[row]):
                    averages[col][i].append(feature)

        # Average features
        avg_mfcc = np.array(
            [
                [np.mean(averages[row][col]) for col in range(cols)]
                for row in range(rows)
            ]
        )

        assert avg_mfcc.shape == base_mfcc.shape, "Wrong MFCC shape"

        return Template(name, duration_sec=base_template.duration_sec, mfcc=avg_mfcc)
