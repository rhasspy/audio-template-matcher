"""Tests for audio_template_matcher"""
import itertools
from pathlib import Path

import numpy as np
from pymicro_features import MicroFrontend
from pysilero_vad import SileroVoiceActivityDetector

from audio_template_matcher import SpeakerData, TemplateMatcher

_DIR = Path(__file__).parent
_AUDIO_DIR = _DIR / "audio"
_BAD_THRESHOLD = 0.1

_VAD = SileroVoiceActivityDetector()
_FRONTEND = MicroFrontend()
_FRONTEND_SAMPLES_PER_CHUNK = 480
_FRONTEND_BYTES_PER_CHUNK = _FRONTEND_SAMPLES_PER_CHUNK * 2


def _features(audio_bytes: bytes) -> np.ndarray:
    features = []

    i = 0
    while (i + _FRONTEND_BYTES_PER_CHUNK) < len(audio_bytes):
        chunk = audio_bytes[i : i + _FRONTEND_BYTES_PER_CHUNK]
        chunk_features = _FRONTEND.ProcessSamples(chunk).features
        if chunk_features:
            features.append(chunk_features)
        i += _FRONTEND_BYTES_PER_CHUNK

    return np.asarray(features, dtype=np.float32)


def _vad(audio: bytes) -> bool:
    return _VAD(audio) > 0.8


def _vad_reset() -> None:
    _VAD.reset()


def test_tune() -> None:
    data_0 = SpeakerData.from_dir(_AUDIO_DIR / "speaker_0")
    assert data_0.train and data_0.positive and data_0.negative

    data_1 = SpeakerData.from_dir(_AUDIO_DIR / "speaker_1")
    assert data_1.train and data_1.positive and data_1.negative

    all_samples = list(
        itertools.chain(
            data_0.positive, data_0.negative, data_1.positive, data_1.negative
        )
    )

    matcher = TemplateMatcher.from_data(
        [data_0, data_1], _features, vad=_vad, vad_reset=_vad_reset
    )
    eval_before = matcher.evaluate(
        all_samples,
        distance_threshold=_BAD_THRESHOLD,  # force some false positives/negatives
    )
    score_before = eval_before.false_negatives + eval_before.false_positives
    assert score_before > 0

    tuned_distance_threshold = matcher.tune(all_samples, step=10)
    eval_after = matcher.evaluate(
        all_samples, distance_threshold=tuned_distance_threshold
    )
    score_after = eval_after.false_negatives + eval_after.false_positives
    assert score_after < score_before  # should improve


def test_speaker_id() -> None:
    """Test that we can distinguish speakers with positive examples after training."""
    speakers = ("speaker_0", "speaker_1")
    speaker_data = [SpeakerData.from_dir(_AUDIO_DIR / speaker) for speaker in speakers]
    matcher = TemplateMatcher.from_data(
        speaker_data, _features, vad=_vad, vad_reset=_vad_reset
    )

    for data in speaker_data:
        assert data.positive
        samples = list(data.positive)

        for sample in samples:
            assert (
                matcher.match_wav(sample.wav_path, probabilty_threshold=0) == data.name
            ), sample
