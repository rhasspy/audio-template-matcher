"""Tests for audio_template_matcher"""
import itertools
from pathlib import Path

import numpy as np
from pysilero_vad import SileroVoiceActivityDetector
from python_speech_features import mfcc

from audio_template_matcher import SpeakerData, TemplateMatcher

_DIR = Path(__file__).parent
_AUDIO_DIR = _DIR / "audio"
_BAD_THRESHOLD = 0.1

_VAD = SileroVoiceActivityDetector()


def _features(audio: np.ndarray) -> np.ndarray:
    return mfcc(audio, winstep=0.02)


def _vad(audio: bytes) -> bool:
    return _VAD(audio) >= 0.5


def _vad_reset() -> None:
    _VAD.reset()


def test_tune() -> None:
    data_0 = SpeakerData.from_dir(_AUDIO_DIR / "speaker_0")
    assert data_0.train and data_0.positive and data_0.negative

    data_3 = SpeakerData.from_dir(_AUDIO_DIR / "speaker_3")
    assert data_3.train and data_3.positive and data_3.negative

    all_samples = list(
        itertools.chain(
            data_0.positive, data_0.negative, data_3.positive, data_3.negative
        )
    )

    matcher = TemplateMatcher.from_data(
        [data_0, data_3], _features, vad=_vad, vad_reset=_vad_reset
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
    """Test that we can distinguish speakers, even with negative samples."""
    speakers = ("speaker_0", "speaker_1", "speaker_2", "speaker_3")
    speaker_data = [SpeakerData.from_dir(_AUDIO_DIR / speaker) for speaker in speakers]
    matcher = TemplateMatcher.from_data(
        speaker_data, _features, vad=_vad, vad_reset=_vad_reset
    )

    for data in speaker_data:
        assert data.positive
        samples = list(data.positive)

        if data.negative:
            samples.extend(data.negative)

        for sample in samples:
            assert (
                matcher.match_wav(sample.wav_path, probabilty_threshold=0) == data.name
            )
