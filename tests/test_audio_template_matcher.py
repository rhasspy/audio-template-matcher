"""Tests for audio_template_matcher"""
from pathlib import Path

from audio_template_matcher import TemplateMatcher

_DIR = Path(__file__).parent
_AUDIO_DIR = _DIR / "audio"
_BAD_THRESHOLD = 0.1


def test_tune() -> None:
    speaker_dir = _AUDIO_DIR / "speaker_0"
    matcher = TemplateMatcher.from_wav_dirs({"speaker_0": speaker_dir / "train"})
    positive_dir = speaker_dir / "positive"
    negative_dir = speaker_dir / "negative"

    eval_before = matcher.evaluate(
        positive_dir, negative_dir, distance_threshold=_BAD_THRESHOLD
    )
    score_before = eval_before.false_negatives + eval_before.false_positives
    assert score_before > 0

    tuned_distance_threshold = matcher.tune(positive_dir, negative_dir, step=10)
    eval_after = matcher.evaluate(
        positive_dir, negative_dir, distance_threshold=tuned_distance_threshold
    )
    score_after = eval_after.false_negatives + eval_after.false_positives
    assert score_after < score_before  # should improve


def test_speaker_id() -> None:
    speakers = ("speaker_0", "speaker_1", "speaker_2")
    matcher = TemplateMatcher.from_wav_dirs(
        {key: _AUDIO_DIR / key / "train" for key in speakers}
    )

    for key in speakers:
        wav_path = _AUDIO_DIR / key / "positive" / "1.wav"
        assert matcher.match_wav(wav_path, probabilty_threshold=0) == key
