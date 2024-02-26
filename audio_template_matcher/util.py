import math
from collections.abc import Callable
from typing import Optional

import numpy as np
import python_speech_features


def get_mfcc(audio: np.ndarray) -> np.ndarray:
    return python_speech_features.mfcc(audio, winstep=0.02)


def distance_to_probability(distance: float, threshold: float) -> float:
    """Compute detection probability using distance and threshold."""
    return 1 / (1 + math.exp((distance - threshold) / threshold))


def trim_silence(
    vad: Callable[[bytes], float],
    audio: bytes,
    samples_per_chunk=480,
    keep_chunks_before: int = 2,
    keep_chunks_after: int = 2,
    sample_width: int = 2,
) -> bytes:
    bytes_per_chunk = samples_per_chunk * sample_width
    num_chunks = len(audio) // bytes_per_chunk

    first_chunk: Optional[int] = None
    last_chunk: Optional[int] = None

    for chunk_idx in range(num_chunks):
        chunk_offset = chunk_idx * bytes_per_chunk
        chunk = audio[chunk_offset : chunk_offset + bytes_per_chunk]
        if not vad(chunk):
            # Silence
            continue

        if first_chunk is None:
            # First speech
            first_chunk = chunk_idx
        else:
            # Last speech so far
            last_chunk = chunk_idx

    if (first_chunk is None) or (last_chunk is None):
        return audio

    first_chunk = max(0, first_chunk - keep_chunks_before)
    last_chunk = min(chunk_idx, last_chunk + keep_chunks_after)

    return audio[(first_chunk * bytes_per_chunk) : ((last_chunk + 1) * bytes_per_chunk)]
