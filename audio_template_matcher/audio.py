try:
    # Use built-in audioop until it's removed in Python 3.13
    import audioop  # pylint: disable=deprecated-module
except ImportError:
    from . import pyaudioop as audioop  # type: ignore[no-redef]


def convert(
    audio_bytes: bytes,
    in_rate: int,
    in_width: int,
    in_channels: int,
    out_rate: int,
    out_width: int,
    out_channels: int,
) -> bytes:
    """Converts sample rate, width, and channels as necessary."""
    width = in_width

    if in_width != out_width:
        # Convert sample width
        audio_bytes = audioop.lin2lin(audio_bytes, in_width, out_width)
        width = out_width

    channels = in_channels
    if in_channels != out_channels:
        # Convert to mono or stereo
        if out_channels == 1:
            audio_bytes = audioop.tomono(audio_bytes, width, 1.0, 1.0)
        elif out_channels == 2:
            audio_bytes = audioop.tostereo(audio_bytes, width, 1.0, 1.0)
        else:
            raise ValueError(f"Cannot convert to channels: {out_channels}")

        channels = out_channels

    if in_rate != out_rate:
        # Resample
        audio_bytes, _ratecv_state = audioop.ratecv(
            audio_bytes,
            width,
            channels,
            in_rate,
            out_rate,
            None,
        )

    return audio_bytes
