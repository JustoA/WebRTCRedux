"""
Audio input and output handling for WebRTC client.
"""
from __future__ import annotations

import asyncio
import logging
from collections import deque

import numpy as np
import sounddevice as sd
from aiortc.contrib.media import MediaPlayer
from aiortc.mediastreams import MediaStreamError
from av import AudioResampler

from config import RATE, CHUNK, MAXBUF
from config import get_mic_defaults

log = logging.getLogger("client.audio")


def open_microphone(device: str | None, fmt: str | None):
    """
    Open microphone for audio input using ffmpeg.

    Args:
        device: Microphone device name (auto-detected on macOS/Linux; required on Windows)
        fmt: Audio format (auto-detected based on OS)

    Returns:
        (player, audio_track) tuple

    Raises:
        RuntimeError: If platform unsupported or device unavailable
    """
    default_fmt, default_device = get_mic_defaults()
    fmt = fmt or default_fmt
    device = device or default_device

    if not fmt:
        raise RuntimeError(
            f"Unsupported platform. Please pass --mic-format. "
            "Examples: avfoundation (macOS), pulse (Linux), dshow (Windows)"
        )
    if not device:
        raise RuntimeError(
            'On Windows --mic-device is required, e.g. "Microphone (Realtek)"\n'
            "List devices:  ffmpeg -f dshow -list_devices true -i dummy"
        )
    if fmt == "dshow" and not device.startswith("audio="):
        device = f"audio={device}"

    log.info("Opening mic  format=%s  device=%r", fmt, device)
    player = MediaPlayer(device, format=fmt)
    if player.audio is None:
        raise RuntimeError(f"No audio stream from {device!r} (format={fmt})")
    return player, player.audio


async def play_remote_track(track) -> None:
    """
    Receive frames from a remote track and play them through the speaker.

    The bounded deque drops the oldest frames if asyncio falls behind, so
    latency is self-correcting rather than growing without bound.

    Args:
        track: Remote audio track from peer
    """
    resampler = AudioResampler(format="s16", layout="mono", rate=RATE)
    buf = deque(maxlen=MAXBUF)
    leftover = np.array([], dtype=np.int16)

    def callback(outdata: np.ndarray, frames: int, _t, _s) -> None:
        nonlocal leftover

        write_pos = 0

        # Fill from leftover data first
        if leftover.size:
            samples_to_write = min(len(leftover), frames)
            outdata[:samples_to_write, 0] = leftover[:samples_to_write]
            leftover = leftover[samples_to_write:]
            write_pos = samples_to_write

        # Fill remaining space from buffer
        while write_pos < frames and buf:
            chunk = buf.popleft()
            samples_to_write = min(len(chunk), frames - write_pos)
            outdata[write_pos : write_pos + samples_to_write, 0] = chunk[:samples_to_write]
            write_pos += samples_to_write

            # Save excess data for next callback
            if samples_to_write < len(chunk):
                leftover = chunk[samples_to_write:]
                return

        # Pad remaining space with silence
        if write_pos < frames:
            outdata[write_pos:, 0] = 0

    def push(frame) -> None:
        for rf in resampler.resample(frame) or []:
            buf.append(rf.to_ndarray().reshape(-1))

    stream = sd.OutputStream(
        samplerate=RATE,
        channels=1,
        dtype="int16",
        latency="low",
        blocksize=CHUNK,
        callback=callback,
    )
    stream.start()
    log.info("Speaker open  (hw latency=%.0f ms)", stream.latency * 1000)
    try:
        while True:
            push(await track.recv())
    except (MediaStreamError, asyncio.CancelledError):
        pass
    finally:
        stream.stop()
        stream.close()
