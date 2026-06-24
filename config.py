"""
Configuration and constants for WebRTC client.
"""
import platform
from aiortc import RTCConfiguration, RTCIceServer

# WebRTC configuration with STUN server for NAT traversal
STUN = RTCConfiguration(iceServers=[RTCIceServer(urls="stun:stun.l.google.com:19302")])

# Audio parameters
RATE = 48_000  # Hz -- Opus clock rate
CHUNK = 960  # samples per frame = 20 ms at 48 kHz
MAXBUF = 12  # playback buffer depth (200 ms headroom for asyncio jitter)

# Platform-specific microphone defaults
# Maps OS → (ffmpeg_format, default_device)
_MIC_DEFAULTS = {
    "Darwin": ("avfoundation", ":0"),
    "Linux": ("pulse", "default"),
    "Windows": ("dshow", None),  # device name required
}


def get_mic_defaults():
    """Get platform-specific microphone defaults."""
    system = platform.system()
    return _MIC_DEFAULTS.get(system, (None, None))
