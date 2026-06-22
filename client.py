#!/usr/bin/env python3
"""
Multi-peer WebRTC voice chat client (full mesh).

Each peer opens a direct RTCPeerConnection to every other participant;
server.py only relays SDP -- no media touches it.

Usage:
    python client.py [--server ws://HOST:8765] [--mic-device NAME] [--mic-format FMT]

Mic device (auto-detected on macOS/Linux; required on Windows):
  macOS   -- ffmpeg -f avfoundation -list_devices true -i ""
  Linux   -- pactl list short sources
  Windows -- ffmpeg -f dshow -list_devices true -i dummy
"""
from __future__ import annotations

import argparse
import asyncio
from collections import deque
import json
import logging
import platform
import sys

import numpy as np
import sounddevice as sd
import websockets
from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer, MediaRelay
from aiortc.mediastreams import MediaStreamError
from av import AudioResampler

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("client")

STUN   = RTCConfiguration(iceServers=[RTCIceServer(urls="stun:stun.l.google.com:19302")])
RATE   = 48_000   # Hz -- Opus clock rate
CHUNK  = 960      # samples per frame = 20 ms at 48 kHz
MAXBUF = 12       # playback buffer depth (200 ms headroom for asyncio jitter)

_MIC_DEFAULTS = {
    "Darwin":  ("avfoundation", ":0"),
    "Linux":   ("pulse",        "default"),
    "Windows": ("dshow",        None),      # device name required
}


def open_microphone(device: str | None, fmt: str | None):
    system = platform.system()
    default_fmt, default_device = _MIC_DEFAULTS.get(system, (None, None))
    fmt = fmt or default_fmt
    device = device or default_device

    if not fmt:
        raise RuntimeError(f"Unsupported platform {system!r} -- pass --mic-format.")
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
    """
    resampler = AudioResampler(format="s16", layout="mono", rate=RATE)
    buf      = deque(maxlen=MAXBUF)
    leftover = np.array([], dtype=np.int16)

    def callback(outdata: np.ndarray, frames: int, _t, _s) -> None:
        nonlocal leftover
        
        pos = 0
        # Use leftover data first
        if len(leftover) > 0:
            take = min(len(leftover), frames)
            outdata[:take, 0] = leftover[:take]
            leftover = leftover[take:]
            pos = take
        
        # Fill remaining from buffer
        while pos < frames and buf:
            chunk = buf.popleft()
            take = min(len(chunk), frames - pos)
            outdata[pos:pos + take, 0] = chunk[:take]
            pos += take
            
            # Save remainder for next callback
            if take < len(chunk):
                leftover = chunk[take:]
                return
        
        # Pad with silence if buffer exhausted
        if pos < frames:
            outdata[pos:, 0] = 0

    def push(frame) -> None:
        for rf in resampler.resample(frame) or []:
            buf.append(rf.to_ndarray().reshape(-1))

    stream = sd.OutputStream(
        samplerate=RATE, channels=1, dtype="int16",
        latency="low", blocksize=CHUNK, callback=callback,
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


async def log_stats(pc: RTCPeerConnection, peer_id: str, interval: float = 5.0) -> None:
    while True:
        await asyncio.sleep(interval)
        try:
            stats = await pc.getStats()
        except Exception:
            return

        metrics = {}
        
        for s in stats.values():
            s_type = getattr(s, "type", "")
            
            # Extract audio inbound stats
            if s_type == "inbound-rtp" and getattr(s, "kind", "") == "audio":
                if (j := getattr(s, "jitter", None)) is not None:
                    metrics["jitter"] = j / RATE * 1000  # ticks -> ms
                if (lost := getattr(s, "packetsLost", None)) is not None:
                    metrics["lost"] = lost
                if (rx := getattr(s, "packetsReceived", None)) is not None:
                    metrics["rx"] = rx
            
            # Extract audio outbound stats
            elif s_type == "outbound-rtp" and getattr(s, "kind", "") == "audio":
                if (tx := getattr(s, "packetsSent", None)) is not None:
                    metrics["tx"] = tx
            
            # Extract connection quality stats
            elif s_type == "candidate-pair" and getattr(s, "nominated", False):
                if (r := getattr(s, "currentRoundTripTime", None)) is not None:
                    metrics["rtt"] = r * 1000  # s -> ms

        # Format and log stats
        if metrics:
            formatters = {
                "rtt": lambda v: f"RTT={v:.1f}ms",
                "jitter": lambda v: f"jitter={v:.1f}ms",
                "lost": lambda v: f"lost={v}",
                "rx": lambda v: f"rx={v}pkts",
                "tx": lambda v: f"tx={v}pkts",
            }
            parts = [formatters[k](metrics[k]) for k in ["rtt", "jitter", "lost", "rx", "tx"] if k in metrics]
            log.info("[stats %s] %s", peer_id, "  ".join(parts))


async def wait_for_ice(pc: RTCPeerConnection) -> None:
    if pc.iceGatheringState == "complete":
        return
    done = asyncio.Event()
    pc.on("icegatheringstatechange", lambda *_: done.is_set() or done.set())
    await done.wait()


class Mesh:
    """One RTCPeerConnection per remote participant, mic fanned out via MediaRelay."""

    def __init__(self, ws, mic_track) -> None:
        self.ws        = ws
        self.relay     = MediaRelay()
        self.mic_track = mic_track
        self.pcs:  dict[str, RTCPeerConnection] = {}
        self.tasks: set[asyncio.Task]           = set()

    def _spawn(self, coro) -> None:
        t = asyncio.ensure_future(coro)
        self.tasks.add(t)
        t.add_done_callback(self.tasks.discard)

    def _new_pc(self, peer_id: str) -> RTCPeerConnection:
        pc = RTCPeerConnection(configuration=STUN)
        pc.addTrack(self.relay.subscribe(self.mic_track))

        @pc.on("track")
        def on_track(track) -> None:
            if track.kind == "audio":
                self._spawn(play_remote_track(track))

        @pc.on("connectionstatechange")
        async def on_state() -> None:
            log.info("peer %s → %s", peer_id, pc.connectionState)
            if pc.connectionState == "connected":
                self._spawn(log_stats(pc, peer_id))

        self.pcs[peer_id] = pc
        return pc

    async def _send(self, **msg) -> None:
        await self.ws.send(json.dumps(msg))

    async def connect_to(self, peer_id: str) -> None:
        pc = self._new_pc(peer_id)
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        await wait_for_ice(pc)
        await self._send(type="offer", to=peer_id, sdp=pc.localDescription.sdp)

    async def handle_offer(self, peer_id: str, sdp: str) -> None:
        pc = self._new_pc(peer_id)
        await pc.setRemoteDescription(RTCSessionDescription(sdp, "offer"))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        await wait_for_ice(pc)
        await self._send(type="answer", to=peer_id, sdp=pc.localDescription.sdp)

    async def handle_answer(self, peer_id: str, sdp: str) -> None:
        if pc := self.pcs.get(peer_id):
            await pc.setRemoteDescription(RTCSessionDescription(sdp, "answer"))

    async def remove_peer(self, peer_id: str) -> None:
        if pc := self.pcs.pop(peer_id, None):
            await pc.close()
            log.info("peer %s left", peer_id)

    async def close(self) -> None:
        for t in self.tasks:
            t.cancel()
        for pc in self.pcs.values():
            await pc.close()


async def run(server_url: str, mic_device: str | None, mic_format: str | None) -> None:
    player, mic_track = open_microphone(mic_device, mic_format)
    async with websockets.connect(server_url, max_size=None) as ws:
        mesh = Mesh(ws, mic_track)
        try:
            async for raw in ws:
                msg = json.loads(raw)
                match msg["type"]:
                    case "welcome":
                        log.info("Joined as peer %s  (%d others present)", msg["id"], len(msg["peers"]))
                        for pid in msg["peers"]:
                            await mesh.connect_to(pid)
                    case "peer-joined":
                        log.info("Peer %s joined", msg["id"])
                    case "peer-left":
                        await mesh.remove_peer(msg["id"])
                    case "offer":
                        await mesh.handle_offer(msg["from"], msg["sdp"])
                    case "answer":
                        await mesh.handle_answer(msg["from"], msg["sdp"])
        except (asyncio.CancelledError, KeyboardInterrupt):
            pass
        finally:
            await mesh.close()
            log.info("Disconnected.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Multi-peer WebRTC voice chat")
    ap.add_argument("--server",     default="ws://localhost:8765")
    ap.add_argument("--mic-device", default=None)
    ap.add_argument("--mic-format", default=None)
    args = ap.parse_args()
    try:
        asyncio.run(run(args.server, args.mic_device, args.mic_format))
    except KeyboardInterrupt:
        pass
    except Exception as e:
        sys.exit(f"Error: {e}")