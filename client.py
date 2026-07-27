#!/usr/bin/env python3
"""
Multi-peer WebRTC voice chat client (full mesh).

Each peer opens a direct RTCPeerConnection to every other participant;
server.py only relays SDP — no media touches it.

Audio is handled entirely through PulseAudio via the `parec` and `pacat`
command-line tools (from pulseaudio-utils) — no PortAudio/sounddevice needed.

Usage:
    python client.py [--server ws://HOST:8765] [--mic-device SOURCE_NAME]

List available PulseAudio sources (mic devices):
    pactl list short sources
"""
from __future__ import annotations

import argparse
import asyncio
import fractions
import json
import logging
import sys

import websockets
from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription, AudioStreamTrack
from aiortc.contrib.media import MediaRelay
from aiortc.mediastreams import MediaStreamError
from av import AudioFrame, AudioResampler

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("client")

STUN   = RTCConfiguration(iceServers=[RTCIceServer(urls="stun:stun.l.google.com:19302")])
RATE   = 48_000   # Hz — Opus clock rate
CHUNK  = 960      # samples per frame = 20 ms at 48 kHz


class MicrophoneTrack(AudioStreamTrack):
    """
    Captures audio from PulseAudio via `parec` and exposes it as a WebRTC
    audio track. Raw s16le PCM is read from parec's stdout one frame at a time.
    """

    def __init__(self, device: str | None = None) -> None:
        super().__init__()
        self._pts  = 0
        self._proc: asyncio.subprocess.Process | None = None
        self._device = device

    async def _start(self) -> None:
        args = ["parec", "--format=s16le", f"--rate={RATE}", "--channels=1", "--latency-msec=20"]
        if self._device:
            args += [f"--device={self._device}"]
        self._proc = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE)
        log.info("Mic open  device=%r", self._device or "default")

    async def recv(self) -> AudioFrame:
        if self._proc is None:
            await self._start()
        data = await self._proc.stdout.readexactly(CHUNK * 2)  # 2 bytes per int16 sample
        frame = AudioFrame(format="s16", layout="mono", samples=CHUNK)
        frame.pts         = self._pts
        frame.sample_rate = RATE
        frame.time_base   = fractions.Fraction(1, RATE)
        frame.planes[0].update(data)
        self._pts += CHUNK
        return frame

    def stop(self) -> None:
        if self._proc:
            self._proc.terminate()


async def play_remote_track(track) -> None:
    """
    Receive frames from a remote track and play them via `pacat`.

    1. Drain ICE-backlog frames first (they arrive instantly; live frames
       take ~20 ms). We discard them before opening pacat so the backlog
       never reaches the speaker.
    2. Open pacat and write live PCM frames to its stdin continuously.

    pacat handles its own internal buffer and timing — no deque or callback needed.
    """
    resampler = AudioResampler(format="s16", layout="mono", rate=RATE)

    # 1. Drain stale frames accumulated during ICE negotiation.
    loop, n = asyncio.get_running_loop(), 0
    while True:
        t0, frame = loop.time(), await track.recv()
        if loop.time() - t0 >= 0.010:   # ≥ 10 ms → live audio
            first_live = frame
            break
        n += 1
    if n:
        log.info("Drained %d stale frames (~%d ms backlog)", n, n * 20)

    # 2. Open pacat and stream PCM to it.
    proc = await asyncio.create_subprocess_exec(
        "pacat", "--format=s16le", f"--rate={RATE}", "--channels=1", "--latency-msec=40",
        stdin=asyncio.subprocess.PIPE,
    )
    log.info("Speaker open")

    def write(frame) -> None:
        for rf in resampler.resample(frame) or []:
            proc.stdin.write(bytes(rf.planes[0]))

    try:
        write(first_live)
        while True:
            frame = await track.recv()
            write(frame)
            await proc.stdin.drain()
    except (MediaStreamError, asyncio.CancelledError):
        pass
    finally:
        proc.stdin.close()
        proc.terminate()
        log.info("Speaker closed")


async def log_stats(pc: RTCPeerConnection, peer_id: str, interval: float = 5.0) -> None:
    while True:
        await asyncio.sleep(interval)
        try:
            stats = await pc.getStats()
        except Exception:
            return

        rtt = jitter = lost = rx = tx = None
        for s in stats.values():
            match getattr(s, "type", ""):
                case "inbound-rtp" if getattr(s, "kind", "") == "audio":
                    j      = getattr(s, "jitter", None)
                    jitter = j / RATE * 1000 if j is not None else None  # ticks → ms
                    lost   = getattr(s, "packetsLost",     None)
                    rx     = getattr(s, "packetsReceived", None)
                case "outbound-rtp" if getattr(s, "kind", "") == "audio":
                    tx = getattr(s, "packetsSent", None)
                case "candidate-pair" if getattr(s, "nominated", False):
                    r   = getattr(s, "currentRoundTripTime", None)
                    rtt = r * 1000 if r is not None else None             # s → ms

        parts = [
            f"RTT={rtt:.1f}ms"       if rtt    is not None else "",
            f"jitter={jitter:.1f}ms" if jitter is not None else "",
            f"lost={lost}"           if lost   is not None else "",
            f"rx={rx}pkts"           if rx     is not None else "",
            f"tx={tx}pkts"           if tx     is not None else "",
        ]
        if line := "  ".join(p for p in parts if p):
            log.info("[stats %s] %s", peer_id, line)


async def wait_for_ice(pc: RTCPeerConnection) -> None:
    if pc.iceGatheringState == "complete":
        return
    done = asyncio.Event()
    pc.on("icegatheringstatechange", lambda *_: pc.iceGatheringState == "complete" and done.set())
    await done.wait()


class Mesh:
    """One RTCPeerConnection per remote participant, mic fanned out via MediaRelay."""

    def __init__(self, ws, mic_track: MicrophoneTrack) -> None:
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


async def run(server_url: str, mic_device: str | None) -> None:
    mic = MicrophoneTrack(device=mic_device)
    async with websockets.connect(server_url, max_size=None) as ws:
        mesh = Mesh(ws, mic)
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
            mic.stop()
            log.info("Disconnected.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Multi-peer WebRTC voice chat")
    ap.add_argument("--server",     default="ws://localhost:8765")
    ap.add_argument("--mic-device", default=None,
                    help="PulseAudio source name (default: PulseAudio default source). "
                         "List sources with: pactl list short sources")
    args = ap.parse_args()
    try:
        asyncio.run(run(args.server, args.mic_device))
    except KeyboardInterrupt:
        pass
    except Exception as e:
        sys.exit(f"Error: {e}")