#!/usr/bin/env python3
"""
client.py - Multi-peer WebRTC voice chat client (full mesh).

Connects to server.py (a thin signalling relay) over a WebSocket. On
joining, the server tells this client who's already in the room, and this
client opens a direct RTCPeerConnection — and sends an offer — to each of
them. When someone else joins later, THEY open a connection to everyone
already present, including us, and we just answer.

Net effect: with N participants each client holds (N-1) PeerConnections,
all negotiated directly and in parallel — no server-side media relay, which
is what was causing the join delay in the SFU version.

Your microphone is captured once (via MediaPlayer) and fanned out to every
peer connection using MediaRelay. Each remote participant's audio gets its
own playback stream (see play_remote_track below).

Run:
    pip install aiortc websockets sounddevice numpy av

    python client.py --server ws://HOST:8765 [--mic-device ...] [--mic-format ...]

Microphone device strings (only needed if auto-detection picks the wrong
device — see `--mic-device` / `--mic-format` below):

  macOS   (avfoundation): default is ":0"  -> device 0 = first audio input
            list devices:  ffmpeg -f avfoundation -list_devices true -i ""
  Linux   (pulse):         default is "default"
            list devices:  pactl list short sources
  Windows (dshow):         REQUIRED, e.g. "Microphone Array (Realtek)"
            list devices:  ffmpeg -f dshow -list_devices true -i dummy
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import platform
import sys

import sounddevice as sd
import websockets
from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.contrib.media import MediaPlayer, MediaRelay
from aiortc.mediastreams import MediaStreamError
from av import AudioResampler

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("client")

STUN_CONFIG  = RTCConfiguration(iceServers=[RTCIceServer(urls="stun:stun.l.google.com:19302")])
OUTPUT_RATE  = 48_000   # sample rate we resample remote audio to before playback


# ─────────────────────────────────────────────────────────────────────────────
# Microphone — high-level MediaPlayer (ffmpeg device capture)
# ─────────────────────────────────────────────────────────────────────────────

def open_microphone(device: str | None, fmt: str | None) -> tuple[MediaPlayer, "MediaStreamTrack"]:
    """
    Open the default microphone via aiortc's MediaPlayer, which reads
    through PyAV/ffmpeg's device-capture demuxers. This handles real-time
    pacing and resampling internally, avoiding the manual queue/PTS bugs
    that cause pitch-shifted or choppy audio.
    """
    system = platform.system()

    if fmt is None:
        fmt = {"Darwin": "avfoundation", "Linux": "pulse", "Windows": "dshow"}.get(system)
        if fmt is None:
            raise RuntimeError(f"Unsupported platform: {system}. Pass --mic-device/--mic-format manually.")

    if device is None:
        if system == "Darwin":
            device = ":0"          # no video, audio device 0 (default input)
        elif system == "Linux":
            device = "default"     # pulse/pipewire default source
        elif system == "Windows":
            raise RuntimeError(
                "On Windows you must pass --mic-device, e.g.\n"
                '  --mic-device "Microphone Array (Realtek)"\n'
                "List devices with:  ffmpeg -f dshow -list_devices true -i dummy"
            )

    if fmt == "dshow" and not device.startswith(("audio=", "video=")):
        # dshow requires this prefix, but `ffmpeg -list_devices` doesn't show
        # it — add it automatically so users can paste the name as-is.
        device = f"audio={device}"

    log.info("Opening microphone: format=%s device=%r", fmt, device)
    player = MediaPlayer(device, format=fmt)
    if player.audio is None:
        raise RuntimeError(
            f"MediaPlayer opened {device!r} (format={fmt}) but it has no audio "
            "track. Check the device name / try --mic-device."
        )
    return player, player.audio


# ─────────────────────────────────────────────────────────────────────────────
# Speaker — one output stream per remote participant
# ─────────────────────────────────────────────────────────────────────────────
#
# Rather than hand-rolling a mixer, each remote track gets its own
# sounddevice OutputStream playing to the default speaker. Audio backends
# (CoreAudio, WASAPI, ALSA/PulseAudio) all happily mix multiple simultaneous
# output streams for you, so this gets correct multi-speaker mixing "for
# free" with far less code, at the cost of one extra audio stream per peer
# (fine for typical voice-chat group sizes).

async def play_remote_track(track) -> None:
    """Pull frames from a remote track, resample to s16/mono, and play them."""
    resampler = AudioResampler(format="s16", layout="mono", rate=OUTPUT_RATE)
    stream = sd.OutputStream(samplerate=OUTPUT_RATE, channels=1, dtype="int16")
    stream.start()
    log.info("Playing remote track (stream open)")

    try:
        while True:
            frame = await track.recv()
            for rframe in (resampler.resample(frame) or []):
                pcm = rframe.to_ndarray().reshape(-1)  # int16, mono
                # write() blocks on PortAudio I/O, so run it off the event loop
                await asyncio.to_thread(stream.write, pcm)
    except MediaStreamError:
        pass
    except asyncio.CancelledError:
        pass
    finally:
        stream.stop()
        stream.close()
        log.info("Remote track ended (stream closed)")


# ─────────────────────────────────────────────────────────────────────────────
# Mesh connection manager
# ─────────────────────────────────────────────────────────────────────────────

async def wait_for_ice(pc: RTCPeerConnection) -> None:
    if pc.iceGatheringState == "complete":
        return
    fut: asyncio.Future = asyncio.get_event_loop().create_future()

    @pc.on("icegatheringstatechange")
    def _on_change() -> None:
        if pc.iceGatheringState == "complete" and not fut.done():
            fut.set_result(None)

    await fut


class Mesh:
    """
    Owns one RTCPeerConnection per remote participant. The local mic track
    is captured once and fanned out to each connection via MediaRelay so
    every peer gets its own copy of the stream.
    """

    def __init__(self, ws, mic_track) -> None:
        self.ws = ws
        self.relay = MediaRelay()
        self.mic_track = mic_track
        self.pcs: dict[str, RTCPeerConnection] = {}
        self.tasks: set[asyncio.Task] = set()

    def _new_pc(self, peer_id: str) -> RTCPeerConnection:
        pc = RTCPeerConnection(configuration=STUN_CONFIG)
        pc.addTrack(self.relay.subscribe(self.mic_track))

        @pc.on("track")
        def on_track(track) -> None:
            if track.kind == "audio":
                t = asyncio.ensure_future(play_remote_track(track))
                self.tasks.add(t)
                t.add_done_callback(self.tasks.discard)

        @pc.on("connectionstatechange")
        async def on_state() -> None:
            log.info("peer %s: connection state -> %s", peer_id, pc.connectionState)

        self.pcs[peer_id] = pc
        return pc

    async def connect_to(self, peer_id: str) -> None:
        """We are joining and peer_id is already in the room: send an offer."""
        pc = self._new_pc(peer_id)
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        await wait_for_ice(pc)
        await self.ws.send(json.dumps({"type": "offer", "to": peer_id, "sdp": pc.localDescription.sdp}))
        log.info("Sent offer to peer %s", peer_id)

    async def handle_offer(self, peer_id: str, sdp: str) -> None:
        """peer_id is joining and sent us an offer: answer it."""
        pc = self._new_pc(peer_id)
        await pc.setRemoteDescription(RTCSessionDescription(sdp, "offer"))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        await wait_for_ice(pc)
        await self.ws.send(json.dumps({"type": "answer", "to": peer_id, "sdp": pc.localDescription.sdp}))
        log.info("Answered offer from peer %s", peer_id)

    async def handle_answer(self, peer_id: str, sdp: str) -> None:
        pc = self.pcs.get(peer_id)
        if pc is not None:
            await pc.setRemoteDescription(RTCSessionDescription(sdp, "answer"))
            log.info("Connected to peer %s", peer_id)

    async def remove_peer(self, peer_id: str) -> None:
        pc = self.pcs.pop(peer_id, None)
        if pc is not None:
            await pc.close()
            log.info("Closed connection to peer %s (left)", peer_id)

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
                mtype = msg["type"]

                if mtype == "welcome":
                    my_id = msg["id"]
                    log.info("Connected as peer %s", my_id)
                    for peer_id in msg["peers"]:
                        await mesh.connect_to(peer_id)

                elif mtype == "peer-joined":
                    log.info("Peer %s joined — waiting for their offer", msg["id"])

                elif mtype == "peer-left":
                    await mesh.remove_peer(msg["id"])

                elif mtype == "offer":
                    await mesh.handle_offer(msg["from"], msg["sdp"])

                elif mtype == "answer":
                    await mesh.handle_answer(msg["from"], msg["sdp"])

        except (asyncio.CancelledError, KeyboardInterrupt):
            pass
        finally:
            await mesh.close()
            player.audio.stop() if hasattr(player.audio, "stop") else None
            log.info("Disconnected.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Multi-peer WebRTC voice chat client")
    ap.add_argument("--server", default="ws://localhost:8765", help="Signalling server URL")
    ap.add_argument("--mic-device", default=None, help="Override microphone device string")
    ap.add_argument("--mic-format", default=None, help="Override ffmpeg input format (avfoundation/pulse/alsa/dshow)")
    args = ap.parse_args()

    try:
        asyncio.run(run(args.server, args.mic_device, args.mic_format))
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)