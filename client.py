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
import json
import logging
import sys

import websockets

from audio import open_microphone
from mesh import Mesh

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("client")


async def run(server_url: str, mic_device: str | None, mic_format: str | None) -> None:
    """
    Main client loop: connect to server and manage mesh network.

    Args:
        server_url: WebSocket server URL
        mic_device: Microphone device name (optional, platform-dependent)
        mic_format: Microphone format (optional, platform-dependent)
    """
    player, mic_track = open_microphone(mic_device, mic_format)
    async with websockets.connect(server_url, max_size=None) as ws:
        mesh = Mesh(ws, mic_track)
        try:
            async for raw in ws:
                msg = json.loads(raw)
                match msg["type"]:
                    case "welcome":
                        log.info(
                            "Joined as peer %s  (%d others present)",
                            msg["id"],
                            len(msg["peers"]),
                        )
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
    ap.add_argument("--server", default="ws://localhost:8765")
    ap.add_argument("--mic-device", default=None)
    ap.add_argument("--mic-format", default=None)
    args = ap.parse_args()
    try:
        asyncio.run(run(args.server, args.mic_device, args.mic_format))
    except KeyboardInterrupt:
        pass
    except Exception as e:
        sys.exit(f"Error: {e}")