"""
Utility functions for WebRTC client.
"""
import asyncio
import logging

from aiortc import RTCPeerConnection

log = logging.getLogger("client.utils")


async def wait_for_ice(pc: RTCPeerConnection) -> None:
    """
    Wait for ICE candidate gathering to complete.

    Args:
        pc: RTCPeerConnection to wait for
    """
    if pc.iceGatheringState == "complete":
        return
    done = asyncio.Event()

    def on_gathering_complete(*_) -> None:
        if not done.is_set():
            done.set()

    pc.on("icegatheringstatechange", on_gathering_complete)
    await done.wait()
