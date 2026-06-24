"""
Mesh network management for multi-peer WebRTC connections.

Each peer maintains a direct RTCPeerConnection to every other participant.
"""
import asyncio
import json
import logging

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay

from config import STUN
from audio import play_remote_track
from stats import log_stats
from utils import wait_for_ice

log = logging.getLogger("client.mesh")


class Mesh:
    """
    Manages one RTCPeerConnection per remote participant.

    The microphone track is fanned out to all peers via MediaRelay.
    """

    def __init__(self, ws, mic_track) -> None:
        self.ws = ws
        self.relay = MediaRelay()
        self.mic_track = mic_track
        self.pcs: dict[str, RTCPeerConnection] = {}
        self.tasks: set[asyncio.Task] = set()

    def _spawn(self, coro) -> None:
        """Spawn and track an async task."""
        t = asyncio.ensure_future(coro)
        self.tasks.add(t)
        t.add_done_callback(self.tasks.discard)

    def _new_pc(self, peer_id: str) -> RTCPeerConnection:
        """Create and configure a new peer connection."""
        pc = RTCPeerConnection(configuration=STUN)
        pc.addTrack(self.relay.subscribe(self.mic_track))

        @pc.on("track")
        def on_remote_track(track) -> None:
            if track.kind == "audio":
                self._spawn(play_remote_track(track))

        @pc.on("connectionstatechange")
        def on_connection_change() -> None:
            log.info("peer %s → %s", peer_id, pc.connectionState)
            if pc.connectionState == "connected":
                self._spawn(log_stats(pc, peer_id))

        self.pcs[peer_id] = pc
        return pc

    async def _send(self, **msg) -> None:
        """Send a message to the server."""
        await self.ws.send(json.dumps(msg))

    async def connect_to(self, peer_id: str) -> None:
        """
        Initiate connection to a peer (create and send offer).

        Args:
            peer_id: ID of the peer to connect to
        """
        pc = self._new_pc(peer_id)
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        await wait_for_ice(pc)
        await self._send(type="offer", to=peer_id, sdp=pc.localDescription.sdp)

    async def handle_offer(self, peer_id: str, sdp: str) -> None:
        """
        Handle incoming offer from a peer.

        Args:
            peer_id: ID of the peer sending the offer
            sdp: Session description protocol string
        """
        pc = self._new_pc(peer_id)
        await pc.setRemoteDescription(RTCSessionDescription(sdp, "offer"))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        await wait_for_ice(pc)
        await self._send(type="answer", to=peer_id, sdp=pc.localDescription.sdp)

    async def handle_answer(self, peer_id: str, sdp: str) -> None:
        """
        Handle incoming answer from a peer.

        Args:
            peer_id: ID of the peer sending the answer
            sdp: Session description protocol string
        """
        if pc := self.pcs.get(peer_id):
            await pc.setRemoteDescription(RTCSessionDescription(sdp, "answer"))

    async def remove_peer(self, peer_id: str) -> None:
        """
        Remove a peer and close its connection.

        Args:
            peer_id: ID of the peer to remove
        """
        if pc := self.pcs.pop(peer_id, None):
            await pc.close()
            log.info("peer %s left", peer_id)

    async def close(self) -> None:
        """Close all peer connections and cancel tasks."""
        for t in self.tasks:
            t.cancel()
        for pc in self.pcs.values():
            await pc.close()
