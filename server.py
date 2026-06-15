#!/usr/bin/env python3
"""
server.py - WebSocket signalling relay for a full-mesh voice chat.

This server does NOT run any WebRTC PeerConnections itself — it only
assigns each connecting client an id and relays JSON signalling messages
between specific clients (by id). All audio flows directly peer-to-peer.

Protocol (JSON over WebSocket)
───────────────────────────────
  Server → Client   {"type": "welcome", "id": "<my_id>", "peers": ["<id>", ...]}
                       Sent once on connect. `peers` lists everyone already
                       in the room — the new client is responsible for
                       initiating a connection to each of them.

  Server → Client   {"type": "peer-joined", "id": "<id>"}
                       Sent to existing clients when someone new connects.
                       No action needed — the new peer initiates.

  Server → Client   {"type": "peer-left", "id": "<id>"}
                       Sent when a client disconnects, so others can close
                       their PeerConnection to it.

  Client → Server   {"type": "offer"|"answer", "to": "<id>", "sdp": "..."}
                       Server adds "from": "<sender_id>" and forwards
                       verbatim to the target client.

Run:
    pip install websockets
    python server.py [--host 0.0.0.0] [--port 8765]
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import logging

import websockets

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("signalling")

clients: dict[str, websockets.WebSocketServerProtocol] = {}
_id_counter = itertools.count(1)


async def handler(ws: websockets.WebSocketServerProtocol) -> None:
    my_id = str(next(_id_counter))
    clients[my_id] = ws
    log.info("%s connected (total=%d)", my_id, len(clients))

    # Tell the newcomer who's already here.
    await ws.send(json.dumps({"type": "welcome", "id": my_id, "peers": [p for p in clients if p != my_id]}))

    # Tell everyone else about the newcomer (informational only — the
    # newcomer is the one who initiates the connection).
    await broadcast({"type": "peer-joined", "id": my_id}, exclude=my_id)

    try:
        async for raw in ws:
            msg = json.loads(raw)
            target = msg.get("to")
            target_ws = clients.get(target)
            if target_ws is None:
                log.warning("%s: unknown target %r", my_id, target)
                continue
            msg["from"] = my_id
            del msg["to"]
            await target_ws.send(json.dumps(msg))
    except websockets.ConnectionClosed:
        pass
    finally:
        clients.pop(my_id, None)
        log.info("%s disconnected (total=%d)", my_id, len(clients))
        await broadcast({"type": "peer-left", "id": my_id})


async def broadcast(msg: dict, exclude: str | None = None) -> None:
    data = json.dumps(msg)
    for pid, ws in list(clients.items()):
        if pid == exclude:
            continue
        try:
            await ws.send(data)
        except websockets.ConnectionClosed:
            pass


async def main(host: str, port: int) -> None:
    async with websockets.serve(handler, host, port, max_size=None):
        log.info("Signalling server listening on ws://%s:%d", host, port)
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="WebRTC mesh signalling relay")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    try:
        asyncio.run(main(args.host, args.port))
    except KeyboardInterrupt:
        pass