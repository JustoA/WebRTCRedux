"""
Statistics collection and logging for WebRTC peer connections.
"""
import asyncio
import logging

from aiortc import RTCPeerConnection

from config import RATE

log = logging.getLogger("client.stats")


async def log_stats(pc: RTCPeerConnection, peer_id: str, interval: float = 5.0) -> None:
    """
    Periodically log connection statistics for a peer connection.

    Collects and formats RTT, jitter, packet loss, and packet counts.

    Args:
        pc: RTCPeerConnection to monitor
        peer_id: Identifier for logging
        interval: Logging interval in seconds
    """
    while True:
        await asyncio.sleep(interval)
        try:
            stats = await pc.getStats()
        except Exception:
            return

        metrics = {}

        for stat in stats.values():
            stat_type = getattr(stat, "type", "")
            is_audio = getattr(stat, "kind", "") == "audio"

            # Collect incoming audio metrics
            if stat_type == "inbound-rtp" and is_audio:
                if (jitter := getattr(stat, "jitter", None)) is not None:
                    metrics["jitter"] = jitter / RATE * 1000  # convert ticks to ms
                if (packets_lost := getattr(stat, "packetsLost", None)) is not None:
                    metrics["lost"] = packets_lost
                if (packets_rx := getattr(stat, "packetsReceived", None)) is not None:
                    metrics["rx"] = packets_rx

            # Collect outgoing audio metrics
            elif stat_type == "outbound-rtp" and is_audio:
                if (packets_tx := getattr(stat, "packetsSent", None)) is not None:
                    metrics["tx"] = packets_tx

            # Collect connection quality metrics
            elif stat_type == "candidate-pair" and getattr(stat, "nominated", False):
                if (rtt := getattr(stat, "currentRoundTripTime", None)) is not None:
                    metrics["rtt"] = rtt * 1000  # convert seconds to ms

        # Format and log stats
        if metrics:
            format_metric = {
                "rtt": lambda v: f"RTT={v:.1f}ms",
                "jitter": lambda v: f"jitter={v:.1f}ms",
                "lost": lambda v: f"lost={v}",
                "rx": lambda v: f"rx={v}pkts",
                "tx": lambda v: f"tx={v}pkts",
            }
            ordered_keys = ["rtt", "jitter", "lost", "rx", "tx"]
            formatted_stats = [
                format_metric[k](metrics[k]) for k in ordered_keys if k in metrics
            ]
            log.info("[stats %s] %s", peer_id, "  ".join(formatted_stats))
