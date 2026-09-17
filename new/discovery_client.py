import argparse
import asyncio
import websockets
import json
import sys
import subprocess


# Change this to whatever port range you want.
PORTS_START_AT = 40000

ACTIVE_OUTGOING_CONNECTIONS: dict[int,subprocess.Popen] = {}
ACTIVE_INCOMING_CONNECTIONS: dict[int,subprocess.Popen] = {}


def connect_to_client_with_gstreamer(ip_addr: str, my_id : int,  other_id: int):
    port_to_connect_to = PORTS_START_AT + my_id
    port_to_listen_on = str(PORTS_START_AT + other_id)

    ACTIVE_OUTGOING_CONNECTIONS[other_id] = subprocess.Popen(["gst-launch-1.0", "-v", "pulsesrc", "!",   "audioconvert", "!",   "audioresample", "!",   "opusenc", "audio-type=restricted-lowdelay", "frame-size=5", "complexity=0", "bandwidth=1102", "bitrate=64000", "!",   "rtpopuspay", "!",  "udpsink", f"host={ip_addr}", f"port={port_to_connect_to}", "sync=false", "async=false"])
    ACTIVE_INCOMING_CONNECTIONS[other_id] = subprocess.Popen(["gst-launch-1.0", "-v",  "udpsrc", f"port={port_to_listen_on}", "buffer-size=524288", "caps=\"application/x-rtp, media=audio, clock-rate=48000, encoding-name=OPUS, payload=96\"", "!",   "rtpjitterbuffer", "latency=20", "!", "rtpopusdepay", "!",   "opusdec", "!",   "audioconvert", "!",   "audioresample", "!",  "pulsesink", "buffer-time=20", "latency-time=20", "sync=false"])

async def run(server_url: str) -> None:
    my_id = 0
    async with websockets.connect(server_url, max_size=None) as ws:
        try:
            await ws.send('{"type":"HELLO"}')
            async for raw in ws:
                msg = json.loads(raw)
                match msg["type"]:
                    case "WELCOME":
                        my_id = int(msg["your_id"])
                        print(f"Joined as peer {my_id}  ({len(msg["clients"])} others present)")   
                        client_dict = dict(msg["clients"])
                        for client_ip, client_id in client_dict.items():
                            connect_to_client_with_gstreamer(client_ip, my_id, client_id)
                    case "NEWFRIEND":
                        print(f"Peer {msg["client_id"]} joined")
                        connect_to_client_with_gstreamer(msg["ip"], my_id, msg("id"))
                    case "CLIENTLEFT":
                        print(f"Terminating connections with client id {msg['id']}")
                        ACTIVE_OUTGOING_CONNECTIONS[msg["id"]].kill()
                        ACTIVE_INCOMING_CONNECTIONS[msg["id"]].kill()
        except (asyncio.CancelledError, KeyboardInterrupt):
            pass
        finally:
            await ws.send('{"type":"BYE"}')



if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Multi-peer WebRTC voice chat")
    ap.add_argument("--server",     default="ws://localhost:8765")
    ap.add_argument("--mic-device", default=0,
                    help="sounddevice device index or name (default: system default)")
    args = ap.parse_args()
    try:
        asyncio.run(run(args.server))
    except KeyboardInterrupt:
        pass
    except Exception as e:
        sys.exit(f"Error: {e}")
