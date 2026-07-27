FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
        pulseaudio-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# ── server ────────────────────────────────────────────────────────────────────
FROM base AS server
COPY server.py .
EXPOSE 8765
CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8765"]


# ── client ────────────────────────────────────────────────────────────────────
# Audio is handled via parec/pacat talking to the host PulseAudio socket.
# network_mode: host is required so WebRTC ICE candidates advertise the real
# host IP — bridge-mode containers would advertise 172.x addresses that remote
# peers cannot reach.
FROM base AS client
COPY client.py .
CMD ["python", "client.py"]