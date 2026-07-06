FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# server
FROM base AS server
COPY server.py .
EXPOSE 8765
CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8765"]


# client
FROM base AS client
COPY client.py .
CMD ["python", "client.py", \
     "--mic-format", "pulse", \
     "--mic-device", "default"]
