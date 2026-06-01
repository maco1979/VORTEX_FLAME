FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /data/soul_memory /data/checkpoints /data/fin_data /data/code_data

ENV VORTEX_MEMORY_DIR=/data/soul_memory
ENV VORTEX_CHECKPOINT_DIR=/data/checkpoints
ENV JEPA_API_PORT=8199

EXPOSE 8199

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8199/health || exit 1

CMD ["python", "jepa_api.py", "--port", "8199", "--memory-dir", "/data/soul_memory"]
