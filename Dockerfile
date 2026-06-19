FROM python:3.12-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl && rm -rf /var/lib/apt/lists/*

# Python deps — PyTorch CUDA 12.4 + torchao + hqq + FastAPI
RUN pip install --no-cache-dir \
    torch torchao --index-url https://download.pytorch.org/whl/cu124 \
    && pip install --no-cache-dir \
    hqq safetensors soundfile numpy fastapi uvicorn huggingface_hub

# Model einbaken (~8 GB)
RUN mkdir -p /app/models/original && \
    huggingface-cli download mistralai/Voxtral-4B-TTS-2603 \
    --local-dir /app/models/original

# App code
COPY src/ /app/src/

WORKDIR /app
EXPOSE 8000

CMD ["python", "src/serve.py", \
     "--model-dir", "/app/models/original", \
     "--port", "8000", \
     "--host", "0.0.0.0", \
     "--flow-steps", "8"]
