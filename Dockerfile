FROM python:3.12-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl && rm -rf /var/lib/apt/lists/*

# Python deps — PyTorch CUDA 12.8 (sm_120 support for RTX 5090 Laptop Blackwell) + torchao + hqq + FastAPI
RUN pip install --no-cache-dir \
    torch torchao==0.17.0+cu128 --index-url https://download.pytorch.org/whl/cu128 \
    && pip install --no-cache-dir \
    hqq safetensors soundfile numpy fastapi uvicorn huggingface_hub

# Model einbaken (~8 GB)
RUN mkdir -p /app/models/original && \
    hf download mistralai/Voxtral-4B-TTS-2603 \
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
