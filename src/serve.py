"""
Voxtral-4B-TTS FastAPI server — drop-in replacement for Orpheus TTS.

OpenAI-compatible /v1/audio/speech endpoint.
Uses int4 HQQ quantized backbone + torch.compile for 57 fps on RTX 3090.

Usage:
    python serve.py                          # Start on port 5005
    python serve.py --port 8000              # Custom port
    python serve.py --no-compile             # Skip torch.compile (faster startup)

API:
    POST /v1/audio/speech  {"input": "text", "voice": "tara"}  → audio/wav
    GET  /v1/audio/voices  → {"voices": [...]}
"""

import argparse
import io
import os
import sys
import time
import wave
import struct
import warnings

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")
os.environ["TORCHDYNAMO_VERBOSE"] = "0"

sys.path.insert(0, str(Path(__file__).parent))

from model import VoxtralConfig
from generate import TekkenTokenizer
from generate_fast import generate_speech_fast, enable_static_cache
from torchao_inference import load_model_int4
from load_model import load_original_model

# ─── Voice Mapping: Orpheus names → Voxtral names ─────────────────────

VOICE_MAP = {
    # English
    "tara": "neutral_female",
    "leah": "cheerful_female",
    "jess": "casual_female",
    "leo": "neutral_male",
    "dan": "casual_male",
    "mia": "neutral_female",
    "zac": "neutral_male",
    "zoe": "cheerful_female",
    # French
    "pierre": "fr_male",
    "amelie": "fr_female",
    "marie": "fr_female",
    # German
    "jana": "de_female",
    "thomas": "de_male",
    "max": "de_male",
    # Korean (no Voxtral equivalent — fallback to English)
    "유나": "neutral_female",
    "준서": "neutral_male",
    # Hindi
    "ऋतिका": "hi_female",
    # Mandarin (no Voxtral equivalent — fallback to English)
    "长乐": "neutral_female",
    "白芷": "neutral_female",
    # Spanish
    "javi": "es_male",
    "sergio": "es_male",
    "maria": "es_female",
    # Italian
    "pietro": "it_male",
    "giulia": "it_female",
    "carlo": "it_male",
    # Direct Voxtral names also accepted
    "neutral_female": "neutral_female",
    "neutral_male": "neutral_male",
    "cheerful_female": "cheerful_female",
    "casual_female": "casual_female",
    "casual_male": "casual_male",
    "fr_male": "fr_male",
    "fr_female": "fr_female",
    "de_male": "de_male",
    "de_female": "de_female",
    "es_male": "es_male",
    "es_female": "es_female",
    "it_male": "it_male",
    "it_female": "it_female",
    "pt_male": "pt_male",
    "pt_female": "pt_female",
    "nl_male": "nl_male",
    "nl_female": "nl_female",
    "ar_male": "ar_male",
    "hi_male": "hi_male",
    "hi_female": "hi_female",
}

DEFAULT_VOICE = "tara"


# ─── Request/Response Models ──────────────────────────────────────────

class SpeechRequest(BaseModel):
    input: str
    model: str = "voxtral"
    voice: str = DEFAULT_VOICE
    response_format: str = "wav"
    speed: float = 1.0


# ─── Audio Utils ──────────────────────────────────────────────────────

def numpy_to_wav_bytes(audio_np: np.ndarray, sample_rate: int = 48000) -> bytes:
    """Convert numpy float array to 16-bit PCM WAV bytes."""
    audio_int16 = (audio_np * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return buf.getvalue()


def split_text_for_batching(text: str, max_chars: int = 1000) -> list:
    """Split long text into sentence-based batches."""
    if len(text) <= max_chars:
        return [text]

    sentences = []
    for sep in [". ", "! ", "? ", "; ", ", "]:
        if sep in text:
            parts = text.split(sep)
            current = ""
            for part in parts:
                candidate = current + sep + part if current else part
                if len(candidate) > max_chars and current:
                    sentences.append(current + sep.rstrip())
                    current = part
                else:
                    current = candidate
            if current:
                sentences.append(current)
            return sentences

    # No sentence boundaries — split at max_chars
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


# ─── Global Model State ──────────────────────────────────────────────

model = None
tokenizer = None
voice_dir = None
_flow_steps = 8


def load_model_global(model_dir: str, device: str = "cuda", compile: bool = True, use_int4: bool = True):
    """Load and optimize the Voxtral model."""
    global model, tokenizer, voice_dir

    model_path = Path(model_dir)
    voice_dir = str(model_path / "voice_embedding")

    t0 = time.time()
    if use_int4:
        print(f"[serve] Loading Voxtral int4 from {model_dir}...")
        model = load_model_int4(str(model_path), device=device)
    else:
        print(f"[serve] Loading Voxtral BF16 from {model_dir}...")
        model = load_original_model(str(model_path), device=device)
    tokenizer = TekkenTokenizer(str(model_path / "tekken.json"))

    # Static cache for pre-allocated KV buffers
    enable_static_cache(model, max_seq_len=800)

    if compile:
        print("[serve] Compiling (first request will be slow)...")
        model.backbone = torch.compile(model.backbone, mode="default", fullgraph=False)
        model.acoustic.predict_velocity = torch.compile(
            model.acoustic.predict_velocity, mode="default", fullgraph=False)

        # Warmup to trigger compilation
        with torch.inference_mode():
            generate_speech_fast(
                model, tokenizer, "Warmup.",
                voice_name="neutral_female", voice_dir=voice_dir,
                max_frames=10, device=device, flow_steps=3, cfg_alpha=1.2)

    vram = torch.cuda.memory_allocated() / 1e9
    print(f"[serve] Ready in {time.time()-t0:.1f}s | VRAM: {vram:.2f} GB")


# ─── FastAPI App ──────────────────────────────────────────────────────

app = FastAPI(title="Voxtral TTS", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/v1/audio/voices")
def list_voices():
    return {"status": "ok", "voices": sorted(VOICE_MAP.keys())}


@app.post("/v1/audio/speech")
def create_speech(req: SpeechRequest):
    if not req.input or not req.input.strip():
        raise HTTPException(status_code=400, detail="Input text is empty")

    # Map voice name
    voxtral_voice = VOICE_MAP.get(req.voice, VOICE_MAP.get(DEFAULT_VOICE))

    # Split long text into batches
    text_batches = split_text_for_batching(req.input.strip())

    all_audio = []
    with torch.inference_mode():
        for batch_text in text_batches:
            try:
                audio, gen_time = generate_speech_fast(
                    model, tokenizer, batch_text,
                    voice_name=voxtral_voice, voice_dir=voice_dir,
                    max_frames=500, device="cuda",
                    flow_steps=_flow_steps, cfg_alpha=1.2)
                if len(audio) > 0:
                    all_audio.append(audio)
            except RuntimeError:
                continue  # Skip failed batches

    if not all_audio:
        raise HTTPException(status_code=500, detail="Audio generation failed")

    # Concatenate batches
    if len(all_audio) > 1:
        combined = np.concatenate(all_audio)
    else:
        combined = all_audio[0]

    wav_bytes = numpy_to_wav_bytes(combined)
    return Response(content=wav_bytes, media_type="audio/wav")


@app.post("/speak")
def speak_legacy(req: dict):
    """Legacy endpoint for backwards compatibility."""
    text = req.get("text", "")
    voice = req.get("voice", DEFAULT_VOICE)
    speech_req = SpeechRequest(input=text, voice=voice)
    return create_speech(speech_req)


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{
            "id": "voxtral-4b-tts",
            "object": "model",
            "owned_by": "mistral",
            "permissions": [],
            "created": 1710000000,
            "root": "voxtral-4b-tts",
            "parent": None
        }]
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": "voxtral-4b-tts",
            "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 2)}


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Voxtral TTS Server")
    parser.add_argument("--model-dir", type=str,
                        default=str(Path(__file__).parent.parent / "models" / "original"))
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--no-compile", action="store_true", help="Skip torch.compile")
    parser.add_argument("--bf16", action="store_true", help="Use BF16 instead of int4 (more VRAM, better quality)")
    parser.add_argument("--flow-steps", type=int, default=8, help="ODE solver steps (8=best quality, 3=fastest)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    global _flow_steps
    _flow_steps = args.flow_steps

    load_model_global(args.model_dir, device=args.device, compile=not args.no_compile, use_int4=not args.bf16)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
