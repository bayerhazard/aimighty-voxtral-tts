"""
Voxtral-4B-TTS text-to-speech generation.
Supports both original BF16 and TurboQuant-quantized models.

Usage:
    python generate.py --text "Hello world" --voice cheerful_female --output hello.wav
    python generate.py --text "Hello world" --quantized models/quantized/turboquant
"""

import argparse
import json
import time
import torch
import soundfile as sf
import numpy as np
from pathlib import Path

from model import VoxtralConfig
from load_model import load_original_model, load_quantized_model


class TekkenTokenizer:
    """Minimal Tekken tokenizer wrapper using tiktoken."""
    def __init__(self, tokenizer_path: str):
        import tiktoken
        import base64

        with open(tokenizer_path) as f:
            data = json.load(f)

        # Build BPE ranks from base64-encoded token_bytes
        ranks = {}
        for item in data.get("vocab", []):
            token_bytes = base64.b64decode(item["token_bytes"])
            ranks[token_bytes] = item["rank"]

        # Special tokens occupy model positions 0..999 (raw rank)
        config = data.get("config", {})
        num_special = config.get("default_num_special_tokens", 1000)

        self.special_tokens = {}
        for st in data.get("special_tokens", []):
            # tiktoken needs unique IDs that don't collide with BPE ranks
            # We map them to (num_bpe_tokens + rank) for tiktoken's internal use
            # but the ENCODE method won't typically hit these (we use raw IDs directly)
            self.special_tokens[st["token_str"]] = len(ranks) + st["rank"]

        # Tekken uses this pattern (from mistral_common)
        pat_str = data.get("config", {}).get("pattern",
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        )

        self.enc = tiktoken.Encoding(
            name="tekken",
            pat_str=pat_str,
            mergeable_ranks=ranks,
            special_tokens=self.special_tokens,
        )
        self._num_special_tokens = num_special

    def encode(self, text: str) -> list:
        """Encode text to token IDs, offset by num_special_tokens for model embedding."""
        tokens = self.enc.encode(text, allowed_special="all")
        # BPE tokens occupy model positions 1000..131071 (offset by num_special_tokens)
        # Clamp to vocab_size-1: tekken has 150K entries but model embedding is 131072
        max_id = 131071
        return [min(t + self._num_special_tokens, max_id) for t in tokens]

    def decode(self, tokens: list) -> str:
        return self.enc.decode(tokens)


@torch.no_grad()
def generate_speech(
    model,
    tokenizer: TekkenTokenizer,
    text: str,
    voice_name: str = "neutral_female",
    voice_dir: str = None,
    max_frames: int = 2000,
    device: str = "cuda",
) -> np.ndarray:
    """
    Generate speech audio from text.

    Args:
        model: VoxtralTTS model
        tokenizer: TekkenTokenizer
        text: Input text to synthesize
        voice_name: Name of preset voice
        voice_dir: Path to voice embedding directory
        max_frames: Maximum number of audio frames (12.5 Hz)
        device: Device

    Returns:
        audio: numpy array of 24kHz audio samples
    """
    config = model.config

    # Build prompt token sequence
    text_tokens = tokenizer.encode(text)
    prompt_ids = (
        [config.bos_id, config.begin_audio_id]
        # voice embeddings inserted at embedding level, not as token IDs
        # We use AUDIO=24 as placeholder for each voice frame
    )

    # Load voice embedding
    voice_embed = None
    if voice_dir:
        voice_path = Path(voice_dir) / f"{voice_name}.pt"
        if voice_path.exists():
            voice_embed = torch.load(voice_path, weights_only=True).to(device=device, dtype=torch.bfloat16)
            # Add placeholder tokens for voice frames
            n_voice_frames = voice_embed.shape[0]
            prompt_ids.extend([config.audio_id] * n_voice_frames)

    prompt_ids.append(config.inst_end_id)  # [/INST] = 36
    prompt_ids.extend(text_tokens)
    prompt_ids.append(config.inst_id)       # [INST] = 35
    prompt_ids.append(config.begin_audio_id)  # [BEGIN_AUDIO] = 25

    # Build input embeddings
    prompt_tensor = torch.tensor([prompt_ids], device=device)
    prompt_embed = model.backbone.tok_embeddings(prompt_tensor)  # (1, T, dim)

    # Replace AUDIO placeholders with voice embeddings
    if voice_embed is not None:
        start_idx = 2  # After BOS and BEGIN_AUDIO
        prompt_embed[0, start_idx:start_idx + n_voice_frames] = voice_embed

    # Prefill: run through backbone
    model.backbone.setup_freqs(max_len=max_frames + len(prompt_ids) + 100, device=device)
    hidden, caches = model.backbone(prompt_embed)

    # Start autoregressive generation
    pos = len(prompt_ids)
    all_codes = []

    # CRITICAL: Feed one AUDIO=24 token as the first decode step after prefill
    # The prompt ends with [BEGIN_AUDIO]. The model expects [AUDIO] next to trigger generation.
    audio_tok_embed = model.backbone.tok_embeddings(
        torch.tensor([[config.audio_id]], device=device)
    )
    hidden, caches = model.backbone(audio_tok_embed, caches=caches, pos=pos)
    pos += 1
    h = hidden[:, -1, :]  # (1, dim)

    t0 = time.time()

    for frame_idx in range(max_frames):
        # Acoustic transformer: generate 37 codes for this frame
        codes, is_end = model.acoustic.decode_one_frame(h)  # (1, 37)

        if is_end.any():
            break

        all_codes.append(codes)

        # Embed codes for next LLM input and run one decode step
        next_embed = model.embed_audio_codes(codes).unsqueeze(1)  # (1, 1, dim)
        hidden, caches = model.backbone(next_embed, caches=caches, pos=pos)
        pos += 1
        h = hidden[:, -1, :]  # (1, dim) - next hidden state

        if (frame_idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  Frame {frame_idx + 1}: {elapsed:.1f}s ({(frame_idx+1)/elapsed:.1f} frames/s)")

    gen_time = time.time() - t0
    n_frames = len(all_codes)

    if n_frames == 0:
        print("Warning: No audio frames generated!")
        return np.zeros(1, dtype=np.float32)

    print(f"  Generated {n_frames} frames in {gen_time:.1f}s ({n_frames/gen_time:.1f} frames/s)")
    print(f"  Audio duration: {n_frames / 12.5:.1f}s, RTF: {gen_time / (n_frames / 12.5):.2f}")

    # Stack all codes and decode with codec
    all_codes_tensor = torch.stack(all_codes, dim=1)  # (1, T, 37)

    print("  Decoding audio with codec...")
    audio = model.codec(all_codes_tensor)  # (1, samples)
    audio = audio[0].float().cpu().numpy()

    # Normalize
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.95

    return audio


def main():
    parser = argparse.ArgumentParser(description="Voxtral-4B-TTS Speech Generation")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--voice", type=str, default="neutral_female", help="Voice name")
    parser.add_argument("--output", type=str, default="output.wav", help="Output WAV file")
    parser.add_argument("--model-dir", type=str, default=str(Path(__file__).parent.parent / "models" / "original"),
                       help="Path to original model")
    parser.add_argument("--quantized", type=str, default=None,
                       help="Path to quantized model dir (if set, uses quantized)")
    parser.add_argument("--max-frames", type=int, default=2000, help="Max audio frames")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Load tokenizer
    if args.quantized:
        tokenizer_path = Path(args.quantized) / "tekken.json"
        voice_dir = Path(args.quantized) / "voice_embedding"
    else:
        tokenizer_path = Path(args.model_dir) / "tekken.json"
        voice_dir = Path(args.model_dir) / "voice_embedding"

    print("Loading tokenizer...")
    tokenizer = TekkenTokenizer(str(tokenizer_path))

    # Load model
    print("Loading model...")
    t0 = time.time()
    if args.quantized:
        model = load_quantized_model(args.quantized, device=args.device)
    else:
        model = load_original_model(args.model_dir, device=args.device)
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    # Check GPU memory
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"GPU memory used: {mem:.2f} GB")

    # Generate
    print(f"\nGenerating speech for: \"{args.text}\"")
    print(f"Voice: {args.voice}")
    audio = generate_speech(
        model, tokenizer, args.text,
        voice_name=args.voice,
        voice_dir=str(voice_dir),
        max_frames=args.max_frames,
        device=args.device,
    )

    # Save
    sf.write(args.output, audio, 24000)
    duration = len(audio) / 24000
    print(f"\nSaved {args.output} ({duration:.1f}s, {len(audio)} samples, 24kHz)")


if __name__ == "__main__":
    main()
