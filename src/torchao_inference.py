"""
Voxtral-4B-TTS fast int4 quantized inference.

Uses torchao int4 weight-only quantization with:
- HQQ algorithm for choosing quantization parameters (quality)
- tinygemm CUDA kernel for fused dequant+matmul (speed)

Performance on RTX 3090:
  - 65-67 fps (RTF=0.19, 5.3x real-time)
  - 3.7 GB VRAM
  - Perfect Whisper transcription quality

Usage:
    from torchao_inference import load_model_int4
    model = load_model_int4("/path/to/original/model")
"""

import torch
import torch.nn as nn
import time
import gc
from pathlib import Path
from safetensors.torch import load_file

from model import VoxtralTTS, VoxtralConfig
from load_model import _assign_weights


def _apply_torchao_int4(model, group_size=64):
    """Apply torchao int4 quantization with HQQ algorithm to backbone only."""
    from torchao.quantization import quantize_, Int4WeightOnlyConfig
    from torchao.quantization.quantize_.workflows.int4.int4_packing_format import Int4PackingFormat

    algo_type = type(Int4WeightOnlyConfig().int4_choose_qparams_algorithm)
    HQQ_ALGO = algo_type['HQQ']

    quantize_(model.backbone, Int4WeightOnlyConfig(
        group_size=group_size,
        int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D,
        int4_choose_qparams_algorithm=HQQ_ALGO,
    ))


# ─── Main API ──────────────────────────────────────────────────────────

def load_model_int4(
    model_dir: str,
    device: str = "cuda",
    group_size: int = 64,
) -> VoxtralTTS:
    """
    Load Voxtral-4B-TTS with int4 quantized backbone.

    Loads BF16 weights, quantizes backbone to int4 with HQQ algorithm,
    uses tinygemm CUDA kernel for inference. Acoustic transformer and
    codec decoder stay in BF16.

    Args:
        model_dir: Path to original model (with consolidated.safetensors)
        device: CUDA device
        group_size: Quantization group size (64 recommended)

    Returns:
        VoxtralTTS model ready for inference at 65+ fps, ~3.7 GB VRAM
    """
    model_dir = Path(model_dir)
    print(f"[int4] Loading Voxtral-4B-TTS (int4 quantized, group_size={group_size})")
    t_start = time.time()

    config = VoxtralConfig()
    model = VoxtralTTS(config)

    state_dict = load_file(str(model_dir / "consolidated.safetensors"))
    _assign_weights(model, state_dict)
    del state_dict; gc.collect()

    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()

    vram_bf16 = torch.cuda.memory_allocated() / 1e9
    print(f"[int4] BF16 loaded: {vram_bf16:.2f} GB")

    print(f"[int4] Quantizing backbone to int4 (HQQ + tinygemm)...")
    tq0 = time.time()
    _apply_torchao_int4(model, group_size)
    print(f"[int4] Quantized in {time.time()-tq0:.1f}s")

    gc.collect(); torch.cuda.empty_cache()

    vram = torch.cuda.memory_allocated() / 1e9
    print(f"[int4] Done in {time.time()-t_start:.1f}s | VRAM: {vram:.2f} GB")
    return model


# ─── Benchmark ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import soundfile as sf
    from generate_fast import generate_speech_fast
    from generate import TekkenTokenizer

    print("=" * 60)
    print("Voxtral-4B-TTS — torchao int4 + HQQ + tinygemm")
    print("=" * 60)

    torch.cuda.reset_peak_memory_stats()

    MODEL_DIR = str(Path(__file__).parent.parent / "models" / "original")
    model = load_model_int4(MODEL_DIR)

    tok = TekkenTokenizer(f"{MODEL_DIR}/tekken.json")
    voice_dir = f"{MODEL_DIR}/voice_embedding"

    with torch.inference_mode():
        # Warmup
        generate_speech_fast(model, tok, "Hi.", voice_name="neutral_female",
            voice_dir=voice_dir, max_frames=5, device="cuda",
            flow_steps=3, cfg_alpha=1.0)

        print(f"\nVRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

        # Benchmark
        tests = [
            "The quick brown fox jumps over the lazy dog.",
            "Hello world, how are you today?",
            "Paris is a beautiful city with many famous landmarks.",
            "Technology advances rapidly in the modern world.",
        ]

        print("\n--- Speed ---")
        for text in tests:
            audio, gen_time = generate_speech_fast(
                model, tok, text, voice_name="neutral_female",
                voice_dir=voice_dir, max_frames=300, device="cuda",
                flow_steps=3, cfg_alpha=1.0)
            dur = len(audio) / 24000
            n = int(dur * 12.5)
            fps = n / gen_time if gen_time > 0 else 0
            print(f"  [{fps:.0f} fps] \"{text}\"")

        # Quality
        print("\n--- Quality (Whisper) ---")
        try:
            import whisper
            wh = whisper.load_model("base")
            for text in tests:
                audio, _ = generate_speech_fast(
                    model, tok, text, voice_name="neutral_female",
                    voice_dir=voice_dir, max_frames=300, device="cuda",
                    flow_steps=3, cfg_alpha=1.0)
                sf.write("/tmp/int4_test.wav", audio, 24000)
                r = wh.transcribe("/tmp/int4_test.wav")
                print(f"  In:  \"{text}\"")
                print(f"  Out: \"{r['text'].strip()}\"")
        except ImportError:
            print("  (whisper not installed)")

    mem = torch.cuda.memory_allocated() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"\nFinal VRAM: {mem:.2f} GB (peak: {peak:.2f} GB)")
