"""
Voxtral-4B-TTS FAST text-to-speech generation.
Optimized for short text, low-latency responses.

Optimizations:
1. Reduced flow steps (8→3) with 2nd-order midpoint ODE solver
2. Optional CFG disable (cfg_alpha=1.0 skips unconditioned pass → 2x acoustic speedup)
3. torch.compile on backbone for fused kernels
4. Pre-allocated KV cache for CUDA graph compatibility
"""

import argparse
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import numpy as np
from pathlib import Path

from model import VoxtralConfig, apply_rotary_emb
from load_model import load_original_model, load_quantized_model
from generate import TekkenTokenizer


# ─── Static BF16 KV Cache ──────────────────────────────────────────────

class StaticGQAAttention(nn.Module):
    """GQAAttention with pre-allocated BF16 KV buffers + padding mask.

    Decode path uses STATIC tensor shapes (full buffer + mask) for CUDA graph compatibility.
    Prefill path uses standard causal attention (dynamic shape, not graphed).
    """

    def __init__(self, original_attn, max_seq_len=700):
        super().__init__()
        self.n_heads = original_attn.n_heads
        self.n_kv_heads = original_attn.n_kv_heads
        self.head_dim = original_attn.head_dim
        self.n_rep = original_attn.n_rep
        self.wq = original_attn.wq
        self.wk = original_attn.wk
        self.wv = original_attn.wv
        self.wo = original_attn.wo
        self.max_seq_len = max_seq_len
        self._k_buf = None
        self._v_buf = None

    def _ensure_buffers(self, device, dtype):
        if self._k_buf is None:
            self._k_buf = torch.zeros(
                1, self.n_kv_heads, self.max_seq_len, self.head_dim,
                device=device, dtype=dtype)
            self._v_buf = torch.zeros(
                1, self.n_kv_heads, self.max_seq_len, self.head_dim,
                device=device, dtype=dtype)

    def reset(self):
        self._k_buf = None
        self._v_buf = None

    def forward(self, x, freqs_cis=None, mask=None, cache=None, pos=0):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis[pos:pos + T])

        # Always write to buffer so decode steps can read prefill KV
        self._ensure_buffers(x.device, x.dtype)
        self._k_buf[:, :, pos:pos + T] = k
        self._v_buf[:, :, pos:pos + T] = v

        if cache is not None:
            # Decode: read from buffer up to current position
            k = self._k_buf[:, :, :pos + T]
            v = self._v_buf[:, :, :pos + T]

        is_causal = (mask is None and cache is None and T > 1)

        new_cache = True  # Sentinel — cache is managed internally

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=is_causal)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out), new_cache


def enable_static_cache(model, max_seq_len=700):
    """Replace GQAAttention with StaticGQAAttention in all backbone layers."""
    for layer in model.backbone.layers:
        old_attn = layer.attention
        if not isinstance(old_attn, StaticGQAAttention):
            layer.attention = StaticGQAAttention(old_attn, max_seq_len)

def reset_static_cache(model):
    """Reset static cache buffers for new generation."""
    for layer in model.backbone.layers:
        if isinstance(layer.attention, StaticGQAAttention):
            layer.attention.reset()


@torch.no_grad()
def generate_speech_fast(
    model,
    tokenizer,
    text: str,
    voice_name: str = "neutral_female",
    voice_dir: str = None,
    max_frames: int = 500,
    device: str = "cuda",
    flow_steps: int = 3,
    cfg_alpha: float = 1.2,  # 1.2 = full CFG (good quality), 1.0 = no CFG (faster but garbled)
):
    """
    Fast speech generation optimized for short text.

    Key speed opts vs generate.py:
    - flow_steps=3 (vs 8): 2.7x fewer acoustic forward passes
    - cfg_alpha=1.0 (vs 1.2): skips unconditioned pass, 2x fewer passes
    - Combined: ~5x faster acoustic transformer
    """
    config = model.config

    # Tokenize
    text_tokens = tokenizer.encode(text)
    prompt_ids = [config.bos_id, config.begin_audio_id]

    # Load voice embedding
    voice_embed = None
    if voice_dir:
        voice_path = Path(voice_dir) / f"{voice_name}.pt"
        if voice_path.exists():
            voice_embed = torch.load(voice_path, weights_only=True).to(device=device, dtype=torch.bfloat16)
            n_voice_frames = voice_embed.shape[0]
            prompt_ids.extend([config.audio_id] * n_voice_frames)

    prompt_ids.append(config.inst_end_id)
    prompt_ids.extend(text_tokens)
    prompt_ids.append(config.inst_id)
    prompt_ids.append(config.begin_audio_id)

    # Build embeddings
    prompt_tensor = torch.tensor([prompt_ids], device=device)
    prompt_embed = model.backbone.tok_embeddings(prompt_tensor)

    if voice_embed is not None:
        prompt_embed[0, 2:2 + n_voice_frames] = voice_embed

    # Prefill
    model.backbone.setup_freqs(max_len=max_frames + len(prompt_ids) + 100, device=device)
    hidden, caches = model.backbone(prompt_embed)
    pos = len(prompt_ids)

    # First decode step: AUDIO token
    audio_tok_embed = model.backbone.tok_embeddings(
        torch.tensor([[config.audio_id]], device=device))
    hidden, caches = model.backbone(audio_tok_embed, caches=caches, pos=pos)
    pos += 1
    h = hidden[:, -1, :]

    all_codes = []
    t0 = time.time()

    for frame_idx in range(max_frames):
        try:
            # Fast acoustic decode
            codes, is_end = _decode_one_frame_fast(
                model.acoustic, h, config, flow_steps=flow_steps, cfg_alpha=cfg_alpha)

            if is_end.any():
                break

            all_codes.append(codes)

            # Embed and advance LLM
            next_embed = model.embed_audio_codes(codes).unsqueeze(1)
            hidden, caches = model.backbone(next_embed, caches=caches, pos=pos)
            pos += 1
            h = hidden[:, -1, :]
        except RuntimeError:
            break  # Return partial audio on CUDA error

    gen_time = time.time() - t0
    n_frames = len(all_codes)

    if n_frames == 0:
        return np.zeros(1, dtype=np.float32), 0.0

    fps = n_frames / gen_time
    duration = n_frames / 12.5
    rtf = gen_time / duration
    print(f"  {n_frames} frames in {gen_time:.1f}s ({fps:.1f} fps, RTF={rtf:.2f})")

    # Trim warmup frames (fixes garbled start, HF discussion #20)
    from audio_postprocess import trim_warmup_frames, postprocess_audio
    all_codes = trim_warmup_frames(all_codes)
    n_frames = len(all_codes)
    if n_frames == 0:
        return np.zeros(1, dtype=np.float32), gen_time

    # Decode to audio — sync first to catch any pending CUDA errors from generation
    try:
        torch.cuda.synchronize()
        all_codes_tensor = torch.stack(all_codes, dim=1)
        audio = model.codec(all_codes_tensor)
        audio = audio[0].float().cpu().numpy()
    except RuntimeError:
        return np.zeros(1, dtype=np.float32), gen_time

    # Post-process: LPF 11kHz + upsample 48kHz + normalize
    audio = postprocess_audio(audio)

    return audio, gen_time


@torch.no_grad()
def _decode_one_frame_fast(acoustic, llm_hidden, config, flow_steps=3, cfg_alpha=1.0):
    """
    Fast acoustic frame decode with reduced steps and optional CFG.

    With flow_steps=3, cfg_alpha=1.0: only 2 acoustic forward passes (vs 14 in default)
    With flow_steps=3, cfg_alpha=1.2: only 4 acoustic forward passes (vs 14)
    Default (flow_steps=8, cfg_alpha=1.2): 14 acoustic forward passes
    """
    B = llm_hidden.shape[0]
    device = llm_hidden.device

    # Semantic code prediction (NaN-safe for numerical stability in long sequences)
    logits = acoustic.semantic_codebook_output(llm_hidden)
    logits = torch.nan_to_num(logits, nan=-1e9)
    logits[:, 0] = -1e9
    logits[:, 8194:] = -1e9
    semantic_code = logits.argmax(dim=-1).clamp(0, 8193)

    is_end = semantic_code <= config.end_audio

    # Flow matching with reduced steps and midpoint solver
    x_t = torch.randn(B, config.n_acoustic_codebooks, device=device) * config.sigma_max
    timesteps = torch.linspace(0, 1, flow_steps, device=device)

    use_cfg = cfg_alpha != 1.0
    zeros = torch.zeros_like(llm_hidden) if use_cfg else None

    for i in range(flow_steps - 1):
        t = timesteps[i].item()
        dt = (timesteps[i + 1] - timesteps[i]).item()

        # Midpoint method (2nd order) for better accuracy with fewer steps
        # .clone() prevents CUDA graph buffer reuse conflicts with reduce-overhead compile
        v1 = acoustic.predict_velocity(x_t, llm_hidden, t).clone()
        if use_cfg:
            v1_uncond = acoustic.predict_velocity(x_t, zeros, t).clone()
            v1 = cfg_alpha * v1 + (1 - cfg_alpha) * v1_uncond

        # Midpoint: evaluate at t + dt/2
        x_mid = x_t + v1 * (dt / 2)
        t_mid = t + dt / 2
        v2 = acoustic.predict_velocity(x_mid, llm_hidden, t_mid).clone()
        if use_cfg:
            v2_uncond = acoustic.predict_velocity(x_mid, zeros, t_mid).clone()
            v2 = cfg_alpha * v2 + (1 - cfg_alpha) * v2_uncond

        # Update using midpoint velocity
        x_t = x_t + v2 * dt

    # FSQ quantize (NaN guard for numerical stability in long sequences)
    x_t = torch.nan_to_num(x_t, nan=0.0, posinf=1.0, neginf=-1.0)
    x_clamp = x_t.clamp(-1, 1)
    scaled = (x_clamp + 1) / 2 * (config.fsq_levels - 1)
    acoustic_codes = scaled.round().long().clamp(0, config.fsq_levels - 1)
    acoustic_codes = acoustic_codes + config.special_count

    acoustic_codes[is_end] = config.empty_audio
    codes = torch.cat([semantic_code.unsqueeze(1), acoustic_codes], dim=1)
    return codes, is_end


def main():
    parser = argparse.ArgumentParser(description="Voxtral-4B-TTS FAST Generation")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--voice", type=str, default="neutral_female")
    parser.add_argument("--output", type=str, default="output.wav")
    parser.add_argument("--model-dir", type=str, default=str(Path(__file__).parent.parent / "models" / "original"))
    parser.add_argument("--quantized", type=str, default=None)
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument("--flow-steps", type=int, default=3, help="Flow matching steps (default 3, original 8)")
    parser.add_argument("--cfg-alpha", type=float, default=1.2, help="CFG strength (1.2=full quality, 1.0=off/faster)")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--static-cache", action="store_true", help="Pre-allocated BF16 KV cache + backbone compile")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Paths
    if args.quantized:
        tokenizer_path = Path(args.quantized) / "tekken.json"
        voice_dir = Path(args.quantized) / "voice_embedding"
    else:
        tokenizer_path = Path(args.model_dir) / "tekken.json"
        voice_dir = Path(args.model_dir) / "voice_embedding"

    print("Loading tokenizer...")
    tokenizer = TekkenTokenizer(str(tokenizer_path))

    print("Loading model...")
    t0 = time.time()
    if args.quantized:
        model = load_quantized_model(args.quantized, device=args.device)
    else:
        model = load_original_model(args.model_dir, device=args.device)
    print(f"Model loaded in {time.time()-t0:.1f}s")

    if args.static_cache:
        print("Enabling static BF16 KV cache...")
        enable_static_cache(model, max_seq_len=args.max_frames + 300)

    if args.compile:
        print("Compiling with torch.compile...")
        model.acoustic.predict_velocity = torch.compile(
            model.acoustic.predict_velocity, mode="default", fullgraph=False)
        if args.static_cache:
            model.backbone = torch.compile(
                model.backbone, mode="default", fullgraph=False)
        # Warmup
        print("Warmup compilation...")
        generate_speech_fast(model, tokenizer, "Hi.", voice_name=args.voice,
                           voice_dir=str(voice_dir), max_frames=5, device=args.device,
                           flow_steps=args.flow_steps, cfg_alpha=args.cfg_alpha)

    mem = torch.cuda.memory_allocated() / 1e9
    print(f"GPU memory: {mem:.2f} GB")

    print(f"\nGenerating: \"{args.text}\"")
    print(f"Voice: {args.voice}, Flow steps: {args.flow_steps}, CFG: {args.cfg_alpha}")

    audio, gen_time = generate_speech_fast(
        model, tokenizer, args.text,
        voice_name=args.voice,
        voice_dir=str(voice_dir),
        max_frames=args.max_frames,
        device=args.device,
        flow_steps=args.flow_steps,
        cfg_alpha=args.cfg_alpha,
    )

    sf.write(args.output, audio, 24000)
    duration = len(audio) / 24000
    print(f"\nSaved {args.output} ({duration:.1f}s audio)")
    print(f"Total time: {gen_time:.2f}s, RTF: {gen_time/max(duration, 0.001):.2f}")


if __name__ == "__main__":
    main()
