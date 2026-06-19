"""
Voxtral-4B-TTS pure PyTorch inference model.
Supports both BF16 and TurboQuant-quantized backbone weights.

Three components:
1. LLM Backbone (26-layer Ministral-3B) - autoregressive semantic token generation
2. Flow-Matching Acoustic Transformer (3-layer) - acoustic code generation via ODE
3. Audio Codec Decoder - codes to 24kHz waveform
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class VoxtralConfig:
    # LLM backbone
    dim: int = 3072
    n_layers: int = 26
    n_heads: int = 32
    n_kv_heads: int = 8
    head_dim: int = 128
    hidden_dim: int = 9216
    vocab_size: int = 131072
    norm_eps: float = 1e-5
    rope_theta: float = 1_000_000.0

    # Acoustic transformer
    acoustic_n_layers: int = 3
    acoustic_rope_theta: float = 10_000.0
    flow_steps: int = 8
    cfg_alpha: float = 1.2
    sigma_max: float = 1.0
    n_acoustic_codebooks: int = 36
    fsq_levels: int = 21
    semantic_codebook_size: int = 8192
    special_count: int = 2  # EMPTY=0, END=1

    # Codec decoder
    codec_dim: int = 1024
    codec_hidden_dim: int = 4096
    codec_n_heads: int = 8
    codec_norm_eps: float = 1e-2
    semantic_embed_dim: int = 256
    patch_size: int = 240
    sample_rate: int = 24000

    # Special tokens: raw rank IDs (special tokens occupy positions 0..999)
    # BPE text tokens are at positions 1000..131071 (tiktoken rank + 1000)
    num_special_tokens: int = 1000
    bos_id: int = 1            # <s> rank=1
    eos_id: int = 2            # </s> rank=2
    audio_id: int = 24         # [AUDIO] rank=24
    begin_audio_id: int = 25   # [BEGIN_AUDIO] rank=25
    inst_id: int = 35          # [REPEAT_AUDIO_TEXT] rank=35
    inst_end_id: int = 36      # [NEXT_AUDIO_TEXT] rank=36

    # Audio special codes
    empty_audio: int = 0
    end_audio: int = 1


# ─── Building blocks ───────────────────────────────────────────────


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() / rms * self.weight.float()).to(x.dtype)


def precompute_freqs_cis(dim: int, max_len: int, theta: float = 1e6, device="cuda"):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_len, device=device).float()
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs = freqs_cis.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, dim/2)
    xq_out = torch.view_as_real(xq_ * freqs).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs).flatten(-2)
    return xq_out.to(xq.dtype), xk_out.to(xk.dtype)


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # gate
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)   # down
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)   # up

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class GQAAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

    def forward(self, x, freqs_cis=None, mask=None, cache=None, pos=0):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis[pos:pos+T])

        # KV cache
        if cache is not None:
            k_cache, v_cache = cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        new_cache = (k, v)

        # Repeat KV for GQA
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Scaled dot-product attention
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=(mask is None and cache is None and T > 1))
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out), new_cache


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, head_dim, hidden_dim, norm_eps):
        super().__init__()
        self.attention = GQAAttention(dim, n_heads, n_kv_heads, head_dim)
        self.feed_forward = SwiGLUFFN(dim, hidden_dim)
        self.attention_norm = RMSNorm(dim, norm_eps)
        self.ffn_norm = RMSNorm(dim, norm_eps)

    def forward(self, x, freqs_cis=None, mask=None, cache=None, pos=0):
        h, new_cache = self.attention(self.attention_norm(x), freqs_cis, mask, cache, pos)
        x = x + h
        x = x + self.feed_forward(self.ffn_norm(x))
        return x, new_cache


# ─── LLM Backbone ──────────────────────────────────────────────────


class LLMBackbone(nn.Module):
    def __init__(self, config: VoxtralConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([
            TransformerBlock(config.dim, config.n_heads, config.n_kv_heads,
                           config.head_dim, config.hidden_dim, config.norm_eps)
            for _ in range(config.n_layers)
        ])
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.freqs_cis = None

    def setup_freqs(self, max_len=4096, device="cuda"):
        self.freqs_cis = precompute_freqs_cis(
            self.config.head_dim, max_len, self.config.rope_theta, device
        )

    def forward(self, x_embed, caches=None, pos=0):
        """
        Args:
            x_embed: (B, T, dim) input embeddings
            caches: list of (k_cache, v_cache) per layer, or None
            pos: position offset for RoPE
        Returns:
            hidden: (B, T, dim)
            new_caches: list of (k_cache, v_cache)
        """
        if self.freqs_cis is None:
            self.setup_freqs(device=x_embed.device)

        h = x_embed
        new_caches = []
        for i, layer in enumerate(self.layers):
            cache = caches[i] if caches else None
            h, new_cache = layer(h, self.freqs_cis, cache=cache, pos=pos)
            new_caches.append(new_cache)

        h = self.norm(h)
        return h, new_caches

    def get_logits(self, hidden):
        """Project hidden states to vocabulary logits using tied embeddings."""
        return hidden @ self.tok_embeddings.weight.T


# ─── Flow-Matching Acoustic Transformer ────────────────────────────


class BidirectionalAttention(nn.Module):
    """Attention without causal mask and without RoPE."""
    def __init__(self, dim, n_heads, n_kv_heads, head_dim):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


class AcousticTransformerBlock(nn.Module):
    def __init__(self, config: VoxtralConfig):
        super().__init__()
        self.attention = BidirectionalAttention(
            config.dim, config.n_heads, config.n_kv_heads, config.head_dim)
        self.feed_forward = SwiGLUFFN(config.dim, config.hidden_dim)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class AcousticTransformer(nn.Module):
    def __init__(self, config: VoxtralConfig):
        super().__init__()
        self.config = config

        self.input_projection = nn.Linear(config.n_acoustic_codebooks, config.dim, bias=False)
        self.time_projection = nn.Linear(config.dim, config.dim, bias=False)
        self.llm_projection = nn.Linear(config.dim, config.dim, bias=False)

        self.layers = nn.ModuleList([
            AcousticTransformerBlock(config) for _ in range(config.acoustic_n_layers)
        ])
        self.norm = RMSNorm(config.dim, config.norm_eps)

        self.semantic_codebook_output = nn.Linear(
            config.dim, config.semantic_codebook_size + config.special_count + 126, bias=True)
        # Output is 8320 = 8192 + 2 + 126 padding
        self.acoustic_codebook_output = nn.Linear(config.dim, config.n_acoustic_codebooks, bias=False)

        # Time embedding frequencies
        half = config.dim // 2
        inv_freq = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer("inv_freq", inv_freq)

    def time_embedding(self, t: float, device):
        angles = t * self.inv_freq.to(device)
        return torch.cat([torch.cos(angles), torch.sin(angles)])  # (dim,)

    def predict_velocity(self, x_t, llm_hidden, t: float):
        """
        Predict velocity for flow matching.
        x_t: (B, 36) noisy acoustic codes
        llm_hidden: (B, dim)
        t: scalar timestep
        """
        B = x_t.shape[0]
        device = x_t.device

        dtype = llm_hidden.dtype
        tok_noise = self.input_projection(x_t.to(dtype))       # (B, dim)
        tok_time = self.time_projection(self.time_embedding(t, device).unsqueeze(0).expand(B, -1).to(dtype))
        tok_llm = self.llm_projection(llm_hidden)     # (B, dim)

        # Stack as 3 tokens: (B, 3, dim)
        x = torch.stack([tok_noise, tok_time, tok_llm], dim=1)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x[:, 0, :])  # Take first token only
        return self.acoustic_codebook_output(x)  # (B, 36)

    @torch.no_grad()
    def decode_one_frame(self, llm_hidden: torch.Tensor):
        """
        Generate one frame of 37 audio codes.
        llm_hidden: (B, dim) from LLM backbone
        Returns: (B, 37) tensor of codes
        """
        B = llm_hidden.shape[0]
        device = llm_hidden.device
        config = self.config

        # Step 1: Predict semantic code (greedy argmax)
        logits = self.semantic_codebook_output(llm_hidden)  # (B, 8320)
        # Mask empty_audio (index 0) and padding (8194+)
        logits = torch.nan_to_num(logits, nan=-1e9)
        logits[:, 0] = -1e9   # empty_audio
        logits[:, 8194:] = -1e9  # padding beyond semantic_codebook_size + special_count
        semantic_code = logits.argmax(dim=-1).clamp(0, 8193)  # (B,)

        # Check for end-of-audio (code 1 = END_AUDIO)
        is_end = semantic_code <= config.end_audio

        # Step 2: Flow matching for acoustic codes
        x_t = torch.randn(B, config.n_acoustic_codebooks, device=device) * config.sigma_max
        timesteps = torch.linspace(0, 1, config.flow_steps, device=device)

        for i in range(config.flow_steps - 1):
            t = timesteps[i].item()
            dt = (timesteps[i + 1] - timesteps[i]).item()

            v_cond = self.predict_velocity(x_t, llm_hidden, t)
            v_uncond = self.predict_velocity(x_t, torch.zeros_like(llm_hidden), t)
            v = config.cfg_alpha * v_cond + (1 - config.cfg_alpha) * v_uncond

            x_t = x_t + v * dt

        # Step 3: FSQ quantize (NaN guard for numerical stability in long sequences)
        x_t = torch.nan_to_num(x_t, nan=0.0, posinf=1.0, neginf=-1.0)
        x_clamp = x_t.clamp(-1, 1)
        scaled = (x_clamp + 1) / 2 * (config.fsq_levels - 1)
        acoustic_codes = scaled.round().long().clamp(0, config.fsq_levels - 1)
        acoustic_codes = acoustic_codes + config.special_count  # offset

        # Set acoustic to EMPTY if end-of-audio
        acoustic_codes[is_end] = config.empty_audio

        # Combine: (B, 37)
        codes = torch.cat([semantic_code.unsqueeze(1), acoustic_codes], dim=1)
        return codes, is_end


# ─── Codec Decoder ──────────────────────────────────────────────────


class WeightNormConv1d(nn.Module):
    """Causal Conv1d with weight normalization stored as original0 (g) and original1 (v)."""
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, transpose=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.transpose = transpose
        self.in_ch = in_ch
        self.out_ch = out_ch

        if transpose:
            self.weight_v = nn.Parameter(torch.randn(in_ch, out_ch, kernel_size))
        else:
            self.weight_v = nn.Parameter(torch.randn(out_ch, in_ch, kernel_size))
        self.weight_g = nn.Parameter(torch.ones(out_ch, 1, 1))

    def get_weight(self):
        # weight_norm: w = g * v / ||v||
        v = self.weight_v
        norm = v.flatten(1).norm(dim=1, keepdim=True).unsqueeze(-1)
        return self.weight_g * v / norm.clamp(min=1e-8)

    def forward(self, x):
        """x: (B, C, T)"""
        w = self.get_weight()
        if self.transpose:
            # Causal transposed conv: pad output
            pad = self.kernel_size - self.stride
            out = F.conv_transpose1d(x, w, stride=self.stride)
            if pad > 0:
                out = out[:, :, :-pad]
            return out
        else:
            # Causal conv: left-pad input
            pad = self.kernel_size - 1
            x_padded = F.pad(x, (pad, 0))
            return F.conv1d(x_padded, w, stride=self.stride)


class CodecQKNorm(nn.Module):
    def __init__(self, full_dim, eps=1e-6):
        super().__init__()
        # QK norm uses eps=1e-6 (not 1e-2 like layer norms) per params.json qk_norm_eps
        # Q/K shape: (B, heads, T, head_dim) -> normalize last dim
        self.q_norm = RMSNorm(full_dim, eps)
        self.k_norm = RMSNorm(full_dim, eps)

    def forward(self, q, k):
        # q, k: (B, n_heads, T, head_dim)
        B, H, T, D = q.shape
        # Reshape to (B, T, H*D), apply norm, reshape back
        q = self.q_norm(q.transpose(1, 2).reshape(B, T, H * D)).reshape(B, T, H, D).transpose(1, 2)
        k = self.k_norm(k.transpose(1, 2).reshape(B, T, H * D)).reshape(B, T, H, D).transpose(1, 2)
        return q, k


class CodecAttention(nn.Module):
    """ALiBi attention with QK norm, layer scale, and sliding window."""
    def __init__(self, dim, n_heads, head_dim, window_size, eps=1e-2):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.window_size = window_size

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)
        self.qk_norm = CodecQKNorm(head_dim, eps)

        # ALiBi slopes
        slopes = torch.tensor([2.0 ** (-8.0 / n_heads * (h + 1)) for h in range(n_heads)])
        self.register_buffer("alibi_slopes", slopes)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # QK norm
        q, k = self.qk_norm(q, k)

        # Attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # ALiBi bias
        positions = torch.arange(T, device=x.device)
        dist = positions.unsqueeze(0) - positions.unsqueeze(1)  # (T, T)
        alibi = self.alibi_slopes.view(1, -1, 1, 1) * dist.unsqueeze(0).unsqueeze(0)
        scores = scores + alibi

        # Causal mask + sliding window
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        window = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool),
                           diagonal=-(self.window_size + 1))
        mask = causal | window
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), -1e9)

        attn = F.softmax(scores, dim=-1).to(v.dtype)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


class CodecTransformerLayer(nn.Module):
    def __init__(self, dim, n_heads, head_dim, hidden_dim, window_size, eps=1e-2):
        super().__init__()
        self.attention = CodecAttention(dim, n_heads, head_dim, window_size, eps)
        self.feed_forward = SwiGLUFFN(dim, hidden_dim)
        self.attention_norm = RMSNorm(dim, eps)
        self.ffn_norm = RMSNorm(dim, eps)
        self.attention_scale = nn.Parameter(torch.ones(dim))
        self.ffn_scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x + self.attention_scale * self.attention(self.attention_norm(x))
        x = x + self.ffn_scale * self.feed_forward(self.ffn_norm(x))
        return x


class CodecDecoder(nn.Module):
    def __init__(self, config: VoxtralConfig):
        super().__init__()
        self.config = config
        code_dim = config.semantic_embed_dim + config.n_acoustic_codebooks  # 256 + 36 = 292

        # Semantic codebook
        self.semantic_embedding_sum = nn.Parameter(
            torch.randn(config.semantic_codebook_size, config.semantic_embed_dim))
        self.semantic_cluster_usage = nn.Parameter(
            torch.ones(config.semantic_codebook_size))

        # Input conv
        self.input_conv = WeightNormConv1d(code_dim, config.codec_dim, kernel_size=3)

        # 4 stages: transformer blocks + upsample convs
        window_sizes = [2, 4, 8, 16]
        self.transformer_stages = nn.ModuleList()
        self.upsample_convs = nn.ModuleList()

        for i, ws in enumerate(window_sizes):
            stage = nn.ModuleList([
                CodecTransformerLayer(
                    config.codec_dim, config.codec_n_heads, config.head_dim,
                    config.codec_hidden_dim, ws, config.codec_norm_eps)
                for _ in range(2)
            ])
            self.transformer_stages.append(stage)
            if i < 3:  # No upsample after last stage
                self.upsample_convs.append(
                    WeightNormConv1d(config.codec_dim, config.codec_dim,
                                   kernel_size=4, stride=2, transpose=True))

        # Output projection
        self.output_proj = WeightNormConv1d(config.codec_dim, config.patch_size, kernel_size=7)

    def embed_codes(self, codes):
        """
        codes: (B, T, 37) - 1 semantic + 36 acoustic
        Returns: (B, T, 292)
        """
        semantic_codes = codes[:, :, 0] - self.config.special_count  # Remove offset
        semantic_codes = torch.nan_to_num(semantic_codes.float(), nan=0.0).long()
        semantic_codes = semantic_codes.clamp(0, self.config.semantic_codebook_size - 1)
        acoustic_codes = codes[:, :, 1:] - self.config.special_count

        # Semantic embedding lookup
        sem_embed = self.semantic_embedding_sum / self.semantic_cluster_usage.unsqueeze(1).clamp(min=1e-8)
        semantic_embed = F.embedding(semantic_codes, sem_embed)

        # Acoustic FSQ decode: map [0, 20] -> [-1, 1]
        acoustic_continuous = acoustic_codes.float() * 2.0 / (self.config.fsq_levels - 1) - 1.0

        return torch.cat([semantic_embed, acoustic_continuous], dim=-1)  # (B, T, 292)

    def forward(self, codes):
        """
        codes: (B, T, 37)
        Returns: (B, samples) waveform at 24kHz
        """
        x = self.embed_codes(codes)  # (B, T, 292)
        x = x.to(self.input_conv.weight_v.dtype)  # match codec weight dtype
        x = x.transpose(1, 2)  # (B, 292, T) channel-first

        x = self.input_conv(x)  # (B, 1024, T)

        for i, stage in enumerate(self.transformer_stages):
            x = x.transpose(1, 2)  # (B, T, 1024)
            for layer in stage:
                x = layer(x)
            x = x.transpose(1, 2)  # (B, 1024, T)

            if i < 3:  # Upsample
                x = self.upsample_convs[i](x)

        x = self.output_proj(x)  # (B, 240, T')
        x = x.transpose(1, 2)    # (B, T', 240)
        return x.reshape(x.shape[0], -1)  # (B, samples)


# ─── Full Voxtral TTS Model ────────────────────────────────────────


class VoxtralTTS(nn.Module):
    def __init__(self, config: VoxtralConfig):
        super().__init__()
        self.config = config
        self.backbone = LLMBackbone(config)
        self.acoustic = AcousticTransformer(config)
        self.codec = CodecDecoder(config)

        # Audio codebook embeddings for feeding codes back to LLM
        # Combined table: semantic (8194) + 36 acoustic (23 each) = 9022 entries
        # Padded to 9088 in the actual model
        self.audio_codebook_embeddings = nn.Embedding(9088, config.dim)

    def embed_audio_codes(self, codes):
        """
        Convert 37 per-frame codes to a single embedding by summing codebook lookups.
        codes: (B, 37) raw codes with offsets
        Returns: (B, dim)

        Embedding table layout (9088 total):
          Codebook 0 (semantic): indices 0..8193 (8192 codes + 2 special), padded to 8320
          Codebook 1..36 (acoustic): 23 entries each (21 FSQ + 2 special)
            Codebook 1: offset=8194, indices 8194..8216
            Codebook 2: offset=8217, indices 8217..8239
            ...
            Codebook k: offset=8194 + (k-1)*23
        """
        B = codes.shape[0]
        embed_sum = torch.zeros(B, self.config.dim, device=codes.device, dtype=torch.bfloat16)

        # Semantic code (codebook 0): offset=0
        sem_idx = torch.nan_to_num(codes[:, 0].float(), nan=0.0).long().clamp(0, 8193)
        embed_sum = embed_sum + self.audio_codebook_embeddings(sem_idx)

        # Acoustic codes (codebooks 1-36): offset = 8194 + (k-1)*23
        for k in range(1, 37):
            offset = 8194 + (k - 1) * 23
            idx = torch.nan_to_num(codes[:, k].float(), nan=0.0).long().clamp(0, 22) + offset
            embed_sum = embed_sum + self.audio_codebook_embeddings(idx)

        return embed_sum
