"""
Load Voxtral-4B-TTS with mixed-precision weights.
Supports both original BF16 and TurboQuant-quantized backbone.
"""

import torch
import json
from pathlib import Path
from safetensors.torch import load_file

from model import VoxtralTTS, VoxtralConfig


def load_original_model(model_dir: str, device="cuda") -> VoxtralTTS:
    """Load the original BF16 model."""
    model_dir = Path(model_dir)
    state_dict = load_file(str(model_dir / "consolidated.safetensors"))

    config = VoxtralConfig()
    model = VoxtralTTS(config)
    _assign_weights(model, state_dict)
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    return model


def load_quantized_model(quantized_dir: str, device="cuda") -> VoxtralTTS:
    """
    Load model with TurboQuant-quantized backbone + BF16 everything else.
    For inference, we dequantize the backbone weights back to BF16.
    This proves the quantization quality while using the same inference code.

    For actual VRAM savings, a custom forward pass with TurboQuantLinear would be needed.
    """
    quantized_dir = Path(quantized_dir)

    config = VoxtralConfig()
    model = VoxtralTTS(config)

    # Load non-quantized weights (acoustic, codec, norms, embeddings)
    non_quantized = load_file(str(quantized_dir / "non_quantized.safetensors"))

    # Load and dequantize backbone weights
    from turboquant_model import TurboQuantConfig, get_codebook
    from turboquant_model.quantize import turboquant_quantize, unpack_4bit
    from turboquant_model.rotation import generate_rotation_matrix
    import math

    tq_config = TurboQuantConfig.load(str(quantized_dir / "turboquant_config.json"))
    layers_dir = quantized_dir / "layers"

    # Reconstruct each backbone linear weight
    backbone_weights = {}
    layer_files = sorted(layers_dir.glob("*.indices.pt"))

    for idx_file in layer_files:
        layer_name = idx_file.name.replace(".indices.pt", "")
        # Convert back: layers_0_attention_wq -> layers.0.attention.wq.weight
        key = layer_name.replace("_", ".", 2)  # careful with underscore conversion
        # More robust conversion
        parts = layer_name.split("_")
        # Reconstruct: layers_N_type_subtype -> layers.N.type.subtype.weight
        key = _layer_name_to_key(layer_name)

        indices_packed = torch.load(layers_dir / f"{layer_name}.indices.pt", weights_only=True)
        norms = torch.load(layers_dir / f"{layer_name}.norms.pt", weights_only=True)

        # Dequantize pass 1
        W1 = _dequantize_pass(indices_packed, norms, tq_config.bit_width,
                              tq_config.group_size, tq_config.seed)

        # Dequantize pass 2 (residual) if exists
        pass2_file = layers_dir / f"{layer_name}.pass2_indices.pt"
        if pass2_file.exists():
            p2_indices = torch.load(pass2_file, weights_only=True)
            p2_norms = torch.load(layers_dir / f"{layer_name}.pass2_norms.pt", weights_only=True)
            p2_codebook = torch.load(layers_dir / f"{layer_name}.pass2_codebook.pt", weights_only=True)
            W2 = _dequantize_pass(p2_indices, p2_norms, tq_config.residual_bit_width,
                                  tq_config.group_size, tq_config.residual_seed)
            W1 = W1 + W2

        backbone_weights[key] = W1.to(torch.bfloat16)

    # Merge all weights
    all_weights = {}
    all_weights.update(backbone_weights)
    all_weights.update(non_quantized)

    _assign_weights(model, all_weights)
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    print(f"Loaded quantized model: {len(backbone_weights)} dequantized + {len(non_quantized)} BF16 tensors")
    return model


def _dequantize_pass(indices_packed, norms, bit_width, group_size, seed):
    """Dequantize a single pass of TurboQuant."""
    from turboquant_model.quantize import unpack_4bit
    from turboquant_model.codebook import get_codebook
    from turboquant_model.rotation import generate_rotation_matrix
    import math

    centroids, boundaries = get_codebook(bit_width)
    centroids_t = centroids.clone().detach().float() if isinstance(centroids, torch.Tensor) else torch.tensor(centroids, dtype=torch.float32)

    M = indices_packed.shape[0]
    N_packed = indices_packed.shape[1]
    N = N_packed * 2  # unpacked size

    # Ensure everything is on CPU for dequantization
    indices_packed = indices_packed.cpu()
    norms = norms.cpu()
    indices = unpack_4bit(indices_packed, N)  # (M, N)

    if norms.dim() == 1:
        norms = norms.unsqueeze(1)  # (M, 1)
        n_groups = 1
    else:
        n_groups = norms.shape[1]

    if group_size is None:
        group_size = N

    W_approx = torch.zeros(M, N, dtype=torch.float32)

    for g_idx in range(n_groups):
        g_start = g_idx * group_size
        g_end = min(g_start + group_size, N)
        g_dim = g_end - g_start

        group_indices = indices[:, g_start:g_end]
        Y_quant = centroids_t[group_indices]

        scale = math.sqrt(g_dim)
        Y_unscaled = Y_quant / scale

        Pi = generate_rotation_matrix(g_dim, seed=seed + g_start)
        W_g = Y_unscaled @ Pi  # inverse rotation

        if n_groups == 1:
            group_norms = norms  # (M, 1)
        else:
            group_norms = norms[:, g_idx:g_idx+1]

        W_approx[:, g_start:g_end] = W_g * group_norms

    return W_approx


def _layer_name_to_key(layer_name: str) -> str:
    """
    Convert sanitized layer name back to original weight key.
    layers_0_attention_wq -> layers.0.attention.wq.weight
    """
    # Split on underscores and reconstruct
    parts = layer_name.split("_")

    # Pattern: layers_N_attention_wX or layers_N_feed_forward_wX
    if parts[0] == "layers":
        layer_num = parts[1]
        if parts[2] == "attention":
            return f"layers.{layer_num}.attention.{parts[3]}.weight"
        elif parts[2] == "feed" and parts[3] == "forward":
            return f"layers.{layer_num}.feed_forward.{parts[4]}.weight"

    return layer_name.replace("_", ".") + ".weight"


def _assign_weights(model: VoxtralTTS, state_dict: dict):
    """Map original Voxtral weight keys to our model structure."""
    config = model.config

    for key, tensor in state_dict.items():
        try:
            _set_weight(model, key, tensor)
        except Exception as e:
            print(f"  Warning: could not assign {key}: {e}")


def _set_weight(model: VoxtralTTS, key: str, tensor: torch.Tensor):
    """Set a single weight in the model by its original key name."""

    # LLM backbone embeddings
    if key == "mm_audio_embeddings.tok_embeddings.weight":
        model.backbone.tok_embeddings.weight.data = tensor
        return
    if key == "mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight":
        model.audio_codebook_embeddings.weight.data = tensor
        return

    # LLM backbone layers
    if key.startswith("layers."):
        parts = key.split(".")
        layer_idx = int(parts[1])
        layer = model.backbone.layers[layer_idx]

        if parts[2] == "attention":
            attn = layer.attention
            if parts[3] == "wq": attn.wq.weight.data = tensor
            elif parts[3] == "wk": attn.wk.weight.data = tensor
            elif parts[3] == "wv": attn.wv.weight.data = tensor
            elif parts[3] == "wo": attn.wo.weight.data = tensor
        elif parts[2] == "attention_norm":
            layer.attention_norm.weight.data = tensor
        elif parts[2] == "feed_forward":
            ff = layer.feed_forward
            if parts[3] == "w1": ff.w1.weight.data = tensor
            elif parts[3] == "w2": ff.w2.weight.data = tensor
            elif parts[3] == "w3": ff.w3.weight.data = tensor
        elif parts[2] == "ffn_norm":
            layer.ffn_norm.weight.data = tensor
        return

    # LLM final norm
    if key == "norm.weight":
        model.backbone.norm.weight.data = tensor
        return

    # Acoustic transformer
    if key.startswith("acoustic_transformer."):
        remainder = key[len("acoustic_transformer."):]

        if remainder.startswith("layers."):
            parts = remainder.split(".")
            layer_idx = int(parts[1])
            layer = model.acoustic.layers[layer_idx]

            if parts[2] == "attention":
                attn = layer.attention
                if parts[3] == "wq": attn.wq.weight.data = tensor
                elif parts[3] == "wk": attn.wk.weight.data = tensor
                elif parts[3] == "wv": attn.wv.weight.data = tensor
                elif parts[3] == "wo": attn.wo.weight.data = tensor
            elif parts[2] == "attention_norm":
                layer.attention_norm.weight.data = tensor
            elif parts[2] == "feed_forward":
                ff = layer.feed_forward
                if parts[3] == "w1": ff.w1.weight.data = tensor
                elif parts[3] == "w2": ff.w2.weight.data = tensor
                elif parts[3] == "w3": ff.w3.weight.data = tensor
            elif parts[2] == "ffn_norm":
                layer.ffn_norm.weight.data = tensor
        elif remainder == "norm.weight":
            model.acoustic.norm.weight.data = tensor
        elif remainder == "input_projection.weight":
            model.acoustic.input_projection.weight.data = tensor
        elif remainder == "time_projection.weight":
            model.acoustic.time_projection.weight.data = tensor
        elif remainder == "llm_projection.weight":
            model.acoustic.llm_projection.weight.data = tensor
        elif remainder == "semantic_codebook_output.weight":
            model.acoustic.semantic_codebook_output.weight.data = tensor
        elif remainder == "semantic_codebook_output.bias":
            model.acoustic.semantic_codebook_output.bias.data = tensor
        elif remainder == "acoustic_codebook_output.weight":
            model.acoustic.acoustic_codebook_output.weight.data = tensor
        return

    # Codec decoder
    if key.startswith("audio_tokenizer."):
        remainder = key[len("audio_tokenizer."):]

        if remainder.startswith("quantizer.semantic_codebook.embedding_sum"):
            model.codec.semantic_embedding_sum.data = tensor
            return
        if remainder.startswith("quantizer.semantic_codebook.cluster_usage"):
            model.codec.semantic_cluster_usage.data = tensor
            return

        if remainder.startswith("decoder_blocks."):
            parts = remainder.split(".")
            block_idx = int(parts[1])

            # Input conv (block 0)
            if block_idx == 0:
                if "original0" in remainder:
                    model.codec.input_conv.weight_g.data = tensor
                elif "original1" in remainder:
                    model.codec.input_conv.weight_v.data = tensor
                return

            # Upsample convs (blocks 2, 4, 6)
            if block_idx in [2, 4, 6]:
                conv_idx = [2, 4, 6].index(block_idx)
                if "original0" in remainder:
                    model.codec.upsample_convs[conv_idx].weight_g.data = tensor
                elif "original1" in remainder:
                    model.codec.upsample_convs[conv_idx].weight_v.data = tensor
                return

            # Transformer stages (blocks 1, 3, 5, 7)
            if block_idx in [1, 3, 5, 7]:
                stage_idx = [1, 3, 5, 7].index(block_idx)
                stage = model.codec.transformer_stages[stage_idx]

                layer_idx = int(parts[3])
                layer = stage[layer_idx]

                submodule = parts[4]
                if submodule == "attention":
                    attn = layer.attention
                    param = parts[5]
                    if param == "wq": attn.wq.weight.data = tensor
                    elif param == "wk": attn.wk.weight.data = tensor
                    elif param == "wv": attn.wv.weight.data = tensor
                    elif param == "wo": attn.wo.weight.data = tensor
                    elif param == "q_norm": attn.qk_norm.q_norm.weight.data = tensor
                    elif param == "k_norm": attn.qk_norm.k_norm.weight.data = tensor
                elif submodule == "attention_norm":
                    layer.attention_norm.weight.data = tensor
                elif submodule == "attention_scale":
                    layer.attention_scale.data = tensor
                elif submodule == "feed_forward":
                    ff = layer.feed_forward
                    param = parts[5]
                    if param == "w1": ff.w1.weight.data = tensor
                    elif param == "w2": ff.w2.weight.data = tensor
                    elif param == "w3": ff.w3.weight.data = tensor
                elif submodule == "ffn_norm":
                    layer.ffn_norm.weight.data = tensor
                elif submodule == "ffn_scale":
                    layer.ffn_scale.data = tensor
                return

        if remainder.startswith("output_proj."):
            if "original0" in remainder:
                model.codec.output_proj.weight_g.data = tensor
            elif "original1" in remainder:
                model.codec.output_proj.weight_v.data = tensor
            return
