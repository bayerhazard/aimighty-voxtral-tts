"""
Voxtral-4B-TTS weight utilities.
Separates the 386 tensors into 3 components and identifies quantizable layers.
"""

import torch
from safetensors.torch import load_file
from pathlib import Path


# Key prefix patterns for each component
BACKBONE_LINEAR_PREFIXES = [
    "layers.",  # layers.{0-25}.attention.{wq,wk,wv,wo}.weight + feed_forward.{w1,w2,w3}.weight
]

BACKBONE_NORM_PREFIXES = [
    "norm.weight",  # final RMSNorm
]

ACOUSTIC_PREFIXES = [
    "acoustic_transformer.",
]

CODEC_PREFIXES = [
    "audio_tokenizer.",
]

EMBEDDING_KEYS = [
    "mm_audio_embeddings.tok_embeddings.weight",              # [131072, 3072] tied with output
    "mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight",  # [9088, 3072]
]


def is_backbone_linear(key: str) -> bool:
    """Check if a key belongs to a backbone nn.Linear layer (quantizable)."""
    if not key.startswith("layers."):
        return False
    # Only attention projections and FFN weights, not norms
    return any(part in key for part in [
        "attention.wq.weight",
        "attention.wk.weight",
        "attention.wv.weight",
        "attention.wo.weight",
        "feed_forward.w1.weight",
        "feed_forward.w2.weight",
        "feed_forward.w3.weight",
    ])


def is_backbone_norm(key: str) -> bool:
    """Check if a key is a backbone norm weight."""
    if key == "norm.weight":
        return True
    if key.startswith("layers.") and ("attention_norm" in key or "ffn_norm" in key):
        return True
    return False


def is_acoustic(key: str) -> bool:
    return key.startswith("acoustic_transformer.")


def is_codec(key: str) -> bool:
    return key.startswith("audio_tokenizer.")


def is_embedding(key: str) -> bool:
    return key in EMBEDDING_KEYS


def separate_weights(state_dict: dict) -> dict:
    """
    Separate state_dict into categorized groups.

    Returns dict with keys:
        'backbone_linear': tensors to quantize (182 nn.Linear weights)
        'backbone_norm': backbone norms to keep BF16 (53 tensors)
        'acoustic': acoustic transformer to keep BF16 (34 tensors)
        'codec': codec decoder to keep BF16 (114 tensors)
        'embedding': embedding tables (2 tensors)
        'unknown': anything not categorized (should be empty)
    """
    groups = {
        'backbone_linear': {},
        'backbone_norm': {},
        'acoustic': {},
        'codec': {},
        'embedding': {},
        'unknown': {},
    }

    for key, tensor in state_dict.items():
        if is_backbone_linear(key):
            groups['backbone_linear'][key] = tensor
        elif is_backbone_norm(key):
            groups['backbone_norm'][key] = tensor
        elif is_acoustic(key):
            groups['acoustic'][key] = tensor
        elif is_codec(key):
            groups['codec'][key] = tensor
        elif is_embedding(key):
            groups['embedding'][key] = tensor
        else:
            groups['unknown'][key] = tensor

    return groups


def print_weight_summary(groups: dict):
    """Print summary of separated weight groups."""
    total_params = 0
    total_bytes = 0

    for group_name, tensors in groups.items():
        params = sum(t.numel() for t in tensors.values())
        bytes_ = sum(t.numel() * t.element_size() for t in tensors.values())
        total_params += params
        total_bytes += bytes_

        print(f"\n{'='*60}")
        print(f"{group_name.upper()}: {len(tensors)} tensors, {params/1e6:.1f}M params, {bytes_/1e9:.3f} GB")
        print(f"{'='*60}")
        for key, tensor in sorted(tensors.items()):
            mb = tensor.numel() * tensor.element_size() / 1e6
            print(f"  {key}: {list(tensor.shape)} {tensor.dtype} ({mb:.1f} MB)")

    print(f"\n{'='*60}")
    print(f"TOTAL: {total_params/1e9:.2f}B params, {total_bytes/1e9:.3f} GB")
    print(f"{'='*60}")


def load_and_separate(model_dir: str) -> dict:
    """Load model weights and separate into components."""
    safetensors_path = Path(model_dir) / "consolidated.safetensors"
    print(f"Loading {safetensors_path}...")
    state_dict = load_file(str(safetensors_path))
    print(f"Loaded {len(state_dict)} tensors")

    groups = separate_weights(state_dict)
    print_weight_summary(groups)

    return groups
