"""
Lightweight audio post-processing for Voxtral-4B-TTS.

Three operations (~1.5ms total for 3s audio, CPU-only):
1. Low-pass filter at 11kHz — removes codec aliasing/hiss above speech band
2. Upsample 24kHz → 48kHz — standard output rate, better DAC compatibility
3. Peak normalization — prevent clipping

Optional:
4. Trim warmup frames — fixes garbled start (known Voxtral bug, HF discussion #20)
"""

import numpy as np
from scipy.signal import butter, sosfilt, resample_poly

# Pre-compute 6th-order Butterworth LPF at 10kHz
# 11kHz was too gentle — codec noise lives in 8-11kHz band
# 10kHz with steeper rolloff (6th order) cuts noise while preserving speech clarity
_LPF_SOS = butter(6, 10000, btype='low', fs=24000, output='sos')


def postprocess_audio(
    audio: np.ndarray,
    input_sr: int = 24000,
    output_sr: int = 48000,
    target_peak: float = 0.95,
) -> np.ndarray:
    """
    Post-process Voxtral TTS output for production quality.

    Cost: ~1.5ms for 3 seconds of audio (CPU-only, negligible vs generation time).

    Steps:
    1. Butterworth LPF at 11kHz — kills hiss from codec upsampling artifacts
    2. Polyphase upsample to 48kHz — standard playback rate
    3. Peak normalize to 0.95

    Args:
        audio: float32 numpy array at input_sr
        input_sr: input sample rate (24000 for Voxtral)
        output_sr: target sample rate (48000 recommended)
        target_peak: peak normalization target

    Returns:
        Processed audio at output_sr
    """
    if len(audio) < 2:
        return audio

    # Step 1: Low-pass filter (removes aliasing above 11kHz)
    audio = sosfilt(_LPF_SOS, audio).astype(np.float32)

    # Step 2: Upsample
    if output_sr != input_sr:
        ratio = output_sr // input_sr
        audio = resample_poly(audio, up=ratio, down=1).astype(np.float32)

    # Step 3: Peak normalize
    peak = np.abs(audio).max()
    if peak > 1e-6:
        audio = audio * (target_peak / peak)

    return audio


def trim_warmup_frames(all_codes: list) -> list:
    """Trim leading run of identical semantic codes before codec decode.

    Fixes garbled audio at start of generation — the model sometimes repeats
    the same semantic code 2-6 times at the start, producing noise bursts.
    Known issue: HuggingFace mistralai/Voxtral-4B-TTS-2603 discussion #20.

    Args:
        all_codes: list of (B, 37) code tensors, one per frame

    Returns:
        Trimmed list with leading repeated codes removed
    """
    if len(all_codes) <= 2:
        return all_codes

    first_code = all_codes[0][0, 0].item()

    # Check if second frame has the same semantic code
    if all_codes[1][0, 0].item() != first_code:
        return all_codes  # No warmup repetition

    # Find where the repeated code ends
    for i in range(2, min(len(all_codes), 30)):
        if all_codes[i][0, 0].item() != first_code:
            return all_codes[i:]

    return all_codes
