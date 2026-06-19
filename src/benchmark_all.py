"""
End-to-end benchmark: all model configurations × short/long text × Whisper accuracy.
Each config runs in a separate subprocess to isolate CUDA errors.
"""

import subprocess
import json
import sys
import os
import numpy as np

VENV_PYTHON = sys.executable  # Use current Python interpreter

WORKER_SCRIPT = """
import torch, gc, time, sys, os, json
import soundfile as sf
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent if '__file__' in dir() else '.'))
sys.path.insert(0, os.environ.get('SRC_DIR', '.'))

from generate_fast import generate_speech_fast
from generate import TekkenTokenizer

MODEL_DIR = os.environ.get("VOXTRAL_MODEL_DIR", str(Path(__file__).parent.parent / "models" / "original") if '__file__' in dir() else "../models/original")
VOICE_DIR = f"{MODEL_DIR}/voice_embedding"
VOICE = "neutral_female"

tok = TekkenTokenizer(f"{MODEL_DIR}/tekken.json")

import whisper
wh = whisper.load_model("base", device="cpu")

def run_gen(model, text, max_frames=300):
    audio, gen_time = generate_speech_fast(
        model, tok, text, voice_name=VOICE, voice_dir=VOICE_DIR,
        max_frames=max_frames, device="cuda", flow_steps=3, cfg_alpha=1.2)
    dur = len(audio) / 24000
    n_frames = int(dur * 12.5)
    fps = n_frames / gen_time if gen_time > 0 else 0
    rtf = gen_time / dur if dur > 0 else 0
    sf.write("/tmp/bench_tts.wav", audio, 24000)
    whisper_out = wh.transcribe("/tmp/bench_tts.wav", language="en")["text"].strip()
    return {"frames": n_frames, "time": round(gen_time, 2), "fps": round(fps, 1),
            "rtf": round(rtf, 3), "duration": round(dur, 1), "whisper": whisper_out}

config_name = sys.argv[1]

torch.cuda.reset_peak_memory_stats()

# Load model
if config_name == "bf16":
    from load_model import load_original_model
    model = load_original_model(MODEL_DIR, device="cuda")
elif config_name == "int4":
    from torchao_inference import load_model_int4
    model = load_model_int4(MODEL_DIR, device="cuda")
elif config_name == "int4_compile":
    import warnings; warnings.filterwarnings('ignore')
    os.environ['TORCHDYNAMO_VERBOSE'] = '0'
    from torchao_inference import load_model_int4
    model = load_model_int4(MODEL_DIR, device="cuda")
    import torch
    model.acoustic.predict_velocity = torch.compile(
        model.acoustic.predict_velocity, mode="default", fullgraph=False)
elif config_name == "int4_static_compile":
    import warnings; warnings.filterwarnings('ignore')
    os.environ['TORCHDYNAMO_VERBOSE'] = '0'
    from torchao_inference import load_model_int4
    model = load_model_int4(MODEL_DIR, device="cuda")
    import torch
    from generate_fast import enable_static_cache
    enable_static_cache(model, max_seq_len=700)
    model.backbone = torch.compile(model.backbone, mode="default", fullgraph=False)
    model.acoustic.predict_velocity = torch.compile(
        model.acoustic.predict_velocity, mode="default", fullgraph=False)

vram = round(torch.cuda.memory_allocated() / 1e9, 2)

# Warmup
with torch.inference_mode():
    generate_speech_fast(model, tok, "Hi.", voice_name=VOICE,
        voice_dir=VOICE_DIR, max_frames=5, device="cuda", flow_steps=3, cfg_alpha=1.2)

results = {"vram_gb": vram, "tests": []}

texts = [
    ("short", "Hello, how are you today?"),
    ("short", "The weather is nice outside."),
    ("long", "Artificial intelligence has transformed the way we interact with technology. "
             "From virtual assistants to autonomous vehicles, machine learning algorithms "
             "are reshaping industries and creating new possibilities for everyone."),
    ("long", "The ancient city of Rome was built over centuries by ambitious engineers. "
             "From the Colosseum to the aqueducts, their innovations continue to inspire "
             "modern builders and designers across the entire globe today."),
]

with torch.inference_mode():
    for length, text in texts:
        try:
            max_f = 200 if length == "short" else 400
            r = run_gen(model, text, max_frames=max_f)
            r["length"] = length
            r["text"] = text
            results["tests"].append(r)
        except Exception as e:
            results["tests"].append({"length": length, "text": text, "error": str(e)})

results["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
print("BENCH_RESULT:" + json.dumps(results))
"""


def run_config(name, config_key):
    """Run one config in a subprocess."""
    print(f"\n{'='*80}")
    print(f"  {name}")
    print(f"{'='*80}")

    proc = subprocess.run(
        [VENV_PYTHON, "-c", WORKER_SCRIPT, config_key],
        capture_output=True, text=True, timeout=600,
        cwd=os.path.dirname(__file__)
    )

    # Print loading output
    for line in proc.stderr.split("\n"):
        if line.strip() and "FutureWarning" not in line and "pynvml" not in line \
           and "Performing inference" not in line and "FP16 is not" not in line \
           and "vectorized_gather" not in line and "Assertion" not in line:
            print(f"  {line.strip()}")

    # Extract result
    for line in proc.stdout.split("\n"):
        if line.startswith("BENCH_RESULT:"):
            return json.loads(line[len("BENCH_RESULT:"):])

    print(f"  FAILED (exit code {proc.returncode})")
    if proc.returncode != 0:
        # Show last few error lines
        err_lines = [l for l in proc.stderr.split("\n") if l.strip()]
        for l in err_lines[-5:]:
            print(f"  ERR: {l.strip()}")
    return None


def main():
    print("=" * 80)
    print("VOXTRAL-4B-TTS — FULL END-TO-END BENCHMARK")
    print("=" * 80)

    configs = [
        ("1. BF16 original", "bf16"),
        ("2. int4 backbone (torchao HQQ)", "int4"),
        ("3. int4 + compile acoustic", "int4_compile"),
        ("4. int4 + static cache + compile all", "int4_static_compile"),
    ]

    all_results = {}
    for name, key in configs:
        try:
            result = run_config(name, key)
            if result:
                all_results[name] = result
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT")
        except Exception as e:
            print(f"  ERROR: {e}")

    # Print summary
    print(f"\n{'='*80}")
    print("SPEED SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Config':<40} {'VRAM':>6} {'Short fps':>10} {'Long fps':>10} {'Short RTF':>10} {'Long RTF':>10}")
    print(f"  {'-'*90}")

    for name, r in all_results.items():
        short_tests = [t for t in r["tests"] if t.get("length") == "short" and "error" not in t]
        long_tests = [t for t in r["tests"] if t.get("length") == "long" and "error" not in t]
        s_fps = np.mean([t["fps"] for t in short_tests]) if short_tests else 0
        l_fps = np.mean([t["fps"] for t in long_tests]) if long_tests else 0
        s_rtf = np.mean([t["rtf"] for t in short_tests]) if short_tests else 0
        l_rtf = np.mean([t["rtf"] for t in long_tests]) if long_tests else 0
        print(f"  {name:<40} {r['vram_gb']:>5.1f}G {s_fps:>9.1f} {l_fps:>9.1f} {s_rtf:>10.3f} {l_rtf:>10.3f}")

    print(f"\n{'='*80}")
    print("WHISPER ACCURACY")
    print(f"{'='*80}")

    for name, r in all_results.items():
        print(f"\n  [{name}]")
        for t in r["tests"]:
            if "error" in t:
                print(f"    ERROR: {t['error'][:60]}")
                continue
            in_t = t["text"][:55] + ("..." if len(t["text"]) > 55 else "")
            out_t = t["whisper"][:55] + ("..." if len(t["whisper"]) > 55 else "")
            print(f"    [{t['length']:<5}] {t['fps']:>5.0f} fps | {t['duration']:>4.1f}s")
            print(f"           In:  {in_t}")
            print(f"           Out: {out_t}")

    # Save
    with open("/tmp/benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to /tmp/benchmark_results.json")


if __name__ == "__main__":
    main()
