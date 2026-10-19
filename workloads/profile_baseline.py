"""
Automatic Baseline VRAM & OOM Profiler for KV-Cache Co-Optimization
Target Testbed: NVIDIA GeForce RTX 4060 (8GB) & CPE Slurm Cluster
KMUTT Computer Engineering · CPE401 Senior Project
"""
import os
import sys
import time
import json
import torch
from datetime import datetime

# Set model from environment or fallback to Qwen2.5-1.5B
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
CONTEXT_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384, 32768]
GENERATE_TOKENS = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_gpu_info():
    if not torch.cuda.is_available():
        return {
            "gpu": "CPU",
            "vram_total_mb": 0,
            "cuda_available": False
        }
    props = torch.cuda.get_device_properties(0)
    return {
        "gpu": props.name,
        "vram_total_mb": round(props.total_memory / (1024 ** 2), 2),
        "cuda_version": torch.version.cuda,
        "cuda_available": True
    }

def run_profiling():
    print("=" * 65)
    print(" 🚀 KV-Cache Co-Optimization: Baseline VRAM & OOM Profiler")
    print("=" * 65)
    
    gpu_info = get_gpu_info()
    print(f"Hardware Target  : {gpu_info['gpu']}")
    print(f"Total VRAM       : {gpu_info['vram_total_mb']} MB")
    print(f"Selected Model   : {MODEL_ID}")
    print(f"Context Sweeps   : {CONTEXT_LENGTHS}")
    print("=" * 65 + "\n")

    if not gpu_info["cuda_available"]:
        print("❌ ERROR: CUDA is not available! A GPU is required for this profiling.")
        return

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("❌ ERROR: 'transformers' is not installed. Run: pip install transformers accelerate")
        return

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline_idle_vram = torch.cuda.memory_allocated() / (1024 ** 2)

    print("⏳ Loading tokenizer and model weights (FP16/BF16)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="cuda"
    )
    model.eval()

    weights_vram = torch.cuda.memory_allocated() / (1024 ** 2)
    print(f"✅ Model Weights VRAM Footprint: {weights_vram:.2f} MB\n")

    results = []
    oom_threshold = None

    for ctx in CONTEXT_LENGTHS:
        print(f"▶ Profiling Context Window: {ctx:,} tokens...")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Construct dummy prompt of target token length
        dummy_text = "The quick brown fox jumps over the lazy dog. " * (ctx // 10 + 1)
        input_ids = tokenizer.encode(dummy_text, return_tensors="pt", max_length=ctx, truncation=True).to(DEVICE)
        actual_input_len = input_ids.shape[1]

        try:
            start_time = time.perf_counter()
            with torch.no_grad():
                _ = model.generate(
                    input_ids,
                    max_new_tokens=GENERATE_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            torch.cuda.synchronize()
            total_time = time.perf_counter() - start_time

            peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
            kv_vram_est = peak_vram - weights_vram
            tpot = (total_time / GENERATE_TOKENS) * 1000  # ms/token

            print(f"   ✓ [PASS] Peak VRAM: {peak_vram:,.2f} MB | Est. KV-Cache: {kv_vram_est:,.2f} MB | TPOT: {tpot:.2f} ms/token")
            results.append({
                "context_length": actual_input_len,
                "status": "PASS",
                "peak_vram_mb": round(peak_vram, 2),
                "kv_vram_est_mb": round(kv_vram_est, 2),
                "tpot_ms_per_token": round(tpot, 2),
                "total_time_sec": round(total_time, 2)
            })

        except torch.cuda.OutOfMemoryError:
            print(f"   💥 [CUDA OOM CRASH] Physical VRAM limit exceeded at {actual_input_len:,} tokens!")
            oom_threshold = actual_input_len
            results.append({
                "context_length": actual_input_len,
                "status": "OOM_CRASH",
                "error": "torch.cuda.OutOfMemoryError"
            })
            torch.cuda.empty_cache()
            print(f"   ⚠️ Stopping further context sweeps (OOM Ceiling reached).\n")
            break

    # Construct Provenance Manifest matching kv_co-op_hub format
    gpu_slug = gpu_info['gpu'].replace(' ', '_').replace('/', '_')
    model_slug = MODEL_ID.split('/')[-1]
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')

    manifest = {
        "run_id": f"{timestamp_str}_{gpu_slug}_{model_slug}_baseline",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "hardware": {
            "gpu": gpu_info["gpu"],
            "vram_total_mb": gpu_info["vram_total_mb"],
            "cuda_version": gpu_info["cuda_version"]
        },
        "model_id": MODEL_ID,
        "weights_vram_mb": round(weights_vram, 2),
        "oom_threshold_ctx": oom_threshold,
        "results": results
    }

    output_filename = f"baseline_{model_slug}_{gpu_slug}.json"
    output_path = os.path.abspath(output_filename)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("=" * 65)
    print(f"📊 SUMMARY OF BASELINE PROFILING")
    print(f" - Model Weights    : {weights_vram:.2f} MB")
    print(f" - OOM Threshold    : {f'{oom_threshold:,} tokens (CRASH)' if oom_threshold else 'Did not crash within sweep range'}")
    print(f" - Golden Manifest  : {output_path}")
    print("=" * 65)

if __name__ == "__main__":
    run_profiling()
