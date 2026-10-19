#!/bin/bash
#SBATCH --job-name=kv_baseline_profile
#SBATCH --output=slurm-%j.log
#SBATCH --error=slurm-%j.err
#SBATCH --partition=gpu4090
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

source /etc/profile
set -euo pipefail

echo "=================================================="
echo " 🚀 CPE Slurm GPU Testbed Baseline Profiler"
echo " Host: $(hostname)"
echo " Date: $(date)"
echo "=================================================="

# 1. Environment & CUDA Setup
module purge
module load cuda/12.2

echo "=== GPU Telemetry ==="
nvidia-smi

# 2. Virtual Environment Setup (Auto-create if not exists)
VENV_DIR="$HOME/.venv_kv_baseline"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install transformers accelerate pynvml psutil
else
    source "$VENV_DIR/bin/activate"
fi

# 3. Run Profiling for Tier 1: Qwen2.5-1.5B
echo "=================================================="
echo "▶ Running Tier 1 Profiling: Qwen/Qwen2.5-1.5B-Instruct"
echo "=================================================="
export MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"
python profile_baseline.py

# 4. Run Profiling for Tier 2: Qwen2.5-3B
echo "=================================================="
echo "▶ Running Tier 2 Profiling: Qwen/Qwen2.5-3B-Instruct"
echo "=================================================="
export MODEL_ID="Qwen/Qwen2.5-3B-Instruct"
python profile_baseline.py

echo "=================================================="
echo "✅ All baseline profiling jobs finished successfully!"
echo "=================================================="
