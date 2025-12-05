#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=2
#SBATCH --job-name=YOLOv11-baseline-enhancement
#SBATCH --account=harshil
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=training-logs/%x-%j.out
#SBATCH --error=training-logs/%x-%j.err

# ============================
# User-configurable section
# ============================
SCRIPT_PATH="/home/harshil/Fish-Detection-Kaggle/yolo11_head_attention.py"
PYTHON_BIN="python3"  # Use system Python
PYTHONPATH_DIR="/home/harshil"
LOG_DIR="./home/harshil/Fish-Detection-Kaggle/training-logs"
EXTRA_ARGS=""  # e.g., "--epochs 300 --batch 64"
# ============================

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Build log file path
SCRIPT_BASENAME="$(basename "$SCRIPT_PATH" .py)"
LOG_FILE="${LOG_DIR}/${SCRIPT_BASENAME}-${SLURM_JOB_ID}.log"

# Export PYTHONPATH (if your script needs local package imports)
export PYTHONPATH="${PYTHONPATH_DIR}:${PYTHONPATH:-}"
export TQDM_DISABLE=True

# Prefer Slurm's assigned GPU(s); otherwise auto-pick the most free GPU on the node
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Using Slurm-assigned GPU(s): ${CUDA_VISIBLE_DEVICES}"
else
  if command -v nvidia-smi >/dev/null 2>&1; then
    BEST_GPU=$(nvidia-smi --query-gpu=memory.free,index --format=csv,noheader,nounits \
               | sort -nr -k1 | head -1 | awk '{print $2}')
    export CUDA_VISIBLE_DEVICES="$BEST_GPU"
    echo "Auto-selected GPU with most free memory: ${BEST_GPU}"
  else
    echo "nvidia-smi not found; proceeding without setting CUDA_VISIBLE_DEVICES."
  fi
fi

echo "Python binary: $PYTHON_BIN ($(command -v "$PYTHON_BIN" || true))"
echo "Launching: $SCRIPT_PATH $EXTRA_ARGS"
srun "$PYTHON_BIN" -u "$SCRIPT_PATH" $EXTRA_ARGS 2>&1 | tee "$LOG_FILE"