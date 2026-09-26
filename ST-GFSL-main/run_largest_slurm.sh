#!/usr/bin/env bash
#SBATCH --job-name=stgfsl_largest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --exclude=gpu07

set -euo pipefail
if [[ -n "${STGFSL_ROOT:-}" ]]; then
  ROOT="$STGFSL_ROOT"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/main.py" ]]; then
  ROOT="$SLURM_SUBMIT_DIR"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$ROOT"
mkdir -p "$ROOT/logs" "$ROOT/output"

TEST_DATASET="${1:?usage: sbatch run_largest_slurm.sh largest_sd|largest_gba|largest_gla}"
MODEL="${MODEL:-GRU}"
SOURCE_EPOCHS="${SOURCE_EPOCHS:-200}"
TARGET_EPOCHS="${TARGET_EPOCHS:-120}"
TARGET_DAYS="${TARGET_DAYS:-3}"
SEED_TAG="${SEED_TAG:-seed7}"

HORIZON_TAG="${HORIZON_TAG:-h12}"
LOG_TAG="${TEST_DATASET}_${MODEL}_${HORIZON_TAG}_s${SOURCE_EPOCHS}_t${TARGET_EPOCHS}_d${TARGET_DAYS}_${SEED_TAG}"
exec >"$ROOT/logs/stgfsl_${LOG_TAG}_${SLURM_JOB_ID:-manual}.out" \
    2>"$ROOT/logs/stgfsl_${LOG_TAG}_${SLURM_JOB_ID:-manual}.err"

PY="${PY:-/home/echoncu/echoncu/conda_envs/oagnn/bin/python}"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[slurm] host=$(hostname) date=$(date '+%F %T') root=${ROOT} test_dataset=${TEST_DATASET} model=${MODEL}"
"$PY" - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA unavailable'
print('cuda', torch.cuda.get_device_name(0), 'mem', round(torch.cuda.get_device_properties(0).total_memory/1024**3,1), 'GB')
PY

"$PY" main.py \
  --test_dataset "${TEST_DATASET}" \
  --model "${MODEL}" \
  --source_epochs "${SOURCE_EPOCHS}" \
  --target_epochs "${TARGET_EPOCHS}" \
  --target_days "${TARGET_DAYS}" \
  --memo "largest_${LOG_TAG}"

echo "[done] $(date '+%F %T')"
