#!/usr/bin/env bash
#SBATCH --job-name=stgfsl_eff
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --exclude=gpu07

set -euo pipefail
if [[ -n "${STGFSL_ROOT:-}" ]]; then
  ROOT="$STGFSL_ROOT"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/bench_efficiency.py" ]]; then
  ROOT="$SLURM_SUBMIT_DIR"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$ROOT"
mkdir -p "$ROOT/logs" "$ROOT/output/efficiency"

TEST_DATASET="${1:?usage: sbatch run_eff_slurm.sh largest_sd|largest_gba|largest_gla}"
MODEL="${MODEL:-GRU}"

exec >"$ROOT/logs/stgfsl_eff_${TEST_DATASET}_${SLURM_JOB_ID:-manual}.out" \
    2>"$ROOT/logs/stgfsl_eff_${TEST_DATASET}_${SLURM_JOB_ID:-manual}.err"

PY="${PY:-/home/echoncu/echoncu/conda_envs/oagnn/bin/python}"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[slurm] host=$(hostname) date=$(date '+%F %T') eff ${TEST_DATASET}"
"$PY" - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA unavailable'
print('cuda', torch.cuda.get_device_name(0))
PY

"$PY" -B bench_efficiency.py --test_dataset "${TEST_DATASET}" --model "${MODEL}"
echo "[done] $(date '+%F %T')"
