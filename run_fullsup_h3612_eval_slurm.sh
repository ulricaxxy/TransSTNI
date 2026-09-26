#!/usr/bin/env bash
#SBATCH --job-name=fullsup_h36
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --exclude=gpu07

# Re-eval existing fullsup ckpts with H3/H6/H12 + Average (no retrain).
set -euo pipefail
ROOT="${OASTID_ROOT:-/home/echoncu/echoncu/iclr27-main/oastid_egbeta_limnbr_k3_full}"
cd "$ROOT"
mkdir -p "$ROOT/logs"
exec >"$ROOT/logs/oastid_fullsup_h3612_${SLURM_JOB_NAME:-job}_${SLURM_JOB_ID:-manual}.out" \
    2>"$ROOT/logs/oastid_fullsup_h3612_${SLURM_JOB_NAME:-job}_${SLURM_JOB_ID:-manual}.err"

PY="${PY:-/home/echoncu/echoncu/conda_envs/oagnn/bin/python}"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
bash scripts/oastid_require_gpu.sh "${PY}"

STAGE="${1:?s1|s2}"
SOURCE_KEY="${2:?sd|gla|gba}"
EVAL=scripts/eval_last_fullsup_indomain.py

case "${SOURCE_KEY}" in
  sd)
    SOURCE=largest_sd
    TAG_PREFIX=full_time_erm_sage_subgraph_r050_075_hd01_sd_n358_537
    S2_CKPT="${ROOT}/output/oastid/${SOURCE}_${TAG_PREFIX}_huber_revin_nostd_calseq_last_h12_mask2_pret35_freezesemv2_b6_egbeta_limnbr_s2i50_abl_nobehav/best.pt"
    S1_CKPT="${ROOT}/output/oastid/${SOURCE}_${TAG_PREFIX}_huber_revin_nostd_calseq_last_h12_mask2_pret35_freezesemv2_b6_egbeta_limnbr_s1only_fullsup_abl_nobehav/best.pt"
    ;;
  gba)
    SOURCE=largest_gba
    TAG_PREFIX=full_time_erm_sage_subgraph_r050_075_hd01_gba_n1176_1764
    S2_CKPT="${ROOT}/output/oastid/${SOURCE}_${TAG_PREFIX}_huber_revin_nostd_calseq_last_h12_mask2_pret35_freezesemv2_b6_egbeta_limnbr_s2i50_abl_nobehav/best.pt"
    S1_CKPT="${ROOT}/output/oastid/${SOURCE}_${TAG_PREFIX}_huber_revin_nostd_calseq_last_h12_mask2_pret35_freezesemv2_b6_egbeta_limnbr_s1only_fullsup_abl_nobehav/best.pt"
    ;;
  gla)
    SOURCE=largest_gla
    TAG_PREFIX=full_time_erm_sage_subgraph_r050_075_hd01_gla_n1917_2875
    S2_CKPT="${ROOT}/output/oastid/${SOURCE}_${TAG_PREFIX}_huber_revin_nostd_calseq_last_frome40_clip5p0_h12_mask2_pret35_freezesemv2_b6_egbeta_limnbr_s2i50_abl_nobehav/best.pt"
    S1_CKPT="${ROOT}/output/oastid/${SOURCE}_${TAG_PREFIX}_huber_revin_nostd_calseq_last_e150_clip5p0_h12_mask2_pret35_freezesemv2_b6_egbeta_limnbr_s2i50_abl_nobehav/epoch_040.pt"
    ;;
  *) echo "unknown source"; exit 1 ;;
esac

if [[ "${STAGE}" == "s1" ]]; then
  CKPT="${S1_CKPT}"
elif [[ "${STAGE}" == "s2" ]]; then
  CKPT="${S2_CKPT}"
else
  echo "unknown stage"; exit 1
fi
[[ -f "${CKPT}" ]] || { echo "missing ${CKPT}"; exit 1; }

echo "[fullsup-h3612] $(hostname) $(date '+%F %T') ${STAGE} ${SOURCE} ${CKPT}"
"${PY}" -B "${EVAL}" --source "${SOURCE_KEY}" --ckpt "${CKPT}" --stage "${STAGE}"
echo "=== DONE ${STAGE} ${SOURCE} === $(date '+%F %T')"
