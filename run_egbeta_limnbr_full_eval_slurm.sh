#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_KEY="${1:?sd|gla|gba}"
exec sbatch --qos="${QOS:-normal}" --chdir="$ROOT" \
  --job-name="egb_full_ec_${SOURCE_KEY}" \
  "${ROOT}/run_oastid_nobehav_egbeta_limnbr_eval_ckpt_slurm.sh" nobehav "${SOURCE_KEY}"
