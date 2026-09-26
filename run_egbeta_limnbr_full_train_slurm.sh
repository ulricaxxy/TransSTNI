#!/usr/bin/env bash
# Thin wrapper: Full nobehav train.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_KEY="${1:?sd|gla|gba}"
exec sbatch --qos="${QOS:-normal}" --chdir="$ROOT" \
  --job-name="egb_full_${SOURCE_KEY}" \
  "${ROOT}/run_oastid_nobehav_egbeta_limnbr_ablation_slurm.sh" nobehav "${SOURCE_KEY}"
