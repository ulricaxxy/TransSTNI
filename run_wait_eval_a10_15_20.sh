#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/echoncu/echoncu/iclr27-main/oastid_egbeta_limnbr_k3_full
cd "$ROOT"
LOG=logs/submit_nobehav_mae_rmsema_sd_a10_15_20_$(date +%Y%m%d_%H%M%S).log
exec >>"$LOG" 2>&1
MAX_PARALLEL=4
POLL=30
PY=/home/echoncu/echoncu/conda_envs/oagnn/bin/python
export OASTID_ROOT=$ROOT PY
ALPHAS=(10.0 15.0 20.0)
TRAIN_NAMES=(egb_maermse_a10.0_sd egb_maermse_a15.0_sd egb_maermse_a20.0_sd)
count(){ squeue -u "$USER" -h 2>/dev/null | grep -c . || true; }
echo "[a10+] waiting trains..."
while squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -qE '^egb_maermse_a(10|15|20)\.0_sd$'; do
  echo "[a10+] trains running n=$(count)"; sleep $POLL
done
echo "[a10+] eval phase"
for a in "${ALPHAS[@]}"; do
  atag="a$(printf %s "$a" | tr . p)"
  tag="full_time_erm_sage_subgraph_r050_075_hd01_sd_n358_537_s1mae_s2maermse_${atag}_mask2_pret35_freezesemv2_b6_egbeta_limnbr_s2i50_abl_nobehav"
  ck="output/oastid/largest_sd_${tag}/best.pt"
  if [[ ! -f $ck ]]; then echo "WARN miss $ck"; continue; fi
  while [[ $(count) -ge $MAX_PARALLEL ]]; do sleep $POLL; done
  jid=$(sbatch --parsable --qos=normal --chdir="$ROOT" --job-name="egb_maermseE_a${a}_sd" \
    --export=ALL,STAGE2_LOSS_ALPHA=$a,META_ITERS=50,SAGE_K=3,FORCE_EVAL=1,PY=$PY,OASTID_ROOT=$ROOT \
    "$ROOT/run_oastid_nobehav_mae_rmsema_eval_ckpt_slurm.sh" nobehav sd)
  echo "[a10+] eval α=$a -> $jid n=$(count)"
done
while [[ $(count) -gt 0 ]]; do sleep $POLL; done
echo "[a10+] all done $(date '+%F %T')"
