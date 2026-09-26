#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/echoncu/echoncu/iclr27-main/oastid_egbeta_limnbr_k3_full
cd "$ROOT"
mkdir -p logs
LOG=logs/submit_mae_rmsema_gba_a50_$(date +%Y%m%d_%H%M%S).log
exec >>"$LOG" 2>&1
PY=/home/echoncu/echoncu/conda_envs/oagnn/bin/python
POLL=30
echo "[gba-a50] waiting train"
while squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -q '^egb_maermse_a50\.0_gba$'; do
  echo "[gba-a50] train running"; sleep $POLL
done
tag=full_time_erm_sage_subgraph_r050_075_hd01_gba_n1176_1764_s1mae_s2maermse_a50p0_mask2_pret35_freezesemv2_b6_egbeta_limnbr_s2i50_abl_nobehav
ck=output/oastid/largest_gba_${tag}/best.pt
if [[ ! -f $ck ]]; then echo "WARN missing $ck"; exit 1; fi
jid=$(sbatch --parsable --qos=normal --chdir="$ROOT" --job-name=egb_maermseE_a50.0_gba \
  --export=ALL,STAGE2_LOSS_ALPHA=50.0,META_ITERS=50,SAGE_K=3,FORCE_EVAL=1,PY=$PY,OASTID_ROOT=$ROOT \
  run_oastid_nobehav_mae_rmsema_eval_ckpt_slurm.sh nobehav gba)
echo "[gba-a50] eval -> $jid"
while squeue -u "$USER" -h | grep -q .; do sleep $POLL; done
echo "[gba-a50] all done $(date '+%F %T')"
