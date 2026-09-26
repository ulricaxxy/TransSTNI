#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
POLL_SEC="${POLL_SEC:-30}"
PY="${PY:-/home/echoncu/echoncu/conda_envs/oagnn/bin/python}"
export PYTHONDONTWRITEBYTECODE=1
export OASTID_ROOT="$ROOT" PY META_ITERS=50 SAGE_K=3 FORCE_EVAL=1 FORCE_TRAIN=0

count(){ squeue -u "$USER" -h 2>/dev/null | grep -c . || true; }

wait_slot(){
  while [[ $(count) -ge $MAX_PARALLEL ]]; do
    echo "[max4] queue=$(count) >= $MAX_PARALLEL, sleep ${POLL_SEC}s" >&2
    sleep "$POLL_SEC"
  done
}

submit_when_slot(){
  local desc=$1; shift
  wait_slot
  local jid
  jid=$(sbatch --parsable --qos=normal --chdir="$ROOT" "$@")
  echo "[max4] $desc -> $jid (n=$(count))" >&2
}

# --- wait for SD α trains already in flight (if any) ---
echo "[max4] waiting for running egb_maermse_a*_sd trains (if any)" >&2
while squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -qE '^egb_maermse_a[0-9]'; do
  echo "[max4] trains still running, n=$(count)" >&2
  sleep "$POLL_SEC"
done

# --- SD α evals ---
for a in 2.0 3.0 5.0; do
  atag="a$(printf %s "$a" | tr . p)"
  tag="full_time_erm_sage_subgraph_r050_075_hd01_sd_n358_537_s1mae_s2maermse_${atag}_mask2_pret35_freezesemv2_b6_egbeta_limnbr_s2i50_abl_nobehav"
  ck="output/oastid/largest_sd_${tag}/best.pt"
  m1="output/cross_domain/oastid_largest_sd_to_largest_gla_${tag}_eval_ckpt/largest_gla_metrics.json"
  m2="output/cross_domain/oastid_largest_sd_to_largest_gba_${tag}_eval_ckpt/largest_gba_metrics.json"
  if [[ ! -f $ck ]]; then
    echo "[max4] WARN skip eval α=$a missing ckpt" >&2
    continue
  fi
  if [[ -f $m1 && -f $m2 && "${FORCE_EVAL:-1}" != "1" ]]; then
    echo "[max4] skip eval α=$a already done" >&2
    continue
  fi
  # force eval even if exists when FORCE_EVAL=1 — but skip if both exist to save GPU unless forced
  if [[ -f $m1 && -f $m2 ]]; then
    echo "[max4] eval α=$a metrics exist; still FORCE_EVAL -> resubmit" >&2
  fi
  submit_when_slot "eval α=$a sd" \
    --job-name="egb_maermseE_a${a}_sd" \
    --export=ALL,STAGE2_LOSS_ALPHA=$a,META_ITERS=50,SAGE_K=3,FORCE_EVAL=1,PY=$PY,OASTID_ROOT=$ROOT \
    "$ROOT/run_oastid_nobehav_mae_rmsema_eval_ckpt_slurm.sh" nobehav sd
done

# --- s1only evals (sd/gla/gba) if ckpt exists and metrics missing ---
s1_tag(){
  case $1 in
    sd) echo "full_time_erm_sage_subgraph_r050_075_hd01_sd_n358_537_huber_h12_mask2_pret35_freezesemv2_b6_egbeta_limnbr_s1only_abl_nobehav";;
    gla) echo "full_time_erm_sage_subgraph_r050_075_hd01_gla_n1917_2875_huber_h12_mask2_pret35_freezesemv2_b6_egbeta_limnbr_s1only_abl_nobehav";;
    gba) echo "full_time_erm_sage_subgraph_r050_075_hd01_gba_n1176_1764_huber_h12_mask2_pret35_freezesemv2_b6_egbeta_limnbr_s1only_abl_nobehav";;
  esac
}
tgts(){ case $1 in sd) echo gla gba;; gla) echo sd gba;; gba) echo sd gla;; esac; }

for s in sd gla gba; do
  tag=$(s1_tag "$s")
  ck="output/oastid/largest_${s}_${tag}/best.pt"
  [[ -f $ck ]] || { echo "[max4] skip s1only eval $s no ckpt"; continue; }
  miss=0
  for t in $(tgts "$s"); do
    [[ -f "output/cross_domain/oastid_largest_${s}_to_largest_${t}_${tag}_eval_ckpt/largest_${t}_metrics.json" ]] || miss=1
  done
  # Prefer formal eval_ckpt; submit if missing
  if [[ $miss -eq 0 ]]; then
    echo "[max4] skip s1only eval $s already have eval_ckpt" >&2
    continue
  fi
  submit_when_slot "eval s1only $s" \
    --job-name="egb_s1E_nobehav_${s}" \
    --export=ALL,FORCE_SKIP_STAGE2=1,META_ITERS=50,SAGE_K=3,FORCE_EVAL=1,PY=$PY,OASTID_ROOT=$ROOT \
    "$ROOT/run_oastid_nobehav_egbeta_limnbr_eval_ckpt_slurm.sh" nobehav "$s"
done

echo "[max4] all queued under cap=${MAX_PARALLEL}; waiting drain" >&2
while [[ $(count) -gt 0 ]]; do sleep "$POLL_SEC"; done
echo "[max4] all done $(date '+%F %T')" >&2
