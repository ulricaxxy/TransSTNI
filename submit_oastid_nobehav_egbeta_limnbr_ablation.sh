#!/usr/bin/env bash
set -euo pipefail
# Train+eval_ckpt for egbeta_limnbr grid (paths relative to this package).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
chmod +x run_oastid_nobehav_egbeta_limnbr_ablation_slurm.sh \
  run_oastid_nobehav_egbeta_limnbr_eval_ckpt_slurm.sh

MAX_PARALLEL="${MAX_PARALLEL:-4}"
POLL_SEC="${POLL_SEC:-30}"
META_ITERS="${META_ITERS:-50}"
SAGE_K="${SAGE_K:-3}"
PY="${PY:-/home/echoncu/echoncu/conda_envs/oagnn/bin/python}"
export PYTHONDONTWRITEBYTECODE=1

VARIANTS=(
  nobehav nosem_nobehav nobsts_nobehav nobsts_nosem_nobehav
  nobeta_nobehav nosem_nobeta_nobehav nobsts_nobeta_nobehav nobsts_nosem_nobeta_nobehav
)
SOURCES=(sd gla gba)

is_noz() { case "$1" in *nobeta*) return 0;; *) return 1;; esac; }

tag_of() {
  local v=$1 s=$2 p proto
  case $s in
    sd) p=full_time_erm_sage_subgraph_r050_075_hd01_sd_n358_537;;
    gla) p=full_time_erm_sage_subgraph_r050_075_hd01_gla_n1917_2875;;
    gba) p=full_time_erm_sage_subgraph_r050_075_hd01_gba_n1176_1764;;
  esac
  if is_noz "$v"; then proto=s1only; else proto="s2i${META_ITERS}"; fi
  echo "${p}_huber_h12_mask2_pret35_freezesemv2_b6_egbeta_limnbr_${proto}_abl_${v}"
}
ckpt_of(){ echo "output/oastid/largest_$1_$(tag_of "$2" "$1")/best.pt"; }
tgts(){ case $1 in sd) echo gla gba;; gla) echo sd gba;; gba) echo sd gla;; esac; }
count(){ squeue -u "$USER" -h 2>/dev/null | grep -c . || true; }

submit(){
  local mode=$1 v=$2 s=$3 jid
  if [[ $mode == train ]]; then
    jid=$(sbatch --parsable --qos=normal --chdir="$ROOT" --job-name="oastid_egb_${v}_${s}" \
      --export=ALL,META_ITERS=$META_ITERS,SAGE_K=$SAGE_K,PY=$PY,OASTID_ROOT=$ROOT \
      "$ROOT/run_oastid_nobehav_egbeta_limnbr_ablation_slurm.sh" "$v" "$s")
  else
    jid=$(sbatch --parsable --qos=normal --chdir="$ROOT" --job-name="oastid_egbec_${v}_${s}" \
      --export=ALL,META_ITERS=$META_ITERS,SAGE_K=$SAGE_K,PY=$PY,OASTID_ROOT=$ROOT \
      "$ROOT/run_oastid_nobehav_egbeta_limnbr_eval_ckpt_slurm.sh" "$v" "$s")
  fi
  echo "[egbeta-limnbr] $mode $v $s -> $jid (n=$(count))" >&2
}

PENDING=()
for v in "${VARIANTS[@]}"; do for s in "${SOURCES[@]}"; do
  [[ -f $(ckpt_of "$s" "$v") ]] || PENDING+=("train $v $s")
done; done
echo "[egbeta-limnbr] train pending ${#PENDING[@]} root=${ROOT}" >&2
while ((${#PENDING[@]}>0)); do
  r=$(count)
  while [[ $r -lt $MAX_PARALLEL && ${#PENDING[@]} -gt 0 ]]; do
    read -r m v s <<<"${PENDING[0]}"; PENDING=("${PENDING[@]:1}")
    submit "$m" "$v" "$s"; r=$(count)
  done
  ((${#PENDING[@]}==0)) && break
  sleep "$POLL_SEC"
done
while [[ $(count) -gt 0 ]]; do sleep "$POLL_SEC"; done

PENDING=()
for v in "${VARIANTS[@]}"; do for s in "${SOURCES[@]}"; do
  ck=$(ckpt_of "$s" "$v"); [[ -f $ck ]] || continue
  tag=$(tag_of "$v" "$s"); miss=0
  for t in $(tgts "$s"); do
    [[ -f output/cross_domain/oastid_largest_${s}_to_largest_${t}_${tag}_eval_ckpt/largest_${t}_metrics.json ]] || miss=1
  done
  [[ $miss -eq 1 ]] && PENDING+=("eval $v $s")
done; done
echo "[egbeta-limnbr] eval pending ${#PENDING[@]}" >&2
while ((${#PENDING[@]}>0)); do
  r=$(count)
  while [[ $r -lt $MAX_PARALLEL && ${#PENDING[@]} -gt 0 ]]; do
    read -r m v s <<<"${PENDING[0]}"; PENDING=("${PENDING[@]:1}")
    submit "$m" "$v" "$s"; r=$(count)
  done
  ((${#PENDING[@]}==0)) && break
  sleep "$POLL_SEC"
done
echo "[egbeta-limnbr] all done" >&2
