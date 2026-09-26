#!/usr/bin/env bash
# Preflight: exit before training if this Slurm step has no usable CUDA GPU.
# Usage: oastid_require_gpu.sh [python_executable]
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
PY="${1:-python3}"
exec "${PY}" -B - <<'PY'
import os
import socket
import sys

try:
    import torch
except Exception as exc:
    print(f"[require_gpu] FATAL: cannot import torch: {exc}", file=sys.stderr)
    raise SystemExit(2)

host = socket.gethostname()
visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
if not torch.cuda.is_available():
    print(
        "[require_gpu] FATAL: torch.cuda.is_available()=False on this node.\n"
        f"  host={host}\n"
        f"  CUDA_VISIBLE_DEVICES={visible}\n"
        "  Ensure the job was submitted with --gres=gpu:1 on partition=gpu.",
        file=sys.stderr,
    )
    raise SystemExit(2)

n = torch.cuda.device_count()
name = torch.cuda.get_device_name(0)
print(f"[require_gpu] OK host={host} gpus={n} cuda:0={name} visible={visible}", flush=True)
PY
