"""Global reproducibility seeding for OA-STID pipelines."""
from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int, *, verbose: bool = True) -> int:
    """Seed Python / NumPy / PyTorch (CPU+CUDA) and enable deterministic mode.

    Covers the RNGs used by this pipeline:
      * ``random`` / ``np.random`` / ``torch`` (incl. CUDA) — DataLoader shuffle,
        ``sample_task`` node/time choice, ``build_sizes``, ``F.dropout``
      * GraphSAGE neighbor sampling uses a **private** ``torch.Generator`` seeded
        from ``args.seed`` at module init (see ``lib/orbit_graphsage.py``)

    Residual non-determinism (usually small): CUDA kernels that ignore
    ``use_deterministic_algorithms(..., warn_only=True)`` (e.g. some scatter
    atomics). Stage1 has bit-matched across reruns on A800; Stage2 was previously
    dominated by unseeded ``build_sizes``.
    """
    seed = int(seed)

    os.environ.setdefault('PYTHONHASHSEED', str(seed))
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    if verbose:
        print(
            f'[repro/seed] seed={seed} '
            f'(py/np/torch/cuda, cudnn.deterministic=True, '
            f'CUBLAS_WORKSPACE_CONFIG={os.environ.get("CUBLAS_WORKSPACE_CONFIG")})',
            flush=True,
        )
    return seed


def seed_from_args(args: Any, *, verbose: bool = True) -> int:
    return seed_everything(int(getattr(args, 'seed', 10)), verbose=verbose)


def patch_parse_args(huber_module: Any) -> None:
    """Call ``seed_everything`` immediately after ``parse_args`` returns."""
    if getattr(huber_module, '_repro_seed_parse_patched', False):
        return

    _orig = huber_module.parse_args

    def parse_args(*args, **kwargs):
        parsed = _orig(*args, **kwargs)
        seed_from_args(parsed)
        return parsed

    parse_args.__wrapped__ = _orig
    huber_module.parse_args = parse_args
    huber_module._repro_seed_parse_patched = True

    print('[repro/seed] patched parse_args → seed_everything', flush=True)
