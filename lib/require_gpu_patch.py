"""Refuse OA-STID Huber runs on CPU when CUDA was requested."""
from __future__ import annotations

import os
import socket
import sys
from typing import Any

import torch


def _cuda_requested(device: Any) -> bool:
    return str(getattr(device, 'device', device)).startswith('cuda')


def enforce_cuda_args(args: Any) -> Any:
    """Fail fast if CLI asks for CUDA but this process cannot use it."""
    dev = str(getattr(args, 'device', 'cuda:0'))
    if dev.startswith('cuda'):
        if not torch.cuda.is_available():
            _fatal_cuda_missing(dev)
        idx = 0
        if dev != 'cuda' and ':' in dev:
            try:
                idx = int(dev.split(':', 1)[1])
            except ValueError:
                idx = 0
        n = torch.cuda.device_count()
        if idx >= n:
            _fatal_cuda_missing(dev, extra=f'invalid device index {idx} (count={n})')
        name = torch.cuda.get_device_name(idx)
        print(
            f'[require_gpu] using {dev} ({name}, count={n})',
            flush=True,
        )
    elif dev == 'cpu' and os.environ.get('OASTID_ALLOW_CPU', '0') != '1':
        print(
            '[require_gpu] FATAL: device=cpu is not allowed '
            '(set OASTID_ALLOW_CPU=1 to override).',
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(2)
    return args


def _fatal_cuda_missing(dev: str, *, extra: str = '') -> None:
    msg = (
        f'[require_gpu] FATAL: requested {dev} but torch.cuda.is_available()=False.\n'
        f'  host={socket.gethostname()}\n'
        f'  CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")}\n'
    )
    if extra:
        msg += f'  detail={extra}\n'
    msg += '  Refusing silent CPU fallback. Re-submit with --gres=gpu:1 on a GPU node.'
    print(msg, file=sys.stderr, flush=True)
    raise SystemExit(2)


def install_require_gpu_patch(huber_module: Any) -> None:
    if getattr(huber_module, '_require_gpu_patched', False):
        return

    _orig_main = huber_module.main
    _orig_parse = huber_module.parse_args

    def parse_args(*args, **kwargs):
        return enforce_cuda_args(_orig_parse(*args, **kwargs))

    def main(*args, **kwargs):
        if not torch.cuda.is_available():
            _fatal_cuda_missing('cuda:0')
        return _orig_main(*args, **kwargs)

    parse_args.__wrapped__ = _orig_parse
    main.__wrapped__ = _orig_main
    huber_module.parse_args = parse_args
    huber_module.main = main
    huber_module._require_gpu_patched = True
    print('[require_gpu] patched parse_args + main (no CPU fallback)', flush=True)
