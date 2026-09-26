"""Support ``node_dim=0`` / ``node_pe_dim=0`` (no spatial Orbit embedding).

GraphSAGE/GCN with zero output dims leaves ``UninitializedParameter`` entries;
the compiled Huber ``main()`` calls ``p.numel()`` before any forward pass and
crashes.  Same remedy as ``stage2/oastid_full_huber_stid_time.py``.
"""
from __future__ import annotations

import torch
from torch.nn.parameter import UninitializedParameter

_PATCHED = False


def install_nospace_orbit_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return

    _orig_numel = torch.Tensor.numel

    def _safe_numel(self):
        if isinstance(self, UninitializedParameter):
            return 0
        try:
            return int(_orig_numel(self))
        except ValueError:
            return 0

    torch.Tensor.numel = _safe_numel  # type: ignore[method-assign]
    UninitializedParameter.numel = _safe_numel  # type: ignore[method-assign]

    _orig_tf = UninitializedParameter.__torch_function__

    @classmethod
    def _safe_tf(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if getattr(func, '__name__', '') == 'numel':
            return 0
        try:
            return _orig_tf(func, types, args, kwargs)
        except ValueError:
            if getattr(func, '__name__', '') == 'numel':
                return 0
            raise

    UninitializedParameter.__torch_function__ = _safe_tf  # type: ignore[method-assign]
    _PATCHED = True
    print('[nospace/orbit] installed UninitializedParameter-safe numel patches', flush=True)
