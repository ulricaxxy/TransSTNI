"""Huber MAML pret35; Stage2 does not *update* sem_proj.

Keep requires_grad=True so second-order MAML's autograd.grad still
sees a valid graph. Outer meta-grads on sem_proj are zeroed by hook,
so the Stage1 MLP (and thus Cross keys / π) stay fixed. Inner-loop
fast weights may still adapt a copy and are discarded after each task.
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

import stage2.oastid_full_huber as H

_orig_maml = H.meta_train_maml


def _drop_grad(g):
    return torch.zeros_like(g)


def _install_sem_proj_outer_freeze(model, tag='[behav/freezesem]'):
    proj = getattr(model, 'sem_proj', None)
    if proj is None:
        print(f'{tag} no sem_proj on model; skip outer freeze', flush=True)
        return []
    handles = []
    n = 0
    for p in proj.parameters():
        if not p.requires_grad:
            p.requires_grad = True
        handles.append(p.register_hook(_drop_grad))
        n += p.numel()
    print(f'{tag} sem_proj stays in the graph ({n} params) but outer '
          f'meta-grads are zeroed; Cross keys stay Stage1-fixed',
          flush=True)
    return handles


def _meta_train_maml_freezesem(args, model, *rest, **kwargs):
    handles = _install_sem_proj_outer_freeze(model)
    try:
        return _orig_maml(args, model, *rest, **kwargs)
    finally:
        for h in handles:
            h.remove()


H.meta_train_maml = _meta_train_maml_freezesem


if __name__ == '__main__':
    print('[behav/freezesem] entry: Stage1 trains sem_proj; Stage2 zeros '
          'its meta-grads (second_order safe)', flush=True)
    H.main()
