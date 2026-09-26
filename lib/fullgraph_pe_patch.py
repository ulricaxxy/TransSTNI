"""Stage2 subgraph training: slice full-graph Laplacian PE instead of recomputing.

When ``--fullgraph_pe_slice 1``, intercept ``build_graph`` during ``sample_task``
so ``node_feat = F_full[S]`` while edges still come from the induced subgraph.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

_PE_CACHE: dict[str, Any] = {'active': False}


def _flag(args: Any) -> bool:
    return bool(int(getattr(args, 'fullgraph_pe_slice', 0)))


def install_fullgraph_pe_patch(huber_module: Any) -> None:
    if getattr(huber_module, '_fullgraph_pe_patched', False):
        return

    build_graph = huber_module.build_graph
    _orig_sample_task = huber_module.sample_task
    _orig_build_graph = build_graph
    _orig_np_choice = np.random.choice

    def build_graph_slice(adj_mx, node_pe_dim, device='cpu'):
        edge_index, edge_weight, _ = _orig_build_graph(adj_mx, node_pe_dim, device)
        if not _PE_CACHE.get('active'):
            _, _, node_feat = _orig_build_graph(adj_mx, node_pe_dim, device)
            return edge_index, edge_weight, node_feat

        full_nf = _PE_CACHE['node_feat']
        n = int(np.asarray(adj_mx).shape[0])
        if n == int(full_nf.shape[0]):
            return edge_index, edge_weight, full_nf

        s = _PE_CACHE.get('last_S')
        if s is None or len(s) != n:
            _, _, node_feat = _orig_build_graph(adj_mx, node_pe_dim, device)
            return edge_index, edge_weight, node_feat

        idx = torch.as_tensor(s, device=full_nf.device, dtype=torch.long)
        return edge_index, edge_weight, full_nf.index_select(0, idx)

    def sample_task(*args, **kwargs):
        if not _PE_CACHE.get('active'):
            return _orig_sample_task(*args, **kwargs)

        def _choice(n, size, replace=False):
            s = _orig_np_choice(n, size, replace=replace)
            _PE_CACHE['last_S'] = np.sort(s)
            return s

        huber_module.build_graph = build_graph_slice
        np.random.choice = _choice
        try:
            return _orig_sample_task(*args, **kwargs)
        finally:
            np.random.choice = _orig_np_choice
            huber_module.build_graph = build_graph_slice

    def _wrap_meta(_orig_meta):
        def meta(args, model, device, Xtr, Ytr, Itr, A, *rest, **kwargs):
            if _flag(args):
                _, _, nf_full = _orig_build_graph(A, args.node_pe_dim, device)
                _PE_CACHE['node_feat'] = nf_full
                _PE_CACHE['active'] = True
                _PE_CACHE.pop('last_S', None)
                print(
                    f'[fullgraph_pe] cached full-graph PE shape={tuple(nf_full.shape)} '
                    f'for Stage2 subgraph slicing',
                    flush=True,
                )
            huber_module.build_graph = build_graph_slice
            try:
                return _orig_meta(args, model, device, Xtr, Ytr, Itr, A, *rest, **kwargs)
            finally:
                _PE_CACHE['active'] = False
                _PE_CACHE.pop('node_feat', None)
                _PE_CACHE.pop('last_S', None)
                huber_module.build_graph = _orig_build_graph

        return meta

    huber_module.build_graph = build_graph_slice
    huber_module.sample_task = sample_task
    if hasattr(huber_module, 'meta_train'):
        huber_module.meta_train = _wrap_meta(huber_module.meta_train)
    if hasattr(huber_module, 'meta_train_maml'):
        huber_module.meta_train_maml = _wrap_meta(huber_module.meta_train_maml)

    huber_module._fullgraph_pe_patched = True
    print('[fullgraph_pe] patched build_graph + sample_task + meta_train(_maml)', flush=True)
