"""Completely remove ``beta_net`` + ``ln_beta`` (Orbit / GraphSAGE spatial Z).

Unlike ``node_dim=0`` nospace (GraphSAGE still runs, output zeroed), this path:

* skips ``_maybe_swap_orbit_gnn`` (no GraphSAGE module),
* deletes ``beta_net`` / ``ln_beta`` after ``build_oastid_model``,
* uses a forward that never calls the spatial encoder,
* keeps ``node_dim=0`` / ``node_pe_dim=0`` so ``hidden_dim`` matches fusion width.

Enable with ``--no_beta_net 1`` (plus the usual ablation flags for BSTS / sem).
"""
from __future__ import annotations

from typing import Any

import torch


def _no_beta_net_flag(args: Any) -> bool:
    return bool(int(getattr(args, 'no_beta_net', 0)))


def _forward_no_beta_net(
    self,
    source,
    edge_index,
    edge_weight,
    node_feat,
    tid_idx=None,
    dow_idx=None,
    sem_emb=None,
    behav_pi=None,
):
    x = source[..., : self.input_dim]
    B, T, N, _ = x.shape
    if T != self.input_len:
        raise ValueError(f'expected input_len={self.input_len}, got T={T}')
    x_series = x
    if getattr(self, 'use_cal_in_series', False):
        if tid_idx is None or dow_idx is None:
            raise ValueError('use_cal_in_series=1 requires tid_idx and dow_idx')
        tod_c, dow_c = self._cal_channels(tid_idx, dow_idx, B, T, N)
        x_series = torch.cat([x, tod_c, dow_c], dim=-1)
    x_flat = (
        x_series.transpose(1, 2)
        .contiguous()
        .view(B, N, -1)
        .transpose(1, 2)
        .unsqueeze(-1)
    )
    ts_emb = self.time_series_emb_layer(x_flat)
    feats = [ts_emb]
    if self.use_time_emb:
        if tid_idx is None or dow_idx is None:
            raise ValueError('use_time_emb=1 requires tid_idx and dow_idx')
        tid_q, dow_q = tid_idx, dow_idx
        if getattr(self, 'time_emb_last_step', False):
            tid_q, dow_q = self._window_last_cal_idx(tid_idx, dow_idx, T)
        tid = self.time_in_day_emb[tid_q]
        dow = self.day_in_week_emb[dow_q]
        feats.append(tid.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, N, 1))
        feats.append(dow.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, N, 1))
    feats.extend(self._sem_feats(sem_emb, B, N))
    feats.extend(self._bsts_feats(x, edge_index, edge_weight))
    feats.extend(self._behav_feats(behav_pi, B, N))
    hidden = torch.cat(feats, dim=1)
    hidden = self.encoder(hidden)
    return self.regression_layer(hidden)


def _strip_beta_modules(model: torch.nn.Module) -> None:
    for name in ('beta_net', 'ln_beta'):
        if name in model._modules:
            del model._modules[name]
    model._no_beta_net = True


def install_nobeta_orbit_patch(huber_module: Any) -> None:
    if getattr(huber_module, '_nobeta_orbit_patched', False):
        return

    base = getattr(huber_module, 'base', None)
    cls = getattr(base, 'OASTIDTransfer', None) if base is not None else None
    if cls is None:
        raise RuntimeError('nobeta patch: OASTIDTransfer not found on huber base')

    _orig_forward = cls.forward

    def forward(self, *args, **kwargs):
        if getattr(self, '_no_beta_net', False):
            return _forward_no_beta_net(self, *args, **kwargs)
        return _orig_forward(self, *args, **kwargs)

    cls.forward = forward  # type: ignore[method-assign]

    def _patch_build_on(mod: Any) -> None:
        _orig_build = mod.build_oastid_model

        def build_oastid_model_nobeta(args, *rest, **kwargs):
            if _no_beta_net_flag(args):
                args.node_dim = 0
                args.node_pe_dim = 0
            model = _orig_build(args, *rest, **kwargs)
            if _no_beta_net_flag(args):
                _strip_beta_modules(model)
                print(
                    '[nobeta] removed beta_net+ln_beta; '
                    f'hidden_dim={model.hidden_dim}',
                    flush=True,
                )
            return model

        mod.build_oastid_model = build_oastid_model_nobeta

    _patch_build_on(huber_module)
    if base is not None and getattr(base, 'build_oastid_model', None) is not huber_module.build_oastid_model:
        _patch_build_on(base)

    def _patch_swap_on(mod: Any) -> None:
        if not hasattr(mod, '_maybe_swap_orbit_gnn'):
            return
        _orig_swap = mod._maybe_swap_orbit_gnn

        def _maybe_swap_orbit_gnn_nobeta(model, args):
            if _no_beta_net_flag(args):
                print('[nobeta] skip orbit_gnn swap (no beta_net)', flush=True)
                return model
            return _orig_swap(model, args)

        mod._maybe_swap_orbit_gnn = _maybe_swap_orbit_gnn_nobeta

    if base is not None:
        _patch_swap_on(base)

    huber_module._nobeta_orbit_patched = True
    print('[nobeta] patched OASTIDTransfer.forward + build_oastid_model', flush=True)
