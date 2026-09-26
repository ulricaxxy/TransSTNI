"""Wrap OA-STID forward with RevIN from ``utils.py``.

Intended pairing: **no dataset-level std scaler** (physical traffic in/out).
Per-sample, per-node mean/std on the lag window [B, T, N]:

  * STID / time-series MLP sees instance-normalized traffic
  * BSTS stats are computed on the **unnormalized** lag window
  * predictions are denormalized with the same (u, s) before Huber / MAE / Cross

``s`` is floored in ``revin_norm`` so overnight zeros do not explode denorm.
"""
from __future__ import annotations

from utils import revin_denorm, revin_norm


def wrap_forward_revin(model, s_min=1.0):
    orig = model.forward
    orig_bsts = model._bsts_feats
    model._revin_s_min = float(s_min)
    model._revin_raw_source = None

    def _bsts_feats(source, edge_index, edge_weight):
        raw = model._revin_raw_source
        return orig_bsts(raw if raw is not None else source, edge_index, edge_weight)

    def forward(source, *args, **kwargs):
        if source.dim() != 4:
            raise ValueError(
                f'RevIN wrap expects source [B,T,N,C], got {tuple(source.shape)}')
        x = source[..., 0]
        x_n, u, s = revin_norm(x, s_min=model._revin_s_min)
        if int(source.shape[-1]) == 1:
            source_n = x_n.unsqueeze(-1)
        else:
            source_n = source.clone()
            source_n[..., 0] = x_n
        model._revin_raw_source = source[..., : getattr(model, 'input_dim', 1)]
        try:
            pred = orig(source_n, *args, **kwargs)
        finally:
            model._revin_raw_source = None
        if pred.dim() != 4:
            raise ValueError(
                f'RevIN wrap expects pred [B,H,N,C], got {tuple(pred.shape)}')
        p_d = revin_denorm(pred[..., 0], u, s)
        if int(pred.shape[-1]) == 1:
            return p_d.unsqueeze(-1)
        out = pred.clone()
        out[..., 0] = p_d
        return out

    model._bsts_feats = _bsts_feats
    model.forward = forward
    return model


def install_noscale_dataloader(huber_module):
    """Force ``normalizer='None'`` on Huber Stage1/2 and ``oastid_full`` eval packs."""
    orig = huber_module.get_dataloader

    def get_dataloader(args, normalizer='std', *rest, **kwargs):
        return orig(args, 'None', *rest, **kwargs)

    huber_module.get_dataloader = get_dataloader
    from stage2 import oastid_full as base
    base.get_dataloader = get_dataloader
    print(
        "[revin] dataset scaler=None (NScaler); Huber/MAE stay on physical scale",
        flush=True,
    )
    return get_dataloader


def install_revin_build(huber_module, s_min=1.0):
    orig_build = huber_module.build_oastid_model

    def build_oastid_model(args):
        model = orig_build(args)
        wrap_forward_revin(model, s_min=float(getattr(args, 'revin_s_min', s_min)))
        print(
            '[revin] wrap model.forward: STID on RevIN(x), BSTS on raw x, '
            f'denorm pred; s_min={float(getattr(args, "revin_s_min", s_min)):g}; '
            'no dataset std',
            flush=True,
        )
        return model

    huber_module.build_oastid_model = build_oastid_model
    return build_oastid_model
