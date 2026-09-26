"""BSTS window statistics (original, unsimplified).

Classic A2TTA / STID stats, every lag-T window:
    [last, mean, std, slope] -> [B, N, 4]

Improved BSTS keeps **four independent groups**. Each group is computed in
full, passed through its **own nn.Linear**, then concatenated
(``stats_grouped=1`` path). Groups are never collapsed into one Linear on
the concatenated raw vector.

  1. s_temp  (8 if stats_v2 else 4)
  2. s_spat  (4, stats_nbr)   neighbor mean-pool on the road graph
  3. s_graph (2, stats_rank)  intra-graph percentile ranks
  4. s_spec  (6, stats_spec)  rFFT energy + near/far multi-scale

Function bodies match the recovered OA-STID BSTS implementation
(temporal_stats / neighbor_stats / rank_stats / spectral_stats /
compute_stats_groups), including keepdim layout [B, N, F], OLS
``clamp_min`` denom, three separate neighbor-mean calls, argsort-argsort
ranks, and spectral ``seg = max(2, T // 3)``.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def temporal_stats(x_btnc: torch.Tensor) -> torch.Tensor:
    """A2TTA-style window stats: last / mean / std / OLS-slope.

    Args:
        x_btnc: [B, T, N, C] history (uses channel 0 when C>1).
    Returns:
        stats: [B, N, 4]
    """
    x = x_btnc[..., 0]
    x_nt = x.permute(0, 2, 1).contiguous()
    T = x_nt.shape[-1]
    last = x_nt[..., -1:]
    mean = x_nt.mean(dim=-1, keepdim=True)
    std = x_nt.std(dim=-1, unbiased=False, keepdim=True)
    t = torch.arange(T, device=x_nt.device, dtype=x_nt.dtype)
    t = t - t.mean()
    denom = (t * t).sum().clamp_min(1e-06)
    slope = ((x_nt - mean) * t).sum(dim=-1, keepdim=True) / denom
    return torch.cat([last, mean, std, slope], dim=-1)


def _temporal_stats_parts(x_btnc: torch.Tensor):
    """Return (x_nt, last, mean, std, slope) for extended stats."""
    x = x_btnc[..., 0]
    x_nt = x.permute(0, 2, 1).contiguous()
    T = x_nt.shape[-1]
    last = x_nt[..., -1:]
    mean = x_nt.mean(dim=-1, keepdim=True)
    std = x_nt.std(dim=-1, unbiased=False, keepdim=True)
    t = torch.arange(T, device=x_nt.device, dtype=x_nt.dtype)
    t = t - t.mean()
    denom = (t * t).sum().clamp_min(1e-06)
    slope = ((x_nt - mean) * t).sum(dim=-1, keepdim=True) / denom
    return (x_nt, last, mean, std, slope)


def temporal_stats_v2_extra(x_nt: torch.Tensor, last: torch.Tensor, mean: torch.Tensor):
    """Extra 4-d window stats: min / max / range / (last - mean)."""
    vmin = x_nt.min(dim=-1, keepdim=True).values
    vmax = x_nt.max(dim=-1, keepdim=True).values
    vrange = vmax - vmin
    delta = last - mean
    return torch.cat([vmin, vmax, vrange, delta], dim=-1)


def _neighbor_mean(x_bnf: torch.Tensor, edge_index: torch.Tensor, num_nodes: int):
    """Mean over graph neighbors for [B, N, F]; isolated nodes fall back to self.

    Aggregation follows the original BSTS path: for each edge (row, col) =
    (edge_index[0], edge_index[1]), add ``x[col]`` onto ``row``, then divide
    by degree. Self-loops are **not** stripped (if the adjacency has them they
    participate in the mean). Isolated nodes (degree 0) keep their own value.
    """
    if edge_index is None or edge_index.numel() == 0:
        return x_bnf
    row, col = edge_index[0], edge_index[1]
    B, N, F = x_bnf.shape
    out = x_bnf.new_zeros(B, N, F)
    count = x_bnf.new_zeros(B, N, 1)
    src = x_bnf[:, col, :]
    row_b = row.view(1, -1, 1).expand(B, -1, F)
    out.scatter_add_(1, row_b, src)
    count.scatter_add_(
        1,
        row.view(1, -1, 1).expand(B, -1, 1),
        x_bnf.new_ones(B, row.numel(), 1),
    )
    has_nbr = count > 0
    out = out / count.clamp_min(1.0)
    return torch.where(has_nbr.expand_as(out), out, x_bnf)


def neighbor_stats(x_btnc: torch.Tensor, edge_index: torch.Tensor, num_nodes: int):
    """4-d neighbor summary: nbr_last / nbr_mean / nbr_slope / (last - nbr_last)."""
    _, last, mean, _std, slope = _temporal_stats_parts(x_btnc)
    nbr_last = _neighbor_mean(last, edge_index, num_nodes)
    nbr_mean = _neighbor_mean(mean, edge_index, num_nodes)
    nbr_slope = _neighbor_mean(slope, edge_index, num_nodes)
    diff_last = last - nbr_last
    return torch.cat([nbr_last, nbr_mean, nbr_slope, diff_last], dim=-1)


def rank_stats(last: torch.Tensor, mean: torch.Tensor):
    """2-d within-graph percentile ranks for last and mean ([B, N, 1] each)."""

    def _rank(v_bn1):
        v = v_bn1.squeeze(-1)
        b, n = v.shape
        if n <= 1:
            return v.new_zeros(b, n, 1)
        order = v.argsort(dim=-1).argsort(dim=-1).float()
        return (order / float(n - 1)).unsqueeze(-1)

    return torch.cat([_rank(last), _rank(mean)], dim=-1)


def _segment_slope(x_seg: torch.Tensor) -> torch.Tensor:
    """OLS slope on a short temporal segment [B, N, seg_len]."""
    T = x_seg.shape[-1]
    if T < 2:
        return x_seg.new_zeros(*x_seg.shape[:-1], 1)
    mean = x_seg.mean(dim=-1, keepdim=True)
    t = torch.arange(T, device=x_seg.device, dtype=x_seg.dtype)
    t = t - t.mean()
    denom = (t * t).sum().clamp_min(1e-06)
    return ((x_seg - mean) * t).sum(dim=-1, keepdim=True) / denom


def spectral_stats(x_nt: torch.Tensor) -> torch.Tensor:
    """6-d FFT + multi-scale stats: low/high/dom energy, near-far delta, segment slopes."""
    B, N, T = x_nt.shape
    xf = torch.fft.rfft(x_nt, dim=-1)
    power = xf.real.pow(2) + xf.imag.pow(2)
    E_total = power.sum(dim=-1, keepdim=True).clamp_min(1e-08)
    n_bins = power.shape[-1]
    k_low = min(2, n_bins)
    E_low = power[..., :k_low].sum(dim=-1, keepdim=True) / E_total
    if n_bins > k_low:
        E_high = power[..., k_low:].sum(dim=-1, keepdim=True) / E_total
        dom = (
            power[..., 1:].amax(dim=-1, keepdim=True) / E_total
            if n_bins > 1
            else x_nt.new_zeros(B, N, 1)
        )
    else:
        E_high = x_nt.new_zeros(B, N, 1)
        dom = x_nt.new_zeros(B, N, 1)
    seg = max(2, T // 3)
    if T >= 2 * seg:
        near = x_nt[..., -seg:]
        far = x_nt[..., :seg]
        ms_delta = near.mean(dim=-1, keepdim=True) - far.mean(dim=-1, keepdim=True)
        slope_near = _segment_slope(near)
        slope_far = _segment_slope(far)
    else:
        z = x_nt.new_zeros(B, N, 1)
        ms_delta, slope_near, slope_far = z, z, z
    return torch.cat([E_low, E_high, dom, ms_delta, slope_near, slope_far], dim=-1)


def compute_stats_groups(
    x_btnc: torch.Tensor,
    edge_index=None,
    num_nodes=None,
    stats_v2=False,
    stats_nbr=False,
    stats_rank=False,
    stats_spec=False,
):
    """BSTS groups: s_temp / s_spat / s_graph / s_spec (each may be None)."""
    x_nt, last, mean, std, slope = _temporal_stats_parts(x_btnc)
    temp = torch.cat([last, mean, std, slope], dim=-1)
    if stats_v2:
        temp = torch.cat([temp, temporal_stats_v2_extra(x_nt, last, mean)], dim=-1)
    spat = None
    if stats_nbr:
        if edge_index is None or num_nodes is None:
            raise ValueError('stats_nbr=1 requires edge_index and num_nodes')
        spat = neighbor_stats(x_btnc, edge_index, int(num_nodes))
    graph = rank_stats(last, mean) if stats_rank else None
    spec = spectral_stats(x_nt) if stats_spec else None
    return {'temp': temp, 'spat': spat, 'graph': graph, 'spec': spec}


def extended_temporal_stats(
    x_btnc: torch.Tensor,
    edge_index=None,
    num_nodes=None,
    stats_v2=False,
    stats_nbr=False,
    stats_rank=False,
    stats_spec=False,
):
    """Compose base 4-d stats with optional v2 / neighbor / rank / spectral extensions."""
    groups = compute_stats_groups(
        x_btnc,
        edge_index=edge_index,
        num_nodes=num_nodes,
        stats_v2=stats_v2,
        stats_nbr=stats_nbr,
        stats_rank=stats_rank,
        stats_spec=stats_spec,
    )
    parts = [groups['temp']]
    for key in ('spat', 'graph', 'spec'):
        if groups[key] is not None:
            parts.append(groups[key])
    return torch.cat(parts, dim=-1)


def resolve_stats_raw_group_dims(args):
    """Raw dims per BSTS group (zeros when group disabled)."""
    if not bool(getattr(args, 'use_bsts', getattr(args, 'use_temporal_stats', 0))):
        return {'temp': 0, 'spat': 0, 'graph': 0, 'spec': 0}
    d_temp = 4 + (4 if bool(getattr(args, 'stats_v2', 0)) else 0)
    d_spat = 4 if bool(getattr(args, 'stats_nbr', 0)) else 0
    d_graph = 2 if bool(getattr(args, 'stats_rank', 0)) else 0
    d_spec = 6 if bool(getattr(args, 'stats_spec', 0)) else 0
    return {'temp': d_temp, 'spat': d_spat, 'graph': d_graph, 'spec': d_spec}


def resolve_stats_in_dim(args):
    """Raw concat dim for temporal stats branch (0 when stats disabled)."""
    g = resolve_stats_raw_group_dims(args)
    return sum(g.values())


def resolve_stats_out_dim(args):
    """Projected stats branch width: sum of the four group Linear output dims."""
    if not bool(getattr(args, 'use_bsts', getattr(args, 'use_temporal_stats', 0))):
        return 0
    g = resolve_stats_raw_group_dims(args)
    out = 0
    if g['temp']:
        out += int(getattr(args, 'stats_proj_temp_dim', 16))
    if g['spat']:
        out += int(getattr(args, 'stats_proj_spat_dim', 8))
    if g['graph']:
        out += int(getattr(args, 'stats_proj_graph_dim', 8))
    if g['spec']:
        out += int(getattr(args, 'stats_proj_spec_dim', 8))
    return out


class BSTSEncoder(nn.Module):
    """Four BSTS groups, each through its own Linear, then concat-ready [B, D, N, 1].

    Projection widths default to the original grouped heads:
        s_temp -> Linear(8, 16), s_spat -> Linear(4, 8),
        s_graph -> Linear(2, 8), s_spec -> Linear(6, 8).
    """

    def __init__(
        self,
        stats_v2=True,
        stats_nbr=True,
        stats_rank=True,
        stats_spec=True,
        stats_proj_temp_dim=16,
        stats_proj_spat_dim=8,
        stats_proj_graph_dim=8,
        stats_proj_spec_dim=8,
    ):
        super().__init__()
        self.stats_v2 = bool(stats_v2)
        self.stats_nbr = bool(stats_nbr)
        self.stats_rank = bool(stats_rank)
        self.stats_spec = bool(stats_spec)
        self.stats_proj_temp_dim = int(stats_proj_temp_dim)
        self.stats_proj_spat_dim = int(stats_proj_spat_dim)
        self.stats_proj_graph_dim = int(stats_proj_graph_dim)
        self.stats_proj_spec_dim = int(stats_proj_spec_dim)

        d_temp = 4 + (4 if self.stats_v2 else 0)
        self.stats_proj_temp = nn.Linear(d_temp, self.stats_proj_temp_dim, bias=True)
        self.out_dim = self.stats_proj_temp_dim
        self.stats_proj_spat = None
        self.stats_proj_graph = None
        self.stats_proj_spec = None
        if self.stats_nbr:
            self.stats_proj_spat = nn.Linear(4, self.stats_proj_spat_dim, bias=True)
            self.out_dim += self.stats_proj_spat_dim
        if self.stats_rank:
            self.stats_proj_graph = nn.Linear(2, self.stats_proj_graph_dim, bias=True)
            self.out_dim += self.stats_proj_graph_dim
        if self.stats_spec:
            self.stats_proj_spec = nn.Linear(6, self.stats_proj_spec_dim, bias=True)
            self.out_dim += self.stats_proj_spec_dim

    def _to_conv(self, feat_bn_d):
        """[B, N, D] -> [B, D, N, 1] for STID concat along channel dim."""
        return feat_bn_d.transpose(1, 2).unsqueeze(-1)

    def forward(self, source, edge_index, edge_weight=None, input_dim=1):
        """
        Args:
            source: [B, T, N, C]
            edge_index: [2, E]
            edge_weight: unused by neighbor mean (kept for call-site compatibility)
        Returns:
            list of [B, proj_dim, N, 1], one per enabled group
            (order: temp, spat, graph, spec).
        """
        x = source[..., :int(input_dim)]
        N = int(x.shape[2])
        groups = compute_stats_groups(
            x,
            edge_index=edge_index,
            num_nodes=N,
            stats_v2=self.stats_v2,
            stats_nbr=self.stats_nbr,
            stats_rank=self.stats_rank,
            stats_spec=self.stats_spec,
        )
        outs = [self._to_conv(self.stats_proj_temp(groups['temp']))]
        if self.stats_proj_spat is not None:
            outs.append(self._to_conv(self.stats_proj_spat(groups['spat'])))
        if self.stats_proj_graph is not None:
            outs.append(self._to_conv(self.stats_proj_graph(groups['graph'])))
        if self.stats_proj_spec is not None:
            outs.append(self._to_conv(self.stats_proj_spec(groups['spec'])))
        return outs


def bsts_kwargs_from_args(args):
    return dict(
        use_bsts=bool(int(getattr(args, 'use_bsts', 1))),
        stats_v2=bool(int(getattr(args, 'stats_v2', 1))),
        stats_nbr=bool(int(getattr(args, 'stats_nbr', 1))),
        stats_rank=bool(int(getattr(args, 'stats_rank', 1))),
        stats_spec=bool(int(getattr(args, 'stats_spec', 1))),
        stats_proj_temp_dim=int(getattr(args, 'stats_proj_temp_dim', 16)),
        stats_proj_spat_dim=int(getattr(args, 'stats_proj_spat_dim', 8)),
        stats_proj_graph_dim=int(getattr(args, 'stats_proj_graph_dim', 8)),
        stats_proj_spec_dim=int(getattr(args, 'stats_proj_spec_dim', 8)),
    )
