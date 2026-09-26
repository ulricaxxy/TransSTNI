"""OA-STID full pipeline (SELF-CONTAINED): Stage1 -> Stage2 (subgraph-MAML) -> Cross.

Self-contained on purpose: the only external deps are the stable original project
modules under ``lib/`` (dataloader, metrics, ydzt_sampler_random). The OA-STID
model + all helpers are inlined here so the pipeline keeps working even if the
separate model/helper files are removed by the environment's sync.

Model = STID backbone (time-series 1x1 conv embed + residual MLP + conv head)
with the ORIGINAL OAGNN Orbit-Adaptive spatial embedding (Laplacian-eigenmap SE
F + 3-layer GCN encoder BETAET) replacing STID's learnable node table. All
parameters are N-independent -> zero-shot transfer to graphs of any size.

Stages:
  * Stage1: supervised pretrain on the full source graph.
  * Stage2: MAML on RANDOM SUBGRAPHS of the source (build_sizes for task node
            counts; inner support-adapt + outer query meta-update via
            torch.func.functional_call). Meta-init used directly for zero-shot.
  * Cross:  zero-shot eval on each target (metrics identical to cross_eval_recover).
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from lib.dataloader import get_dataloader
from lib.metrics import MAE_torch
from lib.ydzt_sampler_random import build_sizes
from lib.bsts_stats import BSTSEncoder
from lib.road_semantic import load_or_encode_road_sem, DEFAULT_QWEN_PATH, META_CSV
# behav_type / behav_zeroshot are imported lazily inside behav-only paths
# (load_behav_pi early-returns when use_behav_emb=0; nobehav never loads them).


# dataset -> (num_nodes, adjacency pickle in data/)
DATASET_INFO = {
    'PEMSD4':           (307,  'adj_mx_undirected_pems04.pkl'),
    'PEMSD7':           (883,  'adj_mx_undirected_pems07.pkl'),
    'PEMSD8':           (170,  'adj_mx_undirected_pems08.pkl'),
    'air_quality_full': (437,  'adj_mx_undirected_air_quality.pkl'),
    'pv_us':            (1082, 'adj_mx_undirected_pv_us.pkl'),
    'korea_covid':      (17,   'adj_mx_undirected_korea_covid.pkl'),
    'largest_sd':       (716,  'adj_mx_undirected_largest_sd.pkl'),
    'largest_gba':      (2352, 'adj_mx_undirected_largest_gba.pkl'),
    'largest_gla':      (3834, 'adj_mx_undirected_largest_gla.pkl'),
    'largest_ca':       (8600, 'adj_mx_undirected_largest_ca.pkl'),
}

# steps per day (for deriving time-of-day from absolute cycle_index):
# 5-min=288, 10-min=144, 15-min=96, hourly=24, daily=1.
STEPS_PER_DAY = {
    'PEMSD4': 288, 'PEMSD7': 288, 'PEMSD8': 288,
    'air_quality_full': 24, 'pv_us': 144, 'korea_covid': 1,
    'largest_sd': 96, 'largest_gba': 96, 'largest_gla': 96, 'largest_ca': 96,
}

# Real calendar start of each series (cycle_index=0). Used for weekday lookup
# (Monday=0 .. Sunday=6), matching OA-MoE / cross_domain_eval conventions.
CALENDAR_START = {
    'PEMSD4': '2018-01-01 00:00:00',
    'PEMSD7': '2017-05-01 00:00:00',
    'PEMSD8': '2016-07-01 00:00:00',
    'air_quality_full': '2014-05-01 00:00:00',
    'pv_us': '2006-01-01 00:00:00',
    'korea_covid': '2020-04-01 00:00:00',
    'largest_sd': '2019-01-01 00:00:00',
    'largest_gba': '2019-01-01 00:00:00',
    'largest_gla': '2019-01-01 00:00:00',
    'largest_ca': '2019-01-01 00:00:00',
}

# minutes per timestep (must match STEPS_PER_DAY: 1440 / spd)
MINUTES_PER_STEP = {
    'PEMSD4': 5, 'PEMSD7': 5, 'PEMSD8': 5,
    'air_quality_full': 60, 'pv_us': 10, 'korea_covid': 1440,
    'largest_sd': 15, 'largest_gba': 15, 'largest_gla': 15, 'largest_ca': 15,
}


# =========================== OA + STID model =============================== #
def laplacian_eigenmap_embedding(adj_mx, embed_dim=8, lap_type="sym", add_self_loop=True):
    """Symmetric-normalized Laplacian eigenmap SE (verbatim from code/OAGNN-main)."""
    A = np.asarray(adj_mx, dtype=np.float64)
    N = A.shape[0]
    A = 0.5 * (A + A.T)
    if add_self_loop:
        A = A + np.eye(N, dtype=np.float64)
    d = np.maximum(A.sum(axis=1), 1e-12)
    if lap_type == "sym":
        d_sqrt_inv = 1.0 / np.sqrt(d)
        L = np.eye(N) - (d_sqrt_inv[:, None] * A) * d_sqrt_inv[None, :]
    else:
        L = np.diag(d) - A
    eigvals, eigvecs = np.linalg.eigh(L)
    eigvecs = eigvecs[:, np.argsort(eigvals)]
    k_use = min(int(embed_dim), N - 1)
    emb = eigvecs[:, 1:1 + k_use]
    if emb.shape[1] < int(embed_dim):
        emb = np.concatenate(
            [emb, np.zeros((N, int(embed_dim) - emb.shape[1]))], axis=1)
    return emb.astype(np.float32)


def _pad_or_trim(emb, embed_dim):
    N, d = emb.shape
    out = np.zeros((N, int(embed_dim)), dtype=np.float32)
    k = min(d, int(embed_dim))
    if k > 0:
        out[:, :k] = emb[:, :k].astype(np.float32)
    return out


def _symmetrize_adj(adj_mx, add_self_loop=True):
    A = np.asarray(adj_mx, dtype=np.float64)
    A = 0.5 * (A + A.T)
    A = np.maximum(A, 0.0)
    if add_self_loop:
        A = A + np.eye(A.shape[0], dtype=np.float64)
    return A


def _degrees(A):
    return np.maximum(A.sum(axis=1), 1e-12)


def _standardize_cols(emb, eps=1e-8):
    mu = emb.mean(axis=0, keepdims=True)
    sd = np.maximum(emb.std(axis=0, keepdims=True), eps)
    return (emb - mu) / sd


def compute_rwse(adj_mx, embed_dim=8, add_self_loop=False):
    """Random-Walk SE / RWPE, faithful to official gnn-lspe (Dwivedi et al., ICLR 2022).

    PE_k = diag((A D^{-1})^k), k=1..embed_dim; no self-loop by default; degree
    clipped to >=1; raw landing probabilities (no standardization). Isolated
    nodes get all-zero rows. add_self_loop=True -> lazy random-walk (A+I) variant.
    """
    A = np.asarray(adj_mx, dtype=np.float64)
    A = 0.5 * (A + A.T)
    A = np.maximum(A, 0.0)
    N = A.shape[0]
    if add_self_loop:
        A = A + np.eye(N, dtype=np.float64)
    deg = np.clip(A.sum(axis=1), 1.0, None)
    Dinv = 1.0 / deg
    RW = A * Dinv[None, :]                     # A D^{-1}
    feats = []
    M_power = RW
    feats.append(np.diagonal(M_power).copy())  # k = 1
    for _ in range(int(embed_dim) - 1):
        M_power = M_power @ RW
        feats.append(np.diagonal(M_power).copy())
    emb = np.stack(feats, axis=1).astype(np.float32)
    return _pad_or_trim(emb, embed_dim)


def compute_hkse(adj_mx, embed_dim=8, add_self_loop=True, times=None):
    """HKdiagSE: diag(exp(-t L_sym)) via sum_i exp(-t λ_i) U_{n,i}^2 (sign-invariant)."""
    A = _symmetrize_adj(adj_mx, add_self_loop=add_self_loop)
    N = A.shape[0]
    d = _degrees(A)
    d_inv_sqrt = 1.0 / np.sqrt(d)
    L = np.eye(N) - (d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :])
    eigvals, eigvecs = np.linalg.eigh(L)
    mask = eigvals > 1e-8
    lam = eigvals[mask]
    U2 = eigvecs[:, mask] ** 2
    if times is None:
        times = np.geomspace(0.1, 10.0, num=int(embed_dim))
    times = np.asarray(times, dtype=np.float64).reshape(-1)[: int(embed_dim)]
    feats = [U2 @ np.exp(-float(t) * lam) for t in times]
    emb = np.stack(feats, axis=1)
    return _pad_or_trim(_standardize_cols(emb), embed_dim)


def compute_frse(adj_mx, embed_dim=8):
    """FRSE / ElstaticSE, faithful to GraphGPS get_electrostatic_function_encoding.

    L = D-A combinatorial Laplacian; E = pinv(L) with hermitian=True; 10 per-node
    stats padded/trimmed to embed_dim. Raw values (no standardization).
    """
    A = np.asarray(adj_mx, dtype=np.float64)
    A = 0.5 * (A + A.T)
    A = np.maximum(A, 0.0)
    np.fill_diagonal(A, 0.0)
    deg = A.sum(axis=1)
    L = np.diag(deg) - A
    Ldiag = np.diag(L).copy()
    with np.errstate(divide='ignore'):
        dinv = 1.0 / Ldiag
    dinv[~np.isfinite(dinv)] = 0.0
    Aabs = np.abs(L)
    np.fill_diagonal(Aabs, 0.0)
    DinvA = dinv[:, None] * Aabs
    E = np.linalg.pinv(L, hermitian=True)
    E = E - np.diag(E)[None, :]
    green = np.stack(
        [
            E.min(axis=0), E.max(axis=0), E.mean(axis=0), E.std(axis=0, ddof=1),
            E.min(axis=1), E.max(axis=0), E.mean(axis=1), E.std(axis=1, ddof=1),
            (DinvA * E).sum(axis=0), (DinvA * E).sum(axis=1),
        ],
        axis=1,
    ).astype(np.float32)
    return _pad_or_trim(green, embed_dim)


# structural-encoding selection (set from CLI in main; build_graph falls back here)
_SE_TYPE = 'lap'
_SE_SELF_LOOP = None


def structural_encoding(adj_mx, embed_dim, se_type=None, add_self_loop=None):
    se_type = _SE_TYPE if se_type is None else se_type
    if se_type == 'rwse':
        sl = False if add_self_loop is None and _SE_SELF_LOOP is None else \
            (_SE_SELF_LOOP if add_self_loop is None else add_self_loop)
        return compute_rwse(adj_mx, embed_dim=embed_dim, add_self_loop=bool(sl))
    if se_type == 'hkse':
        sl = True if add_self_loop is None and _SE_SELF_LOOP is None else \
            (_SE_SELF_LOOP if add_self_loop is None else add_self_loop)
        return compute_hkse(adj_mx, embed_dim=embed_dim, add_self_loop=bool(sl))
    if se_type == 'frse':
        return compute_frse(adj_mx, embed_dim=embed_dim)
    # default: symmetric-normalized Laplacian eigenmap (self-loop on by default)
    sl = True if add_self_loop is None and _SE_SELF_LOOP is None else \
        (_SE_SELF_LOOP if add_self_loop is None else add_self_loop)
    return laplacian_eigenmap_embedding(adj_mx, embed_dim=embed_dim, add_self_loop=bool(sl))


class BETAET(nn.Module):
    """3-layer GCN orbit encoder (architecture from code/OAGNN-main)."""

    def __init__(self, embed_dim, feat_dim=8, hidden_dim=64):
        super().__init__()
        self.gcn = GCNConv(feat_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gcn3 = GCNConv(hidden_dim, embed_dim)

    def forward(self, node_features, edge_index, edge_weight):
        x = self.gcn(node_features, edge_index, edge_weight)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.gcn2(x, edge_index, edge_weight)
        x = self.gcn3(x, edge_index, edge_weight)
        return x


class MultiLayerPerceptron(nn.Module):
    """STID residual MLP block (verbatim from STID-master/stid/arch/mlp.py)."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, x):
        hidden = self.fc2(self.drop(self.act(self.fc1(x))))
        return hidden + x


def build_graph(adj_mx, node_pe_dim=8, device="cpu"):
    A = np.asarray(adj_mx, dtype=np.float32)
    A = 0.5 * (A + A.T)
    node_feat = structural_encoding(A, embed_dim=node_pe_dim)
    edge_index, edge_weight = dense_to_sparse(torch.from_numpy(A))
    return (edge_index.to(device), edge_weight.to(device),
            torch.from_numpy(node_feat).to(device))


class OASTIDTransfer(nn.Module):
    """STID backbone + Orbit-Adaptive spatial embedding (N-independent)."""

    def __init__(
        self,
        input_len,
        output_len,
        input_dim=1,
        embed_dim=32,
        node_dim=32,
        num_layer=3,
        node_pe_dim=8,
        orbit_hidden=64,
        use_time_emb=False,
        temp_dim_tid=32,
        temp_dim_diw=32,
        time_of_day_size=288,
        day_of_week_size=7,
        use_cal_in_series=False,
        time_emb_last_step=True,
        use_sem_emb=False,
        sem_in_dim=1024,
        sem_dim=32,
        use_bsts=True,
        stats_v2=True,
        stats_nbr=True,
        stats_rank=True,
        stats_spec=True,
        stats_proj_temp_dim=16,
        stats_proj_spat_dim=8,
        stats_proj_graph_dim=8,
        stats_proj_spec_dim=8,
        use_behav_emb=False,
        behav_k=32,
        behav_dim=8,
    ):
        super().__init__()
        self.input_len = int(input_len)
        self.output_len = int(output_len)
        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)
        self.node_dim = int(node_dim)
        self.node_pe_dim = int(node_pe_dim)
        self.use_time_emb = bool(use_time_emb)
        self.use_sem_emb = bool(use_sem_emb)
        self.use_bsts = bool(use_bsts)
        self.use_behav_emb = bool(use_behav_emb)
        self.time_of_day_size = int(time_of_day_size)
        self.day_of_week_size = int(day_of_week_size)
        # STID-style: flatten 12-step TOD/DOW [0,1] into the series Conv
        # together with flow. Lookup tables stay a separate concat slot
        # and (by default) index the window's last step, matching
        # STID history_data[:, -1, :].
        self.use_cal_in_series = bool(use_cal_in_series) and bool(use_time_emb)
        self.time_emb_last_step = bool(time_emb_last_step) and bool(use_time_emb)
        series_ch = self.input_dim + (2 if self.use_cal_in_series else 0)
        self.time_series_emb_layer = nn.Conv2d(
            series_ch * self.input_len, self.embed_dim, kernel_size=(1, 1), bias=True)
        self.beta_net = BETAET(
            self.node_dim, self.node_pe_dim, hidden_dim=int(orbit_hidden))
        self.ln_beta = nn.LayerNorm(self.node_dim)
        self.hidden_dim = self.embed_dim + self.node_dim
        if self.use_time_emb:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, int(temp_dim_tid)))
            nn.init.xavier_uniform_(self.time_in_day_emb)
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, int(temp_dim_diw)))
            nn.init.xavier_uniform_(self.day_in_week_emb)
            self.hidden_dim += int(temp_dim_tid) + int(temp_dim_diw)
        if self.use_sem_emb:
            self.sem_dim = int(sem_dim)
            self.sem_proj = nn.Sequential(
                nn.LayerNorm(int(sem_in_dim)),
                nn.Linear(int(sem_in_dim), self.sem_dim),
                nn.ReLU(),
                nn.Linear(self.sem_dim, self.sem_dim),
                nn.LayerNorm(self.sem_dim),
            )
            self.hidden_dim += self.sem_dim
        if self.use_bsts:
            self.bsts = BSTSEncoder(
                stats_v2=bool(stats_v2),
                stats_nbr=bool(stats_nbr),
                stats_rank=bool(stats_rank),
                stats_spec=bool(stats_spec),
                stats_proj_temp_dim=int(stats_proj_temp_dim),
                stats_proj_spat_dim=int(stats_proj_spat_dim),
                stats_proj_graph_dim=int(stats_proj_graph_dim),
                stats_proj_spec_dim=int(stats_proj_spec_dim),
            )
            self.hidden_dim += self.bsts.out_dim
        else:
            self.bsts = None
        if self.use_behav_emb:
            self.behav_k = int(behav_k)
            self.behav_dim = int(behav_dim)
            self.behav_proto = nn.Embedding(self.behav_k, self.behav_dim)
            nn.init.xavier_uniform_(self.behav_proto.weight)
            self.hidden_dim += self.behav_dim
            self._behav_pi = None
        else:
            self._behav_pi = None
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim)
              for _ in range(int(num_layer))])
        self.regression_layer = nn.Conv2d(
            self.hidden_dim, self.output_len, kernel_size=(1, 1), bias=True)

    def _sem_feats(self, sem_emb, B, N):
        if not self.use_sem_emb:
            return []
        if sem_emb is None:
            raise ValueError('use_sem_emb=True requires sem_emb [N, D]')
        if sem_emb.dim() != 2 or int(sem_emb.shape[0]) != int(N):
            raise ValueError(f'sem_emb expected [N={N}, D], got {tuple(sem_emb.shape)}')
        s = self.sem_proj(sem_emb)
        return [s.unsqueeze(0).expand(B, -1, -1).transpose(1, 2).unsqueeze(-1)]

    def _bsts_feats(self, source, edge_index, edge_weight):
        """Per-group BSTS Linear outputs, each [B, proj_dim, N, 1]."""
        if not self.use_bsts:
            return []
        return self.bsts(source, edge_index, edge_weight, input_dim=self.input_dim)

    def set_behav_pi(self, behav_pi):
        """Bind per-graph soft behavior-type assignment [N, behav_k] (frozen)."""
        self._behav_pi = behav_pi

    def _behav_feats(self, behav_pi, B, N):
        """Node embedding = mixture of K learnable prototypes, weights = offline π.

        π is frozen (from 29-d morphology + source K-means + softmax).
        W = behav_proto.weight is learned. Not one-hot, not a Linear on φ.
        Returns one tensor [B, behav_dim, N, 1] (its own concat slot).
        """
        if not self.use_behav_emb:
            return []
        if behav_pi is None:
            behav_pi = self._behav_pi
        if behav_pi is None:
            raise RuntimeError(
                'use_behav_emb=1 but behav_pi was not passed and set_behav_pi() was not called')
        if (behav_pi.dim() != 2
                or int(behav_pi.shape[0]) != int(N)
                or int(behav_pi.shape[1]) != int(self.behav_k)):
            raise RuntimeError(
                f'behav_pi expected [{N}, {self.behav_k}], got {tuple(behav_pi.shape)}')
        b_emb = behav_pi @ self.behav_proto.weight
        return [b_emb.unsqueeze(0).expand(B, -1, -1).transpose(1, 2).unsqueeze(-1)]

    def _cal_channels(self, tid_idx, dow_idx, B, T, N):
        """Build STID-style [B, T, N, 1] TOD/DOW fractions from window-start indices.

        ``tid_idx`` / ``dow_idx`` come from ``time_indices(cycle_index)`` where
        ``cycle_index`` is the first lag step. Slots wrap a day of
        ``time_of_day_size`` (LargeST: 96). DOW increments when the window
        crosses midnight. Values are in [0, 1], matching STID data channels.
        """
        tid = tid_idx.reshape(-1).long()
        dow = dow_idx.reshape(-1).long()
        if int(tid.shape[0]) != int(B) or int(dow.shape[0]) != int(B):
            raise ValueError(
                f'cal channels expect tid/dow [B={B}], got '
                f'{tuple(tid_idx.shape)} / {tuple(dow_idx.shape)}')
        spd = self.time_of_day_size
        steps = torch.arange(T, device=tid.device, dtype=tid.dtype)
        slot = tid.unsqueeze(1) + steps.unsqueeze(0)
        tod = (slot % spd).float() / float(spd)
        days = torch.div(slot, spd, rounding_mode='floor')
        dow_f = ((dow.unsqueeze(1) + days) % self.day_of_week_size).float() / float(
            self.day_of_week_size)
        tod = tod.unsqueeze(-1).unsqueeze(-1).expand(B, T, N, 1)
        dow_f = dow_f.unsqueeze(-1).unsqueeze(-1).expand(B, T, N, 1)
        return tod, dow_f

    def _window_last_cal_idx(self, tid_idx, dow_idx, T):
        """Shift window-start TOD/DOW indices to the last lag step.

        ``cycle_index`` / ``time_indices`` give the first lag. STID looks up
        ``history[:, -1]``, i.e. start + (T-1), wrapping TOD at
        ``time_of_day_size`` and incrementing DOW across midnight.
        """
        tid = tid_idx.reshape(-1).long()
        dow = dow_idx.reshape(-1).long()
        last = tid + int(T) - 1
        spd = self.time_of_day_size
        tid_last = torch.remainder(last, spd)
        days = torch.div(last, spd, rounding_mode='floor')
        dow_last = torch.remainder(dow + days, self.day_of_week_size)
        return tid_last, dow_last

    def forward(
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
        x = source[..., :self.input_dim]
        B, T, N, _ = x.shape
        if T != self.input_len:
            raise ValueError(f'expected input_len={self.input_len}, got T={T}')
        x_series = x
        if self.use_cal_in_series:
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
        z = self.ln_beta(self.beta_net(node_feat, edge_index, edge_weight))
        z_emb = z.unsqueeze(0).expand(B, -1, -1).transpose(1, 2).unsqueeze(-1)
        feats = [ts_emb, z_emb]
        if self.use_time_emb:
            if tid_idx is None or dow_idx is None:
                raise ValueError('use_time_emb=1 requires tid_idx and dow_idx')
            tid_q, dow_q = tid_idx, dow_idx
            if self.time_emb_last_step:
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


def split_low_high_frequency(x, low_k_max=1):
    """Split a length-T series into low / high bands via masked rFFT (differentiable).

    Args:
        x: [B, T, N, C] real-valued history (raw series, NOT a mixed embedding).
        low_k_max: keep frequency bins k=0..low_k_max as the low band; the rest
            form the high band (for T=12, rFFT has 7 bins -> low={0,1}, high={2..6}).

    Returns:
        x_low, x_high: each [B, T, N, C], inverse-FFT reconstructions.
    """
    if x.dim() != 4:
        raise ValueError(f'split_low_high_frequency expects [B,T,N,C], got {tuple(x.shape)}')
    B, T, N, C = x.shape
    # rFFT over the temporal axis; complex spectrum [B, F, N, C], F=T//2+1
    spec = torch.fft.rfft(x, dim=1)
    Fbins = spec.shape[1]
    k_low = int(max(0, min(int(low_k_max), Fbins - 1)))
    low_mask = torch.zeros(Fbins, device=x.device, dtype=spec.real.dtype)
    high_mask = torch.ones(Fbins, device=x.device, dtype=spec.real.dtype)
    low_mask[: k_low + 1] = 1.0
    high_mask[: k_low + 1] = 0.0
    # broadcast masks over [B,F,N,C]
    view = (1, Fbins, 1, 1)
    spec_low = spec * low_mask.view(*view)
    spec_high = spec * high_mask.view(*view)
    x_low = torch.fft.irfft(spec_low, n=T, dim=1)
    x_high = torch.fft.irfft(spec_high, n=T, dim=1)
    return x_low, x_high


class IdentityQueryCrossAttention(nn.Module):
    """Multi-head cross-attention: identity query over temporal (K,V) tokens.

    Operates node-independently by folding (B, N_chunk) into the batch dimension
    so that each sensor attends only over its own temporal token sequence.

    Nodes are processed in chunks (``node_chunk``) to keep the effective MHA batch
    size ``B * node_chunk`` within CUDA scaled-dot-product attention limits.
    This is required for large target graphs (e.g. GLA N=3834, B=64 → B*N≈2.45e5
    which triggers ``CUDA error: invalid configuration argument`` if unchunked).
    """

    def __init__(self, attn_dim, num_heads=4, dropout=0.1, node_chunk=256):
        super().__init__()
        if int(attn_dim) % int(num_heads) != 0:
            raise ValueError(f'attn_dim={attn_dim} must be divisible by num_heads={num_heads}')
        self.attn_dim = int(attn_dim)
        self.num_heads = int(num_heads)
        self.node_chunk = int(max(1, node_chunk))
        self.mha = nn.MultiheadAttention(
            embed_dim=self.attn_dim, num_heads=self.num_heads,
            dropout=float(dropout), batch_first=True)
        self.out_ln = nn.LayerNorm(self.attn_dim)
        self.out_drop = nn.Dropout(p=float(dropout))

    def _attend_flat(self, q, k, v):
        """q/k/v: [BN, L, D] with L_q=1 for query."""
        attn_out, _ = self.mha(q, k, v, need_weights=False)
        attn_out = self.out_ln(q + self.out_drop(attn_out))
        return attn_out[:, 0, :]  # [BN, D]

    def forward(self, query, key, value):
        """
        Args:
            query: [B, N, 1, D]
            key:   [B, N, S, D]
            value: [B, N, S, D]
        Returns:
            out:   [B, N, D]  (single readout token per node)
        """
        B, N, Qlen, D = query.shape
        S = key.shape[2]
        chunk = self.node_chunk
        # Small graphs: one shot (matches original behaviour).
        if N <= chunk:
            q = query.reshape(B * N, Qlen, D)
            k = key.reshape(B * N, S, D)
            v = value.reshape(B * N, S, D)
            return self._attend_flat(q, k, v).reshape(B, N, D)

        outs = []
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            n_c = end - start
            q = query[:, start:end].reshape(B * n_c, Qlen, D)
            k = key[:, start:end].reshape(B * n_c, S, D)
            v = value[:, start:end].reshape(B * n_c, S, D)
            outs.append(self._attend_flat(q, k, v).reshape(B, n_c, D))
        return torch.cat(outs, dim=1)


class OASTIDAttnTransfer(nn.Module):
    """OA-STID with concat fusion replaced by spatial-Query / series-KV attention.

    Drop-in replacement for ``OASTIDTransfer``'s
        [ts_emb ∥ Z ∥ calendar] → MLP → ŷ
    by
        Q=Z,  K/V=raw temporal tokens → CrossAttn → [readout ∥ calendar] → MLP → ŷ

    Two K/V modes (``fusion``):
      * ``tstep``:      K/V length = T (raw timesteps).
      * ``tstep_band``: K/V length = 2T (low/high FFT bands × T steps).

    Calendar (tod/dow) is NOT in Q; it is concatenated after attention, same role
    as in classic STID concat. Default has NO residual base.
    """

    def __init__(self, input_len, output_len, input_dim=1, embed_dim=32,
                 node_dim=32, num_layer=3, node_pe_dim=8, orbit_hidden=64,
                 use_time_emb=False, temp_dim_tid=32, temp_dim_diw=32,
                 time_of_day_size=288, day_of_week_size=7,
                 fusion='tstep', attn_dim=32, attn_heads=4, attn_dropout=0.1,
                 band_low_k_max=1, use_residual_base=False, attn_node_chunk=256):
        super().__init__()
        if fusion not in ('tstep', 'tstep_band'):
            raise ValueError(f'OASTIDAttnTransfer fusion must be tstep|tstep_band, got {fusion}')
        self.input_len = int(input_len)
        self.output_len = int(output_len)
        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)
        self.node_dim = int(node_dim)
        self.node_pe_dim = int(node_pe_dim)
        self.use_time_emb = bool(use_time_emb)
        self.time_of_day_size = int(time_of_day_size)
        self.day_of_week_size = int(day_of_week_size)
        self.fusion = str(fusion)
        self.attn_dim = int(attn_dim)
        self.band_low_k_max = int(band_low_k_max)
        self.use_residual_base = bool(use_residual_base)
        self.attn_node_chunk = int(attn_node_chunk)
        self.temp_dim_tid = int(temp_dim_tid)
        self.temp_dim_diw = int(temp_dim_diw)

        # ----- orbit-adaptive spatial identity (same as concat OA-STID) -----
        self.beta_net = BETAET(embed_dim=self.node_dim, feat_dim=self.node_pe_dim,
                               hidden_dim=int(orbit_hidden))
        self.ln_beta = nn.LayerNorm(self.node_dim)

        # ----- calendar identity tables (optional) -----
        if self.use_time_emb:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        # ----- series -> per-step tokens (from RAW series, not a pre-mixed emb) -----
        self.step_value_proj = nn.Linear(self.input_dim, self.attn_dim, bias=True)
        self.step_pos_emb = nn.Parameter(torch.empty(self.input_len, self.attn_dim))
        nn.init.xavier_uniform_(self.step_pos_emb)
        if self.fusion == 'tstep_band':
            # band-type embedding distinguishes low vs high tokens at the same t
            self.band_type_emb = nn.Parameter(torch.empty(2, self.attn_dim))
            nn.init.xavier_uniform_(self.band_type_emb)
            # optional band-specific value projections (still on raw band series)
            self.step_value_proj_low = nn.Linear(self.input_dim, self.attn_dim, bias=True)
            self.step_value_proj_high = nn.Linear(self.input_dim, self.attn_dim, bias=True)

        # ----- spatial identity query (Z only; calendar is concatenated AFTER attn) -----
        self.query_proj = nn.Sequential(
            nn.Linear(self.node_dim, self.attn_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.attn_dim, self.attn_dim, bias=True),
        )
        self.kv_ln = nn.LayerNorm(self.attn_dim)
        self.cross_attn = IdentityQueryCrossAttention(
            attn_dim=self.attn_dim, num_heads=int(attn_heads),
            dropout=float(attn_dropout), node_chunk=self.attn_node_chunk)

        # ----- MLP head: attn readout ∥ calendar → residual MLP → ŷ -----
        self.hidden_dim = self.attn_dim
        if self.use_time_emb:
            self.hidden_dim += self.temp_dim_tid + self.temp_dim_diw
        self.readout_proj = nn.Conv2d(
            self.attn_dim, self.attn_dim, kernel_size=(1, 1), bias=True)
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim)
              for _ in range(int(num_layer))])
        self.regression_layer = nn.Conv2d(
            self.hidden_dim, self.output_len, kernel_size=(1, 1), bias=True)

        # ----- optional residual baseline (OFF by default; not part of concat→attn) -----
        if self.use_residual_base:
            self.base_ts_emb_layer = nn.Conv2d(
                self.input_dim * self.input_len, self.embed_dim, kernel_size=(1, 1), bias=True)
            self.base_hidden_dim = self.embed_dim
            if self.use_time_emb:
                self.base_hidden_dim += self.temp_dim_tid + self.temp_dim_diw
            self.base_encoder = nn.Sequential(
                *[MultiLayerPerceptron(self.base_hidden_dim, self.base_hidden_dim)
                  for _ in range(int(num_layer))])
            self.base_head = nn.Conv2d(
                self.base_hidden_dim, self.output_len, kernel_size=(1, 1), bias=True)

    def _calendar_feats(self, tid_idx, dow_idx, B, N):
        """Return tid/dow as [B, D, N, 1] lists (empty if time emb disabled)."""
        if not self.use_time_emb:
            return []
        if tid_idx is None or dow_idx is None:
            raise ValueError('use_time_emb=True requires tid_idx and dow_idx')
        tid = self.time_in_day_emb[tid_idx]  # [B, Dt]
        dow = self.day_in_week_emb[dow_idx]  # [B, Dw]
        return [
            tid.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, N, 1),
            dow.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, N, 1),
        ]

    def _build_query(self, z, tid_idx, dow_idx, B, N):
        """Q from spatial orbit Z only (calendar not in Q)."""
        # z: [N, node_dim] -> [B, N, node_dim]
        z_b = z.unsqueeze(0).expand(B, -1, -1)
        q = self.query_proj(z_b)  # [B, N, D]
        return q.unsqueeze(2)  # [B, N, 1, D]

    def _embed_steps(self, series, value_proj, band_id=None):
        """Embed raw series timesteps into tokens.

        Args:
            series: [B, T, N, C]
            value_proj: nn.Linear(C -> attn_dim)
            band_id: None or int in {0,1} for low/high type embedding
        Returns:
            tokens: [B, N, T, D]
        """
        B, T, N, C = series.shape
        # [B, T, N, C] -> [B, N, T, C]
        x = series.permute(0, 2, 1, 3).contiguous()
        tok = value_proj(x)  # [B, N, T, D]
        tok = tok + self.step_pos_emb.view(1, 1, T, self.attn_dim)
        if band_id is not None:
            tok = tok + self.band_type_emb[band_id].view(1, 1, 1, self.attn_dim)
        return self.kv_ln(tok)

    def _build_kv_tstep(self, source):
        # source: [B, T, N, C] raw
        return self._embed_steps(source, self.step_value_proj, band_id=None)

    def _build_kv_tstep_band(self, source):
        x_low, x_high = split_low_high_frequency(source, low_k_max=self.band_low_k_max)
        tok_low = self._embed_steps(x_low, self.step_value_proj_low, band_id=0)
        tok_high = self._embed_steps(x_high, self.step_value_proj_high, band_id=1)
        # concat along sequence: [B, N, 2T, D]
        return torch.cat([tok_low, tok_high], dim=2)

    def _baseline(self, source, tid_idx, dow_idx):
        """Space-agnostic STID-style baseline (no Z)."""
        x = source[..., :self.input_dim]
        B, _, N, _ = x.shape
        flat = x.transpose(1, 2).contiguous().view(B, N, -1).transpose(1, 2).unsqueeze(-1)
        ts_emb = self.base_ts_emb_layer(flat)
        feats = [ts_emb] + self._calendar_feats(tid_idx, dow_idx, B, N)
        hidden = torch.cat(feats, dim=1)
        hidden = self.base_encoder(hidden)
        return self.base_head(hidden)

    def _decode_from_attn(self, attn_readout, tid_idx, dow_idx):
        """attn_readout [B,N,D] ∥ calendar → MLP → forecast [B,H,N,1]."""
        B, N, D = attn_readout.shape
        h = attn_readout.transpose(1, 2).unsqueeze(-1)  # [B, D, N, 1]
        h = self.readout_proj(h)
        feats = [h] + self._calendar_feats(tid_idx, dow_idx, B, N)
        hidden = torch.cat(feats, dim=1)
        hidden = self.encoder(hidden)
        return self.regression_layer(hidden)

    def forward(self, source, edge_index, edge_weight, node_feat, tid_idx=None, dow_idx=None):
        x = source[..., :self.input_dim]
        B, T, N, _ = x.shape
        if T != self.input_len:
            raise ValueError(f'expected input_len={self.input_len}, got T={T}')

        # orbit identity Z = g(A)  (same as concat OA-STID)
        z = self.ln_beta(self.beta_net(node_feat, edge_index, edge_weight))  # [N, d_node]
        query = self._build_query(z, tid_idx, dow_idx, B, N)  # [B, N, 1, D]

        if self.fusion == 'tstep':
            kv = self._build_kv_tstep(x)            # [B, N, T, D]
        else:
            kv = self._build_kv_tstep_band(x)       # [B, N, 2T, D]

        # concat replacement: identity queries series tokens → one vector per node
        attn_readout = self.cross_attn(query, kv, kv)  # [B, N, D]
        pred = self._decode_from_attn(attn_readout, tid_idx, dow_idx)

        if self.use_residual_base:
            # optional experimental path (default OFF)
            return self._baseline(x, tid_idx, dow_idx) + pred
        return pred



def load_sem_emb(args, dataset, device):
    """Qwen road-semantic embeddings [N, 1024], or None if disabled."""
    if not bool(int(getattr(args, 'use_sem_emb', 0))):
        return None
    return load_or_encode_road_sem(
        dataset,
        device,
        model_path=getattr(args, 'qwen_path', None) or DEFAULT_QWEN_PATH,
        expected_n=DATASET_INFO[dataset][0],
        batch_size=int(getattr(args, 'qwen_batch_size', 64)),
    )


def _maybe_swap_orbit_gnn(model, args):
    """CLI switch for Orbit encoder. Default keeps the original BETAET GCN.

    ``--orbit_gnn sage`` replaces ``model.beta_net`` with the GraphSAGE from
    ``code/src/model/graph_sage.py`` (via ``lib.orbit_graphsage``). BETAET
    itself is not edited.
    """
    orbit_gnn = str(getattr(args, 'orbit_gnn', 'gcn')).lower()
    if orbit_gnn in ('', 'gcn'):
        return model
    if orbit_gnn != 'sage':
        raise ValueError(f'unknown orbit_gnn={orbit_gnn} (expected gcn|sage)')
    from lib.orbit_graphsage import OrbitGraphSAGE
    model.beta_net = OrbitGraphSAGE(
        embed_dim=int(args.node_dim),
        feat_dim=int(args.node_pe_dim),
        hidden_dim=int(args.orbit_hidden),
        sage_k=int(getattr(args, 'sage_k', 3)),
        sage_dropout=float(getattr(args, 'sage_dropout', 0.1)),
        sage_norm=bool(int(getattr(args, 'sage_norm', 0))),
        seed=int(getattr(args, 'seed', 10)),
    )
    print(
        f'[orbit] beta_net=GraphSAGE (code/src/model/graph_sage.py) in={args.node_pe_dim}'
        f' hid={args.orbit_hidden} out={args.node_dim}'
        f' sage_k={getattr(args, "sage_k", 3)}'
        f' sage_dropout={getattr(args, "sage_dropout", 0.1)}'
        f' sage_norm={int(getattr(args, "sage_norm", 0))}',
        flush=True,
    )
    return model


def parse_target_list(args):
    return [t.strip() for t in str(getattr(args, 'targets', '')).split(',') if t.strip()]


def _behav_zero_shot_flag(args):
    return bool(int(getattr(args, 'behav_zero_shot', 0)))


def _resolve_sem_flag(args):
    """Keep Qwen road-sem on only when LargeST meta exists for the source."""
    want = bool(int(getattr(args, 'use_sem_emb', 0)))
    if not want:
        args.use_sem_emb = 0
        return args
    if args.source not in META_CSV:
        print(f'[qwen-road] no meta csv for source={args.source}; disable use_sem_emb', flush=True)
        args.use_sem_emb = 0
        return args
    args.use_sem_emb = 1
    return args


def _ckpt_dir_for(args):
    d = os.path.join(PROJECT_ROOT, 'output', 'oastid', f'{args.source.lower()}_{args.tag}')
    os.makedirs(d, exist_ok=True)
    return d


def save_stage1_ckpt(args, model, s1_val):
    path = os.path.join(_ckpt_dir_for(args), 'stage1.pt')
    torch.save({
        'state_dict': model.state_dict(),
        'stage1_val': s1_val,
        'stage2_val': None,
        'args': vars(args),
        'kind': 'stage1',
    }, path)
    print(f'[stage1] saved {path}', flush=True)
    return path


def _load_adj_numpy(dataset):
    path = os.path.join(PROJECT_ROOT, 'data', DATASET_INFO[dataset][1])
    with open(path, 'rb') as f:
        obj = pickle.load(f, encoding='latin1')
    adj_mx = obj[-1] if isinstance(obj, (list, tuple)) else obj
    A = np.asarray(adj_mx, dtype=np.float64)
    return 0.5 * (A + A.T)


def _qwen_numpy(args, dataset):
    """Qwen road-text [N, 1024] on CPU (cache hit, no target flow)."""
    emb = load_or_encode_road_sem(
        dataset,
        'cpu',
        model_path=getattr(args, 'qwen_path', None) or DEFAULT_QWEN_PATH,
        expected_n=DATASET_INFO[dataset][0],
        batch_size=int(getattr(args, 'qwen_batch_size', 64)),
    )
    return np.asarray(emb.detach().cpu().numpy(), dtype=np.float64)


def slice_sem(sem_full, node_ids, device):
    if sem_full is None:
        return None
    idx = torch.as_tensor(np.asarray(node_ids), dtype=torch.long, device=sem_full.device)
    return sem_full.index_select(0, idx).to(device, non_blocking=True)


def slice_behav(behav_full, node_ids, device):
    """Row-slice offline π to the sampled node set. Own helper (not aliased)."""
    if behav_full is None:
        return None
    idx = torch.as_tensor(np.asarray(node_ids), dtype=torch.long, device=behav_full.device)
    return behav_full.index_select(0, idx).to(device, non_blocking=True)


def load_target_eval_pack(args, target, device, model):
    """Full target test pack: loader + graph + Qwen-sem + behav π + scaler + calendar."""
    tgt_args = make_loader_args(args, target)
    ute = args.use_time_emb
    cal_start, spd, mps = calendar_of(target)
    _, _, test_loader, scaler = get_dataloader(
        tgt_args, normalizer='std', tod=ute, dow=ute, weather=False,
        single=False, return_index=ute)
    edge_index, edge_weight, node_feat = load_graph(args, target, device)
    sem = load_sem_emb(args, target, device)
    behav = load_behav_pi(args, target, device, model=model)
    return {
        'test_loader': test_loader,
        'scaler': scaler,
        'ute': ute,
        'cal_start': cal_start,
        'spd': spd,
        'mps': mps,
        'edge_index': edge_index,
        'edge_weight': edge_weight,
        'node_feat': node_feat,
        'sem': sem,
        'behav': behav,
    }


def maybe_run_stage2_cross(args, model, device, step, packs):
    """Every ``cross_every`` Stage2 steps: full zero-shot eval on all targets.

    Uses the current weights (not the best-so-far checkpoint). Target loaders /
    graphs / Qwen-sem are cached in ``packs`` so only the first call reloads.
    """
    every = int(getattr(args, 'cross_every', 20))
    if every <= 0 or int(step) % every != 0:
        return packs
    targets = parse_target_list(args)
    if not targets:
        return packs
    print(f'\n########## STAGE2 CROSS @ step {int(step):04d} ##########', flush=True)
    was_training = model.training
    model.eval()
    row = {'step': int(step), 'targets': {}}
    for tgt in targets:
        if tgt not in packs:
            packs[tgt] = load_target_eval_pack(args, tgt, device, model=model)
        ov = eval_target(args, model, device, tgt, step=step, pack=packs[tgt])
        row['targets'][tgt] = ov
        print(
            f'[stage2/cross] step={int(step):04d} {args.source}->{tgt} '
            f'MAE={ov["MAE"]:.4f} RMSE={ov["RMSE"]:.4f} '
            f'Masked-MAPE={ov["Masked-MAPE%"]:.4f}%',
            flush=True,
        )
    log_dir = os.path.join(
        PROJECT_ROOT, 'output', 'oastid', f'{args.source.lower()}_{args.tag}')
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, 'stage2_cross.jsonl'), 'a') as f:
        f.write(json.dumps(row) + '\n')
    if was_training:
        model.train()
    return packs

def load_behav_pi(args, dataset, device, model=None):
    """Offline behavior-type soft assignment π [N, K], or None if disabled.

    Source K-means centroids + column μ/σ are fit on the source TRAIN split.
    Default Cross (behav_zero_shot=0): target φ from a time window of the
    target series (behav_target_frac; 0.6 = TRAIN, 0.05 = first 5%).
    Zero-shot Cross (behav_zero_shot=1): target π from source-bank retrieval
    with keys = sem_proj(Qwen) || 3 graph scalars (same MLP as concat).
    """
    if not bool(int(getattr(args, 'use_behav_emb', 0))):
        return None
    from lib.behav_type import load_or_build_behav_pi, resolve_behav_alias
    src = str(getattr(args, 'behav_source', '') or '') or str(args.source)
    ds_cal, ds_spd, ds_mps = calendar_of(dataset)
    src_cal, src_spd, src_mps = calendar_of(src)
    if _behav_zero_shot_flag(args) and resolve_behav_alias(dataset) != resolve_behav_alias(src):
        if not bool(int(getattr(args, 'use_sem_emb', 0))):
            raise RuntimeError('behav_zero_shot=1 requires use_sem_emb=1 (Qwen keys for retrieval)')
        if model is None:
            raise RuntimeError('behav_zero_shot target π requires the live model (sem_proj keys)')
        return retrieve_zeroshot_pi_sem_proj(
            args, model, dataset, _qwen_numpy(args, dataset), device,
        )
    pi = load_or_build_behav_pi(
        dataset, DATASET_INFO[dataset][0], args,
        ds_spd, ds_cal, ds_mps,
        src, src_spd, src_cal, src_mps,
    )
    return torch.from_numpy(np.asarray(pi, dtype=np.float32)).to(device)


def build_oastid_model(args):
    """Factory: keep classic concat OA-STID; optionally build attn fusion variants."""
    fusion = getattr(args, 'fusion', 'concat')
    common = dict(
        input_len=args.lag,
        output_len=args.horizon,
        input_dim=args.input_dim,
        embed_dim=args.embed_dim,
        node_dim=args.node_dim,
        num_layer=args.num_layer,
        node_pe_dim=args.node_pe_dim,
        orbit_hidden=args.orbit_hidden,
        use_time_emb=args.use_time_emb,
        temp_dim_tid=args.temp_dim_tid,
        temp_dim_diw=args.temp_dim_diw,
        time_of_day_size=args.time_of_day_size,
        use_sem_emb=bool(int(getattr(args, 'use_sem_emb', 0))),
        sem_in_dim=int(getattr(args, 'sem_in_dim', 1024)),
        sem_dim=int(getattr(args, 'sem_dim', 32)),
        use_bsts=bool(int(getattr(args, 'use_bsts', 1))),
        stats_v2=bool(int(getattr(args, 'stats_v2', 1))),
        stats_nbr=bool(int(getattr(args, 'stats_nbr', 1))),
        stats_rank=bool(int(getattr(args, 'stats_rank', 1))),
        stats_spec=bool(int(getattr(args, 'stats_spec', 1))),
        stats_proj_temp_dim=int(getattr(args, 'stats_proj_temp_dim', 16)),
        stats_proj_spat_dim=int(getattr(args, 'stats_proj_spat_dim', 8)),
        stats_proj_graph_dim=int(getattr(args, 'stats_proj_graph_dim', 8)),
        stats_proj_spec_dim=int(getattr(args, 'stats_proj_spec_dim', 8)),
        use_behav_emb=bool(int(getattr(args, 'use_behav_emb', 1))),
        behav_k=int(getattr(args, 'behav_k', 32)),
        behav_dim=int(getattr(args, 'behav_dim', 8)),
    )
    if fusion == 'concat':
        common['use_cal_in_series'] = bool(int(getattr(args, 'use_cal_in_series', 0)))
        common['time_emb_last_step'] = bool(int(getattr(args, 'time_emb_last_step', 1)))
        model = OASTIDTransfer(**common)
        if model.use_cal_in_series:
            print(
                '[stid/calseq] series Conv in='
                f'{model.input_dim + 2}x{model.input_len} '
                f'(flow+TOD+DOW); lookup tables kept; BSTS/RevIN on flow only',
                flush=True,
            )
        if model.use_time_emb:
            print(
                '[stid] TOD/DOW lookup='
                + ('window last step (history[:, -1])' if model.time_emb_last_step
                   else 'window first step (cycle_index)'),
                flush=True,
            )
        return _maybe_swap_orbit_gnn(model, args)
    if fusion in ('tstep', 'tstep_band'):
        model = OASTIDAttnTransfer(
            **common,
            fusion=fusion,
            attn_dim=getattr(args, 'attn_dim', args.embed_dim),
            attn_heads=getattr(args, 'attn_heads', 4),
            attn_dropout=getattr(args, 'attn_dropout', 0.1),
            band_low_k_max=getattr(args, 'band_low_k_max', 1),
            use_residual_base=bool(getattr(args, 'use_residual_base', 0)),
            attn_node_chunk=getattr(args, 'attn_node_chunk', 256),
        )
        return _maybe_swap_orbit_gnn(model, args)
    raise ValueError(f'unknown fusion={fusion}')


# =============================== helpers ================================== #
def make_loader_args(args, dataset):
    a = argparse.Namespace()
    a.dataset = dataset
    a.num_nodes = DATASET_INFO[dataset][0]
    a.lag, a.horizon = args.lag, args.horizon
    a.val_ratio, a.test_ratio = args.val_ratio, args.test_ratio
    a.batch_size = args.batch_size
    a.normalizer = 'std'
    a.column_wise = False
    a.single = False
    a.input_dim, a.output_dim = args.input_dim, args.output_dim
    a.device = args.device
    a.tod = False
    a.default_graph = True
    return a


def load_graph(args, dataset, device):
    path = os.path.join(PROJECT_ROOT, 'data', DATASET_INFO[dataset][1])
    with open(path, 'rb') as f:
        obj = pickle.load(f, encoding='latin1')
    adj_mx = obj[-1] if isinstance(obj, (list, tuple)) else obj
    return build_graph(adj_mx, node_pe_dim=args.node_pe_dim, device=device)


def to_dev(x, device):
    return x.float().to(device, non_blocking=True)


def calendar_of(dataset, calendar_start=None):
    """Return (calendar_start, steps_per_day, minutes_per_step) for a dataset."""
    start = calendar_start or CALENDAR_START[dataset]
    spd = STEPS_PER_DAY[dataset]
    mps = MINUTES_PER_STEP[dataset]
    return start, spd, mps


def time_indices(cycle_index, steps_per_day, table_size, device,
                 calendar_start=None, minutes_per_step=None):
    """Absolute cycle_index -> (tid_idx, dow_idx).

    * Time-of-day: FRACTION of day so the table stays clock-consistent across
      sampling rates (noon -> same table row).
    * Day-of-week: REAL Gregorian weekday from ``calendar_start + index *
      minutes_per_step`` (pandas Monday=0 .. Sunday=6). Requires calendar_start;
      does NOT use (day_count % 7), which misaligns across years/domains.
    """
    idx = cycle_index.long()
    device_t = torch.device(device) if not isinstance(device, torch.device) else device
    frac = (idx.to(device_t) % steps_per_day).float() / float(steps_per_day)  # [0,1)
    tid = (frac * table_size).long().clamp(0, table_size - 1)

    if calendar_start is None or minutes_per_step is None:
        raise ValueError(
            'time_indices requires calendar_start and minutes_per_step for '
            'real weekday lookup (Monday=0..Sunday=6)')
    # Vectorized timestamp -> weekday on CPU, then move to device.
    idx_np = idx.detach().cpu().numpy().astype(np.int64)
    shape = idx_np.shape
    start = pd.Timestamp(calendar_start)
    stamps = start + pd.to_timedelta(idx_np.ravel() * int(minutes_per_step), unit='m')
    dow_np = np.asarray(stamps.dayofweek, dtype=np.int64).reshape(shape)
    dow = torch.from_numpy(dow_np).to(device_t)
    return tid, dow


def masked_mae_real(pred, y, scaler, mask_value):
    return MAE_torch(scaler.inverse_transform(pred), scaler.inverse_transform(y), mask_value)


# numpy metrics (identical to stage2/cross_eval_recover.py)
def mae_np(p, t):
    return float(np.mean(np.abs(p - t)))


def rmse_np(p, t):
    return float(np.sqrt(np.mean((p - t) ** 2)))


def mape_np(p, t, eps=1e-5):
    return float(np.mean(np.abs((p - t) / np.maximum(np.abs(t), eps))) * 100.0)


def masked_mape_np(p, t, thr):
    mask = np.abs(t) > thr
    if mask.sum() == 0:
        return float('nan'), 0.0
    return float(np.mean(np.abs((p[mask] - t[mask]) / t[mask])) * 100.0), float(mask.mean())


# =============================== Stage 1 ================================== #
def train_source(args, model, device):
    src_args = make_loader_args(args, args.source)
    ute = args.use_time_emb
    cal_start, spd, mps = calendar_of(args.source, getattr(args, 'calendar_start', None))
    train_loader, val_loader, _, scaler = get_dataloader(
        src_args, normalizer='std', tod=ute, dow=ute, weather=False,
        single=False, return_index=ute)
    edge_index, edge_weight, node_feat = load_graph(args, args.source, device)
    sem = load_sem_emb(args, args.source, device)
    behav = load_behav_pi(args, args.source, device)

    def tds(batch):
        if not ute:
            return None, None
        return time_indices(
            batch[2], spd, model.time_of_day_size, device,
            calendar_start=cal_start, minutes_per_step=mps)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(args.stage1_epochs * 0.5), int(args.stage1_epochs * 0.75)], gamma=0.5)

    best_val, best_state, bad = float('inf'), None, 0
    for epoch in range(1, args.stage1_epochs + 1):
        model.train()
        t0, tr_loss, nb = time.perf_counter(), 0.0, 0
        for batch in train_loader:
            x = to_dev(batch[0], device)[..., :args.input_dim]
            y = to_dev(batch[1], device)[..., :args.output_dim]
            tid, dow = tds(batch)
            opt.zero_grad()
            loss = masked_mae_real(model(x, edge_index, edge_weight, node_feat, tid, dow, sem, behav), y, scaler, args.mask_value)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            tr_loss += loss.item()
            nb += 1
        sched.step()
        model.eval()
        vl, vb = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                x = to_dev(batch[0], device)[..., :args.input_dim]
                y = to_dev(batch[1], device)[..., :args.output_dim]
                tid, dow = tds(batch)
                vl += masked_mae_real(model(x, edge_index, edge_weight, node_feat, tid, dow, sem, behav), y, scaler, args.mask_value).item()
                vb += 1
        val_mae = vl / max(vb, 1)
        print(f'[stage1][{args.source}] epoch {epoch:03d} train_mae={tr_loss/max(nb,1):.4f} '
              f'val_mae={val_mae:.4f} ({time.perf_counter()-t0:.1f}s)', flush=True)
        if val_mae < best_val - 1e-4:
            best_val, bad = val_mae, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f'[stage1] early stop (best={best_val:.4f})', flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


# =============================== Stage 2 ================================== #
def _split_data_by_ratio(data, val_ratio, test_ratio):
    """Match ``lib.dataloader.split_data_by_ratio`` (ratio split on time axis)."""
    data_len = int(data.shape[0])
    test_data = data[-int(data_len * test_ratio):]
    val_data = data[-int(data_len * (test_ratio + val_ratio)):-int(data_len * test_ratio)]
    train_data = data[:-int(data_len * (test_ratio + val_ratio))]
    return train_data, val_data, test_data


def phaseformer_preprocess_train_series(series, period_len, week_align=True, rng=None):
    """PhaseFormer-style phase-internal exchange as **offline preprocessing**.

    Operates on the full contiguous TRAIN series ``[T_train, N, C]`` (not on
    lag-sized Stage2 windows). Faithful to PhaseFormer period→phase rearrange:

      1) truncate to a multiple of ``period_len`` (one period = one day when
         ``period_len = steps_per_day``)
      2) view as ``[n_periods, period_len, N, C]``
      3) for each phase ``ell``, randomly permute along the period axis
      4) with ``week_align=True``, only permute periods that share the same
         weekday offset ``period_idx % 7`` (Tue↔Tue only)

    The leftover tail shorter than one period is left unchanged. Spatial
    snapshots move together (all nodes at a time slot share the swap).

    After this, Stage2 merely cuts normal sliding windows from the remixed
    long series — no online small-window phase swap.
    """
    series = np.asarray(series, dtype=np.float32)
    if series.ndim != 3:
        raise ValueError(f'series must be [T,N,C], got {series.shape}')
    period_len = int(period_len)
    if period_len < 1:
        raise ValueError(f'period_len must be >= 1, got {period_len}')
    rng = np.random.RandomState(None if rng is None else int(rng))
    T, N, C = series.shape
    n_periods = T // period_len
    if n_periods < 2:
        print(f'[phase-preprocess] T={T} period_len={period_len} -> n_periods={n_periods}; '
              f'skip (need >=2 periods)', flush=True)
        return series.copy()
    usable = n_periods * period_len
    out = series.copy()
    block = out[:usable].reshape(n_periods, period_len, N, C)
    week_align = bool(week_align)

    if week_align:
        day_offset = np.arange(n_periods, dtype=np.int64) % 7
        n_swap_groups = 0
        for d in range(7):
            idxs = np.flatnonzero(day_offset == d)
            M = int(idxs.size)
            if M < 2:
                continue
            n_swap_groups += 1
            # Independent period-axis perm per phase (PhaseFormer phase-internal).
            for ell in range(period_len):
                perm = rng.permutation(M)
                block[idxs, ell] = block[idxs[perm], ell]
        print(f'[phase-preprocess] week_align=1 periods={n_periods} '
              f'period_len={period_len} weekday_groups_swapped={n_swap_groups} '
              f'T_used={usable}/{T}', flush=True)
    else:
        for ell in range(period_len):
            perm = rng.permutation(n_periods)
            block[:, ell] = block[perm, ell]
        print(f'[phase-preprocess] week_align=0 periods={n_periods} '
              f'period_len={period_len} T_used={usable}/{T}', flush=True)
    out[:usable] = block.reshape(usable, N, C)
    return out


def collect_source_arrays(args, device):
    """Collect Stage2 source windows + val loader + scaler + adjacency.

    Default: unchanged — iterate the train loader (drop_last batches).

    When ``--stage2_phase_preprocess 1``: load the full series, normalize with
    the same scaler as the dataloader, take the contiguous TRAIN split, apply
    PhaseFormer phase-internal exchange **once on the long train series**, then
    rebuild sliding windows. Source val / Cross stay on the original clean
    distribution (val_loader from the untouched dataloader).
    """
    from lib.add_window import Add_Window_Horizon
    from lib.load_dataset import load_st_dataset

    src_args = make_loader_args(args, args.source)
    ute = args.use_time_emb
    train_loader, val_loader, _, scaler = get_dataloader(
        src_args, normalizer='std', tod=ute, dow=ute, weather=False,
        single=False, return_index=ute)

    with open(os.path.join(PROJECT_ROOT, 'data', DATASET_INFO[args.source][1]), 'rb') as f:
        obj = pickle.load(f, encoding='latin1')
    adj_mx = obj[-1] if isinstance(obj, (list, tuple)) else obj
    A = 0.5 * (np.asarray(adj_mx, dtype=np.float32) + np.asarray(adj_mx, dtype=np.float32).T)

    do_preprocess = bool(int(getattr(args, 'stage2_phase_preprocess', 0)))
    if do_preprocess:
        raw = load_st_dataset(args.source)  # [T, N, C] physical scale
        # Same order as dataloader: normalize full series, then ratio-split.
        data_n = scaler.transform(raw)
        if not isinstance(data_n, np.ndarray):
            data_n = np.asarray(data_n, dtype=np.float32)
        data_train, _, _ = _split_data_by_ratio(
            data_n, float(args.val_ratio), float(args.test_ratio))
        _, spd, _ = calendar_of(args.source, getattr(args, 'calendar_start', None))
        period_len = _stage2_period_len(args, spd)
        week_align = _stage2_phase_week_align_flag(args)
        data_train = phaseformer_preprocess_train_series(
            data_train, period_len=period_len, week_align=week_align,
            rng=int(getattr(args, 'seed', 10)))
        x_tra, y_tra = Add_Window_Horizon(
            data_train, args.lag, args.horizon, single=False)
        Xtr = torch.from_numpy(
            np.asarray(x_tra[..., :args.input_dim], dtype=np.float32))
        Ytr = torch.from_numpy(
            np.asarray(y_tra[..., :args.output_dim], dtype=np.float32))
        # Train split starts at global index 0 → window start = window row index.
        Itr = torch.arange(Xtr.shape[0], dtype=torch.long) if ute else None
        print(f'[phase-preprocess] Stage2 windows from remixed train series: '
              f'X={tuple(Xtr.shape)} (online small-window exchange disabled)',
              flush=True)
        return Xtr, Ytr, Itr, val_loader, scaler, A

    # single pass so X, Y (and cycle_index) stay aligned (train loader shuffles)
    xs, ys, idxs = [], [], []
    for b in train_loader:
        xs.append(b[0][..., :args.input_dim].float())
        ys.append(b[1][..., :args.output_dim].float())
        if ute:
            idxs.append(b[2])
    Xtr = torch.cat(xs, 0)
    Ytr = torch.cat(ys, 0)
    Itr = torch.cat(idxs, 0) if ute else None
    return Xtr, Ytr, Itr, val_loader, scaler, A


def stunet_permute_induced_adjacency(A_sub, node_ids):
    """STUNet-style random node-order permutation of an induced subgraph.

    Faithful to STUNet-master:
      * ``src/data/largest/dataset.py`` (SubGraphSampler): optional
        ``node_ids = np.random.permutation(node_ids)`` before building the
        subgraph, so the adjacency row/column order is a random relabeling.
      * ``src/models/STUNet/core/utils.py::generate_permutation_matrix``:
        ``A' = P A P^T`` with a random permutation matrix ``P``.

    For a dense induced adjacency ``A_sub = A[S,S]``, drawing ``perm`` and
    returning ``A_sub[perm][:,perm]`` together with ``S[perm]`` is exactly
    ``P A_sub P^T`` while keeping traffic columns aligned with the new order.

    Args:
        A_sub: [n, n] induced adjacency (numpy).
        node_ids: [n] original sensor indices corresponding to rows/cols of A_sub.

    Returns:
        A_perm: [n, n] permuted adjacency.
        node_ids_perm: [n] original ids in the permuted order (for X/Y indexing).
        perm: [n] permutation applied (int64).
    """
    A_sub = np.asarray(A_sub, dtype=np.float32)
    node_ids = np.asarray(node_ids)
    n = int(A_sub.shape[0])
    if A_sub.shape != (n, n):
        raise ValueError(f'A_sub must be square, got {A_sub.shape}')
    if node_ids.shape[0] != n:
        raise ValueError(f'node_ids length {node_ids.shape[0]} != n={n}')
    perm = np.random.permutation(n).astype(np.int64)
    # A' = P A P^T  <=>  simultaneous row+col gather by perm
    A_perm = A_sub[np.ix_(perm, perm)].astype(np.float32, copy=False)
    node_ids_perm = node_ids[perm]
    return A_perm, node_ids_perm, perm


# -------------------- PhaseFormer-style Stage2 temporal aug -------------------- #
def _phaseformer_to_phase_series(x_periods):
    """PhaseFormer ``_to_phase_series``: (..., P_in, L, ...) -> (..., L, P_in, ...).

    Faithful to PhaseFormer_TSL-main/models/PhaseFormer.py::Model._to_phase_series
    which maps ``(B, C, P_in, L) -> (B, C, L, P_in)``. Here the trailing spatial
    feature dims ``(N, C)`` are carried along unchanged.
    """
    # x_periods: [B, P_in, L, N, C] -> [B, L, P_in, N, C]
    return x_periods.permute(0, 2, 1, 3, 4).contiguous()


def _phaseformer_from_phase_series(phase_series):
    """Inverse of ``_phaseformer_to_phase_series``: (..., L, P_in, ...) -> (..., P_in, L, ...).

    Faithful to PhaseFormer ``_from_phase_steps_to_periods``.
    """
    # phase_series: [B, L, P_in, N, C] -> [B, P_in, L, N, C]
    return phase_series.permute(0, 2, 1, 3, 4).contiguous()


def phaseformer_within_sample_phase_exchange(xy, period_len, starts=None,
                                             week_align=True):
    """Within each sample: PhaseFormer period→phase view, then exchange along periods.

    Faithful pipeline (per batch item), matching PhaseFormer tokenization:
      1) circular-pad length to a multiple of ``period_len`` (PhaseFormer ring pad)
      2) split into periods: view ``[B, P_in, L, N, C]``
      3) rearrange to phase series: ``[B, L, P_in, N, C]``
      4) **phase-internal exchange**: for each phase ``ell``, randomly permute
         the ``P_in`` period-axis (same perm applied to all nodes at that slot)
      5) rearrange back to periods, flatten, crop to original length

    If ``week_align=True`` (default) and ``starts`` is provided, periods are
    only permuted inside the same weekday bucket
    ``day_offset = ((starts[b] + p * period_len) // period_len) % 7``.
    So Tuesday periods exchange with Tuesday periods only — never Tue↔Wed.

    If ``P_in < 2`` after padding, the tensor is returned unchanged (no period
    axis to exchange). This is the temporal analogue of patch-permutation, but
    the segments being exchanged are **same-phase slots across periods**, not
    contiguous time patches.
    """
    if xy is None:
        return xy
    period_len = int(period_len)
    if period_len < 1:
        raise ValueError(f'period_len must be >= 1, got {period_len}')
    B, Ltot, N, C = xy.shape
    n_periods = (Ltot + period_len - 1) // period_len
    pad = n_periods * period_len - Ltot
    if n_periods < 2:
        return xy
    if pad > 0:
        # PhaseFormer uses F.pad(..., mode='circular'); equivalent for dim=1:
        xy_pad = torch.cat([xy, xy[:, :pad]], dim=1)
    else:
        xy_pad = xy
    x_periods = xy_pad.view(B, n_periods, period_len, N, C)
    phase_series = _phaseformer_to_phase_series(x_periods)  # [B, L, P_in, N, C]
    device = xy.device
    week_align = bool(week_align)

    if (not week_align) or starts is None:
        # Original unrestricted period-axis shuffle (may mix weekdays).
        scores = torch.rand(B, period_len, n_periods, device=device)
        perm = torch.argsort(scores, dim=-1)  # [B, L, P_in]
        perm_exp = perm.unsqueeze(-1).unsqueeze(-1).expand(B, period_len, n_periods, N, C)
        phase_series = torch.gather(phase_series, 2, perm_exp)
    else:
        # Weekday-restricted: only shuffle periods that share the same day-of-week
        # offset inside the series week cycle.
        starts = starts.long().view(B).to(device)
        # day_offset[b, p] in {0..6}
        p_idx = torch.arange(n_periods, device=device).unsqueeze(0).expand(B, n_periods)
        day_offset = ((starts.unsqueeze(1) + p_idx * period_len) // period_len) % 7
        out_phase = phase_series.clone()
        for b in range(B):
            for d in range(7):
                members = torch.nonzero(day_offset[b] == d, as_tuple=False).view(-1)
                M = int(members.numel())
                if M < 2:
                    continue
                # one shared perm over periods for all phases of this (b, weekday)
                perm = torch.randperm(M, device=device)
                src = members
                dst = members[perm]
                # out_phase[b, :, dst, ...] <- phase_series[b, :, src, ...]
                out_phase[b, :, dst] = phase_series[b, :, src]
        phase_series = out_phase

    x_periods = _phaseformer_from_phase_series(phase_series)
    xy_out = x_periods.reshape(B, n_periods * period_len, N, C)[:, :Ltot]
    return xy_out


def phaseformer_cross_sample_phase_exchange(xy, starts, period_len, week_align=True):
    """Across-batch PhaseFormer phase-internal exchange under absolute phase.

    Each time index ``t`` of sample ``b`` has absolute time ``starts[b] + t``.

    * ``week_align=False`` (legacy): bucket by daily phase
      ``φ = abs_t mod period_len`` — mixes any weekday at the same clock time
      (Tue 08:00 ↔ Wed 08:00).
    * ``week_align=True`` (default): bucket by **week-cycle position**
      ``ψ = abs_t mod (7 * period_len)`` — same clock time **and** same weekday
      (Tue 08:00 ↔ Tue 08:00 only). This is "一周内同样位置".

    Calendar indices are *not* moved: slots keep their ``starts+t`` identity so
    tid/dow stay aligned with the time axis; only the traffic realization at
    that week-slot is remixed.
    """
    if xy is None:
        return xy
    period_len = int(period_len)
    if period_len < 1:
        raise ValueError(f'period_len must be >= 1, got {period_len}')
    B, Ltot, N, C = xy.shape
    if B < 1 or Ltot < 1:
        return xy
    device = xy.device
    starts = starts.long().view(B).to(device)
    t_grid = torch.arange(Ltot, device=device).unsqueeze(0).expand(B, Ltot)
    abs_t = starts.unsqueeze(1) + t_grid
    if week_align:
        cycle = int(7 * period_len)
        phases = abs_t % cycle  # [B, L] week-slot
        n_buckets = cycle
    else:
        phases = abs_t % period_len
        n_buckets = period_len
    out = xy.clone()
    flat = out.view(B * Ltot, N, C)
    phase_flat = phases.reshape(B * Ltot)
    # Iterate only buckets that appear (week_align has 7*period_len slots).
    present = torch.unique(phase_flat)
    for phi in present.tolist():
        idx = torch.nonzero(phase_flat == int(phi), as_tuple=False).view(-1)
        M = int(idx.numel())
        if M < 2:
            continue
        perm = torch.randperm(M, device=device)
        vals = flat[idx].clone()
        flat[idx] = vals[perm]
    return flat.view(B, Ltot, N, C)


def apply_stage2_phase_exchange(x, y, starts, period_len, mode='both',
                                week_align=True):
    """Construct Stage2 temporal data via PhaseFormer phase-internal exchange.

    Unlike contiguous **patch** permutation (equal-length segment shuffle), this
    operates in the **phase** domain of PhaseFormer:
      * ``within``: per-sample ring-pad → periods → phase series → permute ``P_in``
      * ``cross`` : absolute-phase buckets across the batch → permute within bucket
      * ``both``  : apply ``within`` then ``cross`` (default)

    When ``week_align=True`` (default), exchange is restricted to the **same
    position inside a week** (same weekday + same time-of-day slot): Tuesday
    only swaps with Tuesday, never with Wednesday.

    ``x`` and ``y`` are concatenated along time into one trajectory so the
    exchanged series stays internally consistent, then split back.
    """
    mode = str(mode).lower().strip()
    if mode not in ('within', 'cross', 'both'):
        raise ValueError(f'unknown phase-exchange mode {mode!r}; '
                         f'expected within|cross|both')
    if x is None or y is None:
        return x, y
    T = int(x.shape[1])
    H = int(y.shape[1])
    xy = torch.cat([x, y], dim=1)  # [B, T+H, N, C]
    week_align = bool(week_align)
    if mode in ('within', 'both'):
        xy = phaseformer_within_sample_phase_exchange(
            xy, period_len, starts=starts, week_align=week_align)
    if mode in ('cross', 'both'):
        xy = phaseformer_cross_sample_phase_exchange(
            xy, starts, period_len, week_align=week_align)
    return xy[:, :T].contiguous(), xy[:, T:T + H].contiguous()


def sample_task(Xtr, Ytr, Itr, A, node_size, k_spt, k_qry, node_pe_dim, device,
                spd=None, table_size=None, calendar_start=None, minutes_per_step=None,
                permute_adj=False,
                phase_exchange=False, period_len=None, phase_exchange_mode='both',
                phase_week_align=True,
                sem_full=None, behav_full=None, behav_meta=None):
    """Sample one Stage2 subgraph task.

    Default behaviour (``permute_adj=False``, ``phase_exchange=False``) is
    unchanged from the original OA-STID Stage2 sampler: random node set ``S``
    (sorted), induced ``A[S,S]``, PE/edges via ``build_graph``, random
    support/query time windows.

    When ``permute_adj=True`` (STUNet-style Stage2 augmentation), after forming
    the induced subgraph we apply ``stunet_permute_induced_adjacency`` so that
    adjacency rows/cols and the corresponding X/Y node axes share a random
    relabeling. Topology is identical; only the arbitrary node order changes.

    When ``phase_exchange=True`` (PhaseFormer-style Stage2 temporal construction),
    support and query windows are each passed through
    ``apply_stage2_phase_exchange``: traffic is remixed by **phase-internal
    exchange** (not contiguous patch shuffle). With ``phase_week_align=True``
    (default), only same week-slot (weekday + TOD) may exchange — Tue↔Tue only.
    Calendar ``cycle_index`` / tid / dow are left on the original slots.

    When ``behav_meta`` is set, π is observation-induced / synthesized /
    composition-sampled (zero-shot Stage2). Default path is unchanged.
    """
    if behav_meta is not None:
        return sample_task_behav_induce(
            Xtr, Ytr, Itr, A, node_size, k_spt, k_qry, node_pe_dim, device,
            spd=spd, table_size=table_size, calendar_start=calendar_start,
            minutes_per_step=minutes_per_step, permute_adj=permute_adj,
            phase_exchange=phase_exchange, period_len=period_len,
            phase_exchange_mode=phase_exchange_mode,
            phase_week_align=phase_week_align,
            sem_full=sem_full, behav_full=behav_full, behav_meta=behav_meta,
        )
    N = A.shape[0]
    node_size = int(min(node_size, N))
    S = np.sort(np.random.choice(N, size=node_size, replace=False))
    A_sub = A[np.ix_(S, S)]
    if permute_adj:
        A_sub, S, _perm = stunet_permute_induced_adjacency(A_sub, S)
    edge_index, edge_weight, node_feat = build_graph(A_sub, node_pe_dim, device)
    T = Xtr.shape[0]
    idx = np.random.choice(T, size=min(k_spt + k_qry, T), replace=False)
    S_t = torch.from_numpy(np.asarray(S)).long()
    idx_spt, idx_qry = idx[:k_spt], idx[k_spt:k_spt + k_qry]

    def grab(ix):
        return (Xtr[ix][:, :, S_t, :].to(device), Ytr[ix][:, :, S_t, :].to(device))

    xs, ys = grab(idx_spt)
    xq, yq = grab(idx_qry)

    if phase_exchange:
        pl = int(period_len if period_len is not None else (spd or xs.shape[1]))
        if Itr is not None:
            st_s = Itr[idx_spt].long().reshape(-1).to(device)
            st_q = Itr[idx_qry].long().reshape(-1).to(device)
        else:
            st_s = torch.zeros(xs.shape[0], dtype=torch.long, device=device)
            st_q = torch.zeros(xq.shape[0], dtype=torch.long, device=device)
        xs, ys = apply_stage2_phase_exchange(
            xs, ys, st_s, pl, mode=phase_exchange_mode,
            week_align=phase_week_align)
        xq, yq = apply_stage2_phase_exchange(
            xq, yq, st_q, pl, mode=phase_exchange_mode,
            week_align=phase_week_align)

    tid = dow = None
    if Itr is not None:
        cyc = torch.cat([Itr[idx_spt], Itr[idx_qry]], 0)
        tid, dow = time_indices(
            cyc, spd, table_size, device,
            calendar_start=calendar_start, minutes_per_step=minutes_per_step)
    sem = slice_sem(sem_full, S, device)
    behav = slice_behav(behav_full, S, device)
    return xs, ys, xq, yq, edge_index, edge_weight, node_feat, tid, dow, sem, behav


def _stage2_permute_adj_flag(args):
    return bool(int(getattr(args, 'stage2_permute_adj', 0)))


def _stage2_phase_exchange_flag(args):
    """Online small-window phase exchange.

    Disabled automatically when ``stage2_phase_preprocess=1`` (long-series
    offline remix already applied); the two modes must not stack.
    """
    if bool(int(getattr(args, 'stage2_phase_preprocess', 0))):
        return False
    return bool(int(getattr(args, 'stage2_phase_exchange', 0)))


def _stage2_phase_preprocess_flag(args):
    return bool(int(getattr(args, 'stage2_phase_preprocess', 0)))


def _stage2_period_len(args, spd):
    """Resolve PhaseFormer period_len for Stage2.

    ``stage2_period_len <= 0`` → use source ``steps_per_day`` (LargeST 15-min = 96),
    i.e. one civil day as the phase cycle. Explicit positive values match
    PhaseFormer ``--period_len`` (traffic scripts often use 24).
    """
    pl = int(getattr(args, 'stage2_period_len', 0))
    if pl <= 0:
        return int(spd if spd is not None else 24)
    return pl


def _stage2_phase_exchange_mode(args):
    return str(getattr(args, 'stage2_phase_exchange_mode', 'both'))


def _stage2_phase_week_align_flag(args):
    """Default True: only exchange same week-slot (Tue↔Tue, not Tue↔Wed)."""
    return bool(int(getattr(args, 'stage2_phase_week_align', 1)))


def meta_train(args, model, device, Xtr, Ytr, Itr, A, val_loader, scaler):
    """Stage2: episodic training over RANDOM SOURCE SUBGRAPHS (the OAGNN
    'random-sample' augmentation).

    Each meta-iteration samples ``task_num`` random node-subgraphs of the source
    graph (sizes from ``build_sizes``); for each we recompute the Laplacian-
    eigenmap F and predict a random time batch, accumulating a normalized-space
    MAE, then take one Adam meta-step. This trains the SINGLE shared model to
    forecast on graphs of many sizes/topologies -> improves size-generalization
    for zero-shot transfer, while staying stable (unlike bilevel MAML, which is
    ill-posed here because Laplacian eigenvectors carry sign/basis ambiguity that
    makes cross-subgraph second-order meta-gradients noisy).

    The best source-val checkpoint is kept, seeded with the Stage1 init so Stage2
    can never regress below Stage1.
    """
    node_max = int(min(args.node_max, A.shape[0]))
    node_min = int(min(args.node_min, max(node_max - 1, 1)))
    val_ei, val_ew, val_nf = build_graph(A, node_pe_dim=args.node_pe_dim, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=args.meta_lr, weight_decay=args.weight_decay)
    ute = args.use_time_emb
    cal_start, spd, mps = calendar_of(args.source, getattr(args, 'calendar_start', None))
    tsz = model.time_of_day_size
    permute_adj = _stage2_permute_adj_flag(args)
    phase_exchange = _stage2_phase_exchange_flag(args)
    period_len = _stage2_period_len(args, spd)
    phase_mode = _stage2_phase_exchange_mode(args)
    phase_week_align = _stage2_phase_week_align_flag(args)

    def src_val_mae():
        model.eval()
        with torch.no_grad():
            vl, vb = 0.0, 0
            for b in val_loader:
                x = b[0][..., :args.input_dim].float().to(device)
                y = b[1][..., :args.output_dim].float().to(device)
                tid, dow = (time_indices(
                    b[2], spd, tsz, device,
                    calendar_start=cal_start, minutes_per_step=mps) if ute else (None, None))
                vl += masked_mae_real(model(x, val_ei, val_ew, val_nf, tid, dow), y, scaler, args.mask_value).item()
                vb += 1
        model.train()
        return vl / max(vb, 1)

    # seed with the stage1 init so stage2 can never end up worse
    # Warm-start FROM Stage1, but do NOT keep Stage1 as a checkpoint candidate
    # (no rollback to Stage1). Best = Stage2's own best src-val during Stage2.
    s1_init = src_val_mae()
    best_val, best_state = float('inf'), None
    print(f'[stage2/ERM] warm-start src_val_mae={s1_init:.4f} (no Stage1 rollback) '
          f'permute_adj={int(permute_adj)} phase_exchange={int(phase_exchange)} '
          f'phase_preprocess={int(_stage2_phase_preprocess_flag(args))} '
          f'period_len={period_len} phase_mode={phase_mode} '
          f'week_align={int(phase_week_align)}', flush=True)

    N = A.shape[0]
    for it in range(1, args.meta_iters + 1):
        # always anchor on the FULL source graph (task 0) + random subgraphs, so
        # full-graph performance is directly optimized and cannot drift away while
        # the subgraphs add size/topology augmentation.
        sizes = [N] + build_sizes(num=max(args.task_num - 1, 1), min_n=node_min, max_n=node_max)
        model.train()
        opt.zero_grad()
        loss_sum = 0.0
        for s in sizes:
            # Full-graph anchor (s==N): keep original order (no permute) so Stage2
            # still directly optimizes the true source labeling. Random subgraphs
            # (s < N) optionally get STUNet-style adjacency permutation.
            # PhaseFormer phase-exchange is temporal (not spatial): apply on every
            # Stage2 task including the full-graph anchor when enabled.
            do_perm = bool(permute_adj and int(s) < int(N))
            xs, ys, xq, yq, ei, ew, nf, tid, dow = sample_task(
                Xtr, Ytr, Itr, A, s, args.k_spt, args.k_qry, args.node_pe_dim, device,
                spd=spd, table_size=tsz, calendar_start=cal_start, minutes_per_step=mps,
                permute_adj=do_perm,
                phase_exchange=phase_exchange, period_len=period_len,
                phase_exchange_mode=phase_mode, phase_week_align=phase_week_align)
            xin = torch.cat([xs, xq], 0)
            yin = torch.cat([ys, yq], 0)  # normalized targets
            pred = model(xin, ei, ew, nf, tid, dow)
            loss_sum = loss_sum + torch.mean(torch.abs(pred - yin))
        (loss_sum / len(sizes)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()

        if it % args.meta_eval_every == 0 or it == args.meta_iters:
            val_mae = src_val_mae()
            print(f'[stage2/ERM] iter {it:04d}/{args.meta_iters} '
                  f'subgraph_mae(norm)={float(loss_sum)/len(sizes):.4f} src_val_mae={val_mae:.4f}', flush=True)
            if val_mae < best_val - 1e-4:
                best_val = val_mae
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_val = src_val_mae()
    return model, best_val


# ===================== Stage 2 (OAGNN faithful MAML) ===================== #
def _compute_lr(i, epochs, min_lr, max_lr):
    """Cosine-ish inner/meta lr schedule, verbatim from OAGNN computeLR."""
    if i < epochs * 0.8:
        return max_lr
    e = (i / float(epochs) - 0.5) * 2.0
    e = 0.0 + e * 6.0
    f = 0.5 ** e
    return min_lr + (max_lr - min_lr) * f


def meta_train_maml(args, model, device, Xtr, Ytr, Itr, A, val_loader, scaler):
    """Stage2 = OAGNN's first-order MAML (faithful port of subgraphlearning.Meta).

    Per meta-iter: sample ``task_num`` random source subgraphs; for each, run an
    inner loop of ``update_step`` SGD steps on the SUPPORT set (normalized MAE,
    functional_call fast-weights), then the QUERY loss with the adapted weights is
    the task meta-loss. Average task meta-losses -> one outer Adam step (grad-norm
    clipped). ``second_order`` toggles FOMAML vs full MAML. Inner/meta lr follow
    OAGNN's computeLR schedule. Best source-val checkpoint kept (seed = Stage1).
    """
    from collections import OrderedDict
    from torch.func import functional_call

    N = A.shape[0]
    node_max = int(min(args.node_max, N))
    node_min = int(min(args.node_min, max(node_max - 1, 1)))
    val_ei, val_ew, val_nf = build_graph(A, node_pe_dim=args.node_pe_dim, device=device)
    meta_opt = torch.optim.Adam(model.parameters(), lr=args.meta_lr, eps=1e-8, weight_decay=0)
    ute = args.use_time_emb
    cal_start, spd, mps = calendar_of(args.source, getattr(args, 'calendar_start', None))
    tsz = model.time_of_day_size
    inner_max, inner_min = args.update_lr, args.update_lr * 0.7
    meta_max, meta_min = args.meta_lr, args.meta_lr * 0.7

    def src_val_mae():
        model.eval()
        with torch.no_grad():
            vl, vb = 0.0, 0
            for b in val_loader:
                x = b[0][..., :args.input_dim].float().to(device)
                y = b[1][..., :args.output_dim].float().to(device)
                tid, dow = (time_indices(
                    b[2], spd, tsz, device,
                    calendar_start=cal_start, minutes_per_step=mps) if ute else (None, None))
                vl += masked_mae_real(model(x, val_ei, val_ew, val_nf, tid, dow), y, scaler, args.mask_value).item()
                vb += 1
        model.train()
        return vl / max(vb, 1)

    # Warm-start FROM Stage1, but do NOT keep Stage1 as a checkpoint candidate
    # (no rollback to Stage1). Best = Stage2's own best src-val during Stage2.
    s1_init = src_val_mae()
    best_val, best_state = float('inf'), None
    permute_adj = _stage2_permute_adj_flag(args)
    phase_exchange = _stage2_phase_exchange_flag(args)
    period_len = _stage2_period_len(args, spd)
    phase_mode = _stage2_phase_exchange_mode(args)
    phase_week_align = _stage2_phase_week_align_flag(args)
    print(f'[stage2/MAML] warm-start src_val_mae={s1_init:.4f} (no Stage1 rollback) '
          f'second_order={bool(args.second_order)} update_step={args.update_step} '
          f'permute_adj={int(permute_adj)} phase_exchange={int(phase_exchange)} '
          f'phase_preprocess={int(_stage2_phase_preprocess_flag(args))} '
          f'period_len={period_len} phase_mode={phase_mode} '
          f'week_align={int(phase_week_align)}', flush=True)

    def fwd(params, x, ei, ew, nf, tid, dow):
        return functional_call(model, params, args=(x, ei, ew, nf, tid, dow))

    patience = 0
    for it in range(1, args.meta_iters + 1):
        for g in meta_opt.param_groups:
            g['lr'] = _compute_lr(it, args.meta_iters, meta_min, meta_max)
        inner_lr = _compute_lr(it, args.meta_iters, inner_min, inner_max)
        sizes = build_sizes(num=max(args.task_num, 1), min_n=node_min, max_n=node_max)

        model.train()
        meta_opt.zero_grad()
        outer = torch.zeros((), device=device)
        qsum = 0.0
        for s in sizes:
            # MAML tasks are all random subgraphs (incl. possible s==N). Apply
            # STUNet-style adj permutation whenever enabled so node order is not
            # a spurious cue across meta-tasks. PhaseFormer phase-exchange remixes
            # traffic inside phase buckets (support/query separately).
            xs, ys, xq, yq, ei, ew, nf, tid, dow = sample_task(
                Xtr, Ytr, Itr, A, s, args.k_spt, args.k_qry, args.node_pe_dim, device,
                spd=spd, table_size=tsz, calendar_start=cal_start, minutes_per_step=mps,
                permute_adj=permute_adj,
                phase_exchange=phase_exchange, period_len=period_len,
                phase_exchange_mode=phase_mode, phase_week_align=phase_week_align)
            ns = xs.shape[0]
            tid_s = tid[:ns] if tid is not None else None
            dow_s = dow[:ns] if dow is not None else None
            tid_q = tid[ns:] if tid is not None else None
            dow_q = dow[ns:] if dow is not None else None

            fast = OrderedDict(model.named_parameters())
            for _k in range(args.update_step):
                ps = fwd(fast, xs, ei, ew, nf, tid_s, dow_s)
                loss_s = torch.mean(torch.abs(ps - ys))          # normalized MAE
                grads = torch.autograd.grad(
                    loss_s, fast.values(),
                    create_graph=bool(args.second_order),
                    retain_graph=bool(args.second_order),
                    allow_unused=False)
                fast = OrderedDict(
                    (n, p - inner_lr * g) for (n, p), g in zip(fast.items(), grads))
            pq = fwd(fast, xq, ei, ew, nf, tid_q, dow_q)
            loss_q = torch.mean(torch.abs(pq - yq))
            outer = outer + loss_q
            qsum += float(loss_q)
        outer = outer / len(sizes)
        outer.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        meta_opt.step()

        if it % args.meta_eval_every == 0 or it == args.meta_iters:
            vm = src_val_mae()
            print(f'[stage2/MAML] iter {it:04d}/{args.meta_iters} '
                  f'query_mae(norm)={qsum/len(sizes):.4f} src_val_mae={vm:.4f} '
                  f'inner_lr={inner_lr:.2e}', flush=True)
            if vm < best_val - 1e-4:
                best_val, patience = vm, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= args.meta_patience:
                    print(f'[stage2/MAML] early stop (best={best_val:.4f})', flush=True)
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_val = src_val_mae()
    return model, best_val


# ===================== Stage 2 (Structure-MLDG / Orbit-MLDG) ===================== #
def sample_subgraph_batch(Xtr, Ytr, Itr, A, node_ids, k_batch, node_pe_dim, device,
                          spd=None, table_size=None, calendar_start=None,
                          minutes_per_step=None, permute_adj=False,
                          sem_full=None, behav_full=None):
    """Sample a contiguous Stage2 batch on a *fixed* node set (one MLDG domain).

    Unlike ``sample_task`` (same-graph support/query split for MAML), this returns
    a single batch ``(x, y, ei, ew, nf, tid, dow)`` used as one virtual domain.
    """
    S = np.sort(np.asarray(node_ids))
    A_sub = A[np.ix_(S, S)]
    if permute_adj:
        A_sub, S, _perm = stunet_permute_induced_adjacency(A_sub, S)
    edge_index, edge_weight, node_feat = build_graph(A_sub, node_pe_dim, device)
    T = Xtr.shape[0]
    k_batch = int(min(k_batch, T))
    idx = np.random.choice(T, size=k_batch, replace=False)
    S_t = torch.from_numpy(np.asarray(S)).long()
    x = Xtr[idx][:, :, S_t, :].to(device)
    y = Ytr[idx][:, :, S_t, :].to(device)
    tid = dow = None
    if Itr is not None:
        tid, dow = time_indices(
            Itr[idx], spd, table_size, device,
            calendar_start=calendar_start, minutes_per_step=minutes_per_step)
    sem = slice_sem(sem_full, S, device)
    behav = slice_behav(behav_full, S, device)
    return x, y, edge_index, edge_weight, node_feat, tid, dow, sem, behav


def sample_mldg_domain_pair(Xtr, Ytr, Itr, A, node_min, node_max, k_batch,
                            node_pe_dim, device,
                            spd=None, table_size=None, calendar_start=None,
                            minutes_per_step=None, permute_adj=False,
                            prefer_larger_target=True,
                            sem_full=None, behav_full=None):
    """Sample a disjoint structural domain pair (virtual source / virtual target).

    Domains are induced subgraphs of the *same* source city graph. This is the
    key departure from AAAI'18 MLDG (image-style domains): here domain shift is
    structural (node set / size / topology / recomputed Z=g(A)).
    """
    N = int(A.shape[0])
    node_max = int(min(node_max, N))
    node_min = int(min(node_min, max(node_max - 1, 1)))
    # Need room for two disjoint subsets.
    max_pair = max(N // 2, 1)
    node_max = int(min(node_max, max_pair))
    node_min = int(min(node_min, node_max))

    n_s = int(np.random.randint(node_min, node_max + 1))
    if prefer_larger_target:
        n_t = int(np.random.randint(n_s, node_max + 1))
    else:
        n_t = int(np.random.randint(node_min, node_max + 1))
    # Ensure both fit disjointly.
    if n_s + n_t > N:
        n_t = max(1, N - n_s)
        if n_t < node_min and n_s > node_min:
            # shrink source a bit to keep target usable
            n_s = max(node_min, N - max(node_min, n_t))
            n_t = max(1, N - n_s)

    all_ids = np.arange(N)
    S_v = np.sort(np.random.choice(all_ids, size=n_s, replace=False))
    remain = np.setdiff1d(all_ids, S_v, assume_unique=True)
    if remain.size < n_t:
        n_t = int(remain.size)
    T_v = np.sort(np.random.choice(remain, size=n_t, replace=False))

    sv = sample_subgraph_batch(
        Xtr, Ytr, Itr, A, S_v, k_batch, node_pe_dim, device,
        spd=spd, table_size=table_size, calendar_start=calendar_start,
        minutes_per_step=minutes_per_step, permute_adj=permute_adj,
        sem_full=sem_full, behav_full=behav_full)
    tv = sample_subgraph_batch(
        Xtr, Ytr, Itr, A, T_v, k_batch, node_pe_dim, device,
        spd=spd, table_size=table_size, calendar_start=calendar_start,
        minutes_per_step=minutes_per_step, permute_adj=permute_adj,
        sem_full=sem_full, behav_full=behav_full)
    return sv, tv, n_s, n_t


def meta_train_mldg(args, model, device, Xtr, Ytr, Itr, A, val_loader, scaler):
    """Stage2 = Structure-MLDG (Orbit-MLDG) on source subgraphs.

    Extends AAAI'18 MLDG (Li et al.): virtual meta-train / meta-test domains are
    *disjoint induced subgraphs* of the source city (structural domain shift with
    recomputed Z=g(A)), not same-graph support/query splits as in MAML.

    Per meta-iter, for each of ``task_num`` domain pairs (S_v, T_v):
      L = L_Sv(θ) + β L_Tv(θ') + γ L_Tv(θ) [+ δ L_full(θ)]
    where θ' = θ - α ∇ L_Sv(θ) (FOMAML-style one/few steps). The γ term aligns
    training with strict zero-shot Cross (deploy θ with no adaptation). Noroll:
    best = Stage2's own best clean source-val (no Stage1 rollback).
    """
    from collections import OrderedDict
    from torch.func import functional_call

    N = A.shape[0]
    node_max = int(min(args.node_max, N))
    node_min = int(min(args.node_min, max(node_max - 1, 1)))
    val_ei, val_ew, val_nf = build_graph(A, node_pe_dim=args.node_pe_dim, device=device)
    meta_opt = torch.optim.Adam(model.parameters(), lr=args.meta_lr, eps=1e-8, weight_decay=0)
    ute = args.use_time_emb
    cal_start, spd, mps = calendar_of(args.source, getattr(args, 'calendar_start', None))
    tsz = model.time_of_day_size
    inner_max, inner_min = args.update_lr, args.update_lr * 0.7
    meta_max, meta_min = args.meta_lr, args.meta_lr * 0.7
    beta = float(getattr(args, 'mldg_beta', 1.0))
    gamma = float(getattr(args, 'mldg_gamma', 1.0))
    delta = float(getattr(args, 'mldg_delta', 0.5))
    k_batch = int(getattr(args, 'k_spt', 128)) + int(getattr(args, 'k_qry', 128))
    # MLDG paper uses one meta-train step; keep update_step but default scripts use 1.
    update_step = int(max(getattr(args, 'update_step', 1), 1))
    permute_adj = _stage2_permute_adj_flag(args)
    prefer_larger_tgt = bool(int(getattr(args, 'mldg_prefer_larger_target', 1)))

    def src_val_mae():
        model.eval()
        with torch.no_grad():
            vl, vb = 0.0, 0
            for b in val_loader:
                x = b[0][..., :args.input_dim].float().to(device)
                y = b[1][..., :args.output_dim].float().to(device)
                tid, dow = (time_indices(
                    b[2], spd, tsz, device,
                    calendar_start=cal_start, minutes_per_step=mps) if ute else (None, None))
                vl += masked_mae_real(model(x, val_ei, val_ew, val_nf, tid, dow), y, scaler, args.mask_value).item()
                vb += 1
        model.train()
        return vl / max(vb, 1)

    s1_init = src_val_mae()
    best_val, best_state = float('inf'), None
    print(f'[stage2/MLDG] warm-start src_val_mae={s1_init:.4f} (no Stage1 rollback) '
          f'second_order={bool(args.second_order)} update_step={update_step} '
          f'beta={beta} gamma={gamma} delta={delta} k_batch={k_batch} '
          f'prefer_larger_target={int(prefer_larger_tgt)}', flush=True)

    def fwd(params, x, ei, ew, nf, tid, dow):
        return functional_call(model, params, args=(x, ei, ew, nf, tid, dow))

    # Full-graph anchor batch indices (resampled lightly each time we need it).
    def sample_full_batch():
        return sample_subgraph_batch(
            Xtr, Ytr, Itr, A, np.arange(N), k_batch, args.node_pe_dim, device,
            spd=spd, table_size=tsz, calendar_start=cal_start, minutes_per_step=mps,
            permute_adj=False)

    patience = 0
    for it in range(1, args.meta_iters + 1):
        for g in meta_opt.param_groups:
            g['lr'] = _compute_lr(it, args.meta_iters, meta_min, meta_max)
        inner_lr = _compute_lr(it, args.meta_iters, inner_min, inner_max)
        n_pairs = max(int(args.task_num), 1)

        model.train()
        meta_opt.zero_grad()
        outer = torch.zeros((), device=device)
        ls_sum = lt_ad_sum = lt_z_sum = lf_sum = 0.0

        for _ in range(n_pairs):
            sv, tv, _ns, _nt = sample_mldg_domain_pair(
                Xtr, Ytr, Itr, A, node_min, node_max, k_batch, args.node_pe_dim, device,
                spd=spd, table_size=tsz, calendar_start=cal_start, minutes_per_step=mps,
                permute_adj=permute_adj, prefer_larger_target=prefer_larger_tgt)
            xs, ys, ei_s, ew_s, nf_s, tid_s, dow_s = sv
            xt, yt, ei_t, ew_t, nf_t, tid_t, dow_t = tv

            fast = OrderedDict(model.named_parameters())
            # Meta-train on S_v → θ'
            loss_s = None
            for _k in range(update_step):
                ps = fwd(fast, xs, ei_s, ew_s, nf_s, tid_s, dow_s)
                loss_s = torch.mean(torch.abs(ps - ys))
                grads = torch.autograd.grad(
                    loss_s, fast.values(),
                    create_graph=bool(args.second_order),
                    retain_graph=bool(args.second_order),
                    allow_unused=False)
                fast = OrderedDict(
                    (n, p - inner_lr * g) for (n, p), g in zip(fast.items(), grads))
            # θ terms on S_v / T_v / full (zero-shot aligned)
            # Recompute L_Sv(θ) with current model params (not fast weights).
            theta = OrderedDict(model.named_parameters())
            loss_s_theta = torch.mean(torch.abs(
                fwd(theta, xs, ei_s, ew_s, nf_s, tid_s, dow_s) - ys))
            loss_t_adapt = torch.mean(torch.abs(
                fwd(fast, xt, ei_t, ew_t, nf_t, tid_t, dow_t) - yt))
            loss_t_zero = torch.mean(torch.abs(
                fwd(theta, xt, ei_t, ew_t, nf_t, tid_t, dow_t) - yt))

            pair_loss = loss_s_theta + beta * loss_t_adapt + gamma * loss_t_zero
            if delta > 0:
                xf, yf, ei_f, ew_f, nf_f, tid_f, dow_f = sample_full_batch()
                loss_full = torch.mean(torch.abs(
                    fwd(theta, xf, ei_f, ew_f, nf_f, tid_f, dow_f) - yf))
                pair_loss = pair_loss + delta * loss_full
                lf_sum += float(loss_full)
            else:
                loss_full = None

            outer = outer + pair_loss
            ls_sum += float(loss_s_theta)
            lt_ad_sum += float(loss_t_adapt)
            lt_z_sum += float(loss_t_zero)

        outer = outer / n_pairs
        outer.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        meta_opt.step()

        if it % args.meta_eval_every == 0 or it == args.meta_iters:
            vm = src_val_mae()
            print(f'[stage2/MLDG] iter {it:04d}/{args.meta_iters} '
                  f'Ls={ls_sum/n_pairs:.4f} Lt\'={lt_ad_sum/n_pairs:.4f} '
                  f'Lt0={lt_z_sum/n_pairs:.4f} Lf={lf_sum/max(n_pairs,1):.4f} '
                  f'src_val_mae={vm:.4f} inner_lr={inner_lr:.2e}', flush=True)
            if vm < best_val - 1e-4:
                best_val, patience = vm, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= args.meta_patience:
                    print(f'[stage2/MLDG] early stop (best={best_val:.4f})', flush=True)
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_val = src_val_mae()
    return model, best_val


# =============================== Stage 3 ================================== #
@torch.no_grad()
def eval_target(args, model, device, target, step=None, pack=None):
    """Full zero-shot test eval (same metrics as the original Stage3 path)."""
    if pack is None:
        pack = load_target_eval_pack(args, target, device, model=model)
    else:
        if _behav_zero_shot_flag(args):
            from lib.behav_type import resolve_behav_alias
            if resolve_behav_alias(target) != resolve_behav_alias(str(getattr(args, 'behav_source', '') or '') or str(args.source)):
                pack['behav'] = retrieve_zeroshot_pi_sem_proj(
                    args, model, target, pack['sem'], device)
    ute = pack['ute']
    cal_start, spd, mps = pack['cal_start'], pack['spd'], pack['mps']
    test_loader, scaler = pack['test_loader'], pack['scaler']
    edge_index, edge_weight, node_feat = pack['edge_index'], pack['edge_weight'], pack['node_feat']
    sem = pack['sem']
    behav = pack['behav']
    model.eval()
    preds, trues = [], []
    t0 = time.perf_counter()
    for batch in test_loader:
        x = to_dev(batch[0], device)[..., :args.input_dim]
        y = to_dev(batch[1], device)[..., :args.output_dim]
        tid, dow = (time_indices(
            batch[2], spd, model.time_of_day_size, device,
            calendar_start=cal_start, minutes_per_step=mps) if ute else (None, None))
        preds.append(model(x, edge_index, edge_weight, node_feat, tid, dow, sem, behav).cpu())
        trues.append(y.cpu())
    if device.type == 'cuda':
        torch.cuda.synchronize()
    infer_s = time.perf_counter() - t0
    y_pred = scaler.inverse_transform(torch.cat(preds, 0).numpy())
    y_true = scaler.inverse_transform(torch.cat(trues, 0).numpy())
    thr = args.mape_mask_threshold
    mmape, valid = masked_mape_np(y_pred, y_true, thr)
    overall = {'MAE': mae_np(y_pred, y_true), 'RMSE': rmse_np(y_pred, y_true),
               'MAPE%': mape_np(y_pred, y_true), 'Masked-MAPE%': mmape}
    hm = [mae_np(y_pred[:, h], y_true[:, h]) for h in range(y_true.shape[1])]
    hr = [rmse_np(y_pred[:, h], y_true[:, h]) for h in range(y_true.shape[1])]
    step_s = '' if step is None else f' step={int(step):04d}'
    print(f'\n===== OA-STID (full) ZERO-SHOT {args.source} -> {target} (h={args.horizon}, nodes={DATASET_INFO[target][0]}{step_s}) =====')
    print(f'MAE={overall["MAE"]:.4f} RMSE={overall["RMSE"]:.4f} Masked-MAPE@{thr:g}={overall["Masked-MAPE%"]:.4f}% (valid={valid:.4f})', flush=True)
    out = {
        'source': args.source, 'target': target, 'model': 'OASTIDTransfer',
        'pipeline': 'stage1+stage2+cross', 'tag': args.tag,
        'lag': args.lag, 'horizon': args.horizon, 'num_nodes': DATASET_INFO[target][0],
        'overall': overall, 'horizon_mae': hm, 'horizon_rmse': hr,
        'mape_valid_ratio': valid, 'infer_seconds': infer_s,
        'mape_mask_threshold': float(thr),
        'stage2_step': None if step is None else int(step),
    }
    save_dir = os.path.join(
        PROJECT_ROOT, 'output', 'cross_domain',
        f'oastid_{args.source.lower()}_to_{target.lower()}_{args.tag}')
    os.makedirs(save_dir, exist_ok=True)
    fname = f'{target}_metrics.json' if step is None else f'{target}_metrics_step{int(step):04d}.json'
    with open(os.path.join(save_dir, fname), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Saved: {os.path.join(save_dir, fname)}', flush=True)
    return overall


def parse_args():
    p = argparse.ArgumentParser('OA-STID full pipeline (stage1+stage2+cross)')
    p.add_argument('--source', type = str, required = True, choices = list(DATASET_INFO))
    p.add_argument('--targets', type = str, required = True)
    p.add_argument('--tag', type = str, default = 'oastid_full')
    p.add_argument('--device', type = str, default = 'cuda:0')
    p.add_argument('--lag', type = int, default = 12)
    p.add_argument('--horizon', type = int, default = 12)
    p.add_argument('--val_ratio', type = float, default = 0.2)
    p.add_argument('--test_ratio', type = float, default = 0.2)
    p.add_argument('--batch_size', type = int, default = 64)
    p.add_argument('--mask_value', type = float, default = 0.0)
    p.add_argument('--stage1_epochs', type = int, default = 30)
    p.add_argument('--lr', type = float, default = 0.002)
    p.add_argument('--weight_decay', type = float, default = 0.0001)
    p.add_argument('--patience', type = int, default = 12)
    p.add_argument('--clip', type = float, default = 5.0)
    p.add_argument('--skip_stage2', type = int, default = 0, help = '1: skip stage2 entirely, cross-eval with stage1-only weights')
    p.add_argument('--stage2_mode', type = str, default = 'erm', choices = [
        'erm',
        'maml',
        'mldg'], help = 'erm: episodic subgraph ERM; maml: OAGNN FOMAML; mldg: Structure-MLDG (disjoint subgraph virtual domains)')
    p.add_argument('--second_order', type = int, default = 0, help = 'MAML/MLDG: 1=second-order, 0=first-order')
    p.add_argument('--meta_patience', type = int, default = 20)
    p.add_argument('--meta_iters', type = int, default = 800)
    p.add_argument('--task_num', type = int, default = 4)
    p.add_argument('--update_step', type = int, default = 5)
    p.add_argument('--update_lr', type = float, default = 0.01, help = 'inner SGD lr (normalized space)')
    p.add_argument('--reptile_eps', type = float, default = 0.1, help = 'Reptile meta step size')
    p.add_argument('--meta_lr', type = float, default = 0.0005)
    p.add_argument('--k_spt', type = int, default = 32)
    p.add_argument('--k_qry', type = int, default = 32)
    p.add_argument('--node_min', type = int, default = 128)
    p.add_argument('--node_max', type = int, default = 400)
    p.add_argument('--meta_eval_every', type = int, default = 100)
    p.add_argument('--cross_every', type = int, default = 20, help = 'Stage2: run full zero-shot Cross on --targets every N meta-steps; 0 disables mid-Stage2 cross (Stage3 still runs)')
    p.add_argument('--mldg_beta', type = float, default = 1.0, help = "MLDG: weight for L_Tv(θ') after meta-train step (AAAI'18 β)")
    p.add_argument('--mldg_gamma', type = float, default = 1.0, help = 'MLDG: weight for zero-shot L_Tv(θ) (aligns with Cross; 0=vanilla MLDG)')
    p.add_argument('--mldg_delta', type = float, default = 0.5, help = 'MLDG: weight for full-graph anchor L_full(θ); 0=disable')
    p.add_argument('--mldg_prefer_larger_target', type = int, default = 1, help = '1: sample |T_v| >= |S_v| (small→large structural shift)')
    p.add_argument('--stage2_permute_adj', type = int, default = 0, help = "1: STUNet-style random adjacency permutation on Stage2 random subgraphs (A'=P A P^T + aligned X/Y); 0: original")
    p.add_argument('--stage2_phase_exchange', type = int, default = 0, help = '1: PhaseFormer-style Stage2 temporal construction via phase-internal exchange (not contiguous patch shuffle); 0: original (default)')
    p.add_argument('--stage2_period_len', type = int, default = 0, help = 'PhaseFormer period_len for Stage2 phase exchange; <=0: use source steps_per_day (LargeST 15-min -> 96)')
    p.add_argument('--stage2_phase_exchange_mode', type = str, default = 'both', choices = [
        'within',
        'cross',
        'both'], help = 'within: per-sample period-axis exchange after phase view; cross: absolute-phase buckets across batch; both: within then cross')
    p.add_argument('--stage2_phase_week_align', type = int, default = 1, help = '1 (default): only exchange same week-slot (same weekday + same TOD; Tue↔Tue only); 0: legacy daily-phase buckets (Tue↔Wed allowed)')
    p.add_argument('--stage2_phase_preprocess', type = int, default = 0, help = '1: PhaseFormer phase-internal exchange as offline preprocessing on the full contiguous TRAIN series, then cut Stage2 windows (recommended); disables online small-window --stage2_phase_exchange. 0: off (default)')
    p.add_argument('--input_dim', type = int, default = 1)
    p.add_argument('--output_dim', type = int, default = 1)
    p.add_argument('--embed_dim', type = int, default = 32)
    p.add_argument('--node_dim', type = int, default = 32)
    p.add_argument('--num_layer', type = int, default = 3)
    p.add_argument('--node_pe_dim', type = int, default = 8)
    p.add_argument('--orbit_hidden', type = int, default = 64)
    p.add_argument('--orbit_gnn', type = str, default = 'gcn', choices = [
        'gcn',
        'sage'], help = 'Orbit encoder: gcn = original 3-layer BETAET GCNConv; sage = GraphSAGE from code/src/model/graph_sage.py (3× SAGEConv, train-time neighbor sampling)')
    p.add_argument('--sage_k', type = int, default = 3, help = 'GraphSAGE: neighbors sampled per node during training')
    p.add_argument('--sage_dropout', type = float, default = 0.1, help = 'GraphSAGE: dropout after SAGEConv 1 and 2')
    p.add_argument('--sage_norm', type = int, default = 0, help = 'GraphSAGE: 1 = LayerNorm after SAGEConv 1 and 2')
    p.add_argument('--se_type', type = str, default = 'lap', choices = [
        'lap',
        'rwse',
        'hkse',
        'frse'], help = 'lap / rwse / hkse / frse structural encoding for F')
    p.add_argument('--se_self_loop', type = int, default = -1, help = '-1: use SE default (lap=on, rwse=off); 0/1: force off/on')
    p.add_argument('--use_time_emb', type = int, default = 0, help = '1: add STID time-of-day/day-of-week tables (derived from cycle_index)')
    p.add_argument('--use_cal_in_series', type = int, default = 0, help = '1: STID-style concat 12-step TOD/DOW [0,1] into series Conv (flow still input_dim; RevIN/BSTS stay on flow). Requires --use_time_emb 1')
    p.add_argument('--time_emb_last_step', type = int, default = 1, help = '1: STID-style TOD/DOW table lookup at window last step; 0: window first step (cycle_index). calseq 12-step channels always expand from the first step.')
    p.add_argument('--calendar_start', type = str, default = None, help = 'source-domain calendar start for weekday lookup; default = CALENDAR_START[source]. Targets always use their own start.')
    p.add_argument('--temp_dim_tid', type = int, default = 32)
    p.add_argument('--temp_dim_diw', type = int, default = 32)
    p.add_argument('--time_of_day_size', type = int, default = 288, help = 'fixed time-of-day table size (fraction-of-day slots)')
    p.add_argument('--use_sem_emb', type = int, default = 1, help = '1: concat Qwen3-Embedding-0.6B road-text features; auto-off if source has no LargeST meta csv')
    p.add_argument('--sem_dim', type = int, default = 32, help = 'projected road-semantic dim in concat; zeroshot retrieval keys are this dim + 3 graph scalars (e.g. 8 → 11-d key)')
    p.add_argument('--sem_in_dim', type = int, default = 1024, help = 'raw Qwen embedding dim (0.6B = 1024)')
    p.add_argument('--qwen_path', type = str, default = DEFAULT_QWEN_PATH, help = 'local Qwen3-Embedding-0.6B directory')
    p.add_argument('--qwen_batch_size', type = int, default = 64)
    p.add_argument('--use_bsts', type = int, default = 1, help = '1: concat BSTS (s_temp / s_spat / s_graph / s_spec); each group has its own Linear, then concatenate')
    p.add_argument('--stats_v2', type = int, default = 1, help = '1: s_temp is 8-d (base last/mean/std/slope + min/max/range/last-mean); 0: classic 4-d [last, mean, std, slope] only')
    p.add_argument('--stats_nbr', type = int, default = 1, help = '1: s_spat neighbor mean-pool on the road graph (4-d)')
    p.add_argument('--stats_rank', type = int, default = 1, help = '1: s_graph intra-graph percentile ranks of last/mean (2-d)')
    p.add_argument('--stats_spec', type = int, default = 1, help = '1: s_spec rFFT energy + near/far multi-scale (6-d)')
    p.add_argument('--stats_proj_temp_dim', type = int, default = 16, help = 'BSTS s_temp Linear output width (original grouped head)')
    p.add_argument('--stats_proj_spat_dim', type = int, default = 8, help = 'BSTS s_spat Linear output width')
    p.add_argument('--stats_proj_graph_dim', type = int, default = 8, help = 'BSTS s_graph Linear output width')
    p.add_argument('--stats_proj_spec_dim', type = int, default = 8, help = 'BSTS s_spec Linear output width')
    p.add_argument('--use_behav_emb', type = int, default = 1, help = '1: concat behavior-type embedding (π @ learnable prototypes W)')
    p.add_argument('--behav_k', type = int, default = 32, help = 'number of behavior prototypes / K-means clusters')
    p.add_argument('--behav_dim', type = int, default = 8, help = 'learned prototype dim concatenated into hidden')
    p.add_argument('--behav_tau', type = float, default = 1.0, help = 'softmax temperature: π_ik = softmax_k(-||φ_i-c_k||² / τ)')
    p.add_argument('--behav_n_daily_bins', type = int, default = 24, help = 'Part-A daily-curve bins (96 slots → 24 bins for LargeST)')
    p.add_argument('--behav_source', type = str, default = '', help = 'source dataset whose TRAIN nodes define K-means; empty → --source')
    p.add_argument('--behav_data_root', type = str, default = '', help = 'cache dir for {src}_behav_pack and {ds}_behav_pi; empty → data/largest_meta/.behav_cache')
    p.add_argument('--behav_target_frac', type = float, default = 0.6, help = 'Cross-only (behav_zero_shot=0, behav_target_days=0): fraction of the FULL target series used to estimate φ. Ignored when behav_zero_shot=1 or behav_target_days>0.')
    p.add_argument('--behav_target_days', type = int, default = 0, help = 'Cross-only: if >0, use the first N calendar days (N * spd steps) of the target series for 29-d φ. 7 days on LargeST = 672 steps. Overrides behav_target_frac.')
    p.add_argument('--behav_zero_shot', type = int, default = 0, help = '1: Cross target π from sem_proj(Qwen)+graph retrieval over the source bank. No target traffic. Stage2 uses induce.')
    p.add_argument('--behav_stage2_induce', type = int, default = 0, help = '1: Stage2 mixes gold / short-window 29-d / type-space synthesis / retrieve / type-histogram node sampling')
    p.add_argument('--behav_p_gold', type = float, default = 0.16666666666666666, help = 'Stage2 task fraction with year-train π*')
    p.add_argument('--behav_p_observe', type = float, default = 0.5, help = 'Stage2 task fraction with contiguous-window 29-d π')
    p.add_argument('--behav_p_synth', type = float, default = 0.3333333333333333, help = 'Stage2 task fraction with type-space synthesis')
    p.add_argument('--behav_p_retrieve', type = float, default = 0.0, help = 'Stage2 task fraction with Cross-isomorphic leave-one-out retrieval π (live sem_proj keys)')
    p.add_argument('--behav_p_compose', type = float, default = 0.5, help = 'independent prob of sampling nodes by a random type histogram')
    p.add_argument('--behav_compose_alpha', type = float, default = 0.3, help = 'Dirichlet concentration for type-histogram tasks')
    p.add_argument('--behav_observe_days_min', type = int, default = 0, help = 'min contiguous days for observation-induced φ (0=zero-history)')
    p.add_argument('--behav_observe_days_max', type = int, default = 18, help = 'max contiguous days for observation-induced φ')
    p.add_argument('--behav_retrieve_tau', type = float, default = 0.1, help = 'softmax temperature for static source-bank retrieval')
    p.add_argument('--mape_mask_threshold', type = float, default = 0.001)
    p.add_argument('--seed', type = int, default = 10)
    p.add_argument('--fusion', type = str, default = 'concat', choices = [
        'concat',
        'tstep',
        'tstep_band'], help = 'concat: original STID-style cat; tstep: Q=Z(+cal), K/V=T step tokens; tstep_band: Q=Z(+cal), K/V=2T low/high step tokens')
    p.add_argument('--attn_dim', type = int, default = 32)
    p.add_argument('--attn_heads', type = int, default = 4)
    p.add_argument('--attn_dropout', type = float, default = 0.1)
    p.add_argument('--band_low_k_max', type = int, default = 1, help = 'rFFT bins k=0..band_low_k_max kept as low band')
    p.add_argument('--use_residual_base', type = int, default = 0, help = '0 (default): pure concat→attn replacement; 1: optional ŷ=base(no Z)+attn (experimental)')
    p.add_argument('--attn_node_chunk', type = int, default = 256, help = 'chunk size over nodes for cross-attn (limits B*chunk for CUDA MHA)')
    p.add_argument('--eval_ckpt', type = str, default = '', help = 'if set: skip Stage1/2, load this checkpoint and run Cross only')
    p.add_argument('--stage1_ckpt', type = str, default = '', help = 'if set: skip Stage1 training and load this Stage1 (or compatible) ckpt, then run Stage2+Cross. Matching tensors only (shape-safe).')
    p.add_argument('--skip_stage1', type = int, default = 0, help = '1: skip Stage1 training (requires --stage1_ckpt)')
    return p.parse_args()

def _args_from_ckpt(cli_args, ckpt_args):
    """Rebuild Namespace: checkpoint training args + CLI overrides for eval."""
    merged = dict(ckpt_args)
    for k in ('device', 'targets', 'attn_node_chunk', 'batch_size', 'seed', 'mask_value', 'mape_mask_threshold', 'eval_ckpt', 'tag', 'stage1_ckpt', 'skip_stage1', 'behav_target_frac', 'behav_target_days', 'behav_zero_shot', 'behav_stage2_induce', 'behav_retrieve_tau'):
        if hasattr(cli_args, k):
            merged[k] = getattr(cli_args, k)
    merged.setdefault('use_cal_in_series', 0)
    merged.setdefault('time_emb_last_step', 0)
    merged.setdefault('fusion', 'concat')
    merged.setdefault('attn_dim', merged.get('embed_dim', 32))
    merged.setdefault('attn_heads', 4)
    merged.setdefault('attn_dropout', 0.1)
    merged.setdefault('band_low_k_max', 1)
    merged.setdefault('use_residual_base', 0)
    merged.setdefault('attn_node_chunk', 256)
    merged.setdefault('se_self_loop', -1)
    merged.setdefault('calendar_start', None)
    merged.setdefault('use_sem_emb', 0)
    merged.setdefault('sem_dim', 32)
    merged.setdefault('sem_in_dim', 1024)
    merged.setdefault('qwen_path', DEFAULT_QWEN_PATH)
    merged.setdefault('qwen_batch_size', 64)
    merged.setdefault('use_bsts', 0)
    merged.setdefault('stats_v2', 1)
    merged.setdefault('stats_nbr', 1)
    merged.setdefault('stats_rank', 1)
    merged.setdefault('stats_spec', 1)
    merged.setdefault('stats_proj_temp_dim', 16)
    merged.setdefault('stats_proj_spat_dim', 8)
    merged.setdefault('stats_proj_graph_dim', 8)
    merged.setdefault('stats_proj_spec_dim', 8)
    merged.setdefault('use_behav_emb', 0)
    merged.setdefault('behav_k', 32)
    merged.setdefault('behav_dim', 8)
    merged.setdefault('behav_tau', 1.0)
    merged.setdefault('behav_n_daily_bins', 24)
    merged.setdefault('behav_source', '')
    merged.setdefault('behav_data_root', '')
    merged.setdefault('behav_target_frac', 0.6)
    merged.setdefault('behav_target_days', 0)
    merged.setdefault('behav_zero_shot', 0)
    merged.setdefault('behav_stage2_induce', 0)
    merged.setdefault('behav_p_gold', 0.16666666666666666)
    merged.setdefault('behav_p_observe', 0.5)
    merged.setdefault('behav_p_synth', 0.3333333333333333)
    merged.setdefault('behav_p_retrieve', 0.0)
    merged.setdefault('behav_p_compose', 0.5)
    merged.setdefault('behav_compose_alpha', 0.3)
    merged.setdefault('behav_observe_days_min', 0)
    merged.setdefault('behav_observe_days_max', 18)
    merged.setdefault('behav_retrieve_tau', 0.1)
    merged.setdefault('cross_every', 20)
    merged.setdefault('orbit_gnn', 'gcn')
    merged.setdefault('sage_k', 3)
    merged.setdefault('sage_dropout', 0.1)
    merged.setdefault('sage_norm', 0)
    return argparse.Namespace(**merged)


def main():
    global _SE_TYPE, _SE_SELF_LOOP
    cli_args = parse_args()
    torch.manual_seed(cli_args.seed)
    np.random.seed(cli_args.seed)

    # -------- Cross-only from checkpoint --------
    if cli_args.eval_ckpt:
        ckpt_path = cli_args.eval_ckpt
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(PROJECT_ROOT, ckpt_path)
        print(f'[eval_ckpt] loading {ckpt_path}', flush=True)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        args = _args_from_ckpt(cli_args, ckpt['args'])
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        targets = [t.strip() for t in args.targets.split(',') if t.strip()]
        _SE_TYPE = args.se_type
        _SE_SELF_LOOP = None if int(getattr(args, 'se_self_loop', -1)) < 0 else bool(args.se_self_loop)
        print(f'OA-STID CROSS-ONLY | source={args.source} -> {targets} | h={args.horizon} '
              f'device={device} fusion={args.fusion} attn_node_chunk={args.attn_node_chunk} '
              f'tag={args.tag}', flush=True)
        model = build_oastid_model(args).to(device)
        missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
        print(f'[eval_ckpt] loaded stage1_val={ckpt.get("stage1_val")} '
              f'stage2_val={ckpt.get("stage2_val")} '
              f'missing={len(missing)} unexpected={len(unexpected)}', flush=True)
        if missing:
            print(f'[eval_ckpt] missing keys (first 10): {missing[:10]}', flush=True)
        if unexpected:
            print(f'[eval_ckpt] unexpected keys (first 10): {unexpected[:10]}', flush=True)
        print(f'model params: {sum(p.numel() for p in model.parameters())}', flush=True)
        print('\n########## STAGE 3: cross-domain zero-shot (from ckpt) ##########', flush=True)
        summary = {tgt: eval_target(args, model, device, tgt) for tgt in targets}
        print('\n================ SUMMARY (cross-only) ================', flush=True)
        for tgt, m in summary.items():
            print(f'{args.source} -> {tgt}: MAE={m["MAE"]:.4f} RMSE={m["RMSE"]:.4f} '
                  f'Masked-MAPE={m["Masked-MAPE%"]:.4f}%', flush=True)
        return

    args = cli_args
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    targets = [t.strip() for t in args.targets.split(',') if t.strip()]
    _SE_TYPE = args.se_type
    _SE_SELF_LOOP = None if args.se_self_loop < 0 else bool(args.se_self_loop)
    print(f'OA-STID FULL | source={args.source} -> {targets} | h={args.horizon} '
          f'device={device} | se_type={_SE_TYPE} se_self_loop={_SE_SELF_LOOP} '
          f'fusion={args.fusion}', flush=True)

    model = build_oastid_model(args).to(device)
    print(f'use_time_emb={args.use_time_emb} fusion={args.fusion} '
          f'residual_base={getattr(args, "use_residual_base", 1)}', flush=True)
    if args.use_time_emb:
        src_cal, _, src_mps = calendar_of(args.source, args.calendar_start)
        print(f'[time] weekday=REAL calendar (Mon=0..Sun=6); '
              f'source start={src_cal} min/step={src_mps}', flush=True)
        for t in targets:
            tc, _, tm = calendar_of(t)
            print(f'[time] target {t} start={tc} min/step={tm}', flush=True)
    print(f'model params: {sum(p.numel() for p in model.parameters())}', flush=True)

    print('\n########## STAGE 1: source pretrain ##########', flush=True)
    t0 = time.perf_counter()
    model, s1_val = train_source(args, model, device)
    print(f'[stage1] done best_val_mae={s1_val:.4f} ({time.perf_counter()-t0:.1f}s)', flush=True)

    if args.skip_stage2:
        print('\n########## STAGE 2: SKIPPED (stage1-only weights) ##########', flush=True)
        s2_val = s1_val
    else:
        print(f'\n########## STAGE 2: {args.stage2_mode.upper()} ##########', flush=True)
        t0 = time.perf_counter()
        Xtr, Ytr, Itr, val_loader, scaler, A = collect_source_arrays(args, device)
        print(f'[stage2] windows X={tuple(Xtr.shape)} nodes={A.shape[0]} '
              f'subgraph=[{args.node_min},{min(args.node_max,A.shape[0])}] '
              f'task_num={args.task_num} update_step={args.update_step} '
              f'permute_adj={int(getattr(args, "stage2_permute_adj", 0))} '
              f'phase_exchange={int(getattr(args, "stage2_phase_exchange", 0))} '
              f'phase_preprocess={int(getattr(args, "stage2_phase_preprocess", 0))} '
              f'period_len={int(getattr(args, "stage2_period_len", 0))} '
              f'phase_mode={getattr(args, "stage2_phase_exchange_mode", "both")} '
              f'week_align={int(getattr(args, "stage2_phase_week_align", 1))} '
              f'mldg_beta={getattr(args, "mldg_beta", 1.0)} '
              f'mldg_gamma={getattr(args, "mldg_gamma", 1.0)} '
              f'mldg_delta={getattr(args, "mldg_delta", 0.5)}',
              flush=True)
        if args.stage2_mode == 'maml':
            run_stage2 = meta_train_maml
        elif args.stage2_mode == 'mldg':
            run_stage2 = meta_train_mldg
        else:
            run_stage2 = meta_train
        model, s2_val = run_stage2(args, model, device, Xtr, Ytr, Itr, A, val_loader, scaler)
        print(f'[stage2] done best_src_val_mae={s2_val:.4f} ({time.perf_counter()-t0:.1f}s)', flush=True)

    ckpt_dir = os.path.join(PROJECT_ROOT, 'output', 'oastid', f'{args.source.lower()}_{args.tag}')
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'stage1_val': s1_val,
                'stage2_val': s2_val, 'args': vars(args)}, os.path.join(ckpt_dir, 'best.pt'))

    print('\n########## STAGE 3: cross-domain zero-shot ##########', flush=True)
    summary = {tgt: eval_target(args, model, device, tgt) for tgt in targets}

    print('\n================ SUMMARY (stage1+stage2+cross) ================', flush=True)
    for tgt, m in summary.items():
        print(f'{args.source} -> {tgt}: MAE={m["MAE"]:.4f} RMSE={m["RMSE"]:.4f} '
              f'Masked-MAPE={m["Masked-MAPE%"]:.4f}%', flush=True)


if __name__ == '__main__':
    main()

def _project_qwen_sem(model, qwen):
    """Run the concat-slot MLP: Qwen [N, 1024] → [N, sem_dim] numpy."""
    if model is None or not hasattr(model, 'sem_proj') or model.sem_proj is None:
        raise RuntimeError('zeroshot retrieval keys need model.sem_proj')
    proj = model.sem_proj
    device = next(proj.parameters()).device
    if isinstance(qwen, np.ndarray):
        t = torch.from_numpy(np.asarray(qwen, dtype=np.float32)).to(device)
    else:
        t = qwen.to(device)
    if t.dim() != 2:
        raise ValueError(f'qwen expected [N, D], got {tuple(t.shape)}')
    was = proj.training
    proj.eval()
    with torch.no_grad():
        out = proj(t)
    if was:
        proj.train()
    return np.asarray(out.detach().cpu().numpy(), dtype=np.float64)


def retrieve_zeroshot_pi_sem_proj(args, model, tgt_dataset, qwen_tgt, device):
    """Target π: key = L2([sem_proj(Qwen) || graph_3]). No target flow.

    Uses the same MLP as the concat road-sem slot (sem_dim, e.g. 8 → 11-d key).
    Recomputed from current weights; not the old 1024+3 cache.
    """
    from lib.behav_zeroshot import retrieve_target_pi_static
    from lib.behav_type import load_or_build_behav_pi
    src = str(getattr(args, 'behav_source', '') or '') or str(args.source)
    src_cal, src_spd, src_mps = calendar_of(src)
    pi_src = load_or_build_behav_pi(
        src, DATASET_INFO[src][0], args,
        src_spd, src_cal, src_mps,
        src, src_spd, src_cal, src_mps,
    )
    q8_t = _project_qwen_sem(model, qwen_tgt)
    q8_s = _project_qwen_sem(model, _qwen_numpy(args, src))
    if q8_t.shape[1] != q8_s.shape[1]:
        raise RuntimeError(
            f'sem_proj width tgt {q8_t.shape[1]} vs src {q8_s.shape[1]}'
        )
    key_d = int(q8_t.shape[1]) + 3
    print(
        f'[behav/zeroshot] Cross {src}→{tgt_dataset}: sem_proj '
        f'{q8_t.shape[1]}-d + graph 3-d → {key_d}-d keys, NO target traffic',
        flush=True,
    )
    pi = retrieve_target_pi_static(
        q8_t, _load_adj_numpy(tgt_dataset),
        q8_s, _load_adj_numpy(src),
        pi_src,
        tau_ret=float(getattr(args, 'behav_retrieve_tau', 0.1)),
    )
    return torch.from_numpy(np.asarray(pi, dtype=np.float32)).to(device)


def _source_retrieve_pi_current_sem(ctx, S):
    """Leave-one-out π with live sem_proj keys. Same static path as Cross."""
    from lib.behav_zeroshot import retrieve_source_leave_one_out
    if ctx['qwen'] is None:
        raise RuntimeError('retrieve π needs Qwen keys')
    qwen_key = ctx['qwen']
    mdl = ctx.get('model')
    if mdl is not None and getattr(mdl, 'sem_proj', None) is not None:
        qwen_key = _project_qwen_sem(mdl, ctx['qwen'])
    return retrieve_source_leave_one_out(
        qwen_key,
        ctx['A'],
        ctx['pi_gold'],
        S,
        tau_ret=ctx['tau_ret'],
    )


def build_behav_meta_context(args, A, sem_full, behav_full, Itr):
    """Source-only pack for Stage2 observation / synthesis / composition.

    Built once per Stage2 run. Target cities are never read.
    """
    if not bool(int(getattr(args, 'use_behav_emb', 0))):
        return None
    if not bool(int(getattr(args, 'behav_stage2_induce', 0))) and not _behav_zero_shot_flag(args):
        return None
    from lib.behav_type import (
        compute_phi29, build_source_pack, split_train_raw_flow,
    )
    src = str(getattr(args, 'behav_source', '') or '') or str(args.source)
    cal, spd, mps = calendar_of(src, getattr(args, 'calendar_start', None))
    pack = build_source_pack(
        src, int(spd), cal, int(mps),
        float(args.val_ratio), float(args.test_ratio),
        k=int(getattr(args, 'behav_k', 32)),
        tau=float(getattr(args, 'behav_tau', 1.0)),
        n_bins=int(getattr(args, 'behav_n_daily_bins', 24)),
        seed=int(getattr(args, 'seed', 10)),
    )
    if 'phi_raw' not in pack:
        flow = split_train_raw_flow(
            src, val_ratio=float(args.val_ratio), test_ratio=float(args.test_ratio),
        )
        phi_raw, _daily, _counts = compute_phi29(
            flow, int(spd), cal, int(mps),
            n_bins=int(getattr(args, 'behav_n_daily_bins', 24)),
        )
        pack['phi_raw'] = phi_raw
    if behav_full is None:
        raise RuntimeError('behav_stage2_induce / zero_shot requires source π*')
    pi_gold = np.asarray(behav_full.detach().cpu().numpy(), dtype=np.float64)
    if sem_full is not None:
        qwen = np.asarray(sem_full.detach().cpu().numpy(), dtype=np.float64)
    elif bool(int(getattr(args, 'use_sem_emb', 0))):
        qwen = _qwen_numpy(args, src)
    else:
        qwen = None
    train_flow = split_train_raw_flow(
        src, val_ratio=float(args.val_ratio), test_ratio=float(args.test_ratio),
    )
    itr_np = None
    if Itr is not None:
        itr_np = np.asarray(Itr.detach().cpu().numpy(), dtype=np.int64).reshape(-1)
    p_gold = float(getattr(args, 'behav_p_gold', 0.16666666666666666))
    p_observe = float(getattr(args, 'behav_p_observe', 0.5))
    p_synth = float(getattr(args, 'behav_p_synth', 0.3333333333333333))
    p_retrieve = float(getattr(args, 'behav_p_retrieve', 0.0))
    z = p_gold + p_observe + p_synth + p_retrieve
    if z <= 0.0:
        raise ValueError('behav_p_gold + p_observe + p_synth + p_retrieve must be > 0')
    ctx = dict(
        source_pack=pack,
        pi_gold=pi_gold,
        phi_z=np.asarray(pack['phi'], dtype=np.float64),
        labels=np.asarray(pack['labels'], dtype=np.int64),
        qwen=qwen,
        A=np.asarray(A, dtype=np.float64),
        train_flow=train_flow,
        spd=int(spd),
        cal=cal,
        mps=int(mps),
        itr_np=itr_np,
        lag=int(args.lag),
        horizon=int(args.horizon),
        tau=float(getattr(args, 'behav_tau', 1.0)),
        tau_ret=float(getattr(args, 'behav_retrieve_tau', 0.1)),
        p_gold=p_gold / z,
        p_observe=p_observe / z,
        p_synth=p_synth / z,
        p_retrieve=p_retrieve / z,
        p_compose=float(getattr(args, 'behav_p_compose', 0.5)),
        dirichlet_alpha=float(getattr(args, 'behav_compose_alpha', 0.3)),
        days_min=int(getattr(args, 'behav_observe_days_min', 0)),
        days_max=int(getattr(args, 'behav_observe_days_max', 18)),
        counters={
            'gold': 0, 'observe': 0, 'synth': 0, 'retrieve': 0, 'compose': 0,
            'zero_hist': 0, 'interp': 0, 'qwen_mix': 0, 'convex_mix': 0, 'temp': 0,
        },
    )
    print(
        f"[behav/zeroshot] Stage2 induce on {src}: p_gold={ctx['p_gold']:.3f}"
        f" p_observe={ctx['p_observe']:.3f} p_synth={ctx['p_synth']:.3f}"
        f" p_retrieve={ctx['p_retrieve']:.3f} p_compose={ctx['p_compose']:.3f}"
        f" observe_days=[{ctx['days_min']},{ctx['days_max']}] train_flow_T={train_flow.shape[0]}"
        f" (source only)",
        flush=True,
    )
    return ctx


def load_compatible_state_dict(model, path, log_prefix='[stage1_ckpt]'):
    """Load tensors whose names and shapes match ``model``; leave the rest as-is.

    Used to warm-start Stage2 when the previous Stage1 ckpt has a narrower
    concat hidden (no behavior-type slot): Conv / Orbit / calendar / Qwen /
    BSTS Linears copy over; ``encoder``, ``regression_layer`` and
    ``behav_proto`` stay randomly initialized when shapes differ.
    """
    if not os.path.isabs(path):
        path = os.path.join(PROJECT_ROOT, path)
    print(f'{log_prefix} loading {path}', flush=True)
    ckpt = torch.load(path, map_location='cpu')
    if not isinstance(ckpt, dict):
        raise RuntimeError(f'{log_prefix} expected a dict checkpoint, got {type(ckpt)}')
    sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    if not isinstance(sd, dict):
        raise RuntimeError(f'{log_prefix} missing state_dict')
    model_sd = model.state_dict()
    compatible = {}
    skipped = []
    for k, v in sd.items():
        if k not in model_sd:
            skipped.append((k, 'not_in_model'))
            continue
        if tuple(v.shape) != tuple(model_sd[k].shape):
            skipped.append((k, f'shape {tuple(v.shape)}!={tuple(model_sd[k].shape)}'))
            continue
        compatible[k] = v
    missing = [k for k in model_sd if k not in compatible]
    model.load_state_dict(compatible, strict=False)
    print(
        f'{log_prefix} loaded {len(compatible)}/{len(model_sd)} tensors; skipped={len(skipped)} randomly_init={len(missing)}',
        flush=True,
    )
    if skipped:
        print(f'{log_prefix} skipped (first 12): {skipped[:12]}', flush=True)
    if missing:
        print(f'{log_prefix} randomly-init (first 12): {missing[:12]}', flush=True)
    return ckpt.get('stage1_val', None)


