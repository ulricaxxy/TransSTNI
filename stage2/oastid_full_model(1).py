# -*- coding: utf-8 -*-
"""OA-STID model extracted verbatim from ``stage2/oastid_full.py``.

Scope: ``oastid_egbeta_limnbr_k3_full`` (fusion=concat). Training/eval loops are
not included. Model bodies (SE / BETAET / OASTIDTransfer / orbit swap) are
copied without simplification; this package runtime uses ``--orbit_gnn sage``
via ``_maybe_swap_orbit_gnn`` (GraphSAGE limnbr k=3).
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lib.bsts_stats import BSTSEncoder


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
        use_orbit = True,
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
        self.use.orbit = bool(use_orbit)
        self.time_of_day_size = int(time_of_day_size)
        self.day_of_week_size = int(day_of_week_size)
        self.time_series_emb_layer = nn.Conv2d(
            self.input_dim * self.input_len, self.embed_dim, kernel_size=(1, 1), bias=True)
        self.beta_net = BETAET(
            self.node_dim, self.node_pe_dim, hidden_dim=int(orbit_hidden))
        self.ln_beta = nn.LayerNorm(self.node_dim)
        self.hidden_dim = self.embed_dim 
        if self.use_orbit:
             self.hidden_dim = self.hidden_dim + self.node_dim
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
        B, _, N, _ = x.shape
        x_flat = (
            x.transpose(1, 2)
            .contiguous()
            .view(B, N, -1)
            .transpose(1, 2)
            .unsqueeze(-1)
        )
        ts_emb = self.time_series_emb_layer(x_flat)
        feats = [ts_emb]
        if self.use_orbit:
            z = self.ln_beta(self.beta_net(node_feat, edge_index, edge_weight))
            z_emb = z.unsqueeze(0).expand(B, -1, -1).transpose(1, 2).unsqueeze(-1)
            feats = feats.append(z_emb)
        if self.use_time_emb:
            tid = self.time_in_day_emb[tid_idx]
            dow = self.day_in_week_emb[dow_idx]
            feats.append(tid.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, N, 1))
            feats.append(dow.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, N, 1))
        feats.extend(self._sem_feats(sem_emb, B, N))
        feats.extend(self._bsts_feats(x, edge_index, edge_weight))
        feats.extend(self._behav_feats(behav_pi, B, N))
        hidden = torch.cat(feats, dim=1)
        hidden = self.encoder(hidden)
        return self.regression_layer(hidden)


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


def build_oastid_model(args):
    """Factory: classic concat OA-STID (this package: fusion=concat only).

    Verbatim from ``oastid_full.py`` for the concat path; attn fusions are not
    part of ``oastid_egbeta_limnbr_k3_full`` and are rejected here.
    """
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
        model = OASTIDTransfer(**common)
        return _maybe_swap_orbit_gnn(model, args)
    raise ValueError(
        f'oastid_full_model (egbeta_limnbr_k3_full) only supports fusion=concat, got {fusion}')

