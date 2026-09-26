"""Orbit GraphSAGE encoder, copied from ``code/src/model/graph_sage.py``.

``GraphSAGE`` below is the original class (3× SAGEConv, train-time neighbor
sampling, eval-time full edges, optional LayerNorm / dropout). Do not collapse
the three convolutions, do not drop sampling, do not replace mean+concat SAGE
with a single Linear.

``OrbitGraphSAGE`` is only a calling-convention adapter so this module can sit
in the existing ``beta_net(node_feat, edge_index, edge_weight)`` slot (same
signature as ``BETAET``). It reconstructs the dense ``A`` that the original
``GraphSAGE.forward(A, node_features)`` expects. Cache is cleared when the
graph identity changes (Stage2 subgraphs); the sampling algorithm itself is
unchanged.
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import dense_to_sparse


class GraphSAGE(torch.nn.Module):
    """Exact copy of ``code/src/model/graph_sage.py::GraphSAGE``."""

    def __init__(self, args):
        super(GraphSAGE, self).__init__()
        self.sage_dropout = getattr(args, 'sage_dropout', 0.1)
        self.sage_norm = getattr(args, 'sage_norm', False)

        self.conv1 = SAGEConv(args.node_embed_dim, args.sage_hidden_dim)
        self.conv2 = SAGEConv(args.sage_hidden_dim, args.sage_hidden_dim)
        self.conv3 = SAGEConv(args.sage_hidden_dim, args.embed_dim)

        if self.sage_norm:
            self.norm1 = torch.nn.LayerNorm(args.sage_hidden_dim)
            self.norm2 = torch.nn.LayerNorm(args.sage_hidden_dim)

        self.args = args
        self._neighbor_idx = None
        self._full_edge_index = None

        self._rng = torch.Generator().manual_seed(args.seed)

    def _build_neighbor_cache(self, A):
        N = A.shape[0]
        edge_index, _ = dense_to_sparse(A)
        self._full_edge_index = edge_index
        self._neighbor_idx = []
        for i in range(N):
            mask = edge_index[0] == i
            self._neighbor_idx.append(edge_index[1, mask])

    def sample_neighbors(self, A, k=10):
        if self._neighbor_idx is None:
            self._build_neighbor_cache(A)

        all_src = []
        all_dst = []
        for i, nbrs in enumerate(self._neighbor_idx):
            n = nbrs.size(0)
            if n > k:
                idx = torch.randperm(n, generator=self._rng)[:k]
                chosen = nbrs[idx]
            else:
                chosen = nbrs
            all_src.append(torch.full_like(chosen, i))
            all_dst.append(chosen)

        return torch.stack([torch.cat(all_src), torch.cat(all_dst)], dim=0)

    def forward(self, A: torch.Tensor, node_features: torch.Tensor):
        if self.training:
            edge_index = self.sample_neighbors(A, k=self.args.sage_k)
        else:
            if self._full_edge_index is None:
                self._full_edge_index, _ = dense_to_sparse(A)
            edge_index = self._full_edge_index
        # edge_index, _ = dense_to_sparse(A)
        x = self.conv1(node_features, edge_index)
        x = self.norm1(x) if self.sage_norm else x
        x = F.relu(x)
        x = F.dropout(x, p=self.sage_dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.norm2(x) if self.sage_norm else x
        x = F.relu(x)
        x = F.dropout(x, p=self.sage_dropout, training=self.training)

        x = self.conv3(x, edge_index)
        return x


class OrbitGraphSAGE(torch.nn.Module):
    """Adapter: BETAET slot ``(node_feat, edge_index, edge_weight)`` → GraphSAGE(A, X).

    When ``feat_dim=0`` / ``embed_dim=0`` (nospace ablation), GraphSAGE still runs
    with internal dim-1 I/O so SAGEConv lazy params initialize; the adapter returns
    ``[N, 0]`` so no spatial signal enters fusion.
    """

    def __init__(
        self,
        embed_dim,
        feat_dim,
        hidden_dim,
        sage_k=3,
        sage_dropout=0.1,
        sage_norm=False,
        seed=10,
    ):
        super(OrbitGraphSAGE, self).__init__()
        self._out_dim = int(embed_dim)
        self._in_dim = int(feat_dim)
        self._zero_out = self._out_dim == 0
        self._use_ones_in = self._in_dim == 0
        internal_in = 1 if self._use_ones_in else self._in_dim
        internal_out = int(hidden_dim) if self._zero_out else self._out_dim
        sage_args = argparse.Namespace(
            node_embed_dim=int(internal_in),
            sage_hidden_dim=int(hidden_dim),
            embed_dim=int(internal_out),
            sage_k=int(sage_k),
            sage_dropout=float(sage_dropout),
            sage_norm=bool(sage_norm),
            seed=int(seed),
        )
        self.sage = GraphSAGE(sage_args)
        self._graph_sig = None

    def _invalidate_if_graph_changed(self, edge_index, n_nodes):
        if edge_index is None or edge_index.numel() == 0:
            sig = (int(n_nodes), 0, 0, 0)
        else:
            flat = edge_index.reshape(-1)
            head = flat[:8].to(dtype=torch.int64).sum().item()
            tail = flat[-8:].to(dtype=torch.int64).sum().item()
            sig = (
                int(n_nodes),
                int(edge_index.shape[1]),
                int(head),
                int(tail),
            )
        if self._graph_sig != sig:
            self.sage._neighbor_idx = None
            self.sage._full_edge_index = None
            self._graph_sig = sig

    def forward(self, node_features, edge_index, edge_weight):
        n_nodes = int(node_features.shape[0])
        self._invalidate_if_graph_changed(edge_index, n_nodes)
        A = node_features.new_zeros((n_nodes, n_nodes))
        if edge_index is not None and edge_index.numel() > 0:
            src = edge_index[0].long()
            dst = edge_index[1].long()
            if edge_weight is None:
                A[src, dst] = 1.0
            else:
                A[src, dst] = edge_weight.to(dtype=A.dtype)
        x_in = node_features.new_ones((n_nodes, 1)) if self._use_ones_in else node_features
        out = self.sage(A, x_in)
        if self._zero_out:
            return out.new_zeros((n_nodes, 0))
        return out
