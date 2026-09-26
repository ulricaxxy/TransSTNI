import torch
import torch.nn as nn
import math

def revin_norm(x, u=None, s=None):
    """
    RevIN norm
    x: [B, T, N]
    """
    x = x.permute(0, 2, 1) # [B, N, T]
    if u == None and s == None:
        u = torch.mean(x, dim=2, keepdim=True) # [B, N, 1]
        s = torch.std(x, dim=2, keepdim=True) # [B, N, 1]

    x_norm = (x - u) / (s + 1e-5)
    x_norm = x_norm.permute(0, 2, 1) # [B, T, N]

    return x_norm, u, s

def revin_denorm(x_norm, u, s):
    """
    RevIN denorm
    x_norm: [B, T, N]
    u: mean, [B, N, 1]
    s: std var [B, N, 1]
    """
    x_norm = x_norm.permute(0, 2, 1) # [B, N, T]
    x = x_norm * (s + 1e-5) + u
    x = x.permute(0, 2, 1) # [B, T, N]
    return x

def generate_permutation_matrix(adj):
    """
    Generate a random permutation matrix for each sample in the batch
    """
    B, N, _ = adj.shape
    device = adj.device
    P = torch.zeros((B, N, N), device=device)
    for i in range(B):
        perm = torch.randperm(N, device=device)
        P[i] = torch.eye(N, device=device)[perm]
    adj_permuted = torch.bmm(torch.bmm(P, adj), P.transpose(1, 2)) # P*adj*P^T
    return adj_permuted

def pad_matrix(adj, size):
    """
    Pad adjacency matrix to the desired size with zeros
    """
    B, N, _ = adj.shape
    if N > size:
        raise ValueError("The size to pad must be larger than the current size.")
    padded_adj = torch.zeros((B, size, size), device=adj.device)
    padded_adj[:, :N, :N] = adj
    return padded_adj

class Patcher(nn.Module):
    """
    Class for patching time series
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.patch_len = kwargs.get("patch_len", 16)
        self.stride = kwargs.get("stride", 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, N]
        return: [B, N, num_patches, patch_len]
        """
        B, T, N = x.size()
        x = x.permute(0, 2, 1)  # [B, N, T]
        if (T - self.patch_len) % self.stride != 0:
            pad_len = self.stride - (T - self.patch_len) % self.stride
            pad_values = x[:, :, -1:].expand(B, N, pad_len)
            x = torch.cat([x, pad_values], dim=2)  # [B, N, T + pad_len]
            T = T + pad_len
        patches= x.unfold(dimension=2, size=self.patch_len, step=self.stride)  # [B, N, num_patches, patch_len]
        return patches
    

def get_2d_sincos_pe(n1, n2, dim, device):
    """
    n1: node
    n2: patch
    dim: embedding dim
    return: [n1, n2, dim]
    """
    assert dim % 4 == 0
    dim_each = dim // 2

    pe1 = get_1d_sincos_pe(n1, dim_each, device)  # [n1, dim/2]
    pe2 = get_1d_sincos_pe(n2, dim_each, device)  # [n2, dim/2]

    pe = torch.zeros(n1, n2, dim, device=device)
    pe[:, :, :dim_each] = pe1[:, None, :]
    pe[:, :, dim_each:] = pe2[None, :, :]
    return pe


def get_1d_sincos_pe(length, dim, device):
    pos = torch.arange(length, device=device).float()
    div = torch.exp(
        torch.arange(0, dim, 2, device=device).float()
        * (-math.log(10000.0) / dim)
    )
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(pos[:, None] * div)
    pe[:, 1::2] = torch.cos(pos[:, None] * div)
    return pe


def build_rope(seq_len, dim, device):
    assert dim % 2 == 0
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device) / dim))
    pos = torch.arange(seq_len, device=device)
    freqs = torch.einsum("i,j->ij", pos, inv_freq)  # [T, dim/2]

    sin = freqs.sin()
    cos = freqs.cos()
    return sin, cos


def apply_rope(x, sin, cos):
    # x: [B, H, T, D]
    B, H, T, D = x.shape
    x = x.view(B, H, T, D//2, 2)

    x1 = x[..., 0]
    x2 = x[..., 1]

    sin = sin[None, None, :, :]   # [1,1,T,D/2]
    cos = cos[None, None, :, :]

    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos

    return torch.cat([y1, y2], dim=-1)
