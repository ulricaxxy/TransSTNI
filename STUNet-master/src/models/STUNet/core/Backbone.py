from dataclasses import dataclass
import torch
import torch.nn as nn
from typing import Optional
from math import sqrt
from .layers import QAAttentionLayer
from .Tokenizer import Tokenizer

@dataclass
class BackBoneOutput:
    tokens: Optional[torch.Tensor] = None
    num_per_node: Optional[int] = None
    num_per_graph: Optional[int] = None
    adj_matrix: Optional[torch.Tensor] = None
    

class BackBone(nn.Module):
    def __init__(self, **kwargs):
        super(BackBone, self).__init__()
        self.tokenizer = Tokenizer(**kwargs.get("tokenizer", {}))
        self.num_layers = kwargs.get("num_layers", 4)
        self.num_heads = kwargs.get("num_heads", 4)
        self.d_model = kwargs.get("d_model", 96)
        self.backbone = nn.ModuleList([
            QAAttentionLayer(self.d_model, self.num_heads, self.d_model * 2)
            for _ in range(self.num_layers)
        ])
        self.ln = nn.LayerNorm(self.d_model)
    
    def forward(self, batch, prefix_tokens=None):
        if "valid_N" in batch:
            valid_N = batch.get("valid_N")  # [B]
        B, T, C, N = batch["x"].shape
        token_info = self.tokenizer(batch)
        tokens = token_info.tokens  # [B, (N+1)*num_node_patches + num_graph_patches, D]
        num_node_patches = token_info.node_patch_num
        num_graph_patches = token_info.graph_patch_num
        node_tokens = tokens[:, : -num_graph_patches, :]  # [B, L, D]
        graph_tokens = tokens[:, -num_graph_patches:, :]  # [B, L, D]
        node_tokens, graph_tokens = self.backbone_forward(node_tokens, graph_tokens, N, num_node_patches, int(sqrt(num_graph_patches)))  # [B, L, D]
        if token_info.adj_matrix is not None:
            adj_matrix = token_info.adj_matrix  # [B, N, N]
        else:
            adj_matrix = None
        return BackBoneOutput(
            tokens=torch.cat([node_tokens, graph_tokens], dim=1),  # [B, L, D]
            num_per_node=num_node_patches,
            num_per_graph=num_graph_patches,
            adj_matrix=adj_matrix
        )
        
    def backbone_forward(self, node_tokens, graph_tokens, N, N_T, N_S):
        for layer in self.backbone:
            new_node_tokens, graph_tokens = layer(node_tokens, graph_tokens, N, N_T, N_S)
            node_tokens = new_node_tokens + node_tokens
        return node_tokens, graph_tokens