import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

@dataclass
class GraphGenOutput:
    graph_embedding: Optional[torch.FloatTensor] = None
    adj_matrix: torch.FloatTensor = None
    reconstruction: Optional[torch.FloatTensor] = None


class GraphGeneratorAttn(nn.Module):
    def __init__(self, **kwargs):
        super(GraphGeneratorAttn, self).__init__()
        self.d_model = kwargs.get("d_model", 96)
        self.output_dim = kwargs.get("output_dim", 96)
        self.num_layers = kwargs.get("num_layers", 2)
        self.num_heads = kwargs.get("num_heads", 4)
        self.dropout = kwargs.get("dropout", 0.1)
        self.attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=self.d_model, num_heads=self.num_heads, dropout=self.dropout, batch_first=True) for _ in range(self.num_layers)
        ])
        self.projection = nn.Linear(self.d_model, self.output_dim)

    def forward(self, x):
        """
        x: [B, D, N], batch size, timestamps, channel numbers (nodes)
        generate adjacency matrix using attention mechanism on node features
        """

        attn_weights = []
        for layer in self.attn:
            attn_output, attn_weight = layer(x, x, x)  # attn_weights: [B, N, N]
            attn_weights.append(attn_weight)

        attn_weights = torch.stack(attn_weights, dim=0).mean(dim=0)  # [B, N, N]
        attn_output = attn_output.permute(0, 2, 1)  # [B, N, D]
        attn_output = self.projection(attn_output)  # [B, N, T]
        attn_output = attn_output.permute(0, 2, 1)  # [B, T, N]
 
        return GraphGenOutput(
            adj_matrix=attn_weights,
            reconstruction=attn_output
        )