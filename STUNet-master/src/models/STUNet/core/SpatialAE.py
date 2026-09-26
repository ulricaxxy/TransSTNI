import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass

from .utils import pad_matrix
from .graph_encoder import GraphEncoderModel
from .loss_fn import MSELoss


@dataclass
class SpatialAEOutput:
    loss: Optional[torch.FloatTensor] = None


class SpatialAE(nn.Module):
    def __init__(self, **kwargs):
        super(SpatialAE,self).__init__()
        self.encoder = GraphEncoderModel(**kwargs.get("encoder", {}))
        self.d_model = self.encoder.d_model
        self.graph_patch_len = self.encoder.patch_dim
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.graph_patch_len * self.graph_patch_len)
        )

    def forward(self, batch):
        '''
        Using ViT-like transformer to encode graph structure
        adj: [B, N, N]
        '''
        B, N, _ = batch["adj_matrix"].size()
        num_patches_per_side = int((N + self.graph_patch_len - 1) / self.graph_patch_len)
        num_per_graph = num_patches_per_side * num_patches_per_side
        padded_N = num_patches_per_side * self.graph_patch_len
        graph_token = self.encoder(batch["adj_matrix"]).tokens  # [B, num_per_graph, D]
        graph_recon = self.decoder(graph_token).reshape(B, num_per_graph, self.graph_patch_len, self.graph_patch_len)  # [B, num_per_graph, graph_patch_len*graph_patch_len]
        rows = []
        for i in range(num_patches_per_side):
            row_patches = graph_recon[:, i * num_patches_per_side:(i + 1) * num_patches_per_side, :, :]  # [B, num_patches_per_side, patch_dim, patch_dim]
            row = torch.cat([row_patches[:, j, :, :] for j in range(row_patches.size(1))], dim=2)  # [B, patch_dim, N]
            rows.append(row)
        graph_recon = torch.cat(rows, dim=1)  # [B, N, N]
        adj_matrix = batch["adj_matrix"]
        if adj_matrix.size(1) < padded_N:
            adj_matrix = pad_matrix(adj_matrix, padded_N)
        loss = MSELoss()(graph_recon, adj_matrix)
        return SpatialAEOutput(
            loss=loss
        )
        
    def save_encoder(self):
        self.encoder.save_encoder()