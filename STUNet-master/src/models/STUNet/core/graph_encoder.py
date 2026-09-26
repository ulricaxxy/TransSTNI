import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass

@dataclass
class GraphEncoderOutput:
    graph_embedding: Optional[torch.FloatTensor] = None
    tokens: Optional[torch.FloatTensor] = None

class GraphEncoderModel(nn.Module):
    def __init__(self, **kwargs):
        super(GraphEncoderModel,self).__init__()
        self.d_model = kwargs.get("d_model", 96)
        self.patch_dim = kwargs.get("patch_dim", 16)
        self.encoder = nn.Sequential(
            nn.Linear(self.patch_dim*self.patch_dim, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model)
        )
        self.ln = nn.LayerNorm(self.d_model)
        self.save_path = kwargs.get("save_path", "path-to-ckpt")

    def forward(self, adj):
        '''
        Using ViT-like transformer to encode graph structure
        adj: [B, N, N]
        '''
        B, N, _ = adj.size()
        if N % self.patch_dim != 0:
            # pad adj to make N divisible by patch_dim
            pad_size = self.patch_dim - (N % self.patch_dim)
            adj = F.pad(adj, (0, pad_size, 0, pad_size), "constant", 0)
            N = N + pad_size
        num_patches = (N // self.patch_dim) ** 2
        # [B, N, N] -> [B, num_patches, patch_dim, patch_dim]
        adj_patch = adj.view(
            B,
            N // self.patch_dim, self.patch_dim,
            N // self.patch_dim, self.patch_dim
        ).permute(0, 1, 3, 2, 4).reshape(B, num_patches, self.patch_dim, self.patch_dim)
        x = adj_patch.view(B, num_patches, -1)  # [B, num_patches, patch_dim*patch_dim]
        x = self.encoder(x)  # [B, num_patches, D]
        return GraphEncoderOutput(
            tokens =x  # [B, num_patches, D]
        )
    
    def save_encoder(self):
        if self.save_path is not None:
            cpu_state_dict = {
                k: v.detach().cpu() if torch.is_tensor(v) else v
                for k, v in self.state_dict().items()
            }
            torch.save(cpu_state_dict, self.save_path)
            print(f"Encoder saved to {self.save_path}")
    
    def load_encoder(self):
        self.load_state_dict(torch.load(self.save_path))
        print(f"Encoder loaded from {self.save_path}")