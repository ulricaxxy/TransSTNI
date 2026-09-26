from dataclasses import dataclass
import torch
import torch.nn as nn
from typing import Optional

@dataclass
class TemporalModelOutput:
    tokens: Optional[torch.FloatTensor] = None


class TrafficNodeEncoder(nn.Module):
    def __init__(self, **kwargs):
        super(TrafficNodeEncoder, self).__init__()
        self.patch_len = kwargs.get("patch_len", 16)
        self.d_model = kwargs.get("d_model", 96)
        self.encoder = nn.Sequential(
            nn.Linear(self.patch_len, self.d_model//2),
            nn.ReLU(),
            nn.Linear(self.d_model//2, self.d_model//2)
        )
        self.todencoder = nn.Sequential(
            nn.Linear(self.patch_len, self.d_model//4),
            nn.ReLU(),
            nn.Linear(self.d_model//4, self.d_model//4)
        )
        self.dowencoder = nn.Sequential(
            nn.Linear(self.patch_len, self.d_model//4),
            nn.ReLU(),
            nn.Linear(self.d_model//4, self.d_model//4)
        )
        self.ln = nn.LayerNorm(self.d_model)
        self.save_path = kwargs.get("save_path", None)
    
    def forward(self, x):
        '''
        x: [B, T, C, N]
        '''
        x_emb = self.encoder(x[:, :, 0, :].permute(0, 2, 1))  # [B, N, D/2]
        tod_emb = self.todencoder(x[:, :, 1, :].permute(0, 2, 1))  # [B, N, D/4]
        dow_emb = self.dowencoder(x[:, :, 2, :].permute(0, 2, 1))  # [B, N, D/4]
        x_emb = torch.cat([x_emb, tod_emb, dow_emb], dim=-1)  # [B, N, D]
        return TemporalModelOutput(
            tokens = x_emb # [B, N, D]
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
        if self.save_path is not None:
            self.load_state_dict(torch.load(self.save_path))