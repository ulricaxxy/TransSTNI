from dataclasses import dataclass
import torch
import torch.nn as nn
from typing import Optional

from .temporal_model import TrafficNodeEncoder
from .utils import Patcher
from .loss_fn import MAELossPatchSTG


@dataclass
class TemporalAEOutput:
    loss: Optional[torch.FloatTensor] = None


class TemporalAE(nn.Module):
    
    def __init__(self, **kwargs):
        super(TemporalAE, self).__init__()

        self.patcher = Patcher(**kwargs.get("patch", {}))
        # node-level
        self.encoder = TrafficNodeEncoder(**kwargs.get("encoder", {}))
        self.d_model = kwargs.get('encoder', {}).get('d_model', 96)
        self.seq_len = kwargs.get('encoder', {}).get('patch_len', 12)
        self.pred_len = kwargs.get('pred_len', 12)
        self.save_path = kwargs.get("save_path", None)
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.pred_len)
        )

    def forward(self, batch):
        B, T, C, N = batch["x"].shape
        NP = 1
        # node-level
        x_enc = self.encoder(batch["x"]).tokens # [B, N, num_patches, D]
        # decoder
        y_pred = self.decoder(x_enc)  # [B, N, pred_len]
        y_pred = y_pred.permute(0, 2, 1)
        y_pred = y_pred * batch["std"] + batch["mean"]

        loss = MAELossPatchSTG()(y_pred, batch["y"])

        return TemporalAEOutput(loss=loss)
    
    def save_encoder(self):
        self.encoder.save_encoder()