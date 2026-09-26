from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from typing import Optional

from .loss_fn import MAELossMask
from .GFBackbone import GFBackBone
from .head import ForcastHead


@dataclass
class STUNetOutput:
    loss: Optional[torch.FloatTensor] = None
    y_pred: Optional[torch.FloatTensor] = None
    y_true: Optional[torch.FloatTensor] = None

class STUNet(nn.Module):
    def __init__(self, **kwargs):
        super(STUNet, self).__init__()
        self.backbone = GFBackBone(**kwargs.get("backbone", {}))
        self.head = ForcastHead(**kwargs.get("forcast", {}))
        
    def forward(self, batch):
        backbone_output = self.backbone(batch)
        tokens = backbone_output.tokens  # [B, L, D]
        num_per_node = backbone_output.num_per_node
        num_per_graph = backbone_output.num_per_graph
        output = self.head(tokens, num_per_node, num_per_graph)
        y_pred = output.y_pred  # [B, pred_len, N]
        y_true = batch["y"]  # [B, pred_len, N]
        y_pred = y_pred * batch["std"] + batch["mean"]
        if batch["target_node_mask"] is not None:
            if not self.training:
                target_node_mask = batch["target_node_mask"]  # [N]
                y_pred = y_pred[:, :, target_node_mask]
                y_true = y_true[:, :, target_node_mask]
        total_loss = MAELossMask()(y_pred, y_true)
        return STUNetOutput(
            loss=total_loss,
            y_pred=y_pred,
            y_true=y_true,
        )