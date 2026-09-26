from dataclasses import dataclass
import torch
import torch.nn as nn
from typing import Optional

@dataclass
class ForcastHeadOutput:
    y_pred: Optional[torch.FloatTensor] = None

class ForcastHead(nn.Module):
    def __init__(self, **kwargs):
        super(ForcastHead, self).__init__()
        self.d_model = kwargs.get("d_model", 96)
        self.node_patch_len = kwargs.get("node_patch_len", 10)
        self.stride = kwargs.get("stride", 10)
        self.seq_len = kwargs.get("seq_len", 24)
        self.pred_len = kwargs.get("pred_len", 12)
        self.num_per_node = (self.seq_len - self.node_patch_len) // self.stride + 1
        self.forecaster = nn.Sequential(
            nn.Linear(self.d_model*self.num_per_node, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.pred_len)
        )
        
    def forward(self, tokens, num_per_node, num_per_graph):
        B, L, _ = tokens.size()
        N = int((L - num_per_graph) / num_per_node)  # Number of node
        node_tokens = tokens[:, :-num_per_graph, :].reshape(B, N, num_per_node, self.d_model)  # [B, N, num_per_node, d_model]
        node_tokens = node_tokens.reshape(B, N, -1)  # [B, N, num_per_node*d_model]
        y_pred = self.forecaster(node_tokens)  # [B, N, pred_len]
        y_pred = y_pred.permute(0, 2, 1)  # [B, pred_len, N]
        return ForcastHeadOutput(
            y_pred=y_pred,
        )
        