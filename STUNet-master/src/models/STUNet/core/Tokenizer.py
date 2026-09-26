from dataclasses import dataclass
import torch
import torch.nn as nn
from typing import Optional

from .temporal_model import TrafficNodeEncoder
from .graph_gen import GraphGeneratorAttn
from .graph_encoder import GraphEncoderModel
from .utils import Patcher, pad_matrix

@dataclass
class TokenizerOutput:
    tokens: Optional[torch.FloatTensor] = None
    node_patch_num: Optional[int] = None
    graph_patch_num: Optional[int] = None
    adj_matrix: Optional[torch.FloatTensor] = None
    

class Tokenizer(nn.Module):
    
    def __init__(self, **kwargs):
        super(Tokenizer, self).__init__()

        self.patcher = Patcher(**kwargs.get("patch", {}))
        self.node_encoder = TrafficNodeEncoder(**kwargs.get("temporal", {}))
        self.node_encoder.load_encoder()
        self.d_model = kwargs.get('temporal', {}).get('d_model', 96)
        self.use_adj = kwargs.get('use_adj', False)
        if self.use_adj is False:
            self.graph_generator = GraphGeneratorAttn(**kwargs.get("graph_generator", {}))
        self.graph_encoder = GraphEncoderModel(**kwargs.get("graph_encoder", {}))
        self.graph_encoder.load_encoder()
        for param in self.graph_encoder.parameters():
            param.requires_grad = False
        self.patch_dim = self.graph_encoder.patch_dim

    def forward(self, batch):
        B, T, C, N = batch["x"].shape
        NP = 1
        node_tokens = self.node_encoder(batch["x"]).tokens # [B, N, num_patches, D]
        if self.use_adj:
            adj_matrix = batch.get("adj_matrix")
        else:
            node_embeddings = node_tokens
            graph_gen_output = self.graph_generator(node_embeddings.permute(0, 2, 1))  # [B, N, N]
            adj_matrix = graph_gen_output.adj_matrix
        adj_matrix_raw = adj_matrix
        target_size = ((N + self.patch_dim - 1) // self.patch_dim) * self.patch_dim
        adj_matrix = pad_matrix(adj_matrix, target_size)
        graph_enc_out = self.graph_encoder(adj_matrix)  # [B, D]
        graph_tokens = graph_enc_out.tokens  # [B, num_patches, D]
        tokens = torch.cat([
            node_tokens.reshape(B, N*NP, self.d_model), graph_tokens], dim=1)
        return TokenizerOutput(
            tokens=tokens,
            node_patch_num=NP,
            graph_patch_num=graph_tokens.size(1),
            adj_matrix=adj_matrix_raw
        )