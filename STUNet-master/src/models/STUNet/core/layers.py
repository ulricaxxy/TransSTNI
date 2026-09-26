import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter

from .utils import build_rope, apply_rope
    

class FeedForwardLayer(nn.Module):
    '''
    Feed Forward layer
    input: [B, T, N]
    output: [B, T, N]
    '''
    def __init__(self, d_model, mlp_dim, dropout=0.0):
        super(FeedForwardLayer, self).__init__()
        self.d_model = d_model
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.net = nn.Sequential(
            nn.Linear(self.d_model, self.mlp_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_dim, self.d_model),
        )
    def forward(self, x):
        """
        x: [B, T, N]
        """
        return self.net(x)


class QAAttentionLayer(nn.Module):
    '''
    query-aggregate attention
    '''
    def __init__(self, d_model, num_heads=4, mlp_dim=256):
        super(QAAttentionLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_head = d_model // num_heads
        assert self.d_head % 4 == 0, "d_head must be even for RoPE"
        self.W_qq = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.W_kq = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.W_vq = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.W_qa = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.W_ka = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.W_va = nn.Parameter(torch.FloatTensor(d_model, d_model))
        nn.init.xavier_uniform_(self.W_qq)
        nn.init.xavier_uniform_(self.W_kq)
        nn.init.xavier_uniform_(self.W_vq)
        nn.init.xavier_uniform_(self.W_qa)
        nn.init.xavier_uniform_(self.W_ka)
        nn.init.xavier_uniform_(self.W_va)
        self.query_ffn = FeedForwardLayer(d_model, mlp_dim)
        self.aggregate_ffn = FeedForwardLayer(d_model, mlp_dim)
        self.ln = nn.LayerNorm(d_model)
        
    def split_heads(self, x):
        B, T, _ = x.shape
        x = x.view(B, T, self.num_heads, self.d_head)
        return x.permute(0, 2, 1, 3)  # [B, num_heads, T, d_head]
    
    def forward(self, node_tokens, graph_tokens, N, N_T, N_S):
        '''
        node_tokens: [B, L1, D], L1 = N * N_T
        graph_tokens: [B, L2, D], L2 = N_S * N_S
        '''
        # positional encoding
        # query attention PE
        sinQ_q, cosQ_q = build_rope(N, self.d_head // 2, node_tokens.device)  # [N, d_head / 4]
        sinQ_q = torch.cat([sinQ_q, sinQ_q], dim=-1) # [N, d_head / 2]
        cosQ_q = torch.cat([cosQ_q, cosQ_q], dim=-1) # [N, d_head / 2]
        sinQ_q = sinQ_q.unsqueeze(1).expand(-1, N_T, -1).contiguous().view(N*N_T, self.d_head // 2)  # [N*N_T, d_head / 2]
        cosQ_q = cosQ_q.unsqueeze(1).expand(-1, N_T, -1).contiguous().view(N*N_T, self.d_head // 2)  # [N*N_T, d_head / 2]
        sinK_q, cosK_q = build_rope(N_S, self.d_head // 2, graph_tokens.device)  # [N_S, d_head / 4]
        sinK_q_row = sinK_q.unsqueeze(1).expand(-1, N_S, -1)  # [N_S, N_S, d_head / 4]
        sinK_q_col = sinK_q.unsqueeze(0).expand(N_S, -1, -1)  # [N_S, N_S, d_head / 4]
        sinK_q = torch.cat([sinK_q_row, sinK_q_col], dim=-1).view(N_S*N_S, self.d_head // 2)  # [N_S*N_S, d_head / 2]
        cosK_q_row = cosK_q.unsqueeze(1).expand(-1, N_S, -1)  # [N_S, N_S, d_head / 4]
        cosK_q_col = cosK_q.unsqueeze(0).expand(N_S, -1, -1)  # [N_S, N_S, d_head / 4]
        cosK_q = torch.cat([cosK_q_row, cosK_q_col], dim=-1).view(N_S*N_S, self.d_head // 2)  # [N_S*N_S, d_head / 2]
        # aggregate attention PE
        sin_a_N, cos_a_N = build_rope(N, self.d_head // 2, node_tokens.device)  # [N, d_head / 4]
        sin_a_N = sin_a_N.unsqueeze(1).expand(-1, N_T, -1).contiguous().view(N*N_T, self.d_head // 4)  # [N*N_T, d_head / 4]
        cos_a_N = cos_a_N.unsqueeze(1).expand(-1, N_T, -1).contiguous().view(N*N_T, self.d_head // 4)  # [N*N_T, d_head / 4]
        sin_a_T, cos_a_T = build_rope(N_T, self.d_head // 2, node_tokens.device)  # [N_T, d_head / 4]
        sin_a_T = sin_a_T.unsqueeze(0).expand(N, -1, -1).contiguous().view(N*N_T, self.d_head // 4)  # [N*N_T, d_head / 4]
        cos_a_T = cos_a_T.unsqueeze(0).expand(N, -1, -1).contiguous().view(N*N_T, self.d_head // 4)  # [N*N_T, d_head / 4]
        sin_a = torch.cat([sin_a_N, sin_a_T], dim=-1)  # [N*N_T, d_head / 2]
        cos_a = torch.cat([cos_a_N, cos_a_T], dim=-1)  # [N*N_T, d_head / 2]
        
        # query attention
        Q_q = torch.matmul(node_tokens, self.W_qq)  # [B, L1, D]
        K_q = torch.matmul(graph_tokens, self.W_kq)  # [B, L2, D]
        V_q = torch.matmul(graph_tokens, self.W_vq)  # [B, L2, D]
        Q_q = self.split_heads(Q_q)  # [B, num_heads, L1, d_head]
        K_q = self.split_heads(K_q)  # [B, num_heads, L2, d_head]
        V_q = self.split_heads(V_q)  # [B, num_heads, L2, d_head]
        Q_q = apply_rope(Q_q, sinQ_q, cosQ_q)  # [B, num_heads, L1, d_head]
        K_q = apply_rope(K_q, sinK_q, cosK_q)  # [B, num_heads, L2, d_head]
        attn_scores_q = torch.matmul(Q_q, K_q.transpose(-2, -1)) / (self.d_head ** 0.5)  # [B, num_heads, L1, L2]
        attn_weights_q = F.softmax(attn_scores_q, dim=-1)  # [B, num_heads, L1, L2]
        query_out = torch.matmul(attn_weights_q, V_q)  # [B, num_heads, L1, d_head]
        query_out = query_out.permute(0, 2, 1, 3).contiguous()  # [B, L1, num_heads, d_head]
        query_out = query_out.view(node_tokens.shape[0], node_tokens.shape[1], self.d_model)  # [B, L1, D]
        node_tokens = node_tokens + query_out
        node_tokens = node_tokens + self.query_ffn(node_tokens)
        
        # aggregate attention
        Q_a = torch.matmul(node_tokens, self.W_qa)  # [B, L1, D]
        K_a = torch.matmul(node_tokens, self.W_ka)  # [B, L1, D]
        V_a = torch.matmul(node_tokens, self.W_va)  # [B, L1, D]
        Q_a = self.split_heads(Q_a)  # [B, num_heads, L1, d_head]
        K_a = self.split_heads(K_a)  # [B, num_heads, L1, d_head]
        V_a = self.split_heads(V_a)  # [B, num_heads, L1, d_head]
        Q_a = apply_rope(Q_a, sin_a, cos_a)  # [B, num_heads, L1, d_head]
        K_a = apply_rope(K_a, sin_a, cos_a)  # [B, num_heads, L1, d_head]
        attn_scores_a = torch.matmul(Q_a, K_a.transpose(-2, -1)) / (self.d_head ** 0.5)  # [B, num_heads, L1, L1]
        attn_weights_a = F.softmax(attn_scores_a, dim=-1)  # [B, num_heads, L1, L1]
        aggregate_out = torch.matmul(attn_weights_a, V_a)  # [B, num_heads, L1, d_head]
        aggregate_out = aggregate_out.permute(0, 2, 1, 3).contiguous()  # [B, L1, num_heads, d_head]
        aggregate_out = aggregate_out.view(node_tokens.shape[0], node_tokens.shape[1], self.d_model)  # [B, L1, D]
        node_tokens = node_tokens + aggregate_out
        node_tokens = node_tokens + self.aggregate_ffn(node_tokens)
        
        return node_tokens, graph_tokens


class CrossAttentionLayer(nn.Module):
    '''
    Cross Attention only
    '''
    def __init__(self, d_model, num_heads=4, mlp_dim=256):
        super(CrossAttentionLayer, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.cross_ffn = FeedForwardLayer(d_model, mlp_dim)
    
    def forward(self, node_tokens, graph_tokens):
        '''
        node_tokens: [B, L1, D]
        graph_tokens: [B, L2, D]
        '''
        cross_out, _ = self.cross_attn(node_tokens, graph_tokens, graph_tokens)
        node_tokens = node_tokens + cross_out
        node_tokens = node_tokens + self.cross_ffn(node_tokens)
        return node_tokens


class SelfAttentionLayer(nn.Module):
    '''
    Self Attention only
    '''
    def __init__(self, d_model, num_heads=4, mlp_dim=256):
        super(SelfAttentionLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.self_ffn = FeedForwardLayer(d_model, mlp_dim)
    
    def forward(self, node_tokens):
        '''
        node_tokens: [B, L1, D]
        '''
        self_out, _ = self.self_attn(node_tokens, node_tokens, node_tokens)
        node_tokens = node_tokens + self_out
        node_tokens = node_tokens + self.self_ffn(node_tokens)
        return node_tokens