import torch


def edge_attr2matrix(num_nodes, edge_index, edge_attr=None):
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=edge_attr.dtype if edge_attr is not None else torch.float32)
    src, dst = edge_index
    adj_matrix[src, dst] = 1
    if edge_attr is not None:
        adj_matrix[src, dst] = edge_attr
    return adj_matrix