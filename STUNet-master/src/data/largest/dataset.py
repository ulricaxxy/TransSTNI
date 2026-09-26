from torch.utils.data import BatchSampler
import numpy as np
import torch
import random
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import NeighborSampler
from torch_geometric.utils import subgraph, to_undirected


class SubGraphSampler(object):
    def __init__(self, edge_index, edge_attr, num_nodes, sample_sizes):
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.sample_sizes = sample_sizes
        self.data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
        self._sampler = NeighborSampler(to_undirected(self.edge_index), sizes=self.sample_sizes)

    def sample(self, node_ids):
        # if you want to permute matrix, please uncomment the following line
        # node_ids = np.random.permutation(node_ids)
        _, n_id, _ = self._sampler.sample(node_ids)
        target_node_mask = torch.zeros(len(n_id), dtype=torch.bool)
        target_node_mask[:len(node_ids)] = True
        subgraph_edge_index, subgraph_edge_attr = subgraph(n_id, self.edge_index, self.edge_attr, relabel_nodes=True)
        
        subgraph_ = Data(edge_index=subgraph_edge_index,
                         edge_attr=subgraph_edge_attr,
                         num_nodes=len(n_id),
                         n_id=n_id,
                         target_node_mask=target_node_mask)
        return subgraph_


class Indexer(object):
    def __init__(self, num_time, num_nodes, seq_len, pred_len, step):
        self.num_time = num_time
        self.num_nodes = num_nodes
        self.window_length = seq_len + pred_len
        self.step = step
        self.indexes = self.init_indexes()

    def init_indexes(self):
        indexes = []
        T, num_nodes = self.num_time, self.num_nodes
        num_samples_per_node = (T-self.window_length+1) // self.step
        for j in range(num_samples_per_node):
            for i in range(num_nodes):
                indexes.append([i, j])
        return indexes


class CoherentNodesBatchSampler(BatchSampler):
    def __init__(self, data_source, num_nodes, batch_size):
        self.data_source = data_source
        self.num_nodes = num_nodes
        self.batch_size = batch_size
        self.per_node_sample_num = len(data_source) // self.num_nodes
        self.left_sample_num_per_time = self.num_nodes % self.batch_size

        raw_indices = np.arange(len(self.data_source))
        self.batch_indices = []
        # batch_indices: [[0, 2, 4], [2, 5, 7, 8], [4, 7, 10], [5, 8, 10, 11]]
        for i in range(self.per_node_sample_num):
            num_batch = self.num_nodes // self.batch_size
            for j in range(num_batch):
                self.batch_indices.append(raw_indices[i*self.num_nodes+j*self.batch_size:i*self.num_nodes+(j+1)*self.batch_size])
            self.batch_indices.append(raw_indices[(i+1)*self.num_nodes-self.left_sample_num_per_time:(i+1)*self.num_nodes])
        random.shuffle(self.batch_indices)
    
    def __iter__(self):
        for batch in self.batch_indices:
            yield batch.tolist()
    
    def __len__(self):
        return len(self.batch_indices)


class STDataset(Dataset):
    def __init__(self,
                 x,
                 edge_index,
                 edge_attr,
                 sample_sizes,
                 seq_len,
                 pred_len,
                 step,
                 split,
    ):
        self.x = x
        self.sampler = SubGraphSampler(edge_index=edge_index, edge_attr=edge_attr, num_nodes=x.shape[1], sample_sizes=sample_sizes)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.indexer = Indexer(x.shape[0], x.shape[1], seq_len, pred_len, step)
        self.step = step
        self.split = split
        
    def __len__(self):
        return len(self.indexer.indexes)

    def __getitem__(self, index):
        node_id, start_index = self.indexer.indexes[index]

        start_index = start_index * self.step
        
        return start_index, node_id, self.split


class IndexedDataset(Dataset):
    def __init__(self,
                 x,
                 edge_index,
                 edge_attr,
                 sample_sizes,
                 seq_len,
                 pred_len,
                 step,
                 split,
                 index
    ):
        self.x = x
        self.sampler = SubGraphSampler(edge_index=edge_index, edge_attr=edge_attr, num_nodes=x.shape[1], sample_sizes=sample_sizes)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.index = index
        self.step = step
        self.split = split
    
    def __len__(self):
        return self.index.shape[0]
    
    def __getitem__(self, idx):
        node_id = self.index[idx, 0]
        start_index = self.index[idx, 1]
        
        return start_index, node_id, self.split


class ClusterBatchSampler(BatchSampler):
    def __init__(self, data_source, num_clusters, batch_size):
        self.data_source = data_source
        self.num_clusters = num_clusters
        self.batch_size = batch_size
        self.per_node_sample_num = len(data_source) // self.num_clusters
        self.left_sample_num_per_time = self.per_node_sample_num % self.batch_size

        raw_indices = np.arange(len(self.data_source))
        self.batch_indices = []
        for i in range(self.num_clusters):
            num_batch = self.per_node_sample_num // self.batch_size
            for j in range(num_batch):
                self.batch_indices.append(raw_indices[i*self.per_node_sample_num+j*self.batch_size:i*self.per_node_sample_num+(j+1)*self.batch_size])
        if self.left_sample_num_per_time > 0:
            for i in range(self.num_clusters):
                self.batch_indices.append(raw_indices[(i+1)*self.per_node_sample_num-self.left_sample_num_per_time:(i+1)*self.per_node_sample_num])
        random.shuffle(self.batch_indices)
    
    def __iter__(self):
        for batch in self.batch_indices:
            yield batch.tolist()
    
    def __len__(self):
        return len(self.batch_indices)


class ClusterDataset(Dataset):
    def __init__(self,
                 x,
                 edge_index,
                 edge_attr,
                 seq_len,
                 pred_len,
                 step,
                 split,
                 num_clusters,
                 labels,
                 
    ):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.step = step
        self.split = split
        self.num_clusters = num_clusters
        self.labels = labels
        self.num_samples_per_node = (x.shape[0]-seq_len-pred_len+1) // step
        self.sampler = NeighborSampler(to_undirected(self.edge_index), sizes=[-1])
    
    def __len__(self):
        return self.num_clusters * self.num_samples_per_node
    
    def __getitem__(self, idx):
        start_index = (idx[0] % self.num_samples_per_node) * self.step
        cluster = idx[0] // self.num_samples_per_node
        node_id = np.where(self.labels == cluster)[0]
        np.random.shuffle(node_id)
        node_id = torch.Tensor(node_id).long()
        _, n_id, _ = self.sampler.sample(node_id)
        target_node_mask = torch.zeros(len(n_id), dtype=torch.bool)
        target_node_mask[:len(node_id)] = True
        subgraph_edge_index, subgraph_edge_attr = subgraph(n_id, self.edge_index, self.edge_attr, relabel_nodes=True)
        sample_num = len(idx)
        
        return start_index, self.split, n_id, subgraph_edge_index, subgraph_edge_attr, sample_num, target_node_mask


if __name__ == '__main__':
    ...