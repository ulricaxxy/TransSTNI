import os

from torch.utils.data import DataLoader, default_collate
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from typing import Any, Dict, Optional
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from .dataset import ClusterDataset, ClusterBatchSampler
from .utils import edge_attr2matrix


NODE_NUM = {
    'CA': 8600,
    'GLA': 3834,
    'GBA': 2352,
    'SD': 716,
    'SD-clean': 712,
    'CA-clean': 8554
}


EDGE_NUM = {
    'CA': 209963,
    'GLA': 102537,
    'GBA': 63598,
    'SD': 18035,
    'SD-clean': 17907,
    'CA-clean': 207926
}


class DataModule(LightningDataModule):
    
    def __init__(
        self,
        data_dir,
        dataset_name,
        seq_len,
        pred_len,
        batch_size,
        num_workers,
        num_clusters,
        step,
        tod,
        dow
    ) -> None:
        
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.batch_size_per_device = batch_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_name = dataset_name
        self.step = step
        
        fp = f'{data_dir}/{dataset_name}/'
        num_nodes = NODE_NUM[dataset_name]
        num_step = 35040
        train_steps = int(0.6 * num_step)
        test_steps = int(0.2 * num_step)
        val_steps = num_step - train_steps - test_steps
        
        Traffic = np.memmap(os.path.join(fp, f'full{dataset_name.lower()}.dat'), mode='r', shape=(num_step, num_nodes, 1), dtype=np.float64)
        TE = np.zeros([num_step, 2])
        TE[:,0] = np.array([i % tod for i in range(num_step)])
        TE[:,1] = np.array([(i // tod) % dow for i in range(num_step)])
        TE_tile = np.repeat(np.expand_dims(TE, 1), Traffic.shape[1], 1)
        x = np.concatenate([Traffic, TE_tile], axis=-1)
        edge_index = torch.from_numpy(np.memmap(os.path.join(fp, 'edge_index.dat'), mode='r', shape=(2, EDGE_NUM[dataset_name]), dtype=np.int64)).long()
        edge_attr = torch.from_numpy(np.memmap(os.path.join(fp, 'edge_attr.dat'), mode='r', shape=(EDGE_NUM[dataset_name]), dtype=np.float64)).float()
        
        self.mean = np.mean(Traffic[:train_steps, :, 0])
        self.std = np.std(Traffic[:train_steps, :, 0])
        print(self.mean)
        print(self.std)
        
        meta = pd.read_csv(os.path.join(fp, f'{dataset_name.lower()}_meta.csv'))
        coords = meta[['Lat', 'Lng']].values
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(coords)
        labels = kmeans.labels_

        self.train_dataset = ClusterDataset(x[:train_steps, :, :], edge_index, edge_attr, seq_len, pred_len, step, 'train', num_clusters, labels)
        self.val_dataset = ClusterDataset(x[train_steps:train_steps+val_steps, :, :], edge_index, edge_attr, seq_len, pred_len, step, 'val', num_clusters, labels)
        self.test_dataset = ClusterDataset(x[train_steps+val_steps:, :, :], edge_index, edge_attr, seq_len, pred_len, step, 'test', num_clusters, labels)
    
    def prepare_data(self) -> None:
        ...
        
    def setup(self, stage: Optional[str] = None) -> None:
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size
    
    def train_dataloader(self):
        sampler = ClusterBatchSampler(self.train_dataset, self.hparams.num_clusters, self.batch_size)
        return DataLoader(self.train_dataset, collate_fn=self.collate_fn, num_workers=self.num_workers, sampler=sampler)

    def val_dataloader(self):
        sampler = ClusterBatchSampler(self.val_dataset, self.hparams.num_clusters, self.batch_size)
        return DataLoader(self.val_dataset, collate_fn=self.collate_fn, num_workers=self.num_workers, sampler=sampler)

    def test_dataloader(self):
        sampler = ClusterBatchSampler(self.test_dataset, self.hparams.num_clusters, self.batch_size)
        return DataLoader(self.test_dataset, collate_fn=self.collate_fn, num_workers=self.num_workers, sampler=sampler)
    
    def collate_fn(self, batch):
        start_index, split, n_ids, edge_index, edge_attr, sample_num, target_node_mask = default_collate(batch)
        start_index = start_index[0]
        if n_ids.dim() > 1:
            n_ids = n_ids[0]
        edge_index = edge_index[0]
        edge_attr = edge_attr[0]
        sample_num = sample_num[0]
        target_node_mask = target_node_mask[0] # [N]
        if split[0] == 'train':
            x = self.train_dataset.x
            seq_len = self.train_dataset.seq_len
            pred_len = self.train_dataset.pred_len
        elif split[0] == 'val':
            x = self.val_dataset.x
            seq_len = self.val_dataset.seq_len
            pred_len = self.val_dataset.pred_len
        elif split[0] == 'test':
            x = self.test_dataset.x
            seq_len = self.test_dataset.seq_len
            pred_len = self.test_dataset.pred_len
        
        window_values = x[start_index:start_index+seq_len+pred_len+self.step*(sample_num-1), n_ids, :] # [T, N, C]
        
        base_idx = np.arange(sample_num) * self.step  # [B]
        past_time_idx = base_idx[:, None] + np.arange(seq_len)[None, :]
        future_time_idx = base_idx[:, None] + seq_len + np.arange(pred_len)[None, :]
        past_values = window_values[past_time_idx, :, :]      # [B, seq_len, N, C]
        future_values = window_values[future_time_idx, :, 0]  # [B, pred_len, N]
        
        x = past_values.transpose(0, 1, 3, 2) # [B, T, C, N]
        x[:, :, 0, :] = (x[:, :, 0, :] - self.mean) / self.std
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(future_values).float()
        
        adj_matrix = edge_attr2matrix(len(n_ids), edge_index, edge_attr)
        adj_matrix = adj_matrix.expand(sample_num, -1, -1)

        return {
            "x": x,
            "adj_matrix": adj_matrix,
            "y": y,
            "target_node_mask": target_node_mask,
            "mean": torch.tensor(self.mean).float(),
            "std": torch.tensor(self.std).float()
        }
    
    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    ...