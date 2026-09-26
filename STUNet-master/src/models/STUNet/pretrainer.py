from typing import Any, Dict
import torch
import lightning as pl
import torch
import numpy as np
from torchmetrics import MeanMetric, MinMetric
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

def plot_figures(y_pred, y_true, writer, current_step):
    '''
    y_true, y_pred: [B, T, N]
    '''
    y_pred = y_pred.transpose(0, 2, 1)  # [B, N, T]
    y_true = y_true.transpose(0, 2, 1)  # [B, N, T]
    B, N, T = y_pred.shape

    y_pred = y_pred.reshape(B * N, T)
    y_true = y_true.reshape(B * N, T)

    n = list(range(y_pred.shape[0]))
    sample_size = min(6, len(n))
    sampled_n = np.random.choice(n, size=sample_size, replace=False)

    y_pred = [y_pred[i] for i in sampled_n]
    y_true = [y_true[i] for i in sampled_n]

    fig, axes = plt.subplots(1, sample_size, figsize=(20, 3))
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        axes[i].plot(true, label="true")
        axes[i].plot(pred, label="pred")
        axes[i].legend()
        axes[i].set_title(f"Sample {sampled_n[i]}")

    plt.tight_layout()

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()
    image = np.frombuffer(s, dtype=np.uint8).reshape((height, width, 4))[:, :, :3]
    
    # Convert to CHW format for TensorBoard [C, H, W]
    image = image.transpose(2, 0, 1)

    writer.add_image(f"example#{current_step}", image, global_step=current_step)

    plt.close()
    
def plot_graphs(graphs, writer, current_step):
    '''
    graphs: [B, N, N]
    '''
    n = list(range(graphs.shape[0]))
    sample_size = min(6, len(n))
    sampled_n = np.random.choice(n, size=sample_size, replace=False)

    sampled_graphs = [graphs[i] for i in sampled_n]

    fig, axes = plt.subplots(1, sample_size, figsize=(20, 3))

    if sample_size == 1:
        axes = [axes]

    for i, g in enumerate(sampled_graphs):
        ax = axes[i]
        im = ax.imshow(g, cmap='viridis')
        ax.set_title(f"Graph #{sampled_n[i]}")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()

    # Convert figure to numpy array
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()
    image = np.frombuffer(s, dtype=np.uint8).reshape((height, width, 4))[:, :, :3]

    # Convert to CHW format for TensorBoard [C, H, W]
    image = image.transpose(2, 0, 1)

    # Log to TensorBoard
    writer.add_image(f"graph#{current_step}", image, global_step=current_step)

    plt.close()

class Trainer(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        """Initialize a `Trainer`.

        :param model: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.model = model
        # loss function
        self.criterion = torch.nn.MSELoss()
        
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        
        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()

    def forward(self, batch):
        return self.model(batch)
    
    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
    
    def on_before_batch_transfer(self, batch, dataloader_idx):
        if isinstance(batch, dict):
            batch = {k:v.to(self.device) for k,v in batch.items()}
        elif isinstance(batch, list) or isinstance(batch, tuple):
            batch = {"x":batch[0].to(self.device), "y":batch[1].to(self.device)}
        return batch

    def training_step(self, batch, batch_idx):
        current_step = self.global_step
        output = self(batch)
        loss = output.loss
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        loss = output.loss
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        mse = self.val_loss.compute()  # get current val mse
        self.val_loss_best(mse)  # update best so far val mse
        # log `val_mse_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        output = self(batch)
        loss = output.loss
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=True, on_epoch=True, prog_bar=True)
        
    def on_test_end(self):
        self.model.save_encoder()
        
    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
                'gradient_clip_val': 0.99,
                'gradient_clip_algorithm': 'norm'
            }
        return {
            "optimizer": optimizer,
            'gradient_clip_val': 0.99,
            'gradient_clip_algorithm': 'norm'
            }


if __name__ == "__main__":
    ...