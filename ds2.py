import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchmetrics.functional as plm
import os
from torchmetrics.functional.classification import multiclass_accuracy
from pytorch_lightning.utilities import rank_zero_only
from datetime import datetime

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))

        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, num_blocks, in_channels, channels, classes):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=3, stride=1)

        self.blocks = nn.ModuleList()

        for _ in range(num_blocks):
            self.blocks.append(ResidualBlock(channels, channels, 1))

        self.conv_out = nn.Conv2d(in_channels=channels, out_channels=classes, kernel_size=1, stride=1, bias=False)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        for layer in self.blocks:
            x = self.relu(layer(x))
        return self.softmax(self.avg_pool(self.conv_out(x)).squeeze(-1).squeeze(-1))

def rank_zero_check():
    #if it is not on slurm (maxwell/cirrus) and not
    if not int(os.environ.get('SLURM_PROCID', 0)) > 0 and not int(os.environ.get('LOCAL_RANK', 0)) > 0:
        return True
    else:
        return False

class RN(pl.LightningModule):
    def __init__(self, in_channels, channels, classes, num_blocks):
        super(RN, self).__init__()

        self.ml_model = ResNet(num_blocks=num_blocks, in_channels=in_channels, channels=channels, classes=classes)

    def shared_step(self, batch):
        x, y = batch

        y_hat = self.ml_model(x)

        acc = multiclass_accuracy(y_hat, y, num_classes=10)
        loss = F.cross_entropy(y_hat, y)

        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)

        self.log_dict({'train_acc': acc, 'train_loss': loss},
                prog_bar=True, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)
        self.log_dict({'validation_acc': acc, 'validation_loss': loss},
                prog_bar=True, on_epoch=True, sync_dist=True)

        return

    def test_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)
        self.log_dict({'test_acc': acc, 'test_loss': loss},
                prog_bar=True, on_epoch=True, sync_dist=True)


    def configure_optimizers(self):
        lr = 0.003
        params = self.trainer.model.parameters()

        optimizer = Adam(params, lr=lr)

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)

        return [optimizer], [scheduler]

from torchvision.datasets import MNIST

from torch.utils.data import TensorDataset, DataLoader, random_split

class TimeEpochCallback(pl.Callback):
    def __init__(self):
        self.train_time = datetime.now()
        self.val_time = datetime.now()
        self.test_tune = datetime.now()

    @rank_zero_only
    def on_train_epoch_start(self, trainer, pl_module):
        self.train_time = datetime.now()
        return

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        seconds = (datetime.now() - self.train_time).total_seconds()
        print(f"epoch number: {pl_module.current_epoch} took {seconds}s")
        print(f"Max memory allocated: {torch.cuda.max_memory_allocated()}")
#         pl_module.logger.experiment["train/time_per_epoch"].append(seconds)
        return

class MNISTDataModule(LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size=64, num_workers=8, pin_memory=True, persistent_workers=True):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        self.batch_size=batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.dims = (1, 28, 28)
        self.num_classes = 10

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)



mnist = MNISTDataModule(os.getcwd(), batch_size=32, num_workers=2, pin_memory=True, persistent_workers=True)
model = RN(in_channels=1, channels=64, classes=10, num_blocks=152) #152 is too much for macleod

from pytorch_lightning.strategies import DDPFullyShardedNativeStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

from functools import partial

my_auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=1e5)

my_strategy = DDPFullyShardedNativeStrategy(cpu_offload=CPUOffload(offload_params=True), auto_wrap_policy=my_auto_wrap_policy)

from neptune.new.integrations.pytorch_lightning import NeptuneLogger

print(f'{os.environ["NEPTUNE_PROJECT"]}, {os.environ["NEPTUNE_API_TOKEN"]}')

neptune_logger = NeptuneLogger(
    mode="async",
    api_key=None,
    project=os.environ["NEPTUNE_PROJECT"],
    name="Testing",  # Optional,
    tags=["Test"],  # Optional,
    log_model_checkpoints=False
)

trainer = pl.Trainer(max_epochs=5, fast_dev_run=False, default_root_dir=os.getcwd(), strategy="deepspeed_stage_2", accelerator="gpu", devices=1, log_every_n_steps=25, callbacks=[TimeEpochCallback()], logger=neptune_logger, precision=32, enable_progress_bar=False) #gpus=1

print("PRINTING GPU INFO")

# print(torch.cuda.get_device_properties(0).total_memory)
# print(f"GPUs: {torch.cuda.device_count()}")
# print(torch.cuda.mem_get_info())

print("GPU INFO DONE")


trainer.fit(model, mnist)
