import os
os.environ['WANDB_MODE'] = "offline"

import torch
from lightning.pytorch import LightningModule, Trainer, seed_everything
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only

from torch.nn import functional as F
import torch.nn as nn
import torch.optim as optim

# Seed everything for reproducibility
seed_everything(42)

# Dummy Dataset
class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.data = torch.randn(length, size)
        self.labels = torch.randint(0, 2, (length,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

# Simple Lightning Module
class SimpleModel(LightningModule):
    def __init__(self, input_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.layer = nn.Linear(input_dim, output_dim)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = self.loss_fn(output, target)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)

# Initialize dataset and dataloaders
input_dim = 10
output_dim = 2
dataset = RandomDataset(size=input_dim, length=1000)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
wandb_logger = WandbLogger(project="gpu-test",offline=True)
# Initialize model
model = SimpleModel(input_dim=input_dim, output_dim=output_dim)

if rank_zero_only.rank == 0: # workaround for multi-gpu wandb logging        
    wandb_logger.watch(model, log='all', log_freq=5000)

# Trainer with DDP2 strategy
trainer = Trainer(
    accelerator="gpu",
    devices=10, 
    strategy="ddp",
    logger=wandb_logger, 
    # devices=2,  # Adjust to the number of GPUs on your machine
)

# Train the model
trainer.fit(model, train_loader)
