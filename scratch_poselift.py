"""
dissertation.scratch_poselift.py
Author: Raghhuveer Jaikanth
Date  : 16/08/2023

# Enter Description Here
"""

import glob

import numpy as np
import torch

import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard.writer import SummaryWriter

from torchmetrics.aggregation import MeanMetric


class FFNModel(nn.Module):
    def __init__(self, resume = False):
        super().__init__()

        self.flatten = nn.Flatten()

        # Layer 1 / Input
        self.linear1 = nn.Linear(34, 64)
        self.norm1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()

        # Layer 2
        self.linear2 = nn.Linear(64, 128)
        self.norm2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()

        # Layer 3
        self.linear3 = nn.Linear(128, 64)
        self.norm3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()

        # Layer 4
        # self.linear4 = nn.Linear(256, 128)
        # self.norm4 = nn.BatchNorm1d(128)
        # self.relu4 = nn.ReLU()
        #
        # # Layer 5
        # self.linear5 = nn.Linear(128, 64)
        # self.norm5 = nn.BatchNorm1d(64)
        # self.relu5 = nn.ReLU()

        # Layer 6 / Output
        self.linear6 = nn.Linear(64, 51)

        # Init weights
        if not resume:
            self._init_weights()

    def forward(self, in_):
        in_ = in_.to(torch.float32)

        # Forward
        out_ = self.flatten(in_)
        out_ = self.norm1(self.relu1(self.linear1(out_)))
        out_ = self.norm2(self.relu2(self.linear2(out_)))
        out_ = self.norm3(self.relu3(self.linear3(out_)))
        out_ = self.linear6(out_)

        return out_

    def _init_weights(self):
        nn.init.kaiming_normal_(self.linear1.weight, nonlinearity = 'relu')
        nn.init.kaiming_normal_(self.linear2.weight, nonlinearity = 'relu')
        nn.init.kaiming_normal_(self.linear3.weight, nonlinearity = 'relu')
        nn.init.kaiming_normal_(self.linear6.weight, nonlinearity = 'relu')

        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)
        nn.init.zeros_(self.linear3.bias)
        nn.init.zeros_(self.linear6.bias)

        print("Weights Initialized")


class PoseLiftDataset(Dataset):
    def __init__(self):
        self._arrays = glob.glob("/home/rjaikanth97/myspace/dissertation-final/dissertation/dataset/pose_lift/arrays/Video*.npz")
        self._frame_size = np.array([1200, 1920])

    def __getitem__(self, item):
        try:
            file_ = np.load(self._arrays[item])
            j2d = np.divide(file_['j2d'], self._frame_size)
            j3d = file_['j3d']

            return {
                '2D': j2d,
                '3D': j3d
            }
        except Exception as e:
            print(self._arrays[item])
            return None

    def __len__(self):
        return len(self._arrays)


def train_step(epoch):
    model.train()
    avg_tloss = 0
    tloss_meter = MeanMetric()
    tdl = tqdm.tqdm(train_dl, unit='batch', desc = f"Training {epoch+1}/100")

    for i, data_ in enumerate(tdl):
        inputs = data_['2D'].to(device)
        optimizer.zero_grad()
        prediction = model(inputs)

        # Loss
        target = data_['3D'].to(device)
        target = target.to(torch.float32)
        target = target
        loss = loss_fn(prediction, target)
        tloss_meter.update(loss.item())

        loss.backward()
        optimizer.step()

        avg_tloss = tloss_meter.compute()
        tdl.set_postfix(train_loss = avg_tloss)
        writer.add_scalar("loss/train", avg_tloss, epoch*len(train_dl) + i + 1)

    return avg_tloss


def val_step(epoch):
    model.eval()
    with torch.no_grad():
        avg_vloss = 0
        vloss_meter = MeanMetric()
        vdl = tqdm.tqdm(val_dl, unit = 'batch', desc = f"Validation {epoch + 1}/100")

        for i, data_ in enumerate(vdl):
            inputs = data_['2D'].to(device)
            optimizer.zero_grad()
            prediction = model(inputs)

            # Loss
            target = data_['3D'].to(device)
            target = target.to(torch.float32)
            target = target

            loss = loss_fn(prediction, target)
            vloss_meter.update(loss.item())

            avg_vloss = vloss_meter.compute()
            vdl.set_postfix(val_loss = avg_vloss)
            writer.add_scalar("loss/val", avg_vloss, epoch*len(val_dl) + i + 1)

    return avg_vloss


def safe_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, resume = False):
        super().__init__()

        # Conv2
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, 1, padding = 'same')
        self.norm1 = nn.BatchNorm1d(out_channels)

        # Conv2
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, 1, padding = 'same')
        self.norm2 = nn.BatchNorm1d(out_channels)

        # ReLU
        self.relu = nn.ReLU()

        # Addition
        self.add_ = nn.Conv1d(in_channels, out_channels, 1, 1)

        if not resume:
            self.__init_weights__()

    def forward(self, in_):
        out_ = self.conv1(in_)
        out_ = self.norm1(out_)

        out_ = self.relu(out_)

        out_ = self.conv2(out_)
        out_ = self.norm2(out_)

        out_ += self.add_(in_)

        return out_

    def __init_weights__(self):
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity = 'relu')
        nn.init.zeros_(self.conv1.bias)
        nn.init.uniform_(self.norm1.weight)
        nn.init.zeros_(self.norm1.bias)

        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity = 'relu')
        nn.init.zeros_(self.conv2.bias)
        nn.init.uniform_(self.norm2.weight)
        nn.init.zeros_(self.norm2.bias)

        print("Weights Initialized")


class CNNModel(nn.Module):
    def __init__(self, resume = False):
        super().__init__()

        # Convert to high-dim data
        self.high_dim = nn.Conv1d(2, 16, 1, 1)
        self.norm1 = nn.BatchNorm1d(16)
        self.relu1 = nn.ReLU()

        # Residual blocks
        self.res1 = ResidualBlock(16, 32, resume)
        self.res2 = ResidualBlock(32, 64, resume)
        self.res3 = ResidualBlock(64, 128, resume)
        self.res4 = ResidualBlock(128, 64, resume)
        self.res5 = ResidualBlock(64, 32, resume)
        self.res6 = ResidualBlock(32, 16, resume)

        # Input
        self.out_ = nn.Conv1d(16, 3, 1, 1)

        if not resume:
            self.__init_weights()

    def forward(self, in_):
        in_ = in_.permute((0, 2, 1))
        in_ = in_.to(torch.float32)

        out_ = self.high_dim(in_)

        out_ = self.res1(out_)
        out_ = self.res2(out_)
        out_ = self.res3(out_)
        out_ = self.res4(out_)
        out_ = self.res5(out_)
        out_ = self.res6(out_)

        out_ = self.out_(out_)

        return out_.permute(0, 2, 1)

    def __init_weights(self):
        nn.init.kaiming_normal_(self.high_dim.weight, nonlinearity = 'relu')
        nn.init.zeros_(self.high_dim.bias)
        nn.init.uniform_(self.norm1.weight)
        nn.init.zeros_(self.norm1.bias)

        nn.init.kaiming_normal_(self.out_.weight, nonlinearity = 'relu')
        nn.init.zeros_(self.out_.bias)


if __name__ == "__main__":
    device = "cpu"

    ds = PoseLiftDataset()
    train_ds, val_ds = random_split(ds, [0.8, 0.2], generator = torch.Generator().manual_seed(42))
    
    train_dl = DataLoader(train_ds, batch_size = 64, shuffle = True, num_workers = 6, collate_fn = safe_collate)
    val_dl = DataLoader(val_ds, batch_size = 128, shuffle = False, num_workers = 2, collate_fn = safe_collate)

    torch.manual_seed(42)
    # model = FFNModel(resume = False)
    model = CNNModel(resume = False)
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, weight_decay = 1e-5)
    loss_fn = nn.L1Loss()

    # Writer
    writer = SummaryWriter(comment = "CNN")
    writer.add_graph(model, torch.zeros((1, 17, 2)))

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience = 10)
    best_model = 1_000_000_000.

    for epoch in range(100):
        tloss = train_step(epoch)
        vloss = val_step(epoch)

        scheduler.step(vloss)

        if vloss < best_model:
            best_model = vloss
            torch.save(model, "./models/cnn/best.pt")

        writer.add_scalar("epoch_loss/train", tloss, epoch + 1)
        writer.add_scalar("epoch_loss/val", vloss, epoch + 1)

        torch.save(model, "./models/cnn/epoch-{:03d}.pt".format(epoch+1))
