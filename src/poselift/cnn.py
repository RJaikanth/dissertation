"""
dissertation.cnn.py
Author: Raghhuveer Jaikanth
Date  : 07/08/2023

# Enter Description Here
"""
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


class PoseLiftDataset(Dataset):
    def __init__(self, size=100000):
        super().__init__()
        torch.manual_seed(42)
        self.input_ = torch.randn(size = (size, 17, 2))
        torch.manual_seed(42)
        self.output_ = torch.randn(size = (size, 17, 3))

    def __getitem__(self, item):
        return {
            "2D": self.input_[item],
            "3D": self.output_[item]
        }

    def __len__(self):
        return len(self.input_)


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


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Get data
    for i, data in enumerate(train_loader):
        inputs = data["2D"]
        target = data["3D"]

        # Forward
        optimizer.zero_grad()  # Zero gradients
        outputs = model(inputs)  # Prediction
        loss = loss_fn(outputs, target)  # Loss calculation

        # Backward
        loss.backward()  # Compute gradients
        optimizer.step()  # Adjust weights

        # Gather and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # Loss per batch
            print(f"\tBatch {i + 1:03d} Loss: {last_loss:5.2f}")
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


if __name__ == "__main__":
    torch.manual_seed(42)
    model = CNNModel(resume = False)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(params = model.parameters(), lr = 1e-3, momentum = 0.9, weight_decay = 1e-5)

    train_ds = PoseLiftDataset(20000)
    train_loader = DataLoader(train_ds, batch_size = 2, shuffle = False)

    val_ds = PoseLiftDataset(10000)
    val_loader = DataLoader(val_ds, batch_size = 2, shuffle = True)

    # Initializing in a separate cell, so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/cnn_pose_lift{}'.format(timestamp))

    epoch_number = 0
    EPOCHS = 5
    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer)

        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(val_loader):
                vinputs = vdata["2D"]
                vtarget = vdata["3D"]
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vtarget)
                running_vloss += vloss
        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars(
            'Training vs. Validation Loss',
            {'Training': avg_loss, 'Validation': avg_vloss},
            epoch_number + 1)
        writer.flush()
