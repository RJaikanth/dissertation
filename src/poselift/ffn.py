"""
dissertation.ffn.py
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
        self.linear3 = nn.Linear(128, 256)
        self.norm3 = nn.BatchNorm1d(256)
        self.relu3 = nn.ReLU()

        # Layer 4
        self.linear4 = nn.Linear(256, 128)
        self.norm4 = nn.BatchNorm1d(128)
        self.relu4 = nn.ReLU()

        # Layer 5
        self.linear5 = nn.Linear(128, 64)
        self.norm5 = nn.BatchNorm1d(64)
        self.relu5 = nn.ReLU()

        # Layer 6 / Output
        self.linear6 = nn.Linear(64, 51)

        # Init weights
        if not resume:
            self._init_weights()

    def forward(self, in_):
        # Forward
        in_ = in_.to(torch.float32)

        out_ = self.flatten(in_)
        out_ = self.norm1(self.relu1(self.linear1(out_)))
        out_ = self.norm2(self.relu2(self.linear2(out_)))
        out_ = self.norm3(self.relu3(self.linear3(out_)))
        out_ = self.norm4(self.relu4(self.linear4(out_)))
        out_ = self.norm5(self.relu5(self.linear5(out_)))
        out_ = self.linear6(out_)

        return out_

    def _init_weights(self):
        nn.init.kaiming_normal_(self.linear1.weight, nonlinearity = 'relu')
        nn.init.kaiming_normal_(self.linear2.weight, nonlinearity = 'relu')
        nn.init.kaiming_normal_(self.linear3.weight, nonlinearity = 'relu')
        nn.init.kaiming_normal_(self.linear3.weight, nonlinearity = 'relu')
        nn.init.kaiming_normal_(self.linear4.weight, nonlinearity = 'relu')
        nn.init.kaiming_normal_(self.linear5.weight, nonlinearity = 'relu')
        nn.init.kaiming_normal_(self.linear6.weight, nonlinearity = 'relu')

        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)
        nn.init.zeros_(self.linear3.bias)
        nn.init.zeros_(self.linear4.bias)
        nn.init.zeros_(self.linear5.bias)
        nn.init.zeros_(self.linear6.bias)

        print("Weights Initialized")


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Get data
    for i, data in enumerate(train_loader):
        inputs = data["2D"]
        target = data["3D"].flatten(1)

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
    model = FFNModel(resume = False)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(params = model.parameters(), lr = 1e-3, momentum = 0.9, weight_decay = 1e-5)

    train_ds = PoseLiftDataset(20000)
    train_loader = DataLoader(train_ds, batch_size = 2, shuffle = False)

    val_ds = PoseLiftDataset(10000)
    val_loader = DataLoader(val_ds, batch_size = 2, shuffle = True)

    # Initializing in a separate cell, so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))

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
                vtarget = vdata["3D"].flatten(1)
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

        # Track best performance, and save the model's state
        # if avg_vloss < best_vloss:
        #     best_vloss = avg_vloss
        #     model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        #     torch.save(model.state_dict(), model_path)

        epoch_number += 1
    # print(train_one_epoch(0, None))
