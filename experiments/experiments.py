import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# parameters
epochs = 5


class DummyDataset(Dataset):
    def __init__(self):
        self.images = torch.randn(10, 3, 64, 64)
        self.labels = torch.arange(0, 10)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class DummyModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DummyModel, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 3, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16*16*3, 10, bias=True),
            nn.Liner(10, out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_network(model, optim, dataloader):
    for e in range(epochs):
        for images, labels in dataloader:
            optim.zero_grad()



def main():
    dummy_dataset = DummyDataset()
    dummy_loader = DataLoader(dummy_dataset, batch_size=1)
    model = DummyModel(in_channels=3, out_channels=10)
    train_network(model, dummy_loader)



