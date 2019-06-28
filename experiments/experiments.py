import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from pathlib import Path
from collections import OrderedDict

# parameters
epochs = 5
model_path = Path('dummy_model.pth')
features_weight_path = Path('features.pth')
classifier_weight_path = Path('classifier.pth')


class DummyDataset(Dataset):
    def __init__(self):
        self.images = torch.randn(10, 3, 64, 64)
        self.labels = torch.empty(10, dtype=torch.float).random_(2)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class DummyModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DummyModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=3, padding=1)  # (1,6,64,64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxPool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # (1,6,32,32)
        self.conv2 = nn.Conv2d(6, 3, kernel_size=3, padding=1)  # (1,3,32,32)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # (1,3,16,16)
        self.linear1 = nn.Linear(16*16*3, 128, bias=True)
        self.linear2 = nn.Linear(128, 10)
        self.linear3 = nn.Linear(10, out_channels)
        self.last_layer = nn.Sigmoid()

    def forward(self, x):
        # x = self.features(x)
        # x = self.classifier(x)
        # return x

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxPool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxPool2(x)
        x = x.view(16*16*3)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.last_layer(x)
        return x


class SplitModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SplitModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16*16*3, 128, bias=True),
            nn.Linear(128, 10),
            nn.Linear(10, out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(16*16*3)
        x = self.classifier(x)
        return x


def train_network(model, optim, criterion, dataloader):
    for e in range(epochs):
        training_loss = 0
        for image, label in dataloader:
            optim.zero_grad()
            prob = model(image)
            loss = criterion(prob, label)
            loss.backward()
            optim.step()
            training_loss += loss.item()
            print(f"Epoch: {e+1}/{epochs},"
                  f"Training loss: {loss}")


def save_model(model):
    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    torch.save(model.state_dict(), model_path)


def splitting_weight(weight_path):
    features_weights = OrderedDict()
    classifier_weights = OrderedDict()

    for idx, value in torch.load(weight_path).items():
        if not idx.startswith('linear'):
            features_weights[idx] = value
            print('features extraction --', idx, value.shape)
        else:
            classifier_weights[idx] = value
            print('classifier -- ', idx, value.shape)

    torch.save(features_weights, features_weight_path)
    torch.save(classifier_weights, classifier_weight_path)


def main():
    dummy_dataset = DummyDataset()
    dummy_loader = DataLoader(dummy_dataset, batch_size=1)
    # model = DummyModel(in_channels=3, out_channels=1)
    model = SplitModel(in_channels=3, out_channels=1)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    train_network(model, optimizer, criterion, dummy_loader)
    # save_model(model)


if __name__ == "__main__":
    # main()
    splitting_weight(model_path)

