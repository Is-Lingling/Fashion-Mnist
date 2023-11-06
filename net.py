from torch import nn


class FashionNet(nn.Module):
    def __init__(self):
        super(FashionNet, self).__init__()
        self.Cov1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.MaxPool = nn.MaxPool2d(1)
        self.relu = nn.ReLU()
        self.Cov2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.Cov3 = nn.Conv2d(64, 32, 3, 1, 1)
        self.Cov4 = nn.Conv2d(32, 16, 3, 1, 1)
        self.Cov5 = nn.Conv2d(16, 8, 3, 1, 1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8*28*28, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 10)

    def forward(self,x):
        x = self.relu(self.MaxPool(self.Cov1(x)))
        x = self.relu(self.MaxPool(self.Cov2(x)))
        x = self.relu(self.MaxPool(self.Cov3(x)))
        x = self.relu(self.MaxPool(self.Cov4(x)))
        x = self.relu(self.MaxPool(self.Cov5(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
