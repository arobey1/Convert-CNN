import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(16 * 24 * 24, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(6 * 28 * 28, 10)

    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        x = self.conv1(x)
        # x = self.pool(F.relu(self.conv2(x)))
        # x = F.relu(self.conv2(x))
        # x = x.view(-1, 16 * 24 * 24)
        print(x.detach().numpy())
        x = x.view(-1, 6 * 28 * 28)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        x = self.fc4(x)
        return x