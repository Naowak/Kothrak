import torch.nn as nn
import torch.nn.functional as F


class DeepQNetwork(nn.Module):

    def __init__(self, num_input, num_output):

        super(DeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(num_input, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, num_output)

    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
