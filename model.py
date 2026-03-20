import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, in_states, hidden_nodes, out_actions):
        super().__init__()
        self.fc1 = nn.Linear(in_states, hidden_nodes)
        self.fc2 = nn.Linear(hidden_nodes, hidden_nodes)
        self.out = nn.Linear(hidden_nodes, out_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)
