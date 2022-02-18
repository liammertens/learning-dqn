import torch
import torch.nn as nn
import torch.nn.functional as F


class QModel(nn.Module):
    def __init__(self, observation_dim, nbr_actions):
        super(QModel, self).__init__()
        self.f1 = nn.Linear(observation_dim, 16)
        self.f2 = nn.Linear(16, nbr_actions)

    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.Tensor(state)
        x = self.f1(state)
        x = F.tanh(x)
        x = self.f2(x)
        return x