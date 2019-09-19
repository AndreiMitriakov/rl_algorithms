# -*- coding: UTF-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from graphviz import Digraph
from torchviz import make_dot


class QNetWork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=5e-3): # 3e-3 - 5e-3 are optimal
        super(QNetWork, self).__init__()
        self.num_actions = num_actions
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        # torch.nn.init.zeros_(self.linear1.weight)
        self.linear2 = nn.Linear(hidden_size, 1)
        # torch.nn.init.zeros_(self.linear2.weight)
        self.dropout = nn.Dropout()
        self.eps = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]
        self.epsilon = self.eps[0]
        self.cnt = 0
        self.criterion = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=5e-4)

    def forward(self, state):
        state = torch.from_numpy(state).float()
        state = state.unsqueeze(0)

        x = F.relu(self.linear1(state))
        x = self.dropout(x)
        x = F.relu(self.linear2(x))
        return x

    def act(self, state):
        probs = []

        for act in range(self.num_actions):
            state_try = np.append(state, np.array(act))
            probs.append(self.forward(state_try))
        if np.random.rand() > self.epsilon:
            action = np.argmax(np.array(probs))
        else:
            action = np.random.randint(2)
        return action, probs

    def decrease_epsilon(self):
        self.cnt += 1.
        step = 20
        if self.cnt % step == 0:
            j = int(self.cnt / step)
            if j < len(self.eps):
                self.epsilon = self.eps[j]
            else:
                self.epsilon = self.eps[-1]

        # if self.epsilon > 0.01:
        #     self.epsilon *= 0.99