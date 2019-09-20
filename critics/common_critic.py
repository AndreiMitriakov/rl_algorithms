# -*- coding: UTF-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from graphviz import Digraph
from torchviz import make_dot


class ValueNetWork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=5e-2): # 3e-4
        super(ValueNetWork, self).__init__()
        self.num_actions = num_actions
        eta = 1.
        self.eta = torch.nn.Parameter(torch.Tensor([eta]))

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        torch.nn.init.ones_(self.linear1.weight)

        self.linear2 = nn.Linear(hidden_size, 1)
        torch.nn.init.ones_(self.linear2.weight)

        self.dropout = nn.Dropout()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)# , weight_decay=5e-4)
        self.optimizer_eta = torch.optim.Adam([self.eta], lr=learning_rate*100)

    def forward(self, state):
        # print('input', state)
        state = torch.from_numpy(state).float()
        state = state.unsqueeze(0)
        # print('state INPUT', state)
        x = F.relu(self.linear1(state))
        # print('first output', x)
        x = F.relu(self.linear2(x))
        # print('critic', x)
        # dot = make_dot(x)
        # dot.render()
        # if float(x) == 0:
        #     raise RuntimeError('Output is zero')
        # print('critic', float(x))
        return x 
''''''