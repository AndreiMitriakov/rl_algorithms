# -*- coding: UTF-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from graphviz import Digraph
from torchviz import make_dot


class PolicyREPS(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=1e-2): # 3e-3 - 5e-3 are optimal
        super(PolicyREPS, self).__init__()
        self.num_actions = num_actions
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        self.linear2 = nn.Linear(hidden_size, num_actions)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        self.dropout = nn.Dropout()
        self.criterion = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=5e-5)

    def forward(self, state):
        state = torch.from_numpy(state).float()
        state = state.unsqueeze(0)
        x = F.leaky_relu(self.linear1(state))
        # x = self.dropout(x)
        x = F.softmax(self.linear2(x))
        return x

    def act(self, state):
        probs = self.forward(state)
        try:
            highest_prob_action = np.random.choice(np.array([0,1]), p=np.squeeze(probs.detach().numpy()))
        except:
            print('Error highest probability action is not defined')
        return highest_prob_action

    def get_probabilities(self, state):
        return self.forward(state)
