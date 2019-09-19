# -*- coding: UTF-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from graphviz import Digraph
from torchviz import make_dot


class PolicyNetWork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=5e-4):
        super(PolicyNetWork, self).__init__()
        self.num_actions = num_actions
        self.linear1 = nn.Linear(num_inputs, int(hidden_size))
        # initialization with zeros is useful for REPS
        # initialization with uniform distribution needed for reinforce
        # torch.nn.init.zeros_(self.linear1.weight)
        self.linear2 = nn.Linear(int(hidden_size), num_actions)
        # torch.nn.init.zeros_(self.linear2.weight)
        self.dropout = nn.Dropout()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = torch.nn.MSELoss()

    def forward(self, state):

        x = F.leaky_relu(self.linear1(state))

        x = F.softmax(self.linear2(x))

        return x

    def act(self, state, alg):
        state = torch.from_numpy(state).float()
        state = state.unsqueeze(0)
        probs = self.forward(Variable(state))
        try:
            highest_prob_action = np.random.choice(np.array([0,1]), p=np.squeeze(probs.detach().numpy()))
        except:
            print('Error highest probability action is not defined')
        if alg == 'reinforce':
            log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
            return highest_prob_action, log_prob
        elif alg == 'reps' or 'sarsa':
            return highest_prob_action
        else:
            raise RuntimeError('Algorithm is not defined')

        
    def get_probabilities(self, state):
        state = torch.from_numpy(state).float()
        # print(state.shape)
        state = state.unsqueeze(0)
        # print(state.shape)
        probs = self.forward(Variable(state))
        return probs


