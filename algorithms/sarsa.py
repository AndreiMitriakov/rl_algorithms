# -*- coding: UTF-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from graphviz import Digraph
from torchviz import make_dot


class SARSA:

    def __init__(self):
        np.random.seed()
        self.reward = []
        self.r = 0
        self.gamma = 1.
        self.alpha = 1.

    def huber_loss(self, error):
        if float(error) <= 1.:
            return 0.5*error**2
        else:
            return abs(error)-0.5

    def update(self, actor, sample):
        s, a, s_, r, probs = sample
        Qsa = probs[a]
        a_, probs_ = actor.act(s_)
        Q_sa = probs_[a_]
        actor.optimizer.zero_grad()
        error = self.alpha * (r + self.gamma * Q_sa - Qsa)

        # L1 regularization
        reg_loss = 0
        for param in actor.parameters():
            reg_loss += torch.sum(torch.abs(param))
        factor = 5e-5

        loss = self.huber_loss(error) + factor * reg_loss
        loss.backward(retain_graph=True)
        actor.optimizer.step()
        self.r += 1
        return actor, a_, probs_

