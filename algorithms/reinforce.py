# -*- coding: UTF-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from graphviz import Digraph
from torchviz import make_dot


class REINFORCE:

    def __init__(self):
        np.random.seed()
        self.reward = []
        self.loss_actor = []
        self.samples = []
        self.log_probs = []
        self.gamma = 0.99
        self.n_policy_updates = 0
        self.learning_rate = 0.1
        device = torch.device('cpu')

    def get_G(self, samples):
        GAMMA = 0.9
        discounted_rewards = []
        rewards = [sample[3] for sample in samples]
        self.sum_reward = np.sum(rewards)
        for t in range(len(rewards)):
            Gt = 0
            pw = 0
            for r in rewards[t:]:
                Gt = Gt + GAMMA**pw * r
                pw = pw + 1
            discounted_rewards.append(Gt)        
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9) # normalize discounted rewards
        return discounted_rewards

    def update_policy(self, actor, samples):
        log_probs = []
        for sample in samples:
            log_probs.append(sample[4])

        G = self.get_G(samples)
        policy_gradient = []

        for log_prob, Gt in zip(log_probs, G):
            policy_gradient.append(-log_prob * Gt)
        
        actor.optimizer.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()
        policy_gradient.backward()
        self.loss_actor.append(float(policy_gradient))
        # graph = make_dot(policy_gradient)
        # graph.render('~/Pictures/round.gv', view=True)
        actor.optimizer.step() 
        print('    reward', self.sum_reward)
        self.reward.append(self.sum_reward)
        return actor




