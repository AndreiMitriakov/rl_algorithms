from torch import FloatTensor
from torch.autograd import Variable
import torch
import gym
import numpy as np


class REPS:
    def __init__(self):
        self.eps = 1.
        self.loss_critic = []
        self.loss_actor = []
        self.reward = []
        self.eta = []

    def episode_reward(self, samples):
        '''
        Get the episode reward
        '''
        ep_reward = 0.0
        for sample in samples:
            prev_state, action, new_state, reward = sample
            ep_reward += float(reward)
        #  ep_reward = ep_reward / len(samples)
        self.reward.append(ep_reward)

    def get_bef(self, samples, critic):
        '''
        Calc. Bellman error function according to the article the article
        '''
        bef = []
        for sample in samples:
            prev_state, action, new_state, reward = sample
            # hard coded
            s_0 = np.append(new_state, np.array(0.))
            s0 = np.append(prev_state, np.array(0.))
            s_1 = np.append(new_state, np.array(1.))
            s1 = np.append(prev_state, np.array(1.))
            del0 = reward + critic(s_0) - critic(s0)
            del1 = reward + critic(s_1) - critic(s1)
            bef.append([del0, del1])

        return bef

    def dual_function(self, critic, eta, eps, bef, N, samples):
        bef_samples = []
        for i, sample in enumerate(samples):
            prev_state, action, new_state, reward = sample
            bef_samples.append(bef[i][action])
        bef = torch.stack(bef_samples)
        output = eta * torch.log(torch.sum(torch.exp(eps + bef / eta)) / N )
        return output

    def dual_function_optimization(self, critic, samples):
        N = len(samples)
        def closure():
            critic.optimizer.zero_grad()
            bef = self.get_bef(samples, critic)
            g = self.dual_function(critic, critic.eta, self.eps, bef, N, samples)
            g.backward()
            self.loss_critic.append(float(g))
            self.eta.append(float(critic.eta))
            return g
        critic.optimizer.step(closure)
        return critic

    def get_loss_actor(self, samples, actor, critic):
        '''
        Calc. loss function for the actor according to the formula
        '''
        bef = self.get_bef(samples, critic)
        loss = []
        for i, sample in enumerate(samples):
            prev_state, action, new_state, reward = sample
            action_prob = actor.get_probabilities(new_state)
            action = int(action)
            # actor loss # vef is too huge, eta is huge
            val = action_prob[0][action] * torch.exp(bef[i][action] / critic.eta)
            denom = 0
            for j in range(len(action_prob[0])):
                value = action_prob[0][j] * torch.exp(bef[i][j] / critic.eta)
                denom += value
            # denom = torch.sum(torch.stack(denom)) # if using list comprehensions
            loss.append(actor.criterion(val / denom, action_prob[0][action]))
        return loss

    def update_policy(self, actor, critic, samples):
        #--------Critic--------#
        critic = self.dual_function_optimization(critic, samples)
        #--------Actor--------#
        actor.optimizer.zero_grad()
        loss = self.get_loss_actor(samples, actor, critic)
        loss = torch.mean(torch.stack(loss))
        loss.backward()
        actor.optimizer.step()
        self.episode_reward(samples)
        self.loss_actor.append(float(loss))
        print('loss actor', float(loss), 'reward', float(self.reward[-1]))
        print('########')
        return actor, critic


