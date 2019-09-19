#!/usr/bin/python
# -*- coding: utf-8 -*-

import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding
from actors.common_actor import PolicyNetWork
from actors.actor_sarsa import QNetWork
from critics.common_critic import ValueNetWork
from algorithms.reinforce import REINFORCE
from algorithms.reps import REPS
from algorithms.sarsa import SARSA
import matplotlib.pyplot as plt

'''
0     push left
1     push right
'''
class Learning:

    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.alg = 'sarsa'
        self.algorithms = {
            'reps': REPS(),
            'reinforce': REINFORCE(),
            'sarsa': SARSA()
        }
        print(self.alg)
        if self.alg == 'reps' or self.alg == 'reinforce':
            self.actor = PolicyNetWork(self.env.observation_space.shape[0], self.env.action_space.n, 96)
        elif self.alg == 'sarsa':
            self.actor = QNetWork(self.env.observation_space.shape[0]+1, 2, 96)

        # critic's input is observation_space + 1 where 1 is for the action
        # for reps only
        self.critic = ValueNetWork(self.env.observation_space.shape[0]+1, 1, 96)
        self.algorithms = {
                            'reps': REPS(),
                            'reinforce': REINFORCE(),
                            'sarsa': SARSA()
                            }
        self.actor_critic = False

    def show(self, data, labels):
        plt.figure(1)
        nmbs = [221, 222, 223, 224]
        for i in range(len(data)):
            nmb = nmbs[i]
            dat = data[i] # self.smooth(data[i])
            lbl = labels[i]
            plt.subplot(nmb)
            plt.plot(dat)
            plt.title(lbl, fontsize=8)
            plt.grid(True)
        # smoothed_rewards = [np.mean(rewards[i-window:i+1]) if i > window else np.mean(rewards[:i+1]) for i in range(len(rewards))]
        plt.show()

    def smooth(self, scatters, win=10):
        scat = []
        scatters = np.array(scatters)
        for i in range(len(scatters)-win):
            scat.append(np.mean(scatters[i:i+win]))
        j = win
        for i in range(len(scatters)-win, len(scatters)):
            scat.append(np.mean(scatters[i:i+j]))
            j -= 1
        return scat

    def run(self):
        samples = []
        # Choose algorithm
        # alg = 'reinforce'
        algorithm = self.algorithms[self.alg]
        for episode in range(500):
            state = self.env.reset()
            print('Episode', episode)
            if self.alg == 'sarsa':
                action, probs = self.actor.act(state)
            for _ in range(200):
                if self.alg == 'reinforce':
                    action, log_probs = self.actor.act(state, self.alg)
                elif self.alg == 'reps':
                    action = self.actor.act(state)
                prev_state = state
                state, reward, done, info = self.env.step(action)
                if self.alg == 'reinforce':
                    samples.append([prev_state, action, state, reward, log_probs])
                elif self.alg == 'reps':
                    samples.append([prev_state, action, state, reward])
                elif self.alg == 'sarsa':
                    self.actor, action, probs = algorithm.update(self.actor, [prev_state, action, state, reward, probs])
                if done:
                    if self.alg == 'sarsa':
                        algorithm.reward.append(algorithm.r)
                        print('    episode reward: ', algorithm.r)
                        self.actor.decrease_epsilon()
                        algorithm.r = 0
                    break
            # Episode-based methods policy update
            if self.alg == 'reinforce':
                self.actor = algorithm.update_policy(self.actor, samples)
            elif self.alg == 'reps':
                self.actor, self.critic = algorithm.update_policy(self.actor, self.critic, samples)
            samples = []
        self.env.close()
        if self.alg == 'reps':
            dat = [algorithm.loss_actor, algorithm.reward, algorithm.loss_critic, algorithm.eta]
            labels = ['loss actor', 'reward per episode', 'loss critic', 'eta']
        elif self.alg == 'reinforce':
            dat = [algorithm.loss_actor, algorithm.reward]
            labels = ['loss actor', 'reward per episode']
        elif self.alg == 'sarsa':
            dat = [algorithm.reward]
            labels = ['reward per episode']
        self.show(dat, labels)

if __name__ == '__main__':
    process = Learning()
    process.run()

