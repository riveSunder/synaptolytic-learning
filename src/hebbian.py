import numpy as np
import gym 
import matplotlib.pyplot as plt
import copy
import time
import os

from custom_envs.cartpole_swingup import CartPoleSwingUpEnv
from gym.envs.registration import register

import pybullet
import pybullet_envs


def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))

def softmax(x):
    x = x - np.max(x)

    y = np.exp(x) / np.sum(np.exp(x))

    return y

class Hebbian():
    def __init__(self, input_dim, output_dim, hid=[32], pop_size=10, seed=0, discrete=False):
        
        self.input_dim = input_dim
        self.output_dim = act_dim #output_dim
        self.hid = hid
        self.pop_size = pop_size
        self.seed = seed
        self.discrete = discrete
        self.best_gen = -float("Inf")
        self.best_agent = -float("Inf")
        np.random.seed(self.seed)

        # used for exponential averaging of the Hebbian component
        self.alpha_h = 0.9

        self.init_pop()


    def get_action(self, obs, agent_idx=0, scaler=1.0):

        x = obs        
        for ii in range(len(self.hid)):
            x0 = x
            x = np.matmul(x0, scaler*self.population[agent_idx][ii])
            x = np.tanh(x)

            # compute Hebbian memory
            hebbian_shape = self.hebbian[agent_idx][ii].shape
            for kk in range(hebbian_shape[0]):
                for ll in range(hebbian_shape[1]):

                    
                    self.hebbian[agent_idx][ii][kk,ll] = self.alpha_h\
                            * self.hebbian[agent_idx][ii][kk,ll] \
                            + (1-self.alpha_h) * (x0[kk] * x[ll])

        if self.discrete:
            x0 = x
            x = sigmoid(np.matmul(x0, scaler*self.population[agent_idx][-1]))
            #x = np.where(x > 0.5, 1, 0) 
        else:
            x0 = x
            x = np.matmul(x0, scaler*self.population[agent_idx][-1])

        if self.discrete:
            x = softmax(x)
            act = np.argmax(x)
        else:
            act = np.tanh(x)

        # compute Hebbian memory
        hebbian_shape = self.hebbian[agent_idx][-1].shape
        for kk in range(hebbian_shape[0]):
            for ll in range(hebbian_shape[1]):

                self.hebbian[agent_idx][ii][kk,ll] = self.alpha_h\
                        * self.hebbian[agent_idx][ii][kk,ll] \
                        + (1-self.alpha_h) * (x0[kk] * x[ll])
        return act

    def hebbian_prune(self):

        for ll in range(self.pop_size):
            for mm in range(len(self.population[ll])):
                temp_layer = np.copy(self.population[ll][mm])

                prunes_per_layer = .01 * temp_layer.shape[0]*temp_layer.shape[1]

                temp_layer *= 1.0 * (np.random.random((temp_layer.shape[0],\
                        temp_layer.shape[1])) > (self.hebbian[ll][mm] \
                        * prunes_per_layer))

                self.population[ll][mm] = temp_layer

    def init_pop(self):
        # represent population as a list of lists of np arrays
        self.population = []
        self.hebbian = []

        for jj in range(self.pop_size):
            layers = []
            heb_layers = []
            layer = np.ones((self.input_dim, self.hid[0]))

            layers.append(layer)
            heb_layers.append(layer*0)
            for kk in range(1,len(self.hid)):
                layer = np.ones((self.hid[kk-1], self.hid[kk]))
                layers.append(layer)
                heb_layers.append(layer*0)

            layer = np.ones((self.hid[-1], self.output_dim)) 

            layers.append(layer)
            heb_layers.append(layer*0)

            self.hebbian.append(heb_layers)
            self.population.append(layers)

if __name__ == "__main__":

    population_size = 1
    hid_dim = 32
    env_name = "InvertedPendulumBulletEnv-v0"

    env = gym.make(env_name)

    obs_dim = env.observation_space.shape[0]

    try:
        act_dim = env.action_space.n
        discrete = True
    except:
        act_dim = env.action_space.sample().shape[0]
        discrete = False

    agent = Hebbian(obs_dim, act_dim, hid=[hid_dim, hid_dim], \
            pop_size=population_size, discrete=discrete)
    
    for generation in range(1000):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.get_action(obs, agent_idx=0)

            obs, reward, done, info = env.step(action)

            total_reward += reward
        

        agent.hebbian_prune()
        connections = np.sum([np.sum(layer) for \
                layer in agent.population[0]])
        print(total_reward, connections)
        
