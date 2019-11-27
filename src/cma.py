import numpy as np
import gym 
import matplotlib.pyplot as plt
import copy
import time
import os

from custom_envs.cartpole_swingup import CartPoleSwingUpEnv
from gym.envs.registration import register

register(
    id='CartPoleSwingUp-v0',
    entry_point='custom_envs.cartpole_swingup:CartPoleSwingUpEnv',
    max_episode_steps=200,
    reward_threshold=25.0,
    )


class CMAAgent():
    def __init__(self, obs_dim, act_dim, population_size, \
            seed=0, discrete=False):

        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.hid_dim = 8
        self.population_size = population_size

        self.seed = seed
        self.by = -0.00
        self.discrete = discrete
        self.best_gen = -float("Inf")
        self.best_agent = -float("Inf")
        np.random.seed(self.seed)

        
        self.init_dist()
        self.init_pop()


    def get_action(self, obs, agent_idx):
        x = obs        

        x = np.matmul(x, self.pop[agent_idx][0])
        x = np.tanh(x)

        if self.discrete:
            x = sigmoid(self.by + np.matmul(x, self.pop[agent_idx][-1]))
            #x = np.where(x > 0.5, 1, 0) 
        else:
            x = self.by + np.matmul(x, self.pop[agent_idx][-1])

        if self.discrete:
            x = softmax(x)
            act = np.argmax(x)
        else:
            x = x
            act = np.tanh(x)

        return act

    def get_fitness(self, env, epds=6, render=False):
        fitness = []
        complexity = []
        total_steps = 0

        for agent_idx in range(len(self.pop)):
            #obs = flatten_obs(env.reset())
            accumulated_reward = 0.0
            for epd in range(epds):

                obs = env.reset()
                done=False
                while not done:
                    if render: env.render(); time.sleep(0.005)
                    action = self.get_action(obs, agent_idx=agent_idx)
                    obs, reward, done, info = env.step(action)
                    total_steps += 1
                    accumulated_reward += reward
                render = False 


            fitness.append(accumulated_reward/(epds))
        plt.close("all")

        return fitness, total_steps

    def update_pop(self, fitness, recombine=False):

        # make sure fitnesses aren't equal
        sort_indices = list(np.argsort(fitness))
        sort_indices.reverse()

        sorted_fitness = np.array(fitness)[sort_indices]
        #sorted_pop = self.pop[sort_indices]

        keep = int(np.ceil(0.125*self.population_size))
        if sorted_fitness[0] > self.best_agent:
            # keep best agent
            print("new best elite agent: {} v {}".\
                    format(sorted_fitness[0], self.best_agent))
            self.elite_agent = self.pop[sort_indices[0]]
            self.best_agent = sorted_fitness[0]

        if np.mean(sorted_fitness[:keep]) > -float("Inf"): # self.best_gen:
            # keep best elite population
            #print("new best elite population: {} v {}".\
            #        format(np.mean(sorted_fitness[:keep]), self.best_gen))
            self.best_gen = np.mean(sorted_fitness[:keep])

            self.elite_pop = []
            self.elite_pop.append(self.elite_agent)
            for oo in range(keep):
                self.elite_pop.append(self.pop[sort_indices[oo]])

        self.pop = []
        num_elite = len(self.elite_pop)

        # update parameters
        step_size = 1.0

        sum_layer = np.zeros(self.obs_dim*self.hid_dim)
        sum_cov = np.zeros((self.obs_dim*self.hid_dim, self.obs_dim*self.hid_dim))
        for gg in range(num_elite):
            sum_layer += self.elite_pop[gg][0].ravel()

            for ii in range(self.obs_dim * self.hid_dim):
                for jj in range(self.obs_dim * self.hid_dim):
                    sum_cov[ii,jj] +=\
                            (self.elite_pop[gg][0].ravel()[ii]\
                            - self.layer_dist[0][0][ii]) \
                            * (self.elite_pop[gg][0].ravel()[jj]\
                            - self.layer_dist[0][0][jj])

        mean_layer = sum_layer / num_elite
        mean_cov = sum_cov / num_elite

        self.layer_dist[0] = [mean_layer, mean_cov]

        sum_layer = np.zeros((self.hid_dim * self.act_dim))
        sum_cov = np.zeros((self.act_dim*self.hid_dim, self.act_dim*self.hid_dim))
        for gg in range(num_elite):
            sum_layer += self.elite_pop[gg][1].ravel()

            for ii in range(self.act_dim * self.hid_dim):
                for jj in range(self.act_dim * self.hid_dim):
                    sum_cov[ii,jj] +=\
                            (self.elite_pop[gg][1].ravel()[ii]\
                            - self.layer_dist[1][0][ii]) \
                            * (self.elite_pop[gg][1].ravel()[jj]\
                            - self.layer_dist[1][0][jj])

        mean_layer = sum_layer / num_elite
        mean_cov = sum_cov / num_elite

        self.layer_dist[1] = [mean_layer, mean_cov]

        self.init_pop()

        return sorted_fitness, num_elite

    def init_dist(self):
        self.layer_dist = []
    
        # input to hidden layer distribution

        layer_mew = np.zeros((self.obs_dim * self.hid_dim))
        layer_cov = np.eye(self.obs_dim * self.hid_dim) 

        self.layer_dist.append([layer_mew, layer_cov])

        # hidden layer distribution
        layer_mew = np.zeros((self.hid_dim * self.act_dim))
        layer_cov = np.eye(self.hid_dim * self.act_dim)

        self.layer_dist.append([layer_mew, layer_cov])

    def init_pop(self):
        self.pop = []

        for hh in range(self.population_size):
            layers = []
            layer = np.random.multivariate_normal(self.layer_dist[0][0], \
                    self.layer_dist[0][1]).reshape(self.obs_dim, self.hid_dim)

            layers.append(layer)

            layer = np.random.multivariate_normal(self.layer_dist[1][0], \
                    self.layer_dist[1][1]).reshape(self.hid_dim, self.act_dim)

            layers.append(layer)
            self.pop.append(layers)



if __name__ == "__main__":

    print("good to go")
    population_size = 100

    env_names = ["BipedalWalker-v2","CartPoleSwingUp-v0"]

    for env_name in env_names:

        env = gym.make(env_name)

        obs_dim = env.observation_space.shape[0]
            
        try:
            act_dim = env.action_space.n
            discrete = True
        except:
            act_dim = env.action_space.sample().shape[0]
            discrete = False

        agent = CMAAgent( obs_dim, act_dim, population_size, discrete=discrete)

        for generation in range(100):
            render = generation % 10 == 0 
            fitness, total_steps = agent.get_fitness(env, render=render)

            agent.update_pop(fitness)
            print("gen {}, average fitness: {}".format(generation, np.mean(fitness)))
