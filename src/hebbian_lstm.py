import numpy as np
import gym 
import matplotlib.pyplot as plt
import copy
import time
import os

import pybullet
import pybullet_envs

def sigmoid(x):
    x = np.clip(x, -5e2, 5e2)
    return np.exp(x) / (1 + np.exp(x))

def softmax(x):
    x = x - np.max(x)

    y = np.exp(x) / np.sum(np.exp(x))

    return y

class HebbianLSTMAgent():
    
    def __init__(self, input_dim, act_dim, hid_dim=[128],\
            population_size=10, seed=0, discrete=True, random_init=True):

        self.input_dim = input_dim
        self.output_dim = act_dim #output_dim
        self.hid = hid_dim
        self.random_init = random_init
        
        #self.hid.append(self.output_dim)
        #self.hid.insert(0,self.input_dim)

        self.population_size = population_size
        self.seed = seed
        self.by = -0.00

        self.f_bias = - 1.0


        self.mut_noise = 1e0
        self.discrete = discrete
        
        elitism = 0.125
        num_elite = int( elitism * population_size )
        self.leaderboard = [-float("Inf")] * num_elite
        self.elite_pop = []
        self.best_gen = -float("Inf")
        self.best_agent = -float("Inf")
        np.random.seed(self.seed)

        self.init_pop()
   

    def get_action(self, obs, agent_idx=0, scaler=1.0, hebbian=True, enjoy=False):
        
        cell_state = self.population[agent_idx][0]
        all_bias = 0.0

        x = np.append( obs, cell_state, axis=0 )

        f = sigmoid(np.matmul(x, self.population[agent_idx][1]) + self.f_bias)
        i = np.tanh(np.matmul(x, self.population[agent_idx][2]) + all_bias )
        j = sigmoid(np.matmul(x, self.population[agent_idx][3]) + all_bias )
        o = sigmoid(np.matmul(x, self.population[agent_idx][4]) + all_bias )

        cell_state = np.tanh(cell_state * f + i * j)

        h = cell_state * o 

        action = np.tanh(np.matmul(h, self.population[agent_idx][5]))

        if hebbian:
            f_hebbian = np.matmul(x[np.newaxis,:].T, f[np.newaxis,:])
            i_hebbian = np.matmul(x[np.newaxis,:].T, i[np.newaxis,:])
            j_hebbian = np.matmul(x[np.newaxis,:].T, j[np.newaxis,:])
            o_hebbian = np.matmul(x[np.newaxis,:].T, o[np.newaxis,:])
            a_hebbian = np.matmul(h[np.newaxis,:].T, action[np.newaxis,:])

            self.hebbian[agent_idx][1] += f_hebbian
            self.hebbian[agent_idx][2] += i_hebbian
            self.hebbian[agent_idx][3] += j_hebbian
            self.hebbian[agent_idx][4] += o_hebbian
            self.hebbian[agent_idx][5] += a_hebbian

        self.population[agent_idx][0] = cell_state
        return action
        
       
    def init_pop(self, hebbian=True):

        if not (self.random_init):
            f_forget = np.ones(( self.input_dim + self.hid[0], self.hid[0] ))
            i_input = np.ones(( self.input_dim + self.hid[0], self.hid[0] ))
            j_input = np.ones(( self.input_dim + self.hid[0], self.hid[0] ))
            o_output = np.ones(( self.input_dim + self.hid[0], self.hid[0] ))
            a_action = np.ones(( self.hid[0], self.output_dim ))
            cell_state = np.zeros((self.hid[0]))

        self.population = []
        self.hebbian = []


        for ii in range(self.population_size):
            if self.random_init:

                f_forget = np.random.randn( self.input_dim + self.hid[0], self.hid[0])
                i_input = np.random.randn( self.input_dim + self.hid[0], self.hid[0] )
                j_input = np.random.randn( self.input_dim + self.hid[0], self.hid[0] )
                o_output = np.random.randn( self.input_dim + self.hid[0], self.hid[0] )

                a_action = np.random.randn(self.hid[0], self.output_dim )
                cell_state = np.zeros((self.hid[0]))
        
            self.population.append([np.copy(cell_state), np.copy(f_forget),\
                np.copy(i_input),  np.copy(j_input), \
                np.copy(o_output), np.copy(a_action)])

            if hebbian:
                self.hebbian.append([np.zeros_like(cell_state), np.zeros_like(f_forget),\
                    np.zeros_like(i_input), np.zeros_like(j_input),\
                    np.zeros_like(o_output), np.zeros_like(a_action)])


    def random_prune(self, prune_rate=0.01, keep=0):

        for ii in range(keep, self.population_size):
            for jj in range(1,len(self.population[ii])):
                dim_x, dim_y = np.shape(self.population[ii][jj]) 
                self.population[ii][jj] *= np.random.random((dim_x, dim_y)) \
                    > prune_rate

    def hebbian_prune(self, prune_rate=0.01, keep=0):

        for jj in range(keep,self.population_size):
            for kk in range(1, len(self.population[jj])):

                temp_layer = self.population[jj][kk]

                prunes_per_layer = prune_rate * temp_layer.shape[0]*temp_layer.shape[1]\
                    /(1e-1+np.mean(temp_layer))

                temp_layer *= 1.0 * (np.random.random((temp_layer.shape[0],\
                        temp_layer.shape[1])) > (softmax(-self.hebbian[jj][kk]) \
                        * prunes_per_layer))

                self.hebbian[jj][kk] *= 0

                self.population[jj][kk] = temp_layer

    def get_fitness(self, env, epds=6, values=[1.0], render=False):

        fitness = []
        total_steps = 0

        for agent_idx in range(len(self.population)):
            #obs = flatten_obs(env.reset())
            accumulated_reward = 0.0
            for scaler in values:
                for epd in range(epds):

                    self.population[agent_idx][0] = np.zeros((self.hid[0])) 
                    obs = env.reset()

                    done=False
                    while not done:
                        if render: env.render(); time.sleep(0.05)
                        action = self.get_action(obs, agent_idx=agent_idx,\
                                scaler=scaler) 
                        obs, reward, done, info = env.step(action)
                        total_steps += 1
                        accumulated_reward += reward

            fitness.append(accumulated_reward/(epds*len(values)))
            #complexity.append(complexity_penalty)

        plt.close("all")

        return fitness, total_steps# , complexity 

    def update_pop(self, fitness):

        # make sure fitnesses aren't equal
        fitness = fitness
        sort_indices = list(np.argsort(fitness))
        sort_indices.reverse()

        sorted_fitness = np.array(fitness)[sort_indices]
        sorted_pop = [self.population[idx] for idx in sort_indices]
        
        connections = []
        for pop_idx in range(self.population_size):
            connections.append(np.sum([np.sum(layer) \
                    for layer in self.population[pop_idx][1:]]))

        mean_connections = np.mean(connections)
        std_connections = np.std(connections)

        keep = int(np.ceil(0.125 * self.population_size))

        if sorted_fitness[0] > self.best_agent:
            # keep best agent
            print("new best elite agent: {} v {}".\
                    format(sorted_fitness[0], self.best_agent))
            self.elite_agent = self.population[sort_indices[0]]
            self.best_agent = sorted_fitness[0]

        lb_idx = 0 
        fit_idx = 0
        total_added = 0
        fitness_copy = list(copy.deepcopy(sorted_fitness))

        while fitness_copy[0] > self.leaderboard[-1]:
            
            if fitness_copy[0] > self.leaderboard[lb_idx]:
                total_added +=1
                self.leaderboard.insert(lb_idx, fitness_copy[0])
                self.elite_pop.insert(lb_idx, self.population[sort_indices[fit_idx]])
                fitness_copy.pop(0) 
                fit_idx += 1
            lb_idx += 1

            if lb_idx > keep:
                break
        
        print("added {} agents to elite population".format(total_added))
        self.leaderboard = self.leaderboard[:keep]
        self.elite_pop = self.elite_pop[:keep] + sorted_pop[:keep]
            

        if np.mean(sorted_fitness[:keep]) > self.best_gen:
            # keep best elite population
            print("new best elite population: {} v {}".\
                    format(np.mean(sorted_fitness[:keep]), self.best_gen))
            self.best_gen = np.mean(sorted_fitness[:keep])

        self.population = []
        num_elite = len(self.elite_pop)
        p = np.arange(num_elite,0,-1) / np.sum(np.arange(num_elite,0,-1))
        a = np.arange(num_elite)

        for pp in range(self.population_size):
            idx = np.random.choice(a,size=1,p=p)[0]
            self.population.append(copy.deepcopy(self.elite_pop[idx]))
        
        recombine = True
        num_recombinations = num_elite * 2
        if recombine:
            for ll in range(num_elite):
                agent_0 = np.random.randint(self.population_size)
                agent_1 = np.random.randint(self.population_size)

                swap_gate = np.random.randint(4)

                self.population[agent_0][swap_gate], self.population[agent_1][swap_gate] \
                        = self.population[agent_1][swap_gate], self.population[agent_0][swap_gate] \



        if np.random.randint(2):
            self.random_prune(prune_rate=0.07, keep=2)
        else:
            self.hebbian_prune(prune_rate=0.02, keep=2)

        return sorted_fitness, num_elite, \
                mean_connections, std_connections

                
                        
if __name__ == "__main__":


    env_name = "BipedalWalker-v2"
    env_name = "InvertedPendulumBulletEnv-v0"

    env = gym.make(env_name)
    print("make env", env_name)

    obs_dim = env.observation_space.shape[0]

    try:
        act_dim = env.action_space.n
        discrete = True
    except:
        act_dim = env.action_space.sample().shape[0]
        discrete = False

    population_size = 512 

    hid_dim = [64]
    
    agent = HebbianLSTMAgent(obs_dim, act_dim, hid=hid_dim, \
            population_size=population_size, discrete=discrete)

    #agent.random_prune(prune_rate=0.125)
    obs = env.reset()

    try:
        for gen in range(5000):

            fitness, total_steps = agent.get_fitness(env, epds=2)
            sorted_fitness, num_elite,\
                mean_connections, std_connections = agent.update_pop(fitness)
            keep = 16 
            agent.hebbian_prune(prune_rate=0.1025, keep=keep)
            agent.random_prune(prune_rate=0.10,keep=keep)

            print(mean_connections, " +/- ", std_connections)
            print("{} best/worst/average fitness: {:.3f}/{:.3f}/{:.3f}".format(\
                gen, np.max(fitness), np.min(fitness), np.mean(fitness)))
            gen += 1
    except KeyboardInterrupt:
        pass
    np.save("temp_lstm_weights_2.npy", agent.elite_pop)

