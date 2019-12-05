import numpy as np
import gym 
import matplotlib.pyplot as plt
import copy
import time
import os

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
            self.hebbian[agent_idx][ii] = np.matmul(x0[np.newaxis,:].T, x[np.newaxis,:])

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
        self.hebbian[agent_idx][-1] = np.matmul(x0[np.newaxis,:].T, x[np.newaxis,:])

        return act

    def hebbian_prune(self, prune_rate=0.025):

        for ll in range(self.pop_size):
            for mm in range(len(self.population[ll])):
                temp_layer = np.copy(self.population[ll][mm])

                prunes_per_layer = prune_rate * temp_layer.shape[0]*temp_layer.shape[1]

                temp_layer *= 1.0 * (np.random.random((temp_layer.shape[0],\
                        temp_layer.shape[1])) > (softmax(self.hebbian[ll][mm]) \
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


    def get_fitness(self, env, epds=6, values=[1.0], render=False):

        fitness = []
        total_steps = 0

        for agent_idx in range(len(self.population)):
            #obs = flatten_obs(env.reset())
            accumulated_reward = 0.0
            for scaler in values:
                for epd in range(epds):

                    obs = env.reset()

                    done=False
                    while not done:
                        if render: env.render(); time.sleep(0.05)
                        action = self.get_action(obs, agent_idx=agent_idx,\
                                scaler=scaler) 
                        obs, reward, done, info = env.step(action)
                        total_steps += 1
                        accumulated_reward += reward
                    render = False 

            complexity_penalty =  0 * np.mean([np.mean(layer) \
                    for layer in self.population[agent_idx]])

            fitness.append(accumulated_reward/(epds*len(values))-complexity_penalty)


        return fitness, total_steps# , complexity 

    def update_pop(self, fitness, recombine=False):

        # make sure fitnesses aren't equal
        fitness = fitness
        sort_indices = list(np.argsort(fitness))
        sort_indices.reverse()

        sorted_fitness = np.array(fitness)[sort_indices]
        #sorted_pop = self.population[sort_indices]
        
        connections = []
        for pop_idx in range(self.pop_size):
            connections.append(np.sum([np.sum(layer) \
                    for layer in self.population[pop_idx]]))
        mean_connections = np.mean(connections)
        std_connections = np.std(connections)

        keep = int(np.ceil(0.125*self.pop_size))
        if sorted_fitness[0] > self.best_agent:
            # keep best agent
            print("new best elite agent: {} v {}".\
                    format(sorted_fitness[0], self.best_agent))
            self.elite_agent = self.population[sort_indices[0]]
            self.elite_agent_hebbian = self.hebbian[sort_indices[0]]
            self.best_agent = sorted_fitness[0]

        if np.mean(sorted_fitness[:keep]) > self.best_gen:
            # keep best elite population
            print("new best elite population: {} v {}".\
                    format(np.mean(sorted_fitness[:keep]), self.best_gen))
            self.best_gen = np.mean(sorted_fitness[:keep])

        self.elite_pop = []
        self.elite_pop_hebbian = []
        self.elite_pop.append(self.elite_agent)
        self.elite_pop_hebbian.append(self.elite_agent_hebbian)
        for oo in range(keep):
            self.elite_pop.append(self.population[sort_indices[oo]])
            self.elite_pop_hebbian.append(self.hebbian[sort_indices[oo]])

        self.population = []
        self.hebbian = []
        num_elite = len(self.elite_pop)
        p = np.arange(num_elite,0,-1) / np.sum(np.arange(num_elite,0,-1))
        a = np.arange(num_elite)

        for pp in range(self.pop_size):
            idx = np.random.choice(a,size=1,p=p)[0]
            self.population.append(copy.deepcopy(self.elite_pop[idx]))
            self.hebbian.append(copy.deepcopy(self.elite_pop_hebbian[idx]))

        if(recombine):
            for ll in range(keep,self.pop_size):
                new_layers = []
                for mm in range(len(self.hid)+1):
                    rec_map = np.random.randint(num_elite, size=self.population[0][mm].shape)
                    new_layer = np.zeros_like(self.elite_pop[0][mm])
                    for nn in range(num_elite):
                        new_layer = np.where(rec_map==nn, self.elite_pop[nn][mm],new_layer)
                    new_layers.append(new_layer + self.mut_noise*np.random.randn(\
                            new_layer.shape[0], new_layer.shape[1]))


                self.population.append(new_layers)

        elite_connections = 0.0
        for elite_idx in range(num_elite):
            elite_connections += (mean_connections - np.sum([np.sum(layer)\
                    for layer in self.elite_pop[elite_idx]]))**2

        return sorted_fitness, num_elite, \
                mean_connections, std_connections

if __name__ == "__main__":

    population_size = 100
    hid_dim = 64
    epds = 4
    env_name = "InvertedPendulumSwingupBulletEnv-v0"
    #env_name = "AntBulletEnv-v0"
    env_name = "Walker2DBulletEnv-v0"
    #env_name = "BipedalWalker-v2"

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
    try:
        t0 = time.time()
        total_steps = 0
        for generation in range(2000):
            t1 = time.time()
            fitness, steps = agent.get_fitness(env, epds=epds, values=[1.0], render=False)
            total_steps += steps

            sorted_fitness, num_elite, mean_connections, std_connections \
                    = agent.update_pop(fitness, recombine=False)
            agent.hebbian_prune()

            connections = np.sum([np.sum(layer) for \
                    layer in agent.population[0]])
            
            print("{:.2f}".format(time.time()-t1),generation, np.mean(fitness), np.max(fitness), np.min(fitness), connections)
    except KeyboardInterrupt:        
        pass

    np.save("./temp_hebbian.npy", agent.elite_pop)

