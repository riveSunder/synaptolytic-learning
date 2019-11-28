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

register(
    id='CartPoleSwingUp-v0',
    entry_point='custom_envs.cartpole_swingup:CartPoleSwingUpEnv',
    max_episode_steps=200,
    reward_threshold=25.0,
    )

def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))

def softmax(x):
    x = x - np.max(x)

    y = np.exp(x) / np.sum(np.exp(x))

    return y

class PruneableAgent():

    def __init__(self, input_dim, act_dim, hid=[32,32,32],\
            pop_size=10, seed=0, discrete=True):

        self.input_dim = input_dim
        self.output_dim = act_dim #output_dim
        self.hid = hid
        self.pop_size = pop_size
        self.seed = seed
        self.by = -0.00
        self.mut_noise = 1e0
        self.discrete = discrete
        self.best_gen = -float("Inf")
        self.best_agent = -float("Inf")
        np.random.seed(self.seed)

        self.init_pop()
        #self.mutate_pop(rate=0.25)

    def get_action(self, obs, agent_idx=0, scaler=1.0):

        x = obs        
        nodes = []
        nodes.append(x)

        for ii in range(len(self.hid)):
            x = np.matmul(x, scaler*self.pop[agent_idx][ii])
            x = np.tanh(x)
            nodes.append(x)
            #x[x<0] = 0 # relu

        if self.discrete:
            x = self.by + np.matmul(x, scaler*self.pop[agent_idx][-1])
            #x = np.where(x > 0.5, 1, 0) 
        else:
            x = self.by + np.matmul(x, scaler*self.pop[agent_idx][-1])
        

        nodes.append(np.tanh(x))

        if self.discrete:
            x = softmax(x)
            act = np.argmax(x)
        else:
            x = x
            act = np.tanh(x)

        self.node_buffer.append(nodes)
        return act

    def init_node_buffer(self):
        self.node_buffer = []
        self.node_means = None
        self.node_cov = None

    def get_node_cov(self):



        num_layers = len(self.node_buffer[0])
        
        if self.node_means == None:
            self.node_means = [np.zeros_like(nodes) \
                    for nodes in self.node_buffer[0]]
            
        self.node_cov = [\
                np.zeros((self.node_buffer[0][aa].shape[0], \
                self.node_buffer[0][aa+1].shape[0])) \
                for aa in range(len(self.node_buffer[0])-1)]

        new_node_means = [np.zeros_like(nodes) \
                    for nodes in self.node_buffer[0]]

        len_buffer = len(self.node_buffer)
        for gg in range(len(self.node_buffer)):
            for hh in range(num_layers):
                new_node_means[hh] += self.node_buffer[gg][hh]

                if hh < (num_layers - 1):
                    for ii in range(len(self.node_buffer[gg][hh])):
                        for jj in range(len(self.node_buffer[gg][hh+1])):
                            self.node_cov[hh][ii,jj] += (\
                                    self.node_buffer[gg][hh][ii] \
                                    - self.node_means[hh][ii])\
                                    * (self.node_buffer[gg][hh+1][jj]\
                                    - self.node_means[hh+1][jj]) \

                    self.node_cov[hh] = softmax(- self.node_cov[hh] / len_buffer)
                                     
        self.node_means = [new_node_mean / len_buffer\
                for new_node_mean in new_node_means]
        
    def cov_mutate_pop(self):
        

        for ll in range(self.pop_size):
            for mm in range(len(self.pop[ll])):
                temp_layer = np.copy(self.pop[ll][mm])
                prunes_per_layer = .1 * temp_layer.shape[0]*temp_layer.shape[1]

                temp_layer *= 1.0 * (np.random.random((temp_layer.shape[0],\
                        temp_layer.shape[1])) > (self.node_cov[mm] \
                        * prunes_per_layer))

                self.pop[ll][mm] = temp_layer


    def mutate_pop(self, rate=0.1):
        # mutate population by 
        
        for jj in range(self.pop_size):
            for kk in range(len(self.pop[jj])):
                temp_layer = np.copy(self.pop[jj][kk])
                
                temp_layer *= np.random.random((temp_layer.shape[0],\
                        temp_layer.shape[1])) > rate

                self.pop[jj][kk] = temp_layer

    def get_fitness(self, env, epds=6, values=[1.0], render=False):
        fitness = []
        complexity = []
        total_steps = 0

        for agent_idx in range(len(self.pop)):
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

            complexity_penalty =  np.mean([np.mean(layer) \
                    for layer in self.pop[agent_idx]])

            fitness.append(accumulated_reward/(epds*len(values))- complexity_penalty)
            #complexity.append(complexity_penalty)
        plt.close("all")


        return fitness, total_steps# , complexity 

    def update_pop(self, fitness, recombine=False):

        # make sure fitnesses aren't equal
        fitness = fitness + np.random.randn(len(fitness),)*1e-6    
        sort_indices = list(np.argsort(fitness))
        sort_indices.reverse()

        sorted_fitness = np.array(fitness)[sort_indices]
        #sorted_pop = self.pop[sort_indices]
        
        connections = []
        for pop_idx in range(self.pop_size):
            connections.append(np.sum([np.sum(layer) \
                    for layer in self.pop[pop_idx]]))
        mean_connections = np.mean(connections)
        std_connections = np.std(connections)

        keep = int(np.ceil(0.125*self.pop_size))
        if sorted_fitness[0] > self.best_agent:
            # keep best agent
            print("new best elite agent: {} v {}".\
                    format(sorted_fitness[0], self.best_agent))
            self.elite_agent = self.pop[sort_indices[0]]
            self.best_agent = sorted_fitness[0]

        if np.mean(sorted_fitness[:keep]) > self.best_gen:
            # keep best elite population
            print("new best elite population: {} v {}".\
                    format(np.mean(sorted_fitness[:keep]), self.best_gen))
            self.best_gen = np.mean(sorted_fitness[:keep])

            self.elite_pop = []
            self.elite_pop.append(self.elite_agent)
            for oo in range(keep):
                self.elite_pop.append(self.pop[sort_indices[oo]])

            this_gens_best = None
        else:
            this_gens_best = []
            for oo in range(keep+1):
                this_gens_best.append(self.pop[sort_indices[oo]])

        self.pop = []
        num_elite = len(self.elite_pop)
        p = np.ones((num_elite)) / num_elite
        a = np.arange(num_elite)
        
        if this_gens_best is not None:
            for pp in range(num_elite):
                #idx = np.random.choice(a,size=1,p=p)[0]
                idx = pp
                self.pop.append(copy.deepcopy(this_gens_best[idx]))
            keep += 1
        else:
            keep = 0
        for qq in range(keep, self.pop_size):
            #idx = np.random.choice(a,size=1,p=p)[0]
            idx = qq % num_elite
            self.pop.append(copy.deepcopy(self.elite_pop[idx]))


        return sorted_fitness, num_elite, \
                mean_connections, std_connections

    def init_pop(self):
        # represent population as a list of lists of np arrays
        self.pop = []

        for jj in range(self.pop_size):
            layers = []
            layer = np.ones((self.input_dim, self.hid[0]))

            layers.append(layer)
            for kk in range(1,len(self.hid)):
                    layer = np.ones((self.hid[kk-1], self.hid[kk]))
                    layers.append(layer)
            layer = np.ones((self.hid[-1], self.output_dim)) 
            layers.append(layer)
            self.pop.append(layers)


if __name__ == "__main__":

    save_every = 100
    population_size = 256
    max_generations = 2048
    min_generations = 100
    thresh_connections = 1000
    thresh_performance = {"Walker2DBulletEnv-v0": 3000,\
                        "AntBulletEnv-v0": 3000,\
                        "InvertedDoublePendulumBulletEnv-v0": 800,\
                        "InvertedPendulumBulletEnv-v0": 800,\
                        "CartPoleSwingUp-v0": 500,\
                        "CartPole-v0": 199,\
                        "Acrobot-v1": -85,\
                        "Pendulum-v0": -136,\
                        "BipedalWalker-v2": 300,\
                        "LunarLander-v2": 200}
    mutate_rate = 0.01

    env_names = [\
            "Walker2DBulletEnv-v0",\
            "InvertedPendulumBulletEnv-v0",\
            "InvertedDoublePendulumBulletEnv-v0",\
            "AntBulletEnv-v0",\
            "CartPoleSwingUp-v0",\
            "LunarLander-v2",\
            "BipedalWalker-v2"]
    pop_size = {"Walker2DBulletEnv-v0": 128,\
            "AntBulletEnv-v0": 64,
            "InvertedDoublePendulumBulletEnv-v0": 128,\
            "InvertedPendulumBulletEnv-v0": 32,\
            "CartPole-v0": 128,\
            "CartPoleSwingUp-v0": 1024,\
            "Pendulum-v0": 512,\
            "LunarLander-v2": 512,\
            "BipedalWalker-v2": 512}

    #env_names = [c for c in reversed(env_names)]

    exp_id = "exp" + str(int(time.time()))[4:]

    res_dir = os.listdir("./results/")
    model_dir = os.listdir("./models/")
    if exp_id not in res_dir:
        os.mkdir("./results/"+exp_id)
    if exp_id not in model_dir:
        os.mkdir("./models/"+exp_id)

    render = False
    for my_seed in [2,1,0]:
        np.random.seed(my_seed)
        for env_name in env_names:
            try:
                del(results)
                del(fitness)
                del(sorted_fitness)
                del(population)
            except:
                pass
            
            results = {"generation": [],\
                    "total_env_interacts": [],\
                    "best_agent_fitness": [],\
                    "pop_mean_fit": [],\
                    "pop_std_fit": [],\
                    "pop_max_fit": [],\
                    "pop_min_fit": [],\
                    "mean_agent_sum": [],\
                    "std_agent_sum": [],\
                    "elite_mean_fit": [],\
                    "elite_std_fit": [],\
                    "elite_min_fit": [],\
                    "elite_max_fit": [],\
                    "elite_agent_sum": []}

            env = gym.make(env_name)

            #env._max_episode_steps = 200 #max_env_steps[env_name]

            obs_dim = env.observation_space.shape[0]

            try:
                act_dim = env.action_space.n
                discrete = True
            except:
                act_dim = env.action_space.sample().shape[0]
                discrete = False

            population_size = pop_size[env_name]
            population = PruneableAgent(obs_dim, act_dim, hid=[16,16], \
                    pop_size=population_size, discrete=discrete)

            total_total_steps = 0
            for generation in range(max_generations):
                #if generation % 100 == 0: 
                #    render = True
                #else:
                #    render = False

                population.init_node_buffer()
                fitness, total_steps = population.get_fitness(env, render=render)
                total_total_steps += total_steps


                sorted_fitness, num_elite, \
                        mean_connections, std_connections = \
                        population.update_pop(fitness, recombine=False)

                connections = np.sum([np.sum(layer) for \
                        layer in population.elite_agent])

                population.get_node_cov()
                population.cov_mutate_pop()

                results["generation"].append(generation)
                results["total_env_interacts"].append(total_total_steps)
                results["best_agent_fitness"].append(sorted_fitness[0])
                results["pop_mean_fit"].append(np.mean(fitness))
                results["pop_std_fit"].append(np.std(fitness))
                results["pop_max_fit"].append(np.max(fitness))
                results["pop_min_fit"].append(np.min(fitness))
                results["elite_mean_fit"].append(np.mean(\
                        sorted_fitness[:num_elite]))
                results["elite_std_fit"].append(np.std(\
                        sorted_fitness[:num_elite]))
                results["elite_max_fit"].append(np.max(\
                        sorted_fitness[:num_elite]))
                results["elite_min_fit"].append(np.min(\
                        sorted_fitness[:num_elite]))
                results["elite_agent_sum"].append(connections)
                results["mean_agent_sum"].append(mean_connections)
                results["std_agent_sum"].append(std_connections)

                print("generation {}, mutation rate = {}".format(generation, mutate_rate))
                print(" connections in best agent", connections)
                print("fitness stats: mean {:.3f} | std: {:.3f} | max: {:.3f} | min: {:.3f}"\
                        .format(np.mean(fitness), np.std(fitness),\
                        np.max(fitness), np.min(fitness)))

                if generation % save_every == 0:
                    np.save("./results/{}/env{}_{}_s{}.npy".format(\
                            exp_id, env_name[:6]+env_name[-10:-3], exp_id, my_seed), results)
                    np.save("./models/{}/elite_pop{}_env{}_gen{}_s{}.npy".format(\
                            exp_id, exp_id,env_name[:6]+env_name[-10:-3], generation, my_seed), \
                            population.elite_pop)

                if results["elite_mean_fit"][-1] >= thresh_performance[env_name]\
                        and\
                        results["elite_agent_sum"][-1] <= thresh_connections\
                        and \
                        generation >= min_generations:

                    print("environment solved, ending training")
                    break
