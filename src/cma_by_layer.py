import numpy as np
import gym 
import matplotlib.pyplot as plt
import copy
import time
import os

import pybullet
import pybullet_envs
from pybullet_envs.bullet import MinitaurBulletEnv

def sinc(x):
    return np.where(x == 0, 1.0, np.sin(x) / (1e-3+x))

class CMALayerAgent():
    def __init__(self, obs_dim, act_dim, population_size, \
            seed=0, hid_dim=[16], discrete=False):

        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.hid_dim = hid_dim
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

        for ii in range(len(self.hid_dim)):
            x = np.matmul(x, self.population[agent_idx][ii])
            x = np.tanh(x-1)

        if self.discrete:
            x = (self.by + np.matmul(x, self.population[agent_idx][-1]))
            #x = np.where(x > 0.5, 1, 0) 
        else:
            x = self.by + np.matmul(x, self.population[agent_idx][-1])

        if self.discrete:
            x = softmax(x)
            act = np.argmax(x)
        else:
            x = x
            act = np.tanh(x)

        return act

    def get_fitness(self, env, epds=2, render=False):
        fitness = []
        complexity = []
        total_steps = 0

        for agent_idx in range(len(self.population)):
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
        #sorted_pop = self.population[sort_indices]

        connections = []
        for pop_idx in range(self.population_size):
            connections.append(np.sum([np.sum(np.abs(layer)) \
                    for layer in self.population[pop_idx]]))
        mean_connections = np.mean(connections)
        std_connections = np.std(connections)

        keep = int(np.ceil(0.125*self.population_size))
        if sorted_fitness[0] > self.best_agent:
            # keep best agent
            print("new best elite agent: {:.2f} v {:.2f}".\
                    format(sorted_fitness[0], self.best_agent))
            self.elite_agent = self.population[sort_indices[0]]
            self.best_agent = sorted_fitness[0]

        if np.mean(sorted_fitness[:keep]) > self.best_gen:
            # keep best elite population
            print("new best elite population: {:.2f} v {:.2f}".\
                    format(np.mean(sorted_fitness[:keep]), self.best_gen))
            self.best_gen = np.mean(sorted_fitness[:keep])

        self.elite_pop = []
        self.elite_pop.append(self.elite_agent)
        for oo in range(keep):
            self.elite_pop.append(self.population[sort_indices[oo]])

        self.population = []
        num_elite = len(self.elite_pop)

        # update parameters
        step_size = 1.0
        
        for hh in range(len(self.layer_dist)-1):            
            if hh == 0:
                i_range = self.obs_dim * self.hid_dim[0]
                sum_layer = np.zeros( self.obs_dim * self.hid_dim[0])
                sum_cov = np.zeros(( self.obs_dim * self.hid_dim[0] ,\
                    self.obs_dim * self.hid_dim[-1] ))
            elif hh == (len(self.hid_dim)):
                i_range = self.hid_dim[-1] * self.act_dim
                sum_layer = np.zeros( self.act_dim * self.hid_dim[-1] )
                sum_cov = np.zeros(( self.act_dim * self.hid_dim[-1] ,\
                    self.act_dim * self.hid_dim[-1] ))
            else:
                i_range = self.hid_dim[hh] * self.hid_dim[hh-1]
                sum_layer = np.zeros( self.hid_dim[hh] * self.hid_dim[hh-1] )
                sum_cov = np.zeros(( self.hid_dim[hh]  * self.hid_dim[hh-1] ,\
                    self.hid_dim[hh] * self.hid_dim[hh-1]) )
            for gg in range(num_elite):
                

                sum_layer += self.elite_pop[gg][hh].ravel()
                
                temp_x0 = self.elite_pop[gg][hh].ravel() - self.layer_dist[hh][0]
                sum_cov += np.matmul(temp_x0[np.newaxis,:].T,\
                        temp_x0[np.newaxis,:])

            mean_layer = sum_layer / num_elite
            mean_cov = sum_cov / num_elite

            self.layer_dist[hh] = [mean_layer, mean_cov]

        self.init_pop()
        self.population[-1] = self.elite_agent

        return sorted_fitness, num_elite,\
                mean_connections, std_connections

    def init_dist(self):
        self.layer_dist = []
    
        # input to hidden layer distribution

        layer_mew = np.zeros((self.obs_dim * self.hid_dim[0]))
        layer_cov = np.eye(self.obs_dim * self.hid_dim[0]) 

        self.layer_dist.append([layer_mew, layer_cov])

        # hidden layers distribution
        for ii in range(len(self.hid_dim)-1):
            layer_mew = np.zeros((self.hid_dim[ii] * self.hid_dim[ii+1]))
            layer_cov = np.eye(self.hid_dim[ii] * self.hid_dim[ii+1])

            self.layer_dist.append([layer_mew, layer_cov])

        #output layer
        layer_mew = np.zeros((self.hid_dim[-1] * self.act_dim))
        layer_cov = np.eye(self.hid_dim[-1] * self.act_dim)
        self.layer_dist.append([layer_mew, layer_cov])

    def init_pop(self):
        self.population = []

        for hh in range(self.population_size):
            layers = []
            layer = np.random.multivariate_normal(self.layer_dist[0][0], \
                    self.layer_dist[0][1]).reshape(self.obs_dim, self.hid_dim[0])

            layers.append(layer)

            for ii in range(1,len(self.hid_dim)):
                layer = np.random.multivariate_normal(self.layer_dist[ii][0], \
                        self.layer_dist[ii][1]).reshape(self.hid_dim[ii-1], \
                        self.hid_dim[ii])

                layers.append(layer)


            layer = np.random.multivariate_normal(self.layer_dist[-1][0], \
                    self.layer_dist[-1][1]).reshape(self.hid_dim[-1], self.act_dim)

            layers.append(layer)
            self.population.append(layers)



if __name__ == "__main__":

    min_generations = 10
    epds = 4 
    save_every = 50

    hid_dims = {\
            "InvertedPendulumBulletEnv-v0": [8],\
            "InvertedPendulumSwingupBulletEnv-v0": [16],\
            "InvertedDoublePendulumBulletEnv-v0": [16,16],\
            "HalfCheetahBulletEnv-v0": [8,8],\
            "ReacherBulletEnv-v0": [16,16],\
            "Walker2DBulletEnv-v0": [16,16],\
            "HopperBulletEnv-v0": [32,32,32]}

    env_names = [\
            "HalfCheetahBulletEnv-v0"\
            ]


    pop_size = {\
            "InvertedDoublePendulumBulletEnv-v0": 128,\
            "InvertedPendulumBulletEnv-v0": 128,\
            "InvertedPendulumSwingupBulletEnv-v0": 128,\
            "HalfCheetahBulletEnv-v0": 96,\
            "HopperBulletEnv-v0": 96,\
            "ReacherBulletEnv-v0": 128,\
            "Walker2DBulletEnv-v0": 256}

    thresh_performance = {\
            "InvertedDoublePendulumBulletEnv-v0": 1999,\
            "InvertedPendulumBulletEnv-v0": 999.5,\
            "InvertedPendulumSwingupBulletEnv-v0": 880,\
            "HalfCheetahBulletEnv-v0": 2100,\
            "HopperBulletEnv-v0": 3000,\
            "ReacherBulletEnv-v0": 200,\
            "Walker2DBulletEnv-v0": 2995}

    max_generation = {\
            "InvertedDoublePendulumBulletEnv-v0": 1024,\
            "InvertedPendulumBulletEnv-v0": 1024,\
            "InvertedPendulumSwingupBulletEnv-v0": 1024,\
            "HalfCheetahBulletEnv-v0": 1024,\
            "HopperBulletEnv-v0": 1024,\
            "ReacherBulletEnv-v0": 1024,\
            "Walker2DBulletEnv-v0": 2048}

    res_dir = os.listdir("./results/")
    model_dir = os.listdir("./models/")

    exp_dir = "exp007"
    exp_time = str(int(time.time()))[-7:]
    if exp_dir not in res_dir:
        os.mkdir("./results/"+exp_dir)
    if exp_dir not in model_dir:
        os.mkdir("./models/"+exp_dir)

    smooth_fit = 0.0
    alpha = 0.5
    for my_seed in [2,1,0]:
        np.random.seed(my_seed)
        for env_name in env_names:
            hid_dim = hid_dims[env_name]



            results = {"generation": [],\
                    "total_env_interacts": [],\
                    "wall_time": [],\
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

            exp_id = "exp_" + exp_time + "env_" +\
                    env_name + "_s" + str(my_seed)

            # build env and agent population
            if type(env_name) == str:
                env = gym.make(env_name)
                render = False
            else:
                env = env_name(render=False)

            obs_dim = env.observation_space.shape[0]
            try:
                act_dim = env.action_space.n
                discrete = True
            except:
                act_dim = env.action_space.sample().shape[0]
                discrete = False

            print("make env", env_name)
            population_size = pop_size[env_name]
            agent = CMALayerAgent(obs_dim, act_dim,\
                    population_size, hid_dim=hid_dim, discrete=discrete)

            t0 = time.time()
            total_total_steps = 0
            for generation in range(max_generation[env_name]):


                fitness, total_steps = agent.get_fitness(env, render=render,\
                        epds=epds)
                total_total_steps += total_steps
                sorted_fitness, num_elite,\
                        mean_connections, std_connections\
                        = agent.update_pop(fitness)

                connections = np.sum([np.sum(np.abs(layer)) for \
                        layer in agent.elite_agent])

                results["generation"].append(generation)
                results["total_env_interacts"].append(total_total_steps)
                results["wall_time"].append(time.time()-t0)
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

                smooth_fit = alpha * smooth_fit + ( 1-alpha ) * results["elite_max_fit"][-1]
                print("cma bl gen {} elapsed {:.1f}, mean/max/min fitness: {:.1f}/{:.1f}/{:.1f} elite mean/max/min {:.1f}{:.1f}/{:.1f}/{:.1f}"\
                        .format(generation, results["wall_time"][-1],\
                        results["pop_mean_fit"][-1],\
                        results["pop_max_fit"][-1],\
                        results["pop_min_fit"][-1],\
                        results["elite_mean_fit"][-1],\
                        smooth_fit,\
                        results["elite_max_fit"][-1],\
                        results["elite_min_fit"][-1]))

                if generation % save_every == 0:
                    np.save("./results/{}/cma_bl_{}.npy"\
                            .format(exp_dir, exp_id),results)
                    np.save("./models/{}/cma_bl_{}_gen{}.npy"\
                            .format(exp_dir,exp_id,generation),agent.elite_pop)

                if smooth_fit >= \
                        thresh_performance[env_name]\
                        and\
                        generation >= min_generations:

                    np.save("./results/{}/cma_bl_{}.npy"\
                            .format(exp_dir, exp_id),results)
                    np.save("./models/{}/cma_bl_{}_gen{}.npy"\
                            .format(exp_dir,exp_id, generation), agent.elite_pop)
                    print("environment solved, ending training")
                    break
