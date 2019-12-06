import numpy as np
import gym 
import matplotlib.pyplot as plt
import copy
import time
import os

import pybullet
import pybullet_envs
from pybullet_envs.bullet import MinitaurBulletEnv

class CMAAgent():
    def __init__(self, obs_dim, act_dim, population_size, \
            seed=0, hid_dim=16, discrete=False):

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
            x = np.tanh(x)

        if self.discrete:
            x = sigmoid( np.matmul(x, self.population[agent_idx][-1]))
            #x = np.where(x > 0.5, 1, 0) 
        else:
            x =  np.matmul(x, self.population[agent_idx][-1])

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
            print("new best elite agent: {} v {}".\
                    format(sorted_fitness[0], self.best_agent))
            self.elite_agent = self.population[sort_indices[0]]
            self.best_agent = sorted_fitness[0]

        if np.mean(sorted_fitness[:keep]) > -float("Inf"): # self.best_gen:
            # keep best elite population
            #print("new best elite population: {} v {}".\
            #        format(np.mean(sorted_fitness[:keep]), self.best_gen))
            self.best_gen = np.mean(sorted_fitness[:keep])

            self.elite_pop = []
            self.elite_pop.append(self.elite_agent)
            for oo in range(keep):
                self.elite_pop.append(self.population[sort_indices[oo]])

        num_elite = len(self.elite_pop)

        # update parameters
        step_size = 1.0
        
        for gg in range(self.population_size):
            for hh in range(len(self.hid_dim)+1):            
                if hh == 0:
                    temp_params = self.population[gg][hh].ravel()[np.newaxis,:]
                else:
                    temp_params = np.append(temp_params, self.population[gg][hh].ravel()\
                        [np.newaxis,:], axis=1 )

            if gg == 0:
                params = temp_params
            else:
                params = np.append( params, temp_params, axis=0 )

        for gg in range(num_elite):
            for hh in range(len(self.hid_dim)+1):            
                if hh == 0:
                    temp_params = self.population[gg][hh].ravel()[np.newaxis,:]
                else:
                    temp_params = np.append( temp_params, self.population[gg][hh].ravel()\
                        [np.newaxis,:], axis=1 )

            if gg == 0:
                elite_params = temp_params
            else:
                elite_params = np.append( elite_params, temp_params, axis=0 )

        mean_params = np.mean(params, axis=0)
        mean_cov = np.matmul(( elite_params - self.dist[0]).T,\
                (elite_params-self.dist[0]) )

        self.dist = [mean_params, mean_cov]

        self.init_pop()
        self.population[-1] = self.elite_agent

        return sorted_fitness, num_elite,\
                mean_connections, std_connections

    def init_dist(self):

        num_params = self.obs_dim * self.hid_dim[0]
        num_params += int(np.sum([self.hid_dim[ii] * self.hid_dim[ii+1] \
                for ii in range(len(self.hid_dim)-1)]))
        num_params += self.act_dim * self.hid_dim[-1]

        if num_params >= 500:
            print("Warning: You got a lot of params ({}), CMA will be slow"\
                    .format(num_params))

        dist_mean = np.zeros((num_params))
        dist_cov = np.eye(num_params)
        self.dist = [dist_mean, dist_cov]

    def init_pop(self):

        self.population = []

        for hh in range(self.population_size):
            params = np.random.multivariate_normal(self.dist[0], self.dist[1])

            layers = []
            for layer_idx in range(len(self.hid_dim)+1):
                
                if layer_idx == 0:
                    start_idx = 0
                    dim_x = self.obs_dim 
                    dim_y = self.hid_dim[0]
                elif layer_idx == len(self.hid_dim):
                    start_idx = end_idx
                    dim_x = self.hid_dim[-1]
                    dim_y = self.act_dim
                else:
                    start_idx = end_idx
                    dim_x = self.hid_dim[layer_idx-1]
                    dim_y = self.hid_dim[layer_idx]
                end_idx = start_idx+ dim_x * dim_y

                layer = params[start_idx:end_idx].reshape(dim_x,dim_y)
                    
                layers.append(layer)

            self.population.append(layers)


if __name__ == "__main__":

    min_generations = 100
    epds = 4
    save_every = 50
    hid_dim = [64]

    env_names = [\
            "InvertedPendulumSwingupBulletEnv-v0"]
#            "InvertedDoublePendulumBulletEnv-v0"]
#             "Walker2DBulletEnv-v0"]
#            "InvertedPendulumSwingupBulletEnv-v0"]
#            "ReacherBulletEnv-v0",\
#            "HalfCheetahBulletEnv-v0"]

    pop_size = {\
            "InvertedDoublePendulumBulletEnv-v0": 128,\
            "InvertedPendulumBulletEnv-v0": 128,\
            "InvertedPendulumSwingupBulletEnv-v0": 256,\
            "HalfCheetahBulletEnv-v0": 256,\
            "ReacherBulletEnv-v0": 128,\
            "Walker2DBulletEnv-v0": 128}

    thresh_performance = {\
            "InvertedDoublePendulumBulletEnv-v0": 1999.0,\
            "InvertedPendulumBulletEnv-v0": 999.5,\
            "InvertedPendulumSwingupBulletEnv-v0": 880,\
            "HalfCheetahBulletEnv-v0": 3000,\
            "ReacherBulletEnv-v0": 200,\
            "Walker2DBulletEnv-v0": 3000}
    max_generation = {\
            "InvertedDoublePendulumBulletEnv-v0": 1024,\
            "InvertedPendulumBulletEnv-v0": 1024,\
            "InvertedPendulumSwingupBulletEnv-v0": 1024,\
            "HalfCheetahBulletEnv-v0": 1024,\
            "ReacherBulletEnv-v0": 1024,\
            "Walker2DBulletEnv-v0": 1024}

    res_dir = os.listdir("./results/")
    model_dir = os.listdir("./models/")

    exp_dir = "cma_32_exp004"
    exp_time = str(int(time.time()))[-7:]
    if exp_dir not in res_dir:
        os.mkdir("./results/"+exp_dir)
    if exp_dir not in model_dir:
        os.mkdir("./models/"+exp_dir)

    for my_seed in [2,1,0]:
        np.random.seed(my_seed)
        for env_name in env_names:

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

            population_size = pop_size[env_name]
            agent = CMAAgent(obs_dim, act_dim,\
                    population_size, hid_dim=hid_dim, discrete=discrete)

            t0 = time.time()
            total_total_steps = 0
            for generation in range(max_generation[env_name]):


                t1 = time.time()
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

                print("gen {} elapsed {:.1f}/{:.1f}, mean/max/min fitness: {:.3f}/{:.3f}/{:.3f} elite mean/max/min {:.3f}/{:.3f}/{:.3f}"\
                        .format(generation, time.time()-t1,\
                        results["wall_time"][-1],\
                        results["pop_mean_fit"][-1],\
                        results["pop_max_fit"][-1],\
                        results["pop_min_fit"][-1],\
                        results["elite_mean_fit"][-1],\
                        results["elite_max_fit"][-1],\
                        results["elite_min_fit"][-1]))

                if generation % save_every == 0:
                    np.save("./results/{}/cma_{}.npy"\
                            .format(exp_dir, exp_id),results)
                    np.save("./models/{}/cma_elite_pop_{}_gen{}.npy"\
                            .format(exp_dir,exp_id,generation),agent.elite_pop)

                    if results["elite_max_fit"][-1] >= \
                            thresh_performance[env_name]\
                            and\
                            generation >= min_generations:

                        print("environment solved, ending training")
                        break

