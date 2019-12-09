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
        self.mutate_pop(rate=0.25)

    def get_action(self, obs, agent_idx=0, scaler=1.0, enjoy=False):

        x = obs        
        if enjoy: self.hid = [hid.shape[1] for hid in self.pop[agent_idx]][:-1]
        for ii in range(len(self.hid)):
            x = np.matmul(x, scaler*self.pop[agent_idx][ii])
            x = np.tanh(x)
            #x[x<0] = 0 # relu

        if self.discrete:
            x = sigmoid(self.by + np.matmul(x, scaler*self.pop[agent_idx][-1]))
            #x = np.where(x > 0.5, 1, 0) 
        else:
            x = self.by + np.matmul(x, scaler*self.pop[agent_idx][-1])

        if self.discrete:
            x = softmax(x)
            act = np.argmax(x)
        else:
            x = x
            act = np.tanh(x)

        return act

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

            fitness.append(accumulated_reward/(epds*len(values))-complexity_penalty)
            #complexity.append(complexity_penalty)
        plt.close("all")


        return fitness, total_steps# , complexity 

    def update_pop(self, fitness, recombine=False):

        # make sure fitnesses aren't equal
        fitness = fitness
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

        self.pop = []
        num_elite = len(self.elite_pop)
        p = np.arange(num_elite,0,-1) / np.sum(np.arange(num_elite,0,-1))
        a = np.arange(num_elite)
        for pp in range(self.pop_size):
            idx = np.random.choice(a,size=1,p=p)[0]
            self.pop.append(copy.deepcopy(self.elite_pop[idx]))
        

        if(recombine):
            for ll in range(keep,self.pop_size):
                new_layers = []
                for mm in range(len(self.hid)+1):
                    rec_map = np.random.randint(num_elite, size=self.pop[0][mm].shape)
                    new_layer = np.zeros_like(self.elite_pop[0][mm])
                    for nn in range(num_elite):
                        new_layer = np.where(rec_map==nn, self.elite_pop[nn][mm],new_layer)
                    new_layers.append(new_layer + self.mut_noise*np.random.randn(\
                            new_layer.shape[0], new_layer.shape[1]))


                self.pop.append(new_layers)
        elite_connections = 0.0
        for elite_idx in range(num_elite):
            elite_connections += (mean_connections - np.sum([np.sum(layer)\
                    for layer in self.elite_pop[elite_idx]]))**2

        mutation_rate = np.sqrt(elite_connections / num_elite)
        mutation_rate /= mean_connections
        #mutation_rate *= 0.995
        mutation_rate = np.max([np.min([0.125, mutation_rate]), 0.001])
        return sorted_fitness, num_elite, mutation_rate, \
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

    def mutate_pop(self, rate=0.1):
        # mutate population by 
        
        for jj in range(self.pop_size):
            for kk in range(len(self.pop[jj])):
                temp_layer = np.copy(self.pop[jj][kk])
                
                temp_layer *= np.random.random((temp_layer.shape[0],\
                        temp_layer.shape[1])) > rate

                self.pop[jj][kk] = temp_layer

if __name__ == "__main__":

    min_generations = 100
    epds = 3
    save_every = 50

    hid_dim = 64

    env_names = [\
             "Walker2DBulletEnv-v0"]
#            "InvertedPendulumSwingupBulletEnv-v0"]
#            "ReacherBulletEnv-v0",\
#            "HalfCheetahBulletEnv-v0"]

    pop_size = {\
            "InvertedPendulumBulletEnv-v0": 128,\
            "InvertedPendulumSwingupBulletEnv-v0": 256,\
            "HalfCheetahBulletEnv-v0": 256,\
            "ReacherBulletEnv-v0": 128,\
            "Walker2DBulletEnv-v0": 128}

    thresh_performance = {\
            "InvertedPendulumBulletEnv-v0": 999.5,\
            "InvertedPendulumSwingupBulletEnv-v0": 880,\
            "HalfCheetahBulletEnv-v0": 3000,\
            "ReacherBulletEnv-v0": 200,\
            "Walker2DBulletEnv-v0": 3000}
    max_generation = {\
            "InvertedPendulumBulletEnv-v0": 1024,\
            "InvertedPendulumSwingupBulletEnv-v0": 1024,\
            "HalfCheetahBulletEnv-v0": 1024,\
            "ReacherBulletEnv-v0": 1024,\
            "Walker2DBulletEnv-v0": 1024}

    res_dir = os.listdir("./results/")
    model_dir = os.listdir("./models/")

    exp_dir = "prune_mk1_32_exp004"
    exp_time = str(int(time.time()))[-7:]
    if exp_dir not in res_dir:
        os.mkdir("./results/"+exp_dir)
    if exp_dir not in model_dir:
        os.mkdir("./models/"+exp_dir)

    render = False
    for my_seed in [0,1,2]:
        np.random.seed(my_seed)
        for env_name in env_names:
            try:
                del(results)
                del(fitness)
                del(sorted_fitness)
                del(agent)
            except:
                pass
            
            results = {"generation": [],\
                    "total_env_interacts": [],\
                    "wall_time": [],\
                    "prune_prob": [],\
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

            env = gym.make(env_name)

            obs_dim = env.observation_space.shape[0]

            try:
                act_dim = env.action_space.n
                discrete = True
            except:
                act_dim = env.action_space.sample().shape[0]
                discrete = False

            population_size = pop_size[env_name]
            agent = PruneableAgent(obs_dim, act_dim, hid=[hid_dim, hid_dim], \
                    pop_size=population_size, discrete=discrete)

            total_total_steps = 0
            t0 = time.time()
            for generation in range(max_generation[env_name]):
                #if generation % 100 == 0: 
                #    render = True
                #else:
                #    render = False
                #if "Bullet" in env_name:
                #    env._max_episode_steps = np.max([200, agent.best_agent]) #max_env_steps[env_name]

                fitness, total_steps = agent.get_fitness(env, render=render)
                total_total_steps += total_steps

                sorted_fitness, num_elite, mutate_rate, \
                        mean_connections, std_connections = \
                        agent.update_pop(fitness, recombine=False)

                connections = np.sum([np.sum(layer) for \
                        layer in agent.elite_agent])

                agent.mutate_pop(rate=mutate_rate)

                results["generation"].append(generation)
                results["total_env_interacts"].append(total_total_steps)
                results["wall_time"].append(time.time()-t0)
                results["prune_prob"].append(mutate_rate)
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

                print("mk1 gen {} elapsed {:.3f} mut rate {:.3f}, mean/max/min fitness: {:.3f}/{:.3f}/{:.3f}, elite {:.3f}/{:.3f}/{:.3f}, {:.3f}/{:.3f}"\
                        .format(generation, results["wall_time"][-1],\
                        results["prune_prob"][-1],\
                        results["pop_mean_fit"][-1],\
                        results["pop_max_fit"][-1],\
                        results["pop_min_fit"][-1],\
                        results["elite_mean_fit"][-1],\
                        results["elite_max_fit"][-1],\
                        results["elite_min_fit"][-1],\
                        mean_connections, std_connections))

                if generation % save_every == 0:
                    np.save("./results/{}/prunemk1_{}.npy"\
                            .format(exp_dir, exp_id),results)
                    np.save("./models/{}/prunemk1_elite_pop_{}_gen{}.npy"\
                            .format(exp_dir,exp_id, generation),agent.elite_pop)

                    if results["elite_max_fit"][-1] >= \
                            thresh_performance[env_name]\
                            and\
                            generation >= min_generations:

                        print("environment solved, ending training")
                        break
