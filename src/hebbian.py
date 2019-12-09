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

class HebbianAgent():
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
            self.hebbian[agent_idx][ii] += np.matmul(x0[np.newaxis,:].T, x[np.newaxis,:])

        if self.discrete and self.output_dim == 1:
            x0 = x
            x = sigmoid(np.matmul(x0, scaler*self.population[agent_idx][-1]))
            #x = np.where(x > 0.5, 1, 0) 

        if self.discrete and self.output_dim != 1:
            x = softmax(x)
            act = np.argmax(x)
        else:
            x0 = x
            x = np.matmul(x0, scaler*self.population[agent_idx][-1])
            act = np.tanh(x)

        # compute Hebbian memory
        self.hebbian[agent_idx][-1] += np.matmul(x0[np.newaxis,:].T, x[np.newaxis,:])

        return act

    def hebbian_prune(self, prune_rate=0.05):

        for ll in range(self.pop_size):
            for mm in range(len(self.population[ll])):
                temp_layer = np.copy(self.population[ll][mm])

                prunes_per_layer = prune_rate * temp_layer.shape[0]*temp_layer.shape[1]

                temp_layer *= 1.0 * (np.random.random((temp_layer.shape[0],\
                        temp_layer.shape[1])) > (softmax(self.hebbian[ll][mm]) \
                        * prunes_per_layer))

                self.hebbian[ll][mm] *= 0

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
            self.population.append(new_layers)
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

    def mutate_pop(self, rate=0.1):
        # mutate population by 
        
        for jj in range(self.pop_size):
            for kk in range(len(self.population[jj])):
                temp_layer = np.copy(self.population[jj][kk])
                
                temp_layer *= np.random.random((temp_layer.shape[0],\
                        temp_layer.shape[1])) > rate

                self.population[jj][kk] = temp_layer

if __name__ == "__main__":

    min_generations = 10
    epds = 4
    save_every = 50

    hid_dims = {\
            "InvertedPendulumBulletEnv-v0": [32],\
            "InvertedPendulumSwingupBulletEnv-v0": [32],\
            "InvertedDoublePendulumBulletEnv-v0": [16,16],\
            "HopperBulletEnv-v0": [32,32]}

    env_names = [\
            "InvertedPendulumBulletEnv-v0",\
            "InvertedPendulumSwingupBulletEnv-v0",\
            "InvertedDoublePendulumBulletEnv-v0",\
            "HopperBulletEnv-v0"]
#             "Walker2DBulletEnv-v0"]
#            "InvertedPendulumSwingupBulletEnv-v0"]
#            "ReacherBulletEnv-v0",\
#            "HalfCheetahBulletEnv-v0"]

    pop_size = {\
            "InvertedDoublePendulumBulletEnv-v0": 128,\
            "InvertedPendulumBulletEnv-v0": 128,\
            "InvertedPendulumSwingupBulletEnv-v0": 256,\
            "HalfCheetahBulletEnv-v0": 256,\
            "HopperBulletEnv-v0": 256,\
            "ReacherBulletEnv-v0": 128,\
            "Walker2DBulletEnv-v0": 128}

    thresh_performance = {\
            "InvertedDoublePendulumBulletEnv-v0": 999,\
            "InvertedPendulumBulletEnv-v0": 999.5,\
            "InvertedPendulumSwingupBulletEnv-v0": 880,\
            "HalfCheetahBulletEnv-v0": 3000,\
            "HopperBulletEnv-v0": 3000,\
            "ReacherBulletEnv-v0": 200,\
            "Walker2DBulletEnv-v0": 3000}
    max_generation = {\
            "InvertedDoublePendulumBulletEnv-v0": 1024,\
            "InvertedPendulumBulletEnv-v0": 1024,\
            "InvertedPendulumSwingupBulletEnv-v0": 1024,\
            "HalfCheetahBulletEnv-v0": 1024,\
            "HopperBulletEnv-v0": 1024,\
            "ReacherBulletEnv-v0": 1024,\
            "Walker2DBulletEnv-v0": 1024}

    res_dir = os.listdir("./results/")
    model_dir = os.listdir("./models/")

    exp_dir = "exp005"
    exp_time = str(int(time.time()))[-7:]
    if exp_dir not in res_dir:
        os.mkdir("./results/"+exp_dir)
    if exp_dir not in model_dir:
        os.mkdir("./models/"+exp_dir)

    render = False
    smooth_fit = 0.0
    alpha = 0.5
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

            hid_dim = hid_dims[env_name]  

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
            agent = HebbianAgent(obs_dim, act_dim, hid=hid_dim, \
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

                sorted_fitness, num_elite, \
                        mean_connections, std_connections = \
                        agent.update_pop(fitness, recombine=False)

                connections = np.sum([np.sum(layer) for \
                        layer in agent.elite_agent])

                agent.hebbian_prune(prune_rate=0.05)

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
                print("hebbian gen {} elapsed {:.2f},  mean/max/min fitness: {:.1f}/{:.1f}/{:.1f}, elite {:.1f}/{:.1f}/{:.1f}/{:.1f}, {:.2f}/{:.2f}"\
                        .format(generation, results["wall_time"][-1],\
                        results["pop_mean_fit"][-1],\ 
                        results["pop_max_fit"][-1],\
                        results["pop_min_fit"][-1],\
                        smooth_fit,\
                        results["elite_mean_fit"][-1],\
                        results["elite_max_fit"][-1],\
                        results["elite_min_fit"][-1],\
                        mean_connections, std_connections))

                if generation % save_every == 0:
                    np.save("./results/{}/hebbian_{}.npy"\
                            .format(exp_dir, exp_id),results)
                    np.save("./models/{}/hebbian_elite_pop_{}_gen{}.npy"\
                            .format(exp_dir,exp_id, generation), agent.elite_pop)


                if smooth_fit >= \
                        thresh_performance[env_name]\
                        and\
                        generation >= min_generations:

                    np.save("./results/{}/hebbian_{}.npy"\
                            .format(exp_dir, exp_id),results)
                    np.save("./models/{}/hebbian_elite_pop_{}_gen{}.npy"\
                            .format(exp_dir,exp_id, generation), agent.elite_pop)
                    print("environment solved, ending training")
                    break
