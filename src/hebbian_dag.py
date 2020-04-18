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

def sinc(x):
    return np.where(x == 0, 1.0, np.sin(x) / (1e-3+x))

def softmax(x):
    x = x - np.max(x)

    y = np.exp(x) / np.sum(np.exp(x))

    return y

class HebbianDAG():

    def __init__(self, input_dim, act_dim, hid_dim=[32,32],\
            population_size=10, seed=0, discrete=True, random_init=True, neuromodulate=False):

        self.neuromodulate = neuromodulate
        self.random_init = random_init
        self.input_dim = input_dim
        self.output_dim = act_dim + 1 #output_dim
        self.hid_dim = hid_dim
        
        self.hid_dim.append(self.output_dim)
        self.hid_dim.insert(0,self.input_dim)

        self.population_size = population_size
        self.seed = seed
        self.by = -0.00
        self.mut_noise = 1e0
        self.discrete = discrete
        self.best_gen = -float("Inf")
        self.best_agent = -float("Inf")
        self.keep = int(self.population_size/8)
        self.elite_pop = []
        self.leaderboard = [-float("Inf")] * self.keep
        np.random.seed(self.seed)

        self.init_pop()
        self.mutate_pop(rate=0.025)
        

    def get_action(self, obs, agent_idx=0, scaler=1.0, enjoy=False, \
            get_hebs=False, neuromodulate=True):

        x = obs        
        xs = [x]

        neuromodulate = self.neuromodulate

        if get_hebs: hebs = []
        for ii in range(len(self.hid_dim)-1):
            x = np.zeros((self.hid_dim[ii+1]))
            for jj in range(ii+1):

                if neuromodulate:
                    x_temp = np.matmul(xs[jj], self.population[agent_idx][ii][jj]\
                            + self.hebbian[agent_idx][ii][jj] * self.modulation[agent_idx])
                else:
                    x_temp = np.matmul(xs[jj], self.population[agent_idx][ii][jj])

                self.hebbian[agent_idx][ii][jj] += \
                        self.remember[agent_idx] * \
                        np.matmul(xs[jj][np.newaxis,:].T, x_temp[np.newaxis,:])
                        
                #self.hebbian[agent_idx][ii][jj] = np.clip(self.hebbian[agent_idx][ii][jj],
                #        -10.0, 10.0)

                if get_hebs: 
                    hebs.append(self.remember[agent_idx] * np.abs(np.matmul(xs[jj][np.newaxis,:].T,\
                            x_temp[np.newaxis,:])))
                x += x_temp


            if ii < len(self.hid_dim):
                x = np.tanh(x)

            if self.neuromodulate:
                if ii == len(self.hid_dim)-2:
                    self.modulation[agent_idx] = 0.0 #x[0] #0.0 #np.clip(x[0],-0.001, 0.001)
                    self.remember[agent_idx] = x[1]

            xs.append(x)
            #x[x<0] = 0 # relu

#
#        for ii in range(len(self.hid_dim)-1):
#           x = np.zeros((self.hid_dim[ii+1]))
#            for jj in range(ii+1):
#                
#
#                x += np.matmul(xs[jj], self.population[agent_idx][ii][jj])
#
#            if get_hebs: 
#                hebs.append(np.matmul(xs[jj][np.newaxis,:].T,\
#                        x[np.newaxis,:]))
#
#            x = np.tanh(x-1)
#            #x = sinc(x)
#            #x = np.sin(x)
#
#            xs.append(x)
#            #x[x<0] = 0 # relu
        if self.discrete:
            x = softmax(x)
            act = np.argmax(x)
        else:
            #x = x
            act = np.tanh(x)

        act = x[1:]
        if get_hebs:
            return act, hebs
        else:
            return act

    def get_fitness(self, env, epds=6, values=[1.0], render=False):
        fitness = []
        complexity = []
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

            #complexity_penalty = 0* np.mean([np.mean(layer) \
            #        for layer in self.population[agent_idx]])

            fitness.append(accumulated_reward/(epds*len(values)))
            #complexity.append(complexity_penalty)
        plt.close("all")


        return [fitness, total_steps] # , complexity 

    def get_trajectory(self, num_steps):

        total_steps = 0
        ll_rew = []
        ll_done = []
        ll_hebs = []

        for agent_idx in range(len(self.population)):
            #obs = flatten_obs(env.reset())
            accumulated_reward = 0.0

            l_rew = []
            l_done = []
            l_hebs = []
            step = 0
            while step < num_steps:

                obs = env.reset()
                done=False
                while not done:

                    action, hebs = self.get_action(obs, agent_idx=agent_idx,\
                            get_hebs=True) 
                    obs, reward, done, info = env.step(action)
                    step += 1
                    l_rew.append(reward)
                    l_done.append(done*1.0)
                    l_hebs.append(hebs)
                render = False 
            total_steps += step
            
            ll_rew.append(l_rew)
            ll_done.append(l_done)
            ll_hebs.append(l_hebs)


        return ll_rew, ll_done, ll_hebs  #ll_obs, ll_next_obs 

    def get_advantage(self, l_rew, l_done, gamma=0.9):
        epd_dones = []
        for gg in range(len(l_done)):
            if l_done[gg]:
                epd_dones.append(gg)

        rewards = []
        for hh in range(len(epd_dones)):

            if hh == 0:
                last_done = 0
                epd_done = epd_dones[hh]
            elif hh == len(epd_dones)-1:  
                epd_done = epd_dones[hh] + 1


            epd_rewards = [np.sum([gamma**ii * l_rew[last_done+ii+jj]\
                for ii in range(len(l_rew[last_done+jj:epd_done]))] ) \
                for jj in range(len(l_rew[last_done:epd_done]))]

            last_done = epd_done
        
            rewards.extend(epd_rewards)
                
        advantage = rewards - np.mean(rewards)
        
        return advantage

    def update_pop(self, fitness):

        # make sure fitnesses aren't equal
        fitness = fitness
        sort_indices = list(np.argsort(fitness))
        sort_indices.reverse()

        sorted_fitness = np.array(fitness)[sort_indices]
        #sorted_pop = self.population[sort_indices]
        
        connections = []

        for pop_idx in range(self.population_size):
            connections.append(np.sum([[np.sum(layer) \
                    for layer in layer_layer] \
                    for layer_layer in self.population[pop_idx]]))

        mean_connections = np.mean(connections)
        std_connections = np.std(connections)

        keep = int(np.ceil(0.125*self.population_size))
        if sorted_fitness[0] >= self.best_agent:
            # keep best agent
            print("best elite agent: {} v {}".\
                    format(sorted_fitness[0], self.best_agent))
            self.elite_agent = self.population[sort_indices[0]]
            self.best_agent = sorted_fitness[0]

        if np.mean(sorted_fitness[:keep]) > self.best_gen:
            # keep best elite population
            print("new best elite population: {} v {}".\
                    format(np.mean(sorted_fitness[:keep]), self.best_gen))
            self.best_gen = np.mean(sorted_fitness[:keep])

        lb_idx = 0 
        fit_idx = 0
        fitness_copy = list(copy.deepcopy(sorted_fitness))

        while fitness_copy[0] >= self.leaderboard[-1]:
            
            if fitness_copy[0] >= self.leaderboard[lb_idx]:
                self.leaderboard.insert(lb_idx, fitness_copy[0])
                self.elite_pop.insert(lb_idx, self.population[sort_indices[fit_idx]])
                fitness_copy.pop(0) 
                fit_idx += 1
            lb_idx += 1

            if lb_idx > keep:
                break
        print("added {} agents to leaderboard".format(fit_idx))
        self.leaderboard = self.leaderboard[:keep]
        self.elite_pop = self.elite_pop[:keep]

        #self.elite_pop = []
        self.elite_pop.append(self.elite_agent)

        for oo in range(keep):
            self.elite_pop.insert(2*keep, self.population[sort_indices[oo]])

        self.population = []
        num_elite = len(self.elite_pop)
        p = np.arange(num_elite,0,-1) / np.sum(np.arange(num_elite,0,-1))
        a = np.arange(num_elite)
        for pp in range(self.population_size):
            idx = np.random.choice(a,size=1,p=p)[0]
            self.population.append(copy.deepcopy(self.elite_pop[idx]))


        
        recombine = True
        num_swaps = keep

        if(recombine):
            for ll in range(num_swaps):
                agent_0 = np.random.randint(self.population_size)
                agent_1 = np.random.randint(self.population_size)
                

                el0 = np.random.randint(len(self.population[0]))
                
                # either swap a full segment or single connection matrix
                swap_full = np.random.randint(2)
                if swap_full:
                    self.population[agent_0][el0], self.population[agent_1][el0] \
                            = self.population[agent_1][el0], self.population[agent_0][el0]
                else:
                    el1 = np.random.randint(len(self.population[0][el0]))
                    self.population[agent_0][el0][el1], self.population[agent_1][el0][el1] \
                            = self.population[agent_1][el0][el1], self.population[agent_0][el0][el1]

        elite_connections = 0.0
        for elite_idx in range(num_elite):
            elite_connections += np.sum(np.sum([[np.sum(layer)\
                    for layer in layer_layer] \
                    for layer_layer in self.elite_pop[elite_idx]]))


        for agent_idx in range(self.population_size):
            self.modulation[agent_idx] =  0.0
            self.remember[agent_idx] = 1.0

        self.hebbian_prune(0.01) #, ll_rew, ll_done, ll_hebs)
        if np.random.randint(2): self.mutate_pop(0.01)

        return sorted_fitness, num_elite,\
                mean_connections, std_connections

    def init_pop(self):
        # represent population as a list of lists of np arrays
        self.population = []
        self.hebbian = []

        self.modulation = []
        self.remember = []
        for agent_idx in range(self.population_size):
            self.modulation.append( 0.0)
            self.remember.append(1.0)

        for jj in range(self.population_size):
            layers = [[]]
            heb_layers = [[]]

            for layer_layer in range(len(self.hid_dim)-1):

                for layer in range(layer_layer+1):

                    if self.random_init:
                        layer_weights = np.random.randn(self.hid_dim[layer],\
                                self.hid_dim[layer_layer+1])
                    else:
                        layer_weights = np.ones((self.hid_dim[layer], \
                                self.hid_dim[layer_layer+1])) 

                    if layer == 0 and layer_layer != 0:
                        layers.append([layer_weights])
                        heb_layers.append([layer_weights*0])
                    else: 
                        layers[layer_layer].append(layer_weights)
                        heb_layers[layer_layer].append(layer_weights*0)




            self.population.append(layers)

            self.hebbian.append(heb_layers)
        self.elite_agent = self.population[0]

    def hebbian_prune2(self, prune_rate=0.001, \
            ll_rew=None, ll_done=None, ll_hebs=None):
        keep = self.keep
        for jj in range(keep, self.population_size):
            advantage = self.get_advantage(ll_rew[jj], ll_done[jj], gamma=0.5)
            num_layer_layers = len(self.population[jj])
            heb_idx = 0
            for kk in range(num_layer_layers):
                num_layers = len(self.population[jj][kk])

                for l_l in range(num_layers):
                    ll = l_l
                    
                    dim_x, dim_y = self.population[jj][kk][ll].shape
                    prunes_per_layer = prune_rate * dim_x * dim_y \
                            / len(advantage)

                    num_layers = len(self.population[jj][kk])

                    
                    for mm in range(len(advantage)):
                        
                        self.population[jj][kk][ll] *= \
                            np.random.random((dim_x,dim_y)) \
                            > softmax(\
                            -advantage[mm] * ll_hebs[jj][mm][heb_idx]\
                            ) * prunes_per_layer

                    heb_idx += 1
        self.population[:self.keep] = self.elite_pop[:self.keep]

                       


    def hebbian_prune(self, prune_rate=0.01):

        for jj in range(self.population_size):
            for kk in range(len(self.population[jj])):
                for ll in range(len(self.population[jj][kk])):

                    temp_layer = self.population[jj][kk][ll]

                    prunes_per_layer = prune_rate * temp_layer.shape[0]*temp_layer.shape[1]

                    temp_layer *= 1.0 * (np.random.random((temp_layer.shape[0],\
                            temp_layer.shape[1])) > (softmax(-np.abs(self.hebbian[jj][kk][ll])) \
                            * prunes_per_layer))

                    self.hebbian[jj][kk][ll] *= 0

                    self.population[jj][kk][ll] = temp_layer
        self.population[0] = self.elite_agent

    def mutate_pop(self, rate=0.1):
        # mutate population by 
        
        for jj in range(self.population_size):
            for kk in range(len(self.population[jj])):
                for ll in range(len(self.population[jj][kk])):    

                    temp_layer = self.population[jj][kk][ll]
                    
                    temp_layer *= np.random.random((temp_layer.shape[0],\
                            temp_layer.shape
                            
                            [1])) > rate

                    self.population[jj][kk][ll] = temp_layer

if __name__ == "__main__":

    min_generations = 10
    epds = 4
    save_every = 50
    hid_dims = {\
            "CartPole-v1": [25,12,6],\
            "InvertedPendulumBulletEnv-v0": [16,16],\
            "InvertedPendulumSwingupBulletEnv-v0": [32],\
            "InvertedDoublePendulumBulletEnv-v0": [4,4,4,4],\
            "ReacherBulletEnv-v0": [16,16,16],\
            "BipedalWalker-v2": [16,8,4],\
            "Walker2DBulletEnv-v0": [32,32,32,32],\
            "HopperBulletEnv-v0": [32,32,32]}

    env_names = [\
            "BipedalWalker-v2",\
            "ReacherBulletEnv-v0",\
            "InvertedDoublePendulumBulletEnv-v0"]
#            "CartPole-v1",
#            "BipedalWalker-v2"]
#            "Walker2DBulletEnv-v0"]
#            "HopperBulletEnv-v0"]
#            "InvertedPendulumSwingupBulletEnv-v0"]
#            "HalfCheetahBulletEnv-v0"]


    population_size = {\
            "CartPole-v1": 64,\
            "BipedalWalker-v2": 32,\
            "InvertedDoublePendulumBulletEnv-v0": 64,\
            "InvertedPendulumBulletEnv-v0": 64,\
            "InvertedPendulumSwingupBulletEnv-v0": 64,\
            "HalfCheetahBulletEnv-v0": 256,\
            "HopperBulletEnv-v0": 256,\
            "ReacherBulletEnv-v0": 128,\
            "Walker2DBulletEnv-v0": 256}

    thresh_performance = {\
            "CartPole-v1": 499,\
            "BipedalWalker-v2": 499,\
            "InvertedDoublePendulumBulletEnv-v0": 1999,\
            "InvertedPendulumBulletEnv-v0": 999.5,\
            "InvertedPendulumSwingupBulletEnv-v0": 880,\
            "HalfCheetahBulletEnv-v0": 3000,\
            "HopperBulletEnv-v0": 3000,\
            "ReacherBulletEnv-v0": 200,\
            "Walker2DBulletEnv-v0": 2995}

    max_generation = {\
            "CartPole-v1": 100,\
            "BipedalWalker-v2": 512,\
            "InvertedDoublePendulumBulletEnv-v0": 1024,\
            "InvertedPendulumBulletEnv-v0": 1024,\
            "InvertedPendulumSwingupBulletEnv-v0": 1024,\
            "HalfCheetahBulletEnv-v0": 1024,\
            "HopperBulletEnv-v0": 1024,\
            "ReacherBulletEnv-v0": 1024,\
            "Walker2DBulletEnv-v0": 2048}

    res_dir = os.listdir("./results/")
    model_dir = os.listdir("./models/")

    exp_dir = "exp010"
    exp_time = str(int(time.time()))[-7:]
    if exp_dir not in res_dir:
        os.mkdir("./results/"+exp_dir)
    if exp_dir not in model_dir:
        os.mkdir("./models/"+exp_dir)

    render = False
    smooth_fit = 0.0
    alpha = 0.25
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
            print("make env", env_name)

            obs_dim = env.observation_space.shape[0]

            try:
                act_dim = env.action_space.n
                discrete = True
            except:
                act_dim = env.action_space.sample().shape[0]
                discrete = False

            population_size = population_size[env_name]
            agent = HebbianDAG(obs_dim, act_dim, hid_dim=hid_dim, \
                    population_size=population_size, discrete=discrete)

            total_total_steps = 0
            t0 = time.time()
            
            #agent.mutate_pop(rate=0.35)
            for generation in range(max_generation[env_name]):
                #if generation % 100 == 0: 
                #    render = True
                #else:
                #    render = False
                #if "Bullet" in env_name:
                #    env._max_episode_steps = np.max([200, agent.best_agent]) #max_env_steps[env_name]

                #fitness, total_steps = agent.get_fitness(env, render=render)
                #total_total_steps += total_steps
                
                num_steps = 3.01e3
                #agent.mutate_pop(rate=0.01)
                ll_rew, ll_done, ll_hebs = agent.get_trajectory(num_steps)
                fitness = [np.sum(rew)/(np.sum(dones)) \
                        for rew, dones in zip(ll_rew,ll_done)]

                sorted_fitness, num_elite, \
                        mean_connections, std_connections = \
                        agent.update_pop(fitness)

                connections = np.sum([[np.sum(layer) for \
                        layer in layer_layer] \
                        for layer_layer in agent.elite_agent])

                agent.hebbian_prune2(0.01, ll_rew, ll_done, ll_hebs)

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

                smooth_fit = alpha * smooth_fit + ( 1-alpha ) * results["elite_max_fit"][-1]

                print("hebbian dag gen {} elapsed {:.1f} mut rate {:.1f},  mean/max/min fitness: {:.1f}/{:.1f}/{:.1f}, elite {:.1f}/{:.1f}/{:.1f}/{:.1f}, {:.1f}/{:.1f}"\
                        .format(generation, results["wall_time"][-1],\
                        results["prune_prob"][-1],\
                        results["pop_mean_fit"][-1],\
                        results["pop_max_fit"][-1],\
                        results["pop_min_fit"][-1],\
                        results["elite_mean_fit"][-1],\
                        smooth_fit,\
                        results["elite_max_fit"][-1],\
                        results["elite_min_fit"][-1],\
                        mean_connections, std_connections))

                if generation % save_every == 0:
                    np.save("./results/{}/hb_dag_{}.npy"\
                            .format(exp_dir, exp_id),results)
                    np.save("./models/{}/hb_dag_{}_gen{}.npy"\
                            .format(exp_dir,exp_id, generation), agent.elite_pop)

                if smooth_fit >= \
                        thresh_performance[env_name]\
                        and\
                        generation >= min_generations:
                    np.save("./results/{}/hb_dag_{}.npy"\
                            .format(exp_dir, exp_id),results)
                    np.save("./models/{}/hb_dag_{}_gen{}.npy"\
                            .format(exp_dir,exp_id, generation), agent.elite_pop)
                    print("environment solved, ending training")
                    break
            
            fitness, total_steps = agent.get_fitness(env, epds=100)
            print("fitness over 100 runs max:{:.3f} mean:{:.3f} min:{:.3f}"\
                    .format(np.max(fitness), \
                    np.mean(fitness), np.min(fitness)))
