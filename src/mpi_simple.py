import argparse
import copy
import subprocess
import sys
import os
import numpy as np
import gym 
import time
import matplotlib.pyplot as plt

import pybullet
import pybullet_envs


#mpi paralellization stuff
from mpi4py import MPI
comm = MPI.COMM_WORLD

#rank = comm.Get_rank()
#size = comm.Get_size()

class PruneableAgent():

    def __init__(self, input_dim, act_dim, hid_dim=[32,32,32],\
            population_size=10, seed=0, discrete=True):

        self.input_dim = input_dim
        self.output_dim = act_dim #output_dim
        self.hid = hid_dim
        self.pop_size = population_size
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
        if enjoy: self.hid = [hid.shape[1] for hid in self.population[agent_idx]][:-1]
        for ii in range(len(self.hid)):
            x = np.matmul(x, scaler*self.population[agent_idx][ii])
            x = np.tanh(x)
            #x[x<0] = 0 # relu

        if self.discrete:
            x = sigmoid(self.by + np.matmul(x, scaler*self.population[agent_idx][-1]))
            #x = np.where(x > 0.5, 1, 0) 
        else:
            x = self.by + np.matmul(x, scaler*self.population[agent_idx][-1])

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

            complexity_penalty =  np.mean([np.mean(layer) \
                    for layer in self.population[agent_idx]])

            fitness.append(accumulated_reward/(epds*len(values))-complexity_penalty)
            #complexity.append(complexity_penalty)


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
            self.best_agent = sorted_fitness[0]

        if np.mean(sorted_fitness[:keep]) > self.best_gen:
            # keep best elite population
            print("new best elite population: {} v {}".\
                    format(np.mean(sorted_fitness[:keep]), self.best_gen))
            self.best_gen = np.mean(sorted_fitness[:keep])

        self.elite_pop = []
        self.elite_pop.append(self.elite_agent)

        for oo in range(keep):
            self.elite_pop.append(self.population[sort_indices[oo]])

        self.population = []
        num_elite = len(self.elite_pop)
        p = np.arange(num_elite,0,-1) / np.sum(np.arange(num_elite,0,-1))
        a = np.arange(num_elite)
        for pp in range(self.pop_size):
            idx = np.random.choice(a,size=1,p=p)[0]
            self.population.append(copy.deepcopy(self.elite_pop[idx]))
        

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

        mutation_rate = np.sqrt(elite_connections / num_elite)
        mutation_rate /= mean_connections
        #mutation_rate *= 0.995
        mutation_rate = np.max([np.min([0.05, mutation_rate]), 0.001])
        self.population[:keep] = self.elite_pop
        return sorted_fitness, num_elite, mutation_rate, \
                mean_connections, std_connections

    def init_pop(self):
        # represent population as a list of lists of np arrays
        self.population = []

        for jj in range(self.pop_size):
            layers = []
            layer = np.ones((self.input_dim, self.hid[0]))

            layers.append(layer)
            for kk in range(1,len(self.hid)):
                    layer = np.ones((self.hid[kk-1], self.hid[kk]))
                    layers.append(layer)
            layer = np.ones((self.hid[-1], self.output_dim)) 
            layers.append(layer)
            self.population.append(layers)

    def mutate_pop(self, keep=0, rate=0.01):
        # mutate population by 
        
        for jj in range(keep, self.pop_size):
            for kk in range(len(self.population[jj])):
                temp_layer = np.copy(self.population[jj][kk])
                
                temp_layer *= np.random.random((temp_layer.shape[0],\
                        temp_layer.shape[1])) > rate

                self.population[jj][kk] = temp_layer

def mantle(env_name):

    hid_dim = [64,64,64]
    
    env = gym.make(env_name) 
    obs_dim = env.observation_space.shape[0]
    try:
        act_dim = env.action_space.n
        discrete = True
    except:
        act_dim = env.action_space.sample().shape[0]
        discrete = False

    population_size = 320 
    agent = PruneableAgent(obs_dim, act_dim, hid_dim=hid_dim,\
            population_size=population_size, discrete=discrete)

    for generation in range(20000):
        bb = 0
        fitness = []
        total_steps =0

        t0 = time.time()

        while bb <= population_size: # - nWorker:
            pop_left = population_size - bb
            for cc in range(1, min(nWorker, 1+pop_left)):
                comm.send(agent.population[bb+cc-1], dest=cc)
            
            for cc in range(1, min(nWorker, 1+pop_left)):
                #comm.send(agent.population[bb+cc-1], dest=cc)
                fit = comm.recv(source=cc)
                fitness.extend(fit[0])
                total_steps += fit[1]

            bb += cc
        print(nWorker, len(fitness), population_size, "***")

        sorted_fitness, num_elite, mutation_rate,\
                mean_connections, std_connections = agent.update_pop(fitness)

        keep = 16
        agent.mutate_pop(keep=keep, rate=mutation_rate)

        print("mean/std connections {:.2e}/{:.2e} ".format(mean_connections, std_connections), \
                mutation_rate)
        print("gen {} mean fitness {:.3f}/ max {:.3f} , time elapsed {:.3f}".format(\
                generation, np.mean(fitness), np.max(fitness), time.time()-t0))

    np.save("./best_agent.npy", agent.elite_agent)

    print("time to compute fitness for pop {} on {} workers {:.3f}".format(\
            population_size, nWorker, time.time()-t0))
    for cc in range(1,nWorker):
        comm.send(0, dest=cc)
    data = 0


def arm(env_name):
    
    hid_dim = [64,64,64]
    epds = 16
    env = gym.make(env_name) 

    obs_dim = env.observation_space.shape[0]
    try:
        act_dim = env.action_space.n
        discrete = True
    except:
        act_dim = env.action_space.sample().shape[0]
        discrete = False

    population_size = 1
    agent = PruneableAgent(obs_dim, act_dim,\
            hid_dim=hid_dim, population_size=population_size, discrete=discrete)

    while True:

        my_policy = comm.recv(source=0)

        if my_policy == 0:
            print("worker {} shutting down".format(rank))
            break

        agent.population = [my_policy]

        fitness = agent.get_fitness(env, epds=epds, render=False)


        comm.send(fitness, dest=0)
            
def mpi_fork(n):
  """Re-launches the current script with workers
  Returns "parent" for original parent, "child" for MPI children
  (from https://github.com/garymcintire/mpi_util/)
  """
  if n<=1:
    return "child"

  if os.getenv("IN_MPI") is None:
    env = os.environ.copy()
    env.update(
      MKL_NUM_THREADS="1",
      OMP_NUM_THREADS="1",
      IN_MPI="1"
    )
    print( ["mpirun", "-np", str(n), sys.executable] + sys.argv)
    subprocess.check_call(["mpirun", "-np", str(n), sys.executable] \
        +['-u']+ sys.argv, env=env)
    return "parent"
  else:
    global nWorker, rank
    nWorker = comm.Get_size()
    rank = comm.Get_rank()
    #print('assigning the rank and nworkers', nWorker, rank)
    return "child"

def main(argv):
    
    env_name = argv.env_name

    if rank == 0:
        mantle(env_name)
    else: 
        arm(env_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=\
                "test mpi for running multiple policies")
    parser.add_argument('-n', '--num_workers', type=int, \
            help="number off cores to use, default 2", default=2)
    parser.add_argument('-e', '--env_name', type=str,\
            help="name of environment, default InvertedPendulumSwingupBulletEnv-v0", default="InvertedPendulumSwingupBulletEnv-v0")
    #parser.add_argument('-h', '--hid_dims', type=list,\
    #        help="hidden dims", default=[16])

    args = parser.parse_args()

    num_workers = args.num_workers
    env_name = args.env_name

    #rank = comm.Get_rank()
    #nWorkers = comm.Get_size()

    if mpi_fork(args.num_workers+1) == "parent":
        os._exit(0)

    if rank == 0:
        mantle(env_name)
    else: 
        arm(env_name)
