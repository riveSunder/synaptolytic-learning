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

from hebbian_lstm import HebbianLSTMAgent
from hebbian_dag import HebbianDAG 
from prune_bot import PruneableAgent 

def mantle(args):

    env_name = args.env_name
    max_generations = args.max_generations
    population_size = args.population_size

    disp_every = 25
    my_seeds = args.seed

    exp_time = str(int(time.time()))[-7:]
    for my_seed in my_seeds:
        print("training with seed {}".format(my_seed))
        np.random.seed(my_seed)

        exp_id = "exp_" + exp_time + "env_" +\
                env_name + "_s" + str(my_seed)
        
        np.save("./results/args_{}.npy".format(exp_id), args)
        env = gym.make(env_name) 
        obs_dim = env.observation_space.shape[0]
        try:
            act_dim = env.action_space.n
            discrete = True
        except:
            act_dim = env.action_space.sample().shape[0]
            discrete = False

        hid_dim = [elem for elem in args.hid_dims]
        if "PruneableAgent" in args.agent_type:
            agent = PruneableAgent(obs_dim, act_dim, hid_dim=hid_dim,\
                    population_size=population_size, discrete=discrete)
        elif "Hebbian" in args.agent_type and "LSTM" in args.agent_type:
            agent = HebbianLSTMAgent(obs_dim, act_dim, hid_dim=hid_dim,\
                    population_size=population_size, seed=0, discrete=discrete)
        elif "HebbianDAG" in args.agent_type:
            agent = HebbianDAG(obs_dim, act_dim, hid_dim=hid_dim,\
                    population_size=population_size, seed=0, discrete=discrete)


        t0 = time.time()

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

        total_total_steps = 0

        for generation in range(max_generations):
            bb = 0
            fitness = []
            total_steps =0

            t1 = time.time()

            mod = []
            rem = []
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

            total_total_steps += total_steps
            sorted_fitness, num_elite,\
                    mean_connections, std_connections = agent.update_pop(fitness)

            keep = 16

            connections = mean_connections #np.sum([np.sum(layer) for layer in agent.elite_agent])


            results["generation"].append(generation)
            results["total_env_interacts"].append(total_total_steps)
            results["wall_time"].append(time.time()-t0)
            results["prune_prob"].append(0.01)
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

            if generation % disp_every == 0:
                print(my_seed)
                np.save("./results/mpi_{}.npy"\
                        .format(exp_id), results)
                print("mean/std connections {:.2e}/{:.2e} ".format(mean_connections, std_connections))
                print("gen {} mean fitness {:.3f}/ max {:.3f} , time elapsed/per gen {:.2f}/{:.2f}".\
                        format(generation, np.mean(fitness), np.max(fitness),\
                        time.time()-t0, (time.time() - t0)/(generation+1)))

                np.save("./syn_{}best_agents.npy".format(exp_id), agent.elite_pop)

        np.save("./results/prunemk1_mpi_{}.npy"\
                .format(exp_id), results)
        print("mean/std connections {:.2e}/{:.2e} ".format(mean_connections, std_connections) )
        print("gen {} mean fitness {:.3f}/ max {:.3f} , time elapsed/per gen {:.2f}/{:.2f}".\
                format(generation, np.mean(fitness), np.max(fitness),\
                time.time()-t0, (time.time() - t0)/(generation+1)))
        np.save("./syn_{}best_agents.npy".format(exp_id), agent.elite_pop)

        np.save("./syn_{}best_agent.npy".format(exp_id), agent.elite_agent)

        print("time to compute fitness for pop {} on {} workers {:.3f}".format(\
                population_size, nWorker, time.time()-t0))

        print("evaluating best agent policy....")
        agent.population[0] = agent.elite_agent

        fitness = []
        eval_epds_left = 100 // args.epds
        bb = 0
        while eval_epds_left > 0:
            eval_epds_left = 100 - bb
            for cc in range(1, min(nWorker, 1+eval_epds_left)):
                comm.send(agent.population[0], dest=cc)
            
            for cc in range(1, min(nWorker, 1+eval_epds_left)):
                #comm.send(agent.population[bb+cc-1], dest=cc)
                fit = comm.recv(source=cc)
                fitness.extend(fit[0])
                total_steps += fit[1]


            bb += cc

        print("elite agent evalutated for {} episodes with score {:.2e} +/-{:.2e}".format(\
                cc*args.epds, np.mean(fitness), np.std(fitness)))

    for cc in range(1,nWorker):
        comm.send(0, dest=cc)
    data = 0


def arm(args):
    
    env_name = args.env_name
    max_generations = args.max_generations

    epds = args.epds
    env = gym.make(env_name) 

    obs_dim = env.observation_space.shape[0]
    try:
        act_dim = env.action_space.n
        discrete = True
    except:
        act_dim = env.action_space.sample().shape[0]
        discrete = False

    population_size = 1
    hid_dim = args.hid_dims
    if "PruneableAgent" in args.agent_type:
        agent = PruneableAgent(obs_dim, act_dim, hid_dim=hid_dim,\
                population_size=population_size, discrete=discrete)
    elif "Hebbian" in args.agent_type and "LSTM" in args.agent_type:
        agent = HebbianLSTMAgent(obs_dim, act_dim, hid_dim=hid_dim,\
                population_size=population_size, seed=0, discrete=discrete)
    elif "HebbianDAG" in args.agent_type:
        agent = HebbianDAG(obs_dim, act_dim, hid_dim=hid_dim,\
                population_size=population_size, seed=0, discrete=discrete)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=\
                "test mpi for running multiple policies")
    parser.add_argument('-n', '--num_workers', type=int, \
            help="number off cores to use, default 2", default=2)
    parser.add_argument('-e', '--env_name', type=str,\
            help="name of environment, default InvertedPendulumSwingupBulletEnv-v0", default="InvertedPendulumSwingupBulletEnv-v0")
    parser.add_argument('-g', '--max_generations', type=int,\
            help="training generations", default=10)
    parser.add_argument('-a', '--agent_type', type=str,\
            default="HebbianLSTMAgent")
    parser.add_argument('-p', '--population_size', type=int,\
            help="number of agents in a population", default=92)
    parser.add_argument('-s', '--seed', type=int, nargs='+',\
            help="random seed for initialization", default=[42])

    parser.add_argument('-d', '--hid_dims', type=int, nargs='+',\
            help="hidden layer nodes", default=[256])
    parser.add_argument('-r', '--epds', type=int,\
            help="hidden layer nodes", default=8)

    args = parser.parse_args()

    num_workers = args.num_workers
    env_name = args.env_name
    max_generations = args.max_generations

    #rank = comm.Get_rank()
    #nWorkers = comm.Get_size()

    if mpi_fork(args.num_workers+1) == "parent":
        os._exit(0)

    if rank == 0:
        mantle(args)
    else: 
        arm(args)
