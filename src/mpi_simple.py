import argparse
import subprocess
import sys
import os
import numpy as np
import gym 

import pybullet
import pybullet_envs


#mpi paralellization stuff
from mpi4py import MPI
comm = MPI.COMM_WORLD

#rank = comm.Get_rank()
#size = commm.Get_size()

class Agent():
    def __init__(self, obs_dim, act_dim, seed=0):

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hid_dim = [32]
        
        self.init_policies()
    
    def get_fitness(self, env, agent_idx, total_steps):

        return fitness

    def init_policies(self, pop_size=64):

        self.in2hid = np.random.randn((pop_size, self.obs_dimm, self.hid_dim[0]))
        self.hid2act = np.random.randn((pop_size, self.hid_dim[-1], self.act_dim))

    def get_action(self, obs, agent_idx=0):
        
        
        x = obs        
        x = np.tanh(np.matmul(x, self.in2hid[agent_idx]))

        x = np.matmul(x, self.hid2acti[agent_idx])

        if self.discrete:
            x = softmax(x)
            act = np.random.choice(np.arange(self.act_dim, size=1, p=x))
            #act = np.argmax(x)
        else:
            x = x
            act = np.tanh(x)

        return act

def mantle():
    print("this is the mantle process")

    global nWorker
    
    for aa in range(10):
        data = [np.random.randn(4,32)] * 3
        for worker in range(1,nWorker):
            comm.send(data, worker)

    data = 0
    for worker in range(1,nWorker):
        comm.send(data, worker)


def arm():
    print("this is an arm process") 
    
    while True:

        agent = comm.recv(source=0)

        len_layer = len(agent) if agent is not 0 else 0
        print("number of layers in agent is {}. Also I am worker {}".format(len_layer,rank))
        if len_layer == 0:
           print("worker #{} shutting down".format(rank))
           break
            
def mpi_fork(n):
  env_name = "InvertedPendulumSwingupBulletEnv-v0"
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
    
    if rank == 0:
        mantle()
    else: 
        arm()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=\
                "test mpi for running multiple policies")
    parser.add_argument('-n', '--num_workers', type=int, \
            help="number off cores to use, default 2", default=2)

    args = parser.parse_args()

    if mpi_fork(args.num_workers+1) == "parent":
        os._exit(0)

    main(args)
    
  
