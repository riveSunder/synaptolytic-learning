import argparse
import numpy as np
import gym 
import matplotlib.pyplot as plt
import copy
import time
import os

from prune_bot import PruneableAgent

from cma import CMAAgent
from hebbian_dag import HebbianDAG

import pybullet
import pybullet_envs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=\
            "watch a trained model in a designated environment")
    parser.add_argument('-e', '--env_name', type=str, \
            help="environment to enjoy", default="InvertedPendulumSwingupBulletEnv-v0")
    parser.add_argument('-m', '--model', type=str, \
            help="path to trained model", \
            default="models/cma_32_exp003/cma_5287031InvertedPendulumSwingupBulletEnv-v0_s2_gen100.npy")

    parser.add_argument('-d', '--dag', type=bool,\
            help="use directed acyclic graph policy", default=False)

    parser.add_argument('-p', '--epds', type=int,\
            help="number of episodes to enjoy", default=3)
    parser.add_argument('-a', '--agents', type=int,\
            help="number of agents to enjoy", default=1)

    args = parser.parse_args()

    model_fn = args.model
    env_name = args.env_name
    use_dag = args.dag
    num_agents = args.agents
    epds = args.epds

    # number of agents to test (from top of elite pop.)

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]

    if "Bullet" in env_name:
        env.render()

    act_dim = env.action_space.sample().shape[0]
    discrete = False

    temp_agent = np.load(model_fn, allow_pickle=True)
    if use_dag:
        agent = HebbianDAG(obs_dim, act_dim, discrete=discrete)
    else:
        agent = PruneableAgent(obs_dim, act_dim, discrete=discrete)

    population_size = len(temp_agent)
    agent.pop_size = population_size 
    
    agent.pop = temp_agent

    for agent_idx in range(num_agents):
        for epd in range(epds):
            done = False
            obs = env.reset()
            render = True
            total_reward = 0.0
            while not done:
                if "Bullet" not in env_name: env.render()
                time.sleep(0.01)
                action = agent.get_action(obs, agent_idx=agent_idx, enjoy=True)

                obs, reward, done, info = env.step(action)
                total_reward += reward

            print("agent {}, Episode {}, episode reward: {}".format(agent_idx, epd, total_reward))
    env.close()
