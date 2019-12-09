import argparse
import numpy as np
import gym 
import matplotlib.pyplot as plt
import copy
import time
import os

from prune_bot import PruneableAgent
from cma import CMAAgent

import pybullet
import pybullet_envs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=\
            "watch a trained model in a designated environment")
    parser.add_argument('-e', '--env_name', type=str, \
            help="environment to enjoy", default="InvertedPendulumSwingupBulletEnv-v0")
    parser.add_argument('-m', '--model', type=str, \
            help="path to trained model", default="models/cma_32_exp003/cma_5287031InvertedPendulumSwingupBulletEnv-v0_s2_gen100.npy")

    args = parser.parse_args()

    model_fn = args.model
    env_name = args.env_name

    # number of agents to test (from top of elite pop.)
    num_agents = 1
    epds = 16

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]

    if "Bullet" in env_name:
        env.render()

    act_dim = env.action_space.sample().shape[0]
    discrete = False

    temp_agent = np.load(model_fn, allow_pickle=True)
    
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
