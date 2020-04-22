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
from hebbian_lstm import HebbianLSTMAgent

import pybullet
import pybullet_envs
import skimage

import skimage.io


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=\
            "watch a trained model in a designated environment")
    parser.add_argument('-e', '--env_name', type=str, \
            help="environment to enjoy", default="InvertedPendulumSwingupBulletEnv-v0")
    parser.add_argument('-m', '--model', type=str, \
            help="path to trained model", \
            default="models/cma_32_exp003/cma_5287031InvertedPendulumSwingupBulletEnv-v0_s2_gen100.npy")

    parser.add_argument('-d', '--agent_type', type=str,\
            help="type of agent to use", default="PruneableAgent")

    parser.add_argument('-p', '--epds', type=int,\
            help="number of episodes to enjoy", default=3)
    parser.add_argument('-a', '--agents', type=int,\
            help="number of agents to enjoy", default=1)
    parser.add_argument('-r', '--render', type=int,\
            help="Display environment during eval", default=1)
    parser.add_argument('-s', '--save_images', default=0,
            help="save images, e.g. for a gif")

    args = parser.parse_args()

    model_fn = args.model
    env_name = args.env_name
    agent_type = args.agent_type
    num_agents = args.agents
    epds = args.epds
    render = args.render

    # number of agents to test (from top of elite pop.)

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]

    if "Bullet" in env_name and render:
        env.render()

    act_dim = env.action_space.sample().shape[0]
    discrete = False

    temp_agent = np.load(model_fn, allow_pickle=True)
    if "DAG" in agent_type:
        hid_dim = [1]
        agent = HebbianDAG(obs_dim, act_dim, hid_dim=hid_dim, discrete=discrete)
        #if type(temp_agent) is not list:
        #    temp_agent = [temp_agent]
    elif "LSTM" in agent_type:
        hid_dim = [128]
        agent = HebbianLSTMAgent(obs_dim,act_dim, discrete=discrete)
    else:
        agent = PruneableAgent(obs_dim, act_dim, discrete=discrete)

    #temp_agent = [temp_agent]
    population_size = len(temp_agent)
    agent.population_size = population_size 
    
    agent.population = temp_agent
    #agent.hebbian = temp_agent


    for agent_idx in range(num_agents):
        fitness = []
        for epd in range(epds):
            done = False
            obs = env.reset()
            total_reward = 0.0
            step = 0
            while not done:
                if "Bullet" not in env_name and render:
                    env.render()
                if args.save_images:

                    img = env.render(mode="rgb_array")
                    skimage.io.imsave("./imgs/{}agent{}epd{}step{}.png".format(\
                            env_name[:-3], agent_idx, epd, step), img)

                time.sleep(0.01)
                action = agent.get_action(obs, agent_idx=agent_idx, enjoy=True)

                obs, reward, done, info = env.step(action)
                total_reward += reward
                step += 1
            fitness.append(total_reward)
            #print("agent {}, Episode {}, episode reward: {}".format(agent_idx, epd, total_reward))
        print("agent {} performance over {} epds: {:.2e} +/- {:.2e} s.d".format(\
                agent_idx, epds, np.mean(fitness), np.std(fitness)))
    env.close()

