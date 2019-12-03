
import numpy as np
import gym 
import matplotlib.pyplot as plt
import copy
import time
import os

from prune_bot import PruneableAgent
from custom_envs.cartpole_swingup import CartPoleSwingUpEnv
from gym.envs.registration import register

import pybullet
import pybullet_envs
import skimage
import skimage.io

#register(
#    id='CartPoleSwingUp-v0',
#    entry_point='custom_envs.cartpole_swingup:CartPoleSwingUpEnv',
#    max_episode_steps=200,
#    reward_threshold=25.0,
#    )

if __name__ == "__main__":

    model_fn = "models/cm_prune_16h16h/prunemk2_elite_pop_exp_991477env_InvertetEnv_s2.npy"
    model_fn = "models/prune_mk1_32_exp003/prunemk1_elite_pop_exp_5191527env_InvertedPendulumSwingupBulletEnv-v0_s2_gen100.npy" 
    model_fn = "models/prune_mk2_5_32_exp003/prunemk2_elite_pop_exp_5191522env_InvertedPendulumSwingupBulletEnv-v0_s2_gen100.npy" 
    model_fn = "models/prune_mk1_32_exp003/prunemk1_elite_pop_exp_5191527env_InvertedPendulumBulletSwingupEnv-v0_s2_gen100.npy"
    #model_fn = "models/prune_mk2_5_32_exp003/prunemk2_elite_pop_exp_5241643env_InvertedPendulumSwingupBulletEnv-v0_s2_gen100.npy"
    env_name = "InvertedPendulumSwingupBulletEnv-v0"

    save_renders = True
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]

    if "Bullet" in env_name:
        env.render()

    try:
        act_dim = env.action_space.n
        discrete = True
    except:
        act_dim = env.action_space.sample().shape[0]
        discrete = False


    temp_agent = np.load(model_fn, allow_pickle=True)
    population_size = len(temp_agent)

    agent = PruneableAgent(obs_dim, act_dim, hid=[32], \
        pop_size=population_size, discrete=discrete)
    for agent_idx in range(population_size):
        agent.pop[agent_idx] = temp_agent[agent_idx]

    population_size = 4 

    epds = 1
    count = 0
    for agent_idx in range(population_size):
        for epd in range(epds):
            done = False
            obs = env.reset()
            render = True
            total_reward = 0.0
            while not done:
                if "Bullet" not in env_name: env.render()
                time.sleep(0.0001)
                action = agent.get_action(obs, agent_idx=agent_idx)

                obs, reward, done, info = env.step(action)
                total_reward += reward
                count += 1
                if save_renders:
                    frame = env.render(mode="rgb_array")
                    skimage.io.imsave("./assets/frames/{}frame{}.png".format(env_name[:-3], count), frame)
            print("agent {}, Episode {}, episode reward: {}".format(agent_idx, epd, total_reward))
    env.close()
