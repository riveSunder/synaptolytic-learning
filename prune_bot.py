import numpy as np
import gym 
import matplotlib.pyplot as plt
import copy
import time

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
        np.random.seed(self.seed)

        self.init_pop()
        self.mutate_pop(rate=0.25)

    def get_action(self, obs, agent_idx=0):

        x = obs        
        for ii in range(len(self.hid)):
            x = np.matmul(x, self.pop[agent_idx][ii])
            x = np.tanh(x)
            #x[x<0] = 0 # relu

        if self.discrete:
            x = sigmoid(self.by + np.matmul(x, self.pop[agent_idx][-1]))
            #x = np.where(x > 0.5, 1, 0) 
        else:
            x = self.by + np.matmul(x, self.pop[agent_idx][-1])

        if self.discrete:
            x = softmax(x)
            act = np.argmax(x)
        else:
            x = x
            act = np.tanh(x)

        return act

    def get_fitness(self, env, epds=3, render=False):
        fitness = []
        total_steps = 0

        for agent_idx in range(len(self.pop)):
            #obs = flatten_obs(env.reset())
            accumulated_reward = 0.0
            for epd in range(epds):

                obs = env.reset()
                done=False
                while not done:
                    if render: env.render(); time.sleep(0.05)
                    action = self.get_action(obs, agent_idx=agent_idx) 
                    obs, reward, done, info = env.step(action)
                    total_steps += 1
                    accumulated_reward += reward
                render = False 

            complexity_penalty = -1 * np.mean([np.mean(layer) \
                    for layer in self.pop[agent_idx]])

            fitness.append(accumulated_reward + complexity_penalty)
        plt.close("all")


        return fitness, total_steps

    def update_pop(self, fitness):

        # make sure fitnesses aren't equal
        fitness = fitness + np.random.randn(len(fitness),)*1e-16    
        sort_indices = list(np.argsort(fitness))
        sort_indices.reverse()

        sorted_fitness = np.array(fitness)[sort_indices]
        #sorted_pop = self.pop[sort_indices]
        
        keep = int(np.ceil(0.1*self.pop_size))
        if np.mean(sorted_fitness[:keep]) > self.best_gen:

            print("new best elite population: {} v {}".\
                    format(np.mean(sorted_fitness[:keep]), self.best_gen))
            self.best_gen = np.mean(sorted_fitness[:keep])
            self.elite_pop = []
            for oo in range(keep):
                self.elite_pop.append(self.pop[sort_indices[oo]])
        else:
            # decay recollection of greatest generation 
            pass
            # only the best rep gets in
        #always keep the fittest individual

        self.pop = []
        num_elite = len(self.elite_pop)
        p = np.arange(num_elite,0,-1) / np.sum(np.arange(num_elite,0,-1))
        a = np.arange(num_elite)
        for pp in range(self.pop_size):
            idx = np.random.choice(a,size=1,p=p)[0]
            self.pop.append(copy.deepcopy(self.elite_pop[idx]))
        
        self.mutate_pop()

        if(0):
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

    env_name = "CartPole-v0"
    #env_name = "BipedalWalker-v2"
    env_name = "LunarLander-v2"
    env_name = "Acrobot-v1"
    env_name = "Pendulum-v0"

    env = gym.make(env_name)

    env._max_episode_steps = 100
    print("env made",env.action_space, env.observation_space)
    obs_dim = env.observation_space.shape[0]
    try:
        act_dim = env.action_space.n
        discrete = True
    except:
        act_dim = env.action_space.sample().shape[0]
        discrete = False

    population = PruneableAgent(obs_dim, act_dim, pop_size=500, discrete=discrete)

    total_total_steps = 0
    for generation in range(1000):
        if generation % 100 == 0: 
            render = True
        else:
            render = False
        fitness, total_steps = population.get_fitness(env, render=render)
        total_total_steps += total_steps

        population.update_pop(fitness)
        connections = np.sum([np.sum(layer) for layer in population.pop[0]])
        print("generation {}".format(generation))
        print(" connections in best agent", connections)
        print("fitness stats: mean {:.3f} | std: {:.3f} | max: {:.3f} | min: {:.3f}".format(np.mean(fitness), np.std(fitness), np.max(fitness), np.min(fitness)))

    import pdb; pdb.set_trace()

    population.get_fitness(env, render=True)
