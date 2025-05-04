import pettingzoo
from env.custom_environment import CustomEnvironment

import numpy as np

import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

env = CustomEnvironment()


#set up matplotlib
is_ipython = "inline" in matplotlib.get_backend()
if is_ipython :
    from IPython import display

plt.ion()

#if GPU is to be used 
device = torch.device(
    "cuda" if torch.cuda.is_available() else 
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
observations, info = env.reset()
np_obs = np.stack(list(observations.values()))
tensor_obs = torch.from_numpy(np_obs).float()
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))




class ReplayMemory(object):

    def __init__(self, capacity) : 
        #mémoire faite pour stocker Transition en deque
        self.memory = deque([], maxlen=capacity)

    def push(self, *args) : 
        #ajoute une transition aux selfmemory
        self.memory.append(Transition(*args)) 
    
    def sample(self,batch_size) : 
        return random.sample(self.memory, batch_size)
    
    def __len__(self) : 
        #combien de transition sont stockées
        return len(self.memory)

class DQN(nn.Module) : 

    def __init__(self, obs_space, action_space):
        super(DQN, self).__init__()
        self.layer1= nn.Linear(obs_space, 128)
        self.layer2 = nn.Linear(128,128)
        self.layer3 = nn.Linear(128, action_space)
    
    def forward(self,x): 
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# BATCH_SIZE is the number of transitions sampled from the 
# replay buffer

# GAMMA is the discount factor as mentioned in the previous 
# section

# EPS_START is the starting value of epsilon

# EPS_END is the final value of epsilon

# EPS_DECAY controls the rate of exponential decay of epsilon, 
# higher means a slower decay

# TAU is the update rate of the target network

# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.1
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

env=CustomEnvironment()
env.reset()
agent_names = env.agents
agents={}

def initialize_agents(env) : 
    for agent_name in agent_names : 
        obs_space = env.observation_space(agent_name).shape[0]
        action_space = env.action_space(agent_name).n

        policy_net = DQN(obs_space, action_space).to(device)
        target_net = DQN(obs_space, action_space).to(device)
        target_net.load_state_dict(policy_net.state_dict())

        optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
        memory = ReplayMemory(10000)

        agents[agent_name] = {
            "policy" : policy_net,
            "target" : target_net,
            "optimizer" : optimizer, 
            "memory" : memory 
        }

steps_done = 0

def select_action(observations, agent_name, action_space) :
    global steps_done
    sample = random.random()
    EPS_DECAY * 2 if agent_name == "prisoner" else EPS_DECAY
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of 
            # each row.
            # second column on max result is index of where max 
            # element was
            # found, so we pick action with the larger expected
            # reward.
            return agents[agent_name]["policy"](observations).argmax(dim=1).unsqueeze(0)
    else:
        return torch.tensor([[random.randrange(action_space)]], device=device, dtype=torch.long)

episode_durations = []

def plot_durations(show_result=False) : 
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result :
        plt.title("Result")
    else : 
        plt.clf()
        plt.title("training...") 
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(durations_t.numpy())
    #take 100 episode averages and plot them too

    if len(durations_t) >= 100 :
        means = durations_t.unfold(0,100,1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    
    plt.pause(0.001) #pause a bit so that plots are updated
    if is_ipython : 
        if not show_result :
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else : 
            display.display(plt.gcf())

def optimize_model(agent_name, agents) : 
    memory = agents[agent_name]["memory"]
    policy_net = agents[agent_name]["policy"]
    target_net = agents[agent_name]["target"]
    optimizer = agents[agent_name]["optimizer"]

    if len(memory) < BATCH_SIZE : 
        return
    transitions = memory.sample(BATCH_SIZE)

    #permet de formatter l'array pour que l'on puisse multiplier 
    #les arrays entre eux (voir guide réseaux de neurones)
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate 
    # the batch elements
    # (a final state would've been the one after which 
    # simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

#on prend l'action selon la calculation faite par policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

          # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
