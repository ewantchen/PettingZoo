from env.custom_environment import CustomEnvironment
from agents import ReplayMemory, DQN, select_action, plot_durations, optimize_model, initialize_agents
from agents import agents, TAU, episode_durations
import time
import pettingzoo


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





device = torch.device(
    "cuda" if torch.cuda.is_available() else 
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)



if torch.cuda.is_available() or torch.backends.mps.is_available() : 
    num_episodes = 600
else : 
    num_episodes = 300
win_guard = 0
win_prisoner = 0
BATCH_SIZE = 128
reward_total = {"prisoner" : 0, "guard" : 0}   
action_prisonnier = []   
action_garde = []
initialize_agents(env)
observations, info = env.reset()
agent_names = env.agents  # une liste d'agents
actions = {}
for i_episode in range(num_episodes):

    observations, info = env.reset()

    for agent_name in agent_names:
        observations[agent_name] = torch.from_numpy(
            observations[agent_name]).float().to(device).unsqueeze(0)

    for t in count() : 

        for agent_name in agent_names: 
            action_space = env.action_space(agent_name).n
            action = select_action(observations[agent_name], agent_name, action_space)
            actions[agent_name] = action.item()
        

        next_observation, reward, terminated, truncated, info = env.step(actions)

 
        if i_episode % 150 == 0 :
            print("frame" , t)
            env.render()
            time.sleep(0.2)
            if t > 300 :
                print("Terminated : ", terminated)
                print("truncated : ", truncated)
            for agent_name in agent_names : 
                print(f"{agent_name} -> action : {actions[agent_name]}")
            
        for agent_name in agent_names: 
            if i_episode == 50 and agent_name == "guard":
                action_garde.append(actions[agent_name])
            if i_episode == 50 and agent_name == "prisoner" :
                action_prisonnier.append(actions[agent_name])

        for agent_name in agent_names : 
            r = reward[agent_name]
            r_tensor = torch.tensor([r], device = device)
            done = all(terminated[agent] or truncated[agent] for agent in agent_names)

            if done : 
                next_state = None
            else : next_state = torch.tensor(next_observation[agent_name], dtype=torch.float32, device=device).unsqueeze(0)

        for agent_name in agent_names :
            state = observations[agent_name]
            action_tensor = torch.tensor([[actions[agent_name]]], device=device)
            next_state = (torch.from_numpy(next_observation[agent_name],).float().to(device).unsqueeze(0)
            if not terminated[agent_name] else None)
            memory = agents[agent_name]["memory"]
            memory.push(state, action_tensor, next_state, r_tensor)
            state = next_state

        

        for agent_name in env.agents:
            if len(agents[agent_name]["memory"]) > BATCH_SIZE:
                optimize_model(agent_name, agents)


        for agent_name in agent_names : 
            target_net = agents[agent_name]["target"]
            target_net_state_dict = target_net.state_dict()

            policy_net = agents[agent_name]["policy"]
            policy_net_state_dict = policy_net.state_dict()

            for key in policy_net_state_dict : 
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)

            target_net.load_state_dict(target_net_state_dict)
        
        if done :
            episode_durations.append(t+1)
            print(f"Episode", i_episode," finished after ", t+1 ,"frames")
            if reward['guard'] == -1 and reward['prisoner'] == 1:
                print("Le prisonnier a gagné")
                win_prisoner = win_prisoner + 1
            elif reward['guard'] == 1 and reward['prisoner'] == -1:
                print("Le guard a gagné")
                win_guard = win_guard + 1
            else:
                print("Nulle")
            for agent_name in agent_names :
                reward_total[agent_name] += reward[agent_name]
            plot_durations()
            break
print(action_garde)
print(action_prisonnier)
print(reward_total)
print(win_prisoner / win_guard)
print("Complete")

plot_durations(show_result=True)
plt.ioff()
plt.show()


   





