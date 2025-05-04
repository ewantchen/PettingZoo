import functools
import random
from copy import copy

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete

from pettingzoo import ParallelEnv



class CustomEnvironment(ParallelEnv):
    metadata = {
        "name": "custom_environment_v0",
    }

    def __init__(self):
        """The init method takes in environment arguments.

        Should define the following attributes:
        - escape x and y coordinates
        - guard x and y coordinates
        - prisoner x and y coordinates
        - timestamp
        - possible_agents"""
        self.escape_y = None
        self.escape_x = None

        self.guard_y = None
        self.guard_x = None

        self.prisoner_y = None
        self.prisoner_x = None

        self.old_dist_prisoner_escape = None
        self.old_dist_guard_prisoner = None
        self.new_dist_prisoner_escape = None  
        self.new_dist_guard_prisoner = None

        self.timestep = None
        self.possible_agents = ["prisoner", "guard"]


    def reset(self, seed=None, options=None):
            """Reset set the environment to a starting point.

        It needs to initialize the following attributes:
        - agents
        - timestamp
        - prisoner x and y coordinates
        - guard x and y coordinates
        - escape x and y coordinates
        - observation
        - infos

        And must set up the environment so that render(), step(), 
        and observe() can be called without issues.
        """
            self.agents= copy(self.possible_agents)
            self.timestep = 0

            self.prisoner_x = 0
            self.prisoner_y = 0

            self.guard_x = 6
            self.guard_y = 6

            self.escape_x = random.randint(2,5)
            self.escape_y = random.randint(2,5)

            self.dist_prisoner_escape = abs(self.prisoner_x - self.escape_x) + abs(self.prisoner_y - self.escape_y)
            self.dist_guard_prisoner = abs(self.guard_x - self.prisoner_x) + abs(self.guard_y - self.prisoner_y)


            observations = {
            "prisoner": np.array([
            self.prisoner_x / 7, self.prisoner_y / 7,
            self.guard_x / 7, self.guard_y / 7,
            self.escape_x / 7, self.escape_y / 7
                ], dtype=np.float32),
            "guard": np.array([
            self.guard_x / 7, self.guard_y / 7,
            self.prisoner_x / 7, self.prisoner_y / 7,
            self.escape_x / 7, self.escape_y / 7
            ], dtype=np.float32),
            }

            infos = {a:{} for a in self.agents}

            return observations, infos


    def step(self, actions):
        """Takes in an action for the current agent 
        (specified by agent_selection).

        Needs to update:
        - prisoner x and y coordinates
        - guard x and y coordinates
        - terminations
        - truncations
        - rewards
        - timestamp
        - infos

        And any internal state used by observe() or render()
        """
        self.old_esc = abs(self.prisoner_x - self.escape_x) + abs(self.prisoner_y - self.escape_y)
        self.old_catch = abs(self.guard_x - self.prisoner_x) + abs(self.guard_y - self.prisoner_y)
        #execute actions
        prisoner_action = actions.get("prisoner")
        guard_action = actions.get("guard")
    
        if prisoner_action == 0 and self.prisoner_x > 0:
            self.prisoner_x -= 1 #go left
        elif prisoner_action == 1 and self.prisoner_x < 6:
            self.prisoner_x += 1 #go right
        elif prisoner_action == 2 and self.prisoner_y > 0:
            self.prisoner_y -= 1 #go down
        elif prisoner_action == 3 and self.prisoner_y < 6:
            self.prisoner_y += 1 #go up

        if guard_action == 0 and self.guard_x > 0:
            self.guard_x -= 1 #left
        elif guard_action == 1 and self.guard_x < 6:
            self.guard_x += 1 #right
        elif guard_action == 2 and self.guard_y > 0:
            self.guard_y -= 1 #down
        elif guard_action == 3 and self.guard_y < 6:
            self.guard_y += 1 #up
        
        self.new_esc = abs(self.prisoner_x - self.escape_x) + abs(self.prisoner_y - self.escape_y)
        self.new_catch = abs(self.guard_x - self.prisoner_x) + abs(self.guard_y - self.prisoner_y)

        #check termination conditions
        terminations = {a : False for a in self.agents}
        rewards = {a : 0 for a in self.agents}
        if self.prisoner_x == self.guard_x and self.prisoner_y == self.guard_y : 
            rewards = {"prisoner" : -1 , "guard" : 1}
            terminations = {a : True for a in self.agents}
        elif self.old_esc > self.new_esc :
            rewards = {"prisoner" : 0.5 , "guard" : -0.1}
        elif self.old_catch> self.new_catch :
            rewards = {"prisoner" : -0.3 , "guard" : +0.3}
        elif self.prisoner_x == self.escape_x and self.prisoner_y == self.escape_y :
            rewards = {"prisoner" : 5 , "guard" : -2}
            terminations = {a : True for a in self.agents}


        self.timestep += 1
        #check truncation conditions (overwrites termination conditions)
        truncations = {a : False for a in self.agents}
        if self.timestep > 300 :
            rewards = {"prisoner" : 0 , "guard" : 0}
            truncations = {"prisoner" : True, "guard" : True}
       
        #Get observations
        observations = {
            "prisoner": np.array([
            self.prisoner_x / 7, self.prisoner_y / 7,
            self.guard_x / 7, self.guard_y / 7,
            self.escape_x / 7, self.escape_y / 7
                ], dtype=np.float32),
        "guard": np.array([
            self.guard_x / 7, self.guard_y / 7,
            self.prisoner_x / 7, self.prisoner_y / 7,
            self.escape_x / 7, self.escape_y / 7
            ], dtype=np.float32),
            }

        # Get dummy infos (not used in this example)
        infos = {a: {} for a in self.agents}

        if any(terminations.values()) or all(truncations.values()) :
            self.agents = []
        

        return observations, rewards, terminations, truncations, infos
    
    def render(self):
        grid = np.full((7, 7), "")#le "" crée des cases vides
        grid[self.prisoner_y, self.prisoner_x] = "P"
        grid[self.guard_y, self.guard_x] = "G"
        grid[self.escape_y, self.escape_x] = "E"
        print(f"{grid} \n")

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return MultiDiscrete([7*7] * 6)
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(4)