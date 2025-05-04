import gymnasium as gym
from gymnasium import spaces

from enum import Enum

import numpy as np
import pygame

class actions(Enum) : #on donne toutes les actions possibles pour les agents
    RIGHT = 0 #On met en majuscule pour pas confondre avec des fonctions
    LEFT = 1
    UP = 2
    DOWN = 3

class GridWorldEnv(gym.Env) : #on précise les modes de rendu pour notre environnement
    metadata = {"render_modes" : ["human", "rgb_array"], "render_fps" : 4}


#on utilise __init__ pour notre grille
def __init__(self, render_mode=None, size=5) : #on précise aussi qu'aucun rendu peut-être possible
    self.size = size #taille de notre grille
    self.window_size = 512 #taille de la fenêtre

    self.observation_safe = spaces.Dict(
        {
            "agent": spaces.Box(0, size-1, shape=(2,), dtype=int), 
            "target": spaces.Box(0, size-1, shape=(2), dtype=int),
        }
    )
    self._agent_location = np.array([-1, -1], dtype=int)
    self._target_location = np.array([-1, -1], dtype=int)

    self.action_space = spaces.Discrete(4)
    self._action_to_direction = { #On établi un tableau pour retrouver les actions 
            actions.RIGHT.value: np.array([1, 0]),
            actions.LEFT.value: np.array([0, 1]),
            actions.UP.value: np.array([-1, 0]),
            actions.DOWN.value: np.array([0, -1]),
        }    

    assert render_mode is None or render_mode in self.metadata["render_modes"]
    self.render_mode = render_mode

    self.window = None
    self.clock = None

def _get_obs(self) : #Donne l'information de la position lors de step et reset sur la positon
    return{"agent": self.agent_location, "target" : self._target_location}
def _get_info(self) : #donne l'information de la difference de position lors de step et reset 
    return {
        "distance" : np.linalg.norm(
            self._agent_location - self._target_location, ord = 1
        )
    }

def reset(self, seed= None, options = None) :
    # Permet de changer la seed de l'environnement => position de target et de agent
    super().reset(seed=seed)

    # Décide de la position de l'agent
    self._agent_location = self.np_random.integers(0, self.size, size = 2, dtype = int)

    self._target_location = self._agent_location #On change sa position tant qu'elle est égale à celle de l'agent
    while np.array_equal(self._target_location, self._agent_location) :
        self._target_location = self.np_random.intergers(
            0, self.size, size=2, dtype=int
        )
    observation = self._get_obs()
    info = self._get_info()
    
    if self.render_mode == "human" :
        self._render_frame()

    return observation, info

def step(self,action) : #contient la majorité de la logique de l'environnement
    direction = self._action_to_direction[action] #on défini la direction selon l'action

    self._agent_location = np.clip(
        self._aent_location + direction, 0, self.size-1
    ) #On met des barrières à la grille

    terminated =  np.array_equal(self._agent_location, self._target_location)
    reward = 1 if terminated else 0 #récompense en binaire
    observation = self._get_obs()
    info = self._get_info()
    
    if self.render_mode == "human":
        self._render_frame()

    return observation, reward, terminated, False, info

def render(self) : 
    if self.render_mode == "rgb_array":
        return self._render_frame()

def _render_frame(self):
    if self.window is None and self.render_mode == "human" : #is None veut dire qu'une fenêtre n'existe pas déjà
        pygame.init()
        pygame.display.init()
        self.window = pygame.dsplay.set_mode(
            (self.window_size, self.window_size)
        )
    if self.clock is None and self.render_mode == "human":
        self.clock = pygame.time.Clock()
    
    canvas = pygame.Surface((self.window_size, self.window_size))
    canvas.fill(255,255,255)
    pix_square_size = (
        self.window_size / self.size #taille d'un seul carré en pixel
    )

    #on dessine le target
    pygame.draw.rect(
        canvas, 
        (255,0,0), 
        pygame.Rect(
            pix_square_size * self._target_location,
            (pix_square_size, pix_square_size) ,
        ),
        )

    #on dessine l'agent
    pygame.draw.cirle(
        canvas,
        (0, 0, 255), 
        (self._agent_location + 0.5) * pix_square_size,
        pix_square_size / 3. 
    )
    #On fait des lignes de la grille
    for x in range(self.size + 1) : 
        pygame.draw.line(
            canvas, 
            0,
            (0, pix_square_size * x),
            (self.window_size, pix_square_size * x),
            width = 3
        )

    pygame.draw.line(
            canvas,
             0,
            (pix_square_size * x, 0),
            (pix_square_size * x, self.window_size),
            width=3,
        )

    if self.render_mode == "human" :
        self.window.blit(canvas, canvas.get_rect()) #copie les dessins à la fenêtre
        pygame.event.pump()
        pygame.display.update()

        #on s'assure que le rendu humain respecte les ips, donc on ajoute du délai automatique
        self.clock.tick(self.metadata["render_fps"])
    else : #pour le rendu rgb
        return np.transpose(
            np.array(pygame.surfarray.pixes3d(canvas)), axes = (1,0,2)

        )
def close(self) : 
    if self.window is not None :
        pygame.display.quit()
        pygame.quit()



