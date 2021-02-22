import sys
import asyncio
from PyQt5.QtWidgets import QApplication

import gym
import numpy as np

from envs.game.MyApp import MyApp, style
from envs.game.Utils import GRID_RAY, NB_CELLS, NB_PLAYERS
from envs.game.Cell import Cell

class KothrakEnv(gym.Env):
    
    def __init__(self):
        # Initalise l'app
        self.qapp = QApplication(sys.argv)
        self.qapp.setStyleSheet(style)
        self.game = MyApp()
        self.game.show()

        # Initialise les actions et observations
        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Dict({
            'players': gym.spaces.Box(low=-GRID_RAY, high=GRID_RAY, shape=(2*NB_PLAYERS,)),
            'cells': gym.spaces.Box(low=1, high=Cell.MAX_STAGE, shape=(NB_CELLS,)),
            'step': gym.spaces.Discrete(2)
        })
    
    def reset(self):
        self.game.new_game()
        

    def step(self, action):
        pass
    
    def render(self, mode='human'):
        pass

    
    
