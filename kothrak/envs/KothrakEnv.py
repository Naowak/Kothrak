from PyQt5.QtWidgets import QApplication
import sys
import time

import gym
import numpy as np

from kothrak.envs.game.MyApp import MyApp, style
from kothrak.envs.game.Utils import GRID_RAY, NB_CELLS, NB_PLAYERS
from kothrak.envs.game.Cell import Cell

TIME_TO_SLEEP = 1

def transform_action(action):
    coord_actions = [(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)]
    return coord_actions[action]


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
        obs = self._get_observation()
        return obs

    def step(self, action):
        q, r = transform_action(action)
        self.game.play(q, r)
        # time.sleep(TIME_TO_SLEEP)

        obs = self._get_observation()
        reward = self.game.evaluate()
        done = self.game.is_game_over()

        return obs, reward, done, {}
    
    def render(self, mode='human'):
        pass

    def _get_observation(self):
        obs = self.game.state()
        observations = []
        for v in obs.values():
            observations += v
        observations = np.array(observations)
        return observations
    
    
