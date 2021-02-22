import gym
from gym import spaces
import numpy as np
import sys
from envs.game.MyApp import MyApp, style

class KothrakEnv(gym.Env):
    
    def __init__(self):
        self.qapp = QApplication(sys.argv)
        self.qapp.setStyleSheet(style)
        self.game = MyApp()
        self.game = 'hello'
    
    def reset(self):
        pass
    
    def step(self, action):
        pass
    
    def render(self, mode='human'):
        pass

    
    