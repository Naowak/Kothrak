import gym
import numpy as np

from kothrak.envs.game.Utils import GRID_RAY, NB_CELLS, NB_PLAYERS
from kothrak.envs.game.Cell import Cell

def transform_action(action):
    coord_actions = [(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)]
    return coord_actions[action]


class KothrakEnv(gym.Env):
    
    def __init__(self):
        self.game = None

        # Initialise les actions et observations
        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Dict({
            'cells_stage': gym.spaces.Box(low=0, high=1, shape=(NB_CELLS,)),
            'cells_taken': gym.spaces.Discrete(NB_CELLS),
            'step': gym.spaces.Discrete(2)
        })
    
    def set_game(self, game):
        self.game = game
    
    def reset(self):
        self.game.new_game()
        obs = self._get_observation()
        return obs

    def step(self, action):
        q, r = transform_action(action)
        self.game.play(q, r)

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
        print(observations)
        return observations
    
    
