from kothrak.game.Utils import NB_CELLS
from kothrak.game.MyApp import MyApp

def transform_action(action):
    coord_actions = [(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)]
    return coord_actions[action]


class KothrakEnv():
    
    def __init__(self, qapp, window=None, state_mode='relative'):
        """Initialize the environnement.
        - qapp : main QApplication
        - window : main window (optional)
        """
        # Create the game
        self.game = MyApp(qapp, window, state_mode=state_mode)

        # Initialise les actions et observations
        self.num_actions = 6
        self.num_observations = 3*NB_CELLS+2
    
    def reset(self):
        """Reset the environnement for a new game."""
        self.game.new_game()
        obs = self._get_observation()
        return obs

    def step(self, action):
        """Make an action to the game and return observations, reward, done
        and a list of informations (set to null for now).
        - action : An integer representing the action to make
        """
        q, r = transform_action(action)
        self.game.play(q, r)

        obs = self._get_observation()
        players_reward = self.game.evaluate()
        done = self.game.is_game_over()

        return obs, players_reward, done, {}
    
    def render(self, mode='human'):
        """Not implemented.
        """
        pass

    def _get_observation(self):
        obs = self.game.state()
        observations = []
        for v in obs.values():
            observations += v
        return observations
