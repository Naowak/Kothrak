from envs.game.Game import Game, GRID_RAY, NB_PLAYERS

def transform_action(action):
    coord_actions = [(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)]
    move = int(action / len(coord_actions))
    build = int(action % len(coord_actions))
    return coord_actions[move], coord_actions[build]


class KothrakEnv():
    
    def __init__(self):
        """Initialize the environnement.
        """
        # Init game and rewards
        self.game = Game()
        self.rewards = {pid: 0 for pid in range(NB_PLAYERS)}

        # Initialise les actions et observations
        nb_cells = 3*GRID_RAY**2 + 3*GRID_RAY + 1  # (3n^2 + 3n + 1)
        self.num_actions = 6*6
        self.num_observations = 3*nb_cells

    
    def reset(self):
        """Reset the environnement for a new game."""
        self.game.new_game()
        self.rewards = {pid: 0 for pid in range(NB_PLAYERS)}
        
        state, infos = self._get_observation()
        self._update_rewards(infos)
        
        return state, infos


    def step(self, action):
        """Make an action to the game and return observations, reward, done
        and a list of informations (set to null for now).
        - action : An integer representing the action to make
        """
        # Make action
        action_move, action_build = transform_action(action)
        self.game.play(action_move, action_build)

        # Get 
        state, infos = self._get_observation()
        rewards = self._update_rewards(infos)
        done = self.game.is_game_over()

        return state, rewards, done, infos
    

    def render(self, mode='human'):
        """Not implemented.
        """
        pass


    def _get_observation(self):
        """Retrieve state and informations from game and return vectorized state.
        """
        state_dict, infos = self.game.observations()
        state = []
        for dico in state_dict.values():
            state += list(dico.values())
        return state, infos


    def _update_rewards(self, infos):
        """Récompense et/ou puni les joueurs en fonction des informations
        de la game.
        """
        reward_values = {'new_game': {'player': 0, 'others': 0},
                        'win': {'player': 100, 'others': 0},  # noqa: F841
                        'eliminated': {'player': -100, 'others': 1},
                        'playing': {'player': 0, 'others': 0}}

        player_id = infos['player_id']
        status = infos['status']

        # Set reward to all players
        for pid in range(NB_PLAYERS):
            if pid == player_id:
                self.rewards[pid] = reward_values[status]['player']
            else:
                self.rewards[pid] = reward_values[status]['others']

        return self.rewards
