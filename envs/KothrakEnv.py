from envs.game.Game import Game
from envs.game.Grid import DIR_COORDS

def transform_number_into_actions(action):
    move = int(action / len(DIR_COORDS))
    build = int(action % len(DIR_COORDS))
    return DIR_COORDS[move], DIR_COORDS[build]

def transform_actions_into_number(move, build):
    action_move = -1
    action_build = -1

    for i, coord in enumerate(DIR_COORDS):
        if coord[0] == move[0] and coord[1] == move[1]:
            action_move = i

    if build is not None:
        for i, coord in enumerate(DIR_COORDS):
            if coord[0] == build[0] and coord[1] == build[1]:
                action_build = i
    else:
        action_build = 0

    return action_move*len(DIR_COORDS) + action_build


class KothrakEnv():
    
    def __init__(self, nb_players, grid_ray):
        """Initialize the environnement.
        """
        # Init game and rewards
        self.game = Game(nb_players, grid_ray)
        self.rewards = {pid: 0 for pid in range(self.game.nb_players)}

        # Initialise les actions et observations
        # (3n^2 + 3n + 1)
        nb_cells = 3*self.game.grid_ray**2 + 3*self.game.grid_ray + 1  
        self.num_observations = 3*nb_cells
        self.num_actions = 6*6

    
    def reset(self):
        """Reset the environnement for a new game."""
        self.game.new_game()
        self.rewards = {pid: 0 for pid in range(self.game.nb_players)}
        
        state, infos = self._get_observation()
        self._update_rewards(infos)
        
        return state, infos


    def step(self, action):
        """Make an action to the game and return observations, reward, done
        and a list of informations (set to null for now).
        - action : An integer representing the action to make
        """
        # Make action
        action_move, action_build = transform_number_into_actions(action)
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

    
    def get_mask_play(self, infos):
        """Return the mask for all the 36 actions in function of possible
        plays"""

        loc = [self.game.current_player.cell.q, 
                                        self.game.current_player.cell.r]
        plays = []
        for play in infos['possible_plays']:
            # Possible plays are in absolute location, 
            # transform them to relative
            rel_move = [play['move'][0] - loc[0], play['move'][1] - loc[1]]
            if play['build'] is None:
                rel_build = None 
            else:
                rel_build = [play['build'][0] - play['move'][0], 
                                    play['build'][1] - play['move'][1]]

            plays += [[rel_move, rel_build]]
        
        actions = [transform_actions_into_number(*p) for p in plays]

        mask = [1 if n in actions else 0 for n in range(36)]
        print('\n')
        print('coord', loc)
        print('pp', infos['possible_plays'])
        print('plays', plays)
        print('actions', actions)
        print(mask)
        return mask


    def _get_observation(self):
        """Retrieve state and informations from game and return vectorized state.
        """
        state_dict, infos = self.game.observations()
        # Vectorize state
        state = []
        for dico in state_dict.values():
            for q, values in dico.items():
                for r in values:
                    state += [dico[q][r]]
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
        for pid in range(self.game.nb_players):
            if pid == player_id:
                self.rewards[pid] = reward_values[status]['player']
            else:
                self.rewards[pid] = reward_values[status]['others']

        return self.rewards


