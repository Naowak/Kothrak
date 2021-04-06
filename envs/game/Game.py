# -*- coding: utf-8 -*-
import random

from envs.game.Grid import Grid, MAX_STAGE, DIR_COORDS

class Player:
    def __init__(self, _id, _cell):
        self.id = _id
        self.cell = _cell


class Game:
  
    def __init__(self, nb_players=2, grid_ray=2):
        """Initialize the game.
        - a value between 'relative', 'absolute'
        """ 
        self.nb_players = nb_players
        self.grid_ray = grid_ray
        # Initialisation des variables
        self.grid = None
        self.current_player = None
        self.players = []
        self.next_player_id = -1
        self.game_over = False
        self.infos = {}


    def new_game(self):
        """ Init and run a new game, clear the display if a previous game has 
        been launched.
        """
        # Initialisation des Cells
        self.grid = Grid(self, ray=self.grid_ray)

        # Instanciate the players
        cells = random.sample(self.grid.get_all_cells(), self.nb_players)
        self.players = [Player(i, cell) for i, cell in enumerate(cells)]
        
        # Lancement de la partie 
        self.next_player_id = random.randrange(self.nb_players)
        self._next_player()

        # Init infos
        settings = self._settings()
        players_location = self._players_location()
        self._update_infos('new_game', 
                            settings=settings, 
                            players_location=players_location)
        self.game_over = False


    def play(self, rel_coord_move, rel_coord_build):
        """Make the play (move or build) on the cell on relative coordinates [q, r] 
        from the player.
        - rel_coord_move : relative [q, r] from current_player 
        - rel_coord_build : relative [q, r] from current_player
        """        
        # Do nothing if game is over 
        if self.game_over:
            return
        
        # Retrieve move cell
        q_move = self.current_player.cell.q + rel_coord_move[0]
        r_move = self.current_player.cell.r + rel_coord_move[1]
        cell_move = self.grid.get_cell_from_coord(q_move, r_move)

        if not self._is_move_correct(cell_move, self.current_player.cell):
            self._end_game('invalid move')
            return

        # Retrieve build cell
        q_build = cell_move.q + rel_coord_build[0]
        r_build = cell_move.r + rel_coord_build[1]
        cell_build = self.grid.get_cell_from_coord(q_build, r_build)

        if not self._is_build_correct(cell_build, cell_move):
            self._end_game('invalid build')
            return

        # Make the move
        self.current_player.cell = cell_move

        # Verify if current player won
        if self.current_player.cell.stage == MAX_STAGE:
            self._end_game('win')
            return

        # Make the build
        cell_build.stage += 1

        # Prepare next turn
        self._next_player()

        # Update infos
        self._update_infos('playing', move=rel_coord_move, 
            build=rel_coord_build)


    def state(self): 
        """Return the state of the game.
        """
        # Retrieve all cells
        state = {}
        cell_from = self.grid.get_cell_from_coord(0, 0)
        cells = self.grid.get_neighbors(cell_from, 
                                        ray=self.grid_ray, with_none=True)

        # Height of each cell between 0 and 1
        cells_stage = {}
        for c in cells:
            if c.q not in cells_stage.keys():
                cells_stage[c.q] = {}
            cells_stage[c.q][c.r] = (c.stage - 1)/(MAX_STAGE - 1)
        state['cells_stage'] = cells_stage

        # Boolean if cell is taken
        opponents = {}
        for c in cells:
            if c.q not in opponents.keys():
                opponents[c.q] = {}

            player = self._get_player_on_cell(c)
            if player is not None and player != self.current_player:
                opponents[c.q][c.r] = 1
            else:
                opponents[c.q][c.r] = 0                
        state['opponents'] = opponents

        # Boolean if current_player is on cell
        current_player = {}
        for c in cells:
            if c.q not in current_player.keys():
                current_player[c.q] = {}

            player = self._get_player_on_cell(c)
            if player == self.current_player:
                current_player[c.q][c.r] = 1
            else:
                current_player[c.q][c.r] = 0                
        state['current_player'] = current_player

        return state


    def observations(self):
        """Return state and infos.
        """
        return self.state(), self.infos


    def is_game_over(self):
        return self.game_over


    def _get_possible_plays(self):
        """Return combinaisons of all possibles moves and build.
        """
        plays = []
        for move in DIR_COORDS:
            for build in DIR_COORDS:

                q_move = self.current_player.cell.q + move[0]
                r_move = self.current_player.cell.r + move[1]
                cell_move = self.grid.get_cell_from_coord(q_move, r_move)

                # Move incorrect, next_move
                if not self._is_move_correct(cell_move, 
                                                self.current_player.cell):
                    continue

                # Play won, set build to None
                if cell_move.stage == MAX_STAGE:
                    c_move = [cell_move.q, cell_move.r]
                    c_build = None
                    plays += [{'move': c_move, 'build': c_build}]
                    continue

                # Correct move but player didn't win
                q_build = cell_move.q + build[0]
                r_build = cell_move.r + build[1]
                cell_build = self.grid.get_cell_from_coord(q_build, r_build)

                # Correct build, set move and build
                if self._is_build_correct(cell_build, cell_move):
                    c_move = [cell_move.q, cell_move.r]
                    c_build = [cell_build.q, cell_build.r]
                    plays += [{'move': c_move, 'build': c_build}]

        return plays



    def _settings(self):
        """Return the settings of the game.
        """
        return {'MAX_STAGE': MAX_STAGE, 'RAY': self.grid_ray, 
            'NB_PLAYERS': self.nb_players}


    def _players_location(self):
        """Return the location for each player.
        """
        players_location = {}
        for player in self.players:
            location = [player.cell.q, player.cell.r]
            players_location[player.id] = location
        return players_location


    def _end_game(self, reason):
        """Update self.infos and set self.game_over to True.
        """
        if reason == 'win':
            print(f'{self.current_player.id} won the game.')
            self._update_infos('win')
        else:
            print(f'{self.current_player.id} eliminated : {reason}')
            self._update_infos('eliminated')

        self.game_over = True


    def _update_infos(self, status, **infos):
        """Update the informations of the turn. 
        - status indicate in which state of the game we are
        a. If status in new_game, player_location inficate the beginning position
        for all players.
        b. If the game isn't over (status win or eliminated), then player_id
        indicates which player is about to play, and possible_plays indicate
        which plays he can do.
        c. Else player_id indicate who won or lose the game.
        d. If status is in playing, move and build are available and indicates
        the coord of the play.
        """
        self.infos = {}
        self.infos['player_id'] = self.current_player.id
        self.infos['status'] = status

        if status == 'new_game' or status == 'playing':
            self.infos['possible_plays'] = self._get_possible_plays()
        
        for k, v in infos.items():
            self.infos[k] = v


    def _is_move_correct(self, cell, cell_from):
        return cell is not None and \
            cell in self.grid.get_neighbors(cell_from) and \
            self._get_player_on_cell(cell) is None and \
            cell.stage <= self.current_player.cell.stage + 1


    def _is_build_correct(self, cell, cell_from):
        return cell is not None and \
            cell in self.grid.get_neighbors(cell_from) and \
            self._get_player_on_cell(cell) in [None, self.current_player] and \
            cell.stage < MAX_STAGE


    def _next_player(self):
        """Change current_player to the next one.
        """
        self.current_player = self.players[self.next_player_id]
        self.next_player_id = (self.next_player_id + 1) % self.nb_players


    def _get_player_on_cell(self, cell):
        """Get the player on the cell, None if no player on the cell.
        - cell : instance of the Cell
        """
        for p in self.players:
            if p.cell == cell:
                return p
