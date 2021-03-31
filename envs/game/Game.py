# -*- coding: utf-8 -*-
import random

from envs.game.Grid import Grid, MAX_STAGE

class Player:
    def __init__(self, _id, _cell):
        self.id = _id
        self.cell = _cell


NB_PLAYERS = 2
GRID_RAY = 2


class Game:
  
    def __init__(self):
        """Initialize the game.
        - a value between 'relative', 'absolute'
        """ 
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
        self.grid = Grid(self, ray=GRID_RAY)

        # Instanciate the players
        cells = random.sample(self.grid.get_all_cells(), NB_PLAYERS)
        self.players = [Player(i, cell) for i, cell in enumerate(cells)]
        
        # Lancement de la partie 
        self.next_player_id = random.randrange(NB_PLAYERS)
        self._next_player()

        # Init infos
        self._update_infos('playing')
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

        # Update infos
        self._update_infos('playing')
        
        # Prepare next turn
        self._next_player()


    def state(self): 
        """Return the state of the game.
        """
        # Retrieve all cells
        state = {}
        cell_from = self.grid.get_cell_from_coord(0, 0)
        cells = self.grid.get_neighbors(cell_from, 
                                        ray=GRID_RAY, with_none=True)

        # Height of each cell between 0 and 1
        cells_stage = [(c.stage - 1)/(MAX_STAGE - 1) if c is not None 
            else 0 for c in cells]
        state['cells_stage'] = cells_stage

        # Boolean if cell is taken
        opponents = []
        for c in cells:
            player = self._get_player_on_cell(c)
            if player is not None and player != self.current_player:
                opponents += [1]
            else:
                opponents += [0]                
        state['opponents'] = opponents

        # Boolean if current_player is on cell
        current_player = []
        for c in cells:
            player = self._get_player_on_cell(c)
            if player == self.current_player:
                current_player += [1]
            else:
                current_player += [0]                
        state['current_player'] = current_player

        return state


    def observations(self):
        """Return state and infos.
        """
        return self.state(), self.infos


    def is_game_over(self):
        return self.game_over


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


    def _update_infos(self, status):
        self.infos['player_id'] = self.current_player.id
        self.infos['status'] = status


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
        self.next_player_id = (self.next_player_id + 1) % NB_PLAYERS


    def _get_player_on_cell(self, cell):
        """Get the player on the cell, None if no player on the cell.
        - cell : instance of the Cell
        """
        for p in self.players:
            if p.cell == cell:
                return p
