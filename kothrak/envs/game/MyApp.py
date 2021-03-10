# -*- coding: utf-8 -*-

import sys
import random

from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtCore import Qt

from kothrak.envs.game.Utils import APP_PIXDIM, MESSAGE_PIXDIM, NB_PLAYERS, GRID_RAY
from kothrak.envs.game.Grid import Grid
from kothrak.envs.game.Player import Player
from kothrak.envs.game.Cell import Cell



class MyApp:

    REWARDS = {'init': {'current': 0, 'others': 0}, 
               'win': {'current': 100, 'others': -100},
               'invalid_attempt': {'current': -100, 'others': 20}}
    
    def __init__(self, parent=None):
        # Initialisation de la fenetre
        self.window = QWidget(parent)
        self.window.resize(APP_PIXDIM[0], APP_PIXDIM[1])
        self.window.setWindowTitle('MyApp')
        self.window.mouseReleaseEvent = lambda event: self._on_click(event)
        self.window.setObjectName('game_bg')
        # self.window.keyReleaseEvent=lambda event:self.on_keyboard(event)

        # Initialisation du message
        self.message = QLabel(self.window)
        self.message.setGeometry(0, 30, MESSAGE_PIXDIM[0], MESSAGE_PIXDIM[1])
        self.message.setText("")
        self.message.setAlignment(Qt.AlignCenter)
        self.message.setObjectName('message')

        # Initialisation des variables
        self.players = []
        self.reward_dict = {}
        self.next_player_id = -1
        self.current_player = None
        self.current_step = 'init'

        # Init new game
        self.new_game()


    def new_game(self):
        """ Init and run a new game, clear the display if a previous game has been launched."""

        def create_players(self, nb_players=NB_PLAYERS):
            
            cells = self.grid.get_all_cells()
            cells_selected = []
            
            for player_id in range(nb_players):
                c = random.choice(cells)
                cells.remove(c)
                cells_selected += [(player_id, c)]

            cells_selected = sorted(cells_selected, key=lambda x: x[1].r)

            self.players = []
            for player_id, cell in cells_selected:
                self.players += [Player(player_id, cell, self)]

        def init_game(self):
            self.next_player_id = random.choice(range(len(self.players)))
            self._next_player()
            self.current_step = 'move'         
            self._update_rewards('init')
        

        # Nettoyage de l'affichage
        if self.current_step != 'init':
            self._clear_display()

        # Initialisation des Cells
        self.grid = Grid(self)

        # Initialisation des Players
        create_players(self)

        # Initialisation des joueurs et récompense
        # Lancement de la partie 
        init_game(self)
        self._update_message()


    def play(self, q, r):
        """Make the play (move or build) on the cell on relative coordinates [q, r] 
        from the player."""

        def invalid_attempt(self, attempt):
            """The player made an invalid attempt, end the game and punish him."""
            self.current_step = 'game_over'
            self._update_rewards('invalid_attempt')       
            print(f'Invalid {attempt}. Player {self.current_player.player_id} got eliminated.')

        # Retrieve the corresponding cell
        q = self.current_player.cell.q + q
        r = self.current_player.cell.r + r 
        cell = self.grid.get_cell_from_coord(q, r)

        # If cell is out of map, game is over
        if cell is None:
            invalid_attempt(self, 'cell')

        # Make the move is the game is still playing
        if not self.is_game_over():

            if self.current_step == 'move':
                if cell in self.grid.get_neighbors(self.current_player.cell) \
                    and self._get_player_on_cell(cell) is None \
                        and cell.stage <= self.current_player.cell.stage + 1:
                    self.current_player.move(cell)
                    self.current_step = 'build'
                    if self._player_on_top():
                        self.current_step = 'game_over'
                        self._update_rewards('win')
                        print('Player {} won the game.'.format(self.current_player.player_id))
                else:
                    invalid_attempt(self, 'move')

            elif self.current_step == 'build':
                if cell in self.grid.get_neighbors(self.current_player.cell) \
                    and self._get_player_on_cell(cell) is None \
                        and cell.stage < cell.MAX_STAGE:
                    cell.grew()
                    self._next_player()
                    self.current_step = 'move'
                else:
                    invalid_attempt(self, 'build')

            self._update_message()


    def state(self): 
        """Return the state of the grid."""
        state = {}

        cell_from = self.current_player.cell
        cells_around = self.grid.get_neighbors(cell_from, ray=GRID_RAY, with_none=True) 

        # Hauteur de chaque cellule
        cells_stage = [c.stage/Cell.MAX_STAGE if c is not None else 0 for c in cells_around]
        state['cells_stage'] = cells_stage

        # Boolean if cell is taken
        cells_taken = []
        for c in cells_around:
            player = self._get_player_on_cell(c)
            cells_taken += [1] if player else [0]                
        state['cells_taken'] = cells_taken

        # Step
        if self.current_step == 'move':
            state['step'] = [1, 0]
        elif self.current_step == 'build':
            state['step'] = [0, 1]
        elif self.current_step == 'game_over':
            state['step'] = [0, 0]
        else:
            raise Exception(f'Error in state from MyApp: current step invalid:\
                 {self.current_step}.')

        return state
    
    def evaluate(self, player_id=0):
        return self.reward_dict[player_id]

    def show(self):
        """Display the window on the screen."""
        self.window.show()
    
    def is_game_over(self):
        return self.current_step == 'game_over'
    


    def _update_rewards(self, reason):
        """Récompense et/ou puni les joueurs en fonction du motif."""
        pid = self.current_player.player_id
        players_ids = [p.player_id for p in self.players]
        for k in players_ids:
            if k == pid:
                self.reward_dict[k] = self.REWARDS[reason]['current']
            else:
                self.reward_dict[k] = self.REWARDS[reason]['others']

    def _update_message(self):
        if not self.is_game_over():
            text = 'Player {}: {}'.format(self.current_player.player_id + 1, self.current_step)
        else:
            text = 'Game Over'
        self.message.setText(text)

    def _player_on_top(self):
        return self.current_player.cell.stage == Cell.MAX_STAGE


    def _next_player(self):
        self.current_player = self.players[self.next_player_id]
        self.next_player_id = (self.next_player_id + 1) % NB_PLAYERS



    def _on_click(self, event):
        x, y = event.pos().x(), event.pos().y()
        cell = self.grid.get_cell_from_pos(x, y)
        if cell is not None:
            q_relative = cell.q - self.current_player.cell.q
            r_relative = cell.r - self.current_player.cell.r
            self.play(q_relative, r_relative)
        
        # Uncomment to restart a game right after the end off the previous one
        # if self.is_game_over():
        #     self.new_game()

    def _get_player_on_cell(self, cell):
        for p in self.players:
            if p.cell == cell:
                return p

    
    def _clear_display(self):
        for p in self.players:
            p.delete()
        self.grid.delete()
        del(self.grid)
        del(self.players)


style = '''
QWidget {
    background-color: rgb(40, 41, 35);
} 
QLabel {
    background-color: rgba(255, 255, 255, 0);
}
QLabel#message {
    color:rgb(210, 90, 20);
    font:30pt;
}
'''

def run():
    qapp = QApplication(sys.argv)
    qapp.setStyleSheet(style)
    main_window = MyApp()
    main_window.show()
    qapp.exec_()
