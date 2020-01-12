# -*- coding: utf-8 -*-

import sys
import numpy as np
import random

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtCore import Qt


COEF = 0.3
GRID_RAY = 2

APP_PIXDIM = (2800 * COEF, 2300 * COEF)
MESSAGE_PIXDIM = (APP_PIXDIM[0], 200 * COEF)

IMGCELL_PIXDIM = (456*COEF, 342*COEF)
IMGPLAYER_PIXDIM = (590 * COEF * 0.8, 632 * COEF * 0.8)
PIXSIZE_STAGE_CELL = [0, 80 * COEF]
PIXSIZE_SHADOW_PLAYER = [90 * COEF, 0]

PIXSIZE_VECTOR_PLAYER_CELL = [IMGCELL_PIXDIM[0]/2 - IMGPLAYER_PIXDIM[0]/2 - PIXSIZE_SHADOW_PLAYER[0]/2, IMGCELL_PIXDIM[1]/2 - IMGPLAYER_PIXDIM[1]]
POS_CENTER = [(APP_PIXDIM[0] - IMGCELL_PIXDIM[0])/2, MESSAGE_PIXDIM[1] + (APP_PIXDIM[1] - IMGCELL_PIXDIM[1])/2]

MV_R = [450 * COEF, 0]
MV_DR = [228 * COEF, 190 * COEF]



class MyApp :
    
    def __init__(self) :
        
        def create_players(self, nb_players=2) :
            
            cells = self.grid.get_all_cells()
            cells_selected = []
            
            for player_id in range(nb_players) :
                c = random.choice(cells)
                cells.remove(c)
                cells_selected += [(player_id, c)]

            cells_selected = sorted(cells_selected, key=lambda x : x[1].r)

            for player_id, cell in cells_selected :
                self.players += [Player(player_id, cell, self)]

        def init_game(self) :
            self.next_player_id = random.choice(range(len(self.players)))
            self.next_player()
            self.current_step = 'move'


        # Initialisation de la fenetre
        self.window = QWidget()
        self.window.resize(APP_PIXDIM[0], APP_PIXDIM[1])
        self.window.setWindowTitle('MyApp')
        self.window.mouseReleaseEvent=lambda event:self.on_click(event)
        self.window.keyReleaseEvent=lambda event:self.on_keyboard(event)

        # Initialisation du message
        self.message = QLabel(self.window)
        self.message.setGeometry(0, 0, MESSAGE_PIXDIM[0], MESSAGE_PIXDIM[1])
        self.message.setText("")
        self.message.setAlignment(Qt.AlignCenter)
        self.message.setObjectName('Hello')

        # Initialisation des Cells
        self.grid = Grid(self)

        # Initialisation de la partie
        self.players = []
        self.next_player_id = -1
        self.current_player = None
        self.current_step = ''
        create_players(self)
        init_game(self)
        self.update_message()

    def update_message(self) :
        if self.current_step != 'game_over' :
            text = 'Player {} : {}'.format(self.current_player.player_id + 1, self.current_step)
        else :
            text = 'Game Over'
        self.message.setText(text)

    def is_game_over(self) :
        return self.current_player.cell.stage == Cell.MAX_STAGE

    def get_neighbors(self, cell) :
        dir_coords = [(0, -1), (-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1)]
        neighbors_coord = [(cell.q + c[0], cell.r + c[1]) for c in dir_coords]
        neighbors = [self.grid.get_cell_from_coord(q, r) for q, r in neighbors_coord]
        neighbors = [c for c in neighbors if c is not None]
        return neighbors

    def next_player(self) :
        self.current_player = self.players[self.next_player_id]
        self.next_player_id = (self.next_player_id + 1) % 2

    def play(self, q, r) :

        q = self.current_player.cell.q + q
        r = self.current_player.cell.r + r 
        cell = self.grid.get_cell_from_coord(q, r)

        if self.current_step != 'game_over' :

            if self.current_step == 'move' :
                if cell in self.get_neighbors(self.current_player.cell) \
                    and self.get_player_on_cell(cell) is None \
                    and cell.stage <= self.current_player.cell.stage + 1 :
                    self.current_player.move(cell)
                    self.current_step = 'build'
                    if self.is_game_over() :
                        self.current_step = 'game_over'
                        print('Game over !')
                        print('Player {} won the game.'.format(self.current_player.player_id))

            elif self.current_step == 'build' and self.get_player_on_cell(cell) is None :
                if cell in self.get_neighbors(self.current_player.cell) :
                    if cell.stage < cell.MAX_STAGE :
                        cell.grew()
                        self.next_player()
                        self.current_step = 'move'

            self.update_message()


    def on_click(self, event) :
        x, y = event.pos().x(), event.pos().y()
        cell = self.grid.get_cell_from_pos(x, y)
        print(cell.q, cell.r)
        if cell is not None :
            q_relative = cell.q - self.current_player.cell.q
            r_relative = cell.r - self.current_player.cell.r
            self.play(q_relative, r_relative)

    def get_player_on_cell(self, cell) :
        for p in self.players :
            if p.cell == cell :
                return p

    def show(self) :
        self.window.show()




class Grid :

    def __init__(self, app, ray=GRID_RAY) :
        self.ray = ray
        self.app = app
        self.grid = []
        self.create_grid()

    def create_grid(self) :
        """Create the map"""

        def create_one_cell(self, q, r) :
            pos_x = POS_CENTER[0] + MV_R[0]*q + MV_DR[0]*r
            pos_y = POS_CENTER[1] + MV_DR[1]*r
            return Cell(pos_x, pos_y, q, r, self.app) 

        def create_one_line(self, nb_cell, q, r) :
            for i in range(nb_cell) :
                c = create_one_cell(self, q, r)
                self.grid[r + self.ray] += [c]
                q += 1
            return q, r

        nb_cell = self.ray + 1
        q = 0
        r = -self.ray
        for i in range(self.ray) :
            self.grid += [list()]
            q, r = create_one_line(self, nb_cell, q, r)
            nb_cell += 1
            r += 1
            q = -self.ray -r
        for i in range(self.ray + 1) :
            self.grid += [list()]
            q, r = create_one_line(self, nb_cell, q, r)
            r += 1
            q = -self.ray
            nb_cell -= 1

    def get_cell_from_coord(self, q, r) :
        for line in self.grid :
            for cell in line :
                if cell.q == q and cell.r == r :
                    return cell

    def get_cell_from_pos(self, x, y) :
        cells = []
        for r in reversed(range(-self.ray, self.ray+1)) :
            for line in self.grid :
                for c in line :
                    if c.r == r :
                        cells += [c]

        for c in cells :
            if c.is_pos_in_cell(x, y) :
                return c

    def get_all_cells(self) :
        cells = []
        for line in self.grid :
            for cell in line :
                 cells += [cell]
        return cells

class Player :

    PATH_IMG = "img/player_img{}.png"

    def __init__(self, player_id, cell, app) :
        self.app = app
        self.player_id = player_id
        self.cell = None
        self.x = 0
        self.y = 0
        self.size_x = IMGPLAYER_PIXDIM[0]
        self.size_y = IMGPLAYER_PIXDIM[1]

        self.img = QLabel(self.app.window)
        # self.img.setGeometry(QtCore.QRect(self.x, self.y, self.size_x, self.size_y))
        self.img.setPixmap(QtGui.QPixmap(self.PATH_IMG.format(self.player_id)))
        self.img.setScaledContents(True)
        self.img.setObjectName("Player_{}".format(self.player_id))
        self.move(cell)

    def move(self, cell) :
        self.cell = cell
        self.x = cell.x + PIXSIZE_VECTOR_PLAYER_CELL[0]
        self.y = cell.y + PIXSIZE_VECTOR_PLAYER_CELL[1]
        self.img.setGeometry(QtCore.QRect(self.x, self.y, self.size_x, self.size_y))
        


class Cell :

    PATH_IMG = "img/cell_{}.png"
    # PATH_IMG = 'cell1_resized.png'
    X_MIN_LIM = 10 * COEF
    X_MAX_LIM = 450 * COEF
    Y_MIN_LIM = 0 * COEF
    Y_MAX_LIM = 260 * COEF
    CORNER_LEFTUP_PIXPOS = [X_MIN_LIM, 70 * COEF]
    CORNER_LEFTDOWN_PIXPOS = [X_MIN_LIM, 190 * COEF]
    CORNER_UP_PIXPOS = [230 * COEF, Y_MIN_LIM]
    CORNER_DOWN_PIXPOS = [230 * COEF, Y_MAX_LIM]
    CORNER_RIGHTUP_PIXPOS = [X_MAX_LIM, 70 * COEF]
    CORNER_RIGHTDOWN_PIXPOS = [X_MAX_LIM, 190 * COEF]

    MAX_STAGE = 4

    def __init__(self, x, y, q, r, app) :
        self.app = app
        self.q = q
        self.r = r
        self.x = x
        self.y = y
        self.size_x = IMGCELL_PIXDIM[0]
        self.size_y = IMGCELL_PIXDIM[1]
        self.stage = 1

        self.img = QLabel(self.app.window)
        self.img.setGeometry(QtCore.QRect(self.x, self.y, self.size_x, self.size_y))
        self.img.setPixmap(QtGui.QPixmap(self.PATH_IMG.format(self.stage)))
        self.img.setScaledContents(True)
        self.img.setObjectName("cell_{}_{}".format(q, r))

    def grew(self) :
        if self.stage >= Cell.MAX_STAGE :
            print("Cannot grow anymore.")
            return
        self.stage += 1
        self.change_img()

        player = self.app.get_player_on_cell(self)
        if player != None :
            player.move(self)

    def change_img(self) :
        self.y -= PIXSIZE_STAGE_CELL[1]
        self.size_y += PIXSIZE_STAGE_CELL[1]
        self.img.setGeometry(QtCore.QRect(self.x, self.y, self.size_x, self.size_y))
        self.img.setPixmap(QtGui.QPixmap(self.PATH_IMG.format(self.stage)))

    def absolute_to_relative_position(self, x, y) :
        return (x - self.x, y - self.y)

    def is_pos_in_cell(self, x, y) :
        """On vérifie que x et y sont bien dans l'image et non dans le fond transparent."""

        x, y = self.absolute_to_relative_position(x, y)
        
        # Up
        segments_up = [(self.CORNER_LEFTUP_PIXPOS, self.CORNER_UP_PIXPOS),
                     (self.CORNER_UP_PIXPOS, self.CORNER_RIGHTUP_PIXPOS)]
        for p, q in segments_up :
             coef_dir = (q[1] - p[1])/(q[0] - p[0])
             b = p[1] - coef_dir * p[0]
             if coef_dir * x + b - y > 0 :
                 return False

        # Down
        segments_down = [(self.CORNER_DOWN_PIXPOS, self.CORNER_RIGHTDOWN_PIXPOS),
                         (self.CORNER_LEFTDOWN_PIXPOS, self.CORNER_DOWN_PIXPOS)]
        for m, n in segments_down :
            p = (m[0], m[1] + PIXSIZE_STAGE_CELL[1] * self.stage)
            q = (n[0], n[1] + PIXSIZE_STAGE_CELL[1] * self.stage)
            coef_dir = (q[1] - p[1])/(q[0] - p[0])
            b = p[1] - coef_dir*p[0] 
            if coef_dir*x + b - y < 0 :
                return False

        # Sides
        if x < self.X_MIN_LIM or x > self.X_MAX_LIM :
            return False

        return True


style = '''
QWidget {
    background-color: rgb(40, 41, 35);
} 
QLabel {
    background-color: rgba(255, 255, 255, 0);
}
QLabel#Hello {
    color:rgb(210, 90, 20);
    font:30pt;
}
'''

if __name__ == '__main__' :
    qapp = QApplication(sys.argv)
    qapp.setStyleSheet(style)
    main_window = MyApp()
    main_window.show()
    sys.exit(qapp.exec_())