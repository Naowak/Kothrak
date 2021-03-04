from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QLabel

from kothrak.envs.game.Utils import PIXSIZE_STAGE_CELL, IMGCELL_PIXDIM, COEF

class Cell:

    PATH_IMG = "kothrak/envs/game/img/cell_{}.png"
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

    def __init__(self, x, y, q, r, app):
        self.app = app
        self.q = q
        self.r = r
        self.x = x
        self.y = y
        self.size_x = IMGCELL_PIXDIM[0]
        self.size_y = IMGCELL_PIXDIM[1]
        self.stage = 1

        self.img = QLabel(self.app.window)
        self.img.setGeometry(QtCore.QRect(
            self.x, self.y, self.size_x, self.size_y))
        self.img.setPixmap(QtGui.QPixmap(self.PATH_IMG.format(self.stage)))
        self.img.setScaledContents(True)
        self.img.setObjectName("cell_{}_{}".format(q, r))
        self.img.show()

    def grew(self):
        if self.stage >= Cell.MAX_STAGE:
            print("Cannot grow anymore.")
            return
        self.stage += 1
        self.change_img()

        player = self.app._get_player_on_cell(self)
        if player != None:
            player.move(self)

    def change_img(self):
        self.y -= PIXSIZE_STAGE_CELL[1]
        self.size_y += PIXSIZE_STAGE_CELL[1]
        self.img.setGeometry(QtCore.QRect(
            self.x, self.y, self.size_x, self.size_y))
        self.img.setPixmap(QtGui.QPixmap(self.PATH_IMG.format(self.stage)))

    def absolute_to_relative_position(self, x, y):
        return (x - self.x, y - self.y)

    def is_pos_in_cell(self, x, y):
        """On vérifie que x et y sont bien dans l'image et non dans le fond transparent."""

        x, y = self.absolute_to_relative_position(x, y)

        # Up
        segments_up = [(self.CORNER_LEFTUP_PIXPOS, self.CORNER_UP_PIXPOS),
                       (self.CORNER_UP_PIXPOS, self.CORNER_RIGHTUP_PIXPOS)]
        for p, q in segments_up:
            coef_dir = (q[1] - p[1])/(q[0] - p[0])
            b = p[1] - coef_dir * p[0]
            if coef_dir * x + b - y > 0:
                return False

        # Down
        segments_down = [(self.CORNER_DOWN_PIXPOS, self.CORNER_RIGHTDOWN_PIXPOS),
                         (self.CORNER_LEFTDOWN_PIXPOS, self.CORNER_DOWN_PIXPOS)]
        for m, n in segments_down:
            p = (m[0], m[1] + PIXSIZE_STAGE_CELL[1] * self.stage)
            q = (n[0], n[1] + PIXSIZE_STAGE_CELL[1] * self.stage)
            coef_dir = (q[1] - p[1])/(q[0] - p[0])
            b = p[1] - coef_dir*p[0]
            if coef_dir*x + b - y < 0:
                return False

        # Sides
        if x < self.X_MIN_LIM or x > self.X_MAX_LIM:
            return False

        return True
    
    def delete(self):
        self.img.setParent(None)
