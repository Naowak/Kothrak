from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QLabel

from kothrak.envs.game.Utils import IMGPLAYER_PIXDIM, PIXSIZE_VECTOR_PLAYER_CELL

class Player:

    PATH_IMG = "envs/game/img/player_img{}.png"

    def __init__(self, player_id, cell, app):
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
        self.img.show()
        self.move(cell)

    def move(self, cell):
        self.cell = cell
        self.x = cell.x + PIXSIZE_VECTOR_PLAYER_CELL[0]
        self.y = cell.y + PIXSIZE_VECTOR_PLAYER_CELL[1]
        self.img.setGeometry(QtCore.QRect(
            self.x, self.y, self.size_x, self.size_y))
    
    def delete(self):
        self.img.deleteLater()
