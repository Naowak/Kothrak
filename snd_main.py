# -*- coding: utf-8 -*-

import sys

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QLabel

APP_PIXDIM = (800, 600)
IMGCELL_PIXDIM = (586, 454)

POS_CENTER = (APP_PIXDIM[0]/2, APP_PIXDIM[1]/2)
MV_R = [117, 0]
MV_DR = [57, 51]
COEF = 0.2
GRID_RAY = 2


class MyApp :
	
	def __init__(self) :
		# Initialisation de la fenetre
		self.window = QWidget()
		self.window.resize(APP_PIXDIM[0], APP_PIXDIM[1])
		self.window.setWindowTitle('MyApp')
		self.window.mouseReleaseEvent=lambda event:self.click(event)
		self.window.keyReleaseEvent=lambda event:self.keyboard(event)

		# Initialisation des cells
		self.grid = Grid(self)
		# self.a=Cell(200, 200, self)
		# self.b=Cell(500, 500, self)

	def click(self, event) :
		x, y = event.pos().x(), event.pos().y()
		print(x, y)

	def keyboard(self, event) :
		key = event.key()
		print(key)

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
			return Cell(pos_x, pos_y, self.app) 

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


class Cell :

	PATH_IMG = '/home/naowak/Images/cell_basic.png'
	# PATH_IMG = 'cell1_resized.png'

	def __init__(self, x, y, app) :
		self.app = app
		self.x = x
		self.y = y
		self.size_x = IMGCELL_PIXDIM[0]*COEF
		self.size_y = IMGCELL_PIXDIM[1]*COEF
		self.img = QLabel(self.app.window)
		self.img.setGeometry(QtCore.QRect(x, y, self.size_x, self.size_y))
		self.img.setPixmap(QtGui.QPixmap(self.PATH_IMG))
		self.img.setText("")
		self.img.setScaledContents(True)
		self.img.setObjectName("cell_{}_{}".format(x, y))

	def move(self, direction) :
		if direction == 'up' :
			self.y -= 1
		elif direction == 'down' :
			self.y += 1
		elif direction == 'right' :
			self.x += 1
		elif direction == 'left' :
			self.x -= 1
		else :
			raise Exception('That direction doesn\'t mean anything : {}'.format(direction))
		print(self.x, self.y)
		self.img.setGeometry(self.x, self.y, self.size_x, self.size_y)

qapp = QApplication(sys.argv)
main_window = MyApp()
main_window.show()
sys.exit(qapp.exec_())