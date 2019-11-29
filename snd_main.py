# -*- coding: utf-8 -*-

import sys
import numpy as np

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QLabel


COEF = 0.3
GRID_RAY = 2

APP_PIXDIM = (2800 * COEF, 2100 * COEF)

IMGCELL_PIXDIM = (456*COEF, 342*COEF)
PIXSIZE_STAGE_CELL = [0, 80 * COEF]

POS_CENTER = [(APP_PIXDIM[0] - IMGCELL_PIXDIM[0])/2, (APP_PIXDIM[1] - IMGCELL_PIXDIM[1])/2]

MV_R = [450 * COEF, 0]
MV_DR = [228 * COEF, 190 * COEF]



class MyApp :
	
	def __init__(self) :
		# Initialisation de la fenetre
		self.window = QWidget()
		self.window.resize(APP_PIXDIM[0], APP_PIXDIM[1])
		self.window.setWindowTitle('MyApp')
		self.window.mouseReleaseEvent=lambda event:self.on_click(event)
		self.window.keyReleaseEvent=lambda event:self.on_keyboard(event)

		# Initialisation des cells
		self.grid = Grid(self)

	def on_click(self, event) :
		x, y = event.pos().x(), event.pos().y()
		cell = self.grid.get_cell_from_pos(x, y)
		cell.grew()
		print(cell.q, cell.r)

	def on_keyboard(self, event) :
		key = event.key()
		msg = ''
		if key == QtCore.Qt.Key_Up :
			msg = 'up'
		elif key == QtCore.Qt.Key_Down :
			msg = 'down'
		elif key == QtCore.Qt.Key_Right :
			msg = 'right'
		elif key == QtCore.Qt.Key_Left :
			msg = 'left'
		else :
			print(key)
		# self.b.move(msg)

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


class Cell :

	PATH_IMG = "/home/naowak/Documents/Kothrak/cell_{}.png"
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
		self.img.setGeometry(QtCore.QRect(x, y, self.size_x, self.size_y))
		self.img.setPixmap(QtGui.QPixmap(self.PATH_IMG.format(self.stage)))
		self.img.setText("")
		self.img.setScaledContents(True)
		self.img.setObjectName("cell_{}_{}".format(x, y))

	def grew(self) :
		if self.stage >= Cell.MAX_STAGE :
			print("Cannot grow anymore.")
			return
		self.stage += 1
		self.change_img()

	def change_img(self) :
		self.y -= PIXSIZE_STAGE_CELL[1]
		self.size_y += PIXSIZE_STAGE_CELL[1]
		self.img.setGeometry(QtCore.QRect(self.x, self.y, self.size_x, self.size_y))
		self.img.setPixmap(QtGui.QPixmap(self.PATH_IMG.format(self.stage)))

	def absolute_to_relative_position(self, x, y) :
		return (x - self.x, y - self.y)

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
		self.img.setGeometry(self.x, self.y, self.size_x, self.size_y)

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
	background-color: rgba(255, 255, 255, 0)
}
'''

if __name__ == '__main__' :
	qapp = QApplication(sys.argv)
	qapp.setStyleSheet(style)
	main_window = MyApp()
	main_window.show()
	sys.exit(qapp.exec_())