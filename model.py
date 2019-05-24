# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

# from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
import sys

class Cell :

    POS_CELL_CORNER_LEFT = (0,45)
    POS_CELL_CORNER_RIGHT = (149,45)
    POS_CELL_CORNER_UP = (75,0)
    POS_CELL_CORNER_DOWN = (75,88)
    
    def __init__(self, x, y, img, app) :
        self.img = img
        self.x = x
        self.y = y
        self.app = app

    def is_click_in_corner_img(self, click_x, click_y) :
        p, q = (Cell.POS_CELL_CORNER_LEFT, Cell.POS_CELL_CORNER_UP)
        coef = (q[1] - p[1])/(q[0] - p[0])
        b = p[1] - coef*p[0]
        if coef*click_x + b - click_y > 0 :
            return "upleft"
        p, q = (Cell.POS_CELL_CORNER_UP, Cell.POS_CELL_CORNER_RIGHT)
        coef = (q[1] - p[1])/(q[0] - p[0])
        b = p[1] - coef*p[0]
        if coef*click_x + b - click_y > 0 :
            return "upright"
        p, q = (Cell.POS_CELL_CORNER_DOWN, Cell.POS_CELL_CORNER_RIGHT)
        coef = (q[1] - p[1])/(q[0] - p[0])
        b = p[1] - coef*p[0]
        if coef*click_x + b - click_y < 0 :
            return "downright"
        p, q = (Cell.POS_CELL_CORNER_DOWN, Cell.POS_CELL_CORNER_RIGHT)
        coef = (q[1] - p[1])/(q[0] - p[0])
        b = p[1] - coef*p[0]
        if coef*click_x + b - click_y < 0 :
            return "downleft"
        return False

    
    def click_cell(self, event) :
        click_x, click_y = (event.pos().x(), event.pos().y())
        corner = self.is_click_in_corner_img(click_x, click_y)
        if corner :
            correct_cell = self.correct_click_cell(corner)
            if correct_cell != None :
                correct_cell.delete()
        else :
            self.delete()
    
    def correct_click_cell(self, corner) :
        if corner == "upleft" :
            x = self.x
            y = self.y - 1
        elif corner == "upright" :
            x = self.x + 1
            y = self.y
        elif corner == "downright" :
            x = self.x 
            y = self.y + 1
        elif corner == "downleft" :
            x = self.x - 1
            y = self.y
        else :
            raise Exception("Incorrect corner.")
        # PEUT ETRE AMELIORER
        cell = None
        for c in self.app.cells :
            if c.x == x and c.y == y :
                cell = c
        return cell

    def delete(self) :
        self.app.cells.remove(self)
        self.img.setParent(None)
        del(self.img)
        del(self)    


class MyApp() :

    TRANSLATE_X = (78, -46)
    TRANSLATE_Y = (78, 46)
    SIZE_CELL_X = 150
    SIZE_CELL_Y = 116
    POS_INITIAL_X = 50
    POS_INITIAL_Y = 242

    NB_CELLS = 5

    def __init__(self) :
        self.window = QWidget()
        self.window.resize(1000, 600)
        self.window.setWindowTitle('MyApp')

        self.cells = list()
        for i in reversed(list(range(MyApp.NB_CELLS))) :
            for j in range(MyApp.NB_CELLS) :
                self.add_cell(i, j)


    def add_cell(self, abs, ord) :
        img = QLabel(self.window)
        cell = Cell(abs, ord, img, self)
        img.setGeometry(QtCore.QRect(MyApp.POS_INITIAL_X + abs * MyApp.TRANSLATE_X[0] + ord * MyApp.TRANSLATE_X[0],
                                MyApp.POS_INITIAL_Y + abs * MyApp.TRANSLATE_X[1] + ord * MyApp.TRANSLATE_Y[1], 
                                MyApp.SIZE_CELL_X,
                                MyApp.SIZE_CELL_Y))
        img.setMaximumSize(QtCore.QSize(323, 16777215))
        img.setText("")
        img.setPixmap(QtGui.QPixmap("cell.png"))
        img.setScaledContents(True)
        img.setObjectName("cell_{}_{}".format(abs, ord))
        img.mouseReleaseEvent=lambda event:cell.click_cell(event)
        self.cells += [cell]

    def show(self) :
        self.window.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MyApp()
    main_window.show()
    sys.exit(app.exec_())