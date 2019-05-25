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

COEF_RESIZE = 0.5

class MyApp() :


    TRANSLATE_X = (162 * COEF_RESIZE, -96 * COEF_RESIZE)
    TRANSLATE_Y = (162 * COEF_RESIZE, 96 * COEF_RESIZE)
    SIZE_STAGE = 60 * COEF_RESIZE
    SIZE_CELL_X = 312 * COEF_RESIZE
    SIZE_CELL_Y = [240 * COEF_RESIZE + 60 * COEF_RESIZE * i for i in range(4)]
    POS_INITIAL_X = 100 * COEF_RESIZE
    POS_INITIAL_Y = [500 * COEF_RESIZE - 60 * COEF_RESIZE * i for i in range(4)]

    NB_CELLS = 5

    def __init__(self) :
        self.window = QWidget()
        self.window.resize(2000 * COEF_RESIZE, 1200 * COEF_RESIZE)
        self.window.setWindowTitle('MyApp')
        self.window.mouseReleaseEvent=lambda event:self.click(event)

        self.cells = list()
        for i in reversed(list(range(MyApp.NB_CELLS))) :
            for j in range(MyApp.NB_CELLS) :
                self.add_cell(i, j)

    def click(self, event) :
        click_x, click_y = (event.pos().x(), event.pos().y())
        for i in range(MyApp.NB_CELLS) :
            for j in reversed(list(range(MyApp.NB_CELLS))) :
                cell = self.get_cell(i, j)
                relative_x = click_x - cell.pos_x
                relative_y = click_y - cell.pos_y
                if cell.is_pos_in_cell(relative_x, relative_y) :
                    cell.grew()
                    return

    def get_cell(self, abs, ord) :
        for c in self.cells :
            if c.x == abs and c.y == ord :
                return c

    def add_cell(self, abs, ord) :
        cell = Cell(abs, ord, self)
        self.cells += [cell]

    def show(self) :
        self.window.show()


class Cell :

    POS_CELL_CORNER_LEFT = (0 * COEF_RESIZE,94 * COEF_RESIZE)
    POS_CELL_CORNER_RIGHT = (311 * COEF_RESIZE,94 * COEF_RESIZE)
    POS_CELL_CORNER_UP = (156 * COEF_RESIZE,0 * COEF_RESIZE)
    POS_CELL_CORNER_DOWN = (156 * COEF_RESIZE,183 * COEF_RESIZE)

    MAX_HEIGHT = 3
    
    def __init__(self, x, y, app) :
        self.x = x
        self.y = y
        self.app = app
        self.height = 0
        self.pos_x = None
        self.pos_y = None
        self.size_x = None
        self.size_y = None
        self.create_img_cell()

    def is_pos_in_cell(self, x, y) :
        """On vérifie que x et y sont bien dans l'image et non dans le fond transparent."""
        # up left
        p, q = (Cell.POS_CELL_CORNER_LEFT, Cell.POS_CELL_CORNER_UP)
        coef = (q[1] - p[1])/(q[0] - p[0])
        b = p[1] - coef*p[0]
        if coef*x + b - y > 0 :
            return False
        # up right 
        p, q = (Cell.POS_CELL_CORNER_UP, Cell.POS_CELL_CORNER_RIGHT)
        coef = (q[1] - p[1])/(q[0] - p[0])
        b = p[1] - coef*p[0]
        if coef*x + b - y > 0 :
            return False
        # down right
        m, n = (Cell.POS_CELL_CORNER_DOWN, Cell.POS_CELL_CORNER_RIGHT)
        p = (m[0], m[1] + MyApp.SIZE_STAGE * self.height)
        q = (n[0], n[1] + MyApp.SIZE_STAGE * self.height)
        coef = (q[1] - p[1])/(q[0] - p[0])
        b = p[1] - coef*p[0] 
        if coef*x + b - y < 0 :
            return False
        # down left
        p, q = (Cell.POS_CELL_CORNER_LEFT, Cell.POS_CELL_CORNER_DOWN)
        p = (m[0], m[1] + MyApp.SIZE_STAGE * self.height)
        q = (n[0], n[1] + MyApp.SIZE_STAGE * self.height)
        coef = (q[1] - p[1])/(q[0] - p[0])
        b = p[1] - coef*p[0] 
        if coef*x + b - y < 0 :
            return False
        # right
        if x < 0 :
            return False
        # left
        if x > self.size_x :
            return False
        return True

    def grew(self) :
        if self.height >= Cell.MAX_HEIGHT :
            print("Cannot grow anymore.")
            return
        self.height += 1
        self.change_img()

    def delete_img(self) :
        self.img.setParent(None)
        del(self.img)

    def delete(self) :
        self.app.cells.remove(self)
        self.delete_img()
        del(self)  

    def change_img(self) :
        abs = self.x
        ord = self.y
        self.pos_x = MyApp.POS_INITIAL_X + abs * MyApp.TRANSLATE_X[0] + ord * MyApp.TRANSLATE_X[0]
        self.pos_y = MyApp.POS_INITIAL_Y[self.height] + abs * MyApp.TRANSLATE_X[1] + ord * MyApp.TRANSLATE_Y[1]
        self.size_x = MyApp.SIZE_CELL_X
        self.size_y = MyApp.SIZE_CELL_Y[self.height]
        self.img.setGeometry(QtCore.QRect(self.pos_x, self.pos_y, self.size_x, self.size_y))
        self.img.setPixmap(QtGui.QPixmap("cell{}_resized.png".format(self.height)))

    def create_img_cell(self) :
        abs = self.x
        ord = self.y
        self.pos_x = MyApp.POS_INITIAL_X + abs * MyApp.TRANSLATE_X[0] + ord * MyApp.TRANSLATE_X[0]
        self.pos_y = MyApp.POS_INITIAL_Y[self.height] + abs * MyApp.TRANSLATE_X[1] + ord * MyApp.TRANSLATE_Y[1]
        self.size_x = MyApp.SIZE_CELL_X
        self.size_y = MyApp.SIZE_CELL_Y[self.height]
        self.img = QLabel(self.app.window)
        self.img.setGeometry(QtCore.QRect(self.pos_x, self.pos_y, self.size_x, self.size_y))
        self.img.setText("")
        self.img.setPixmap(QtGui.QPixmap("cell{}_resized.png".format(self.height)))
        self.img.setScaledContents(True)
        self.img.setObjectName("cell_{}_{}".format(abs, ord))
    
    


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MyApp()
    main_window.show()
    sys.exit(app.exec_())