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

class MyApp() :

    TRANSLATE_X = (75, -44)
    TRANSLATE_Y = (75, 44)
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

    def click_cell(self, cell) :
        cell.setParent(None)
        del(cell)

    def add_cell(self, abs, ord) :
        cell = QLabel(self.window)
        cell.setGeometry(QtCore.QRect(MyApp.POS_INITIAL_X + abs * MyApp.TRANSLATE_X[0] + ord * MyApp.TRANSLATE_X[0],
                                MyApp.POS_INITIAL_Y + abs * MyApp.TRANSLATE_X[1] + ord * MyApp.TRANSLATE_Y[1], 
                                MyApp.SIZE_CELL_X,
                                MyApp.SIZE_CELL_Y))
        cell.setMaximumSize(QtCore.QSize(323, 16777215))
        cell.setText("")
        cell.setPixmap(QtGui.QPixmap("cell.png"))
        cell.setScaledContents(True)
        cell.setObjectName("cell_{}_{}".format(abs, ord))
        cell.mouseReleaseEvent=lambda event:self.click_cell(cell)
        self.cells += [cell]

    def show(self) :
        self.window.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MyApp()
    main_window.show()
    sys.exit(app.exec_())