from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QTextEdit
from PyQt5 import uic
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore    import *
from PyQt5.QtGui     import *


class UI(QMainWindow):
    def __init__(self):
        super(UI, self).__init__()
        uic.loadUi("face_anti.ui", self)
        self.show()


app = QApplication(sys.argv)
window = UI()
app.exec_()