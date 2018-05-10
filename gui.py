from videocheckgui import GUI
from PyQt5.QtWidgets import *
import sys
import os

def gui():
    app = QApplication(sys.argv)
    window = GUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    #training()
    gui()
