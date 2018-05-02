from preprocess import PreProcess
from cnn import CNN
from videocheckgui import GUI
from PyQt5.QtWidgets import *
import sys


width = 64
height = 64
width_number_area = 28
height_number_area = 28
#videopath = '60fpx.MP4'
videopath = 'videoplayback.mp4'

def main():
    cnn = CNN(width, height, videopath)
    cnn.predict(save_csv=True, save_for_retaraindata=False, refresh=True)

def training():
    cnn = CNN(width, height, videopath)
    #cnn.train()
    cnn.retrain()

def gui():
    app = QApplication(sys.argv)
    window = GUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    #training()
    main()
    #gui()