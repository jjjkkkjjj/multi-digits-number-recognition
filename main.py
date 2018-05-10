from preprocess import PreProcess
from cnn import CNN
from videocheckgui import GUI
from PyQt5.QtWidgets import *
import sys
import os


width = 64
height = 64
width_number_area = 28
height_number_area = 28
#videopath = '60fpx.MP4'
videopath = 'video/'

def main(videoname):
    global videopath
    videolist = os.listdir(videopath)
    if videoname not in videolist:
         print("video's name is incorrect")
    videopath += videoname
    cnn = CNN(width, height, videopath)
    cnn.predict(save_csv=True, save_for_retaraindata=False, refresh=False)

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
    args = sys.argv
    if len(args) != 2:
        print("argument error")
    else:
        main(args[1])
    #training()
    #gui()
