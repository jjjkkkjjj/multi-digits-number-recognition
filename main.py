from preprocess import PreProcess
from cnn import CNN

width = 64
height = 64
width_number_area = 28
height_number_area = 28
videopath = '60fpx.MP4'

def main():
    cnn = CNN(width, height, videopath)
    cnn.predict()

def training():
    cnn = CNN(width, height, videopath)
    cnn.train()

if __name__ == '__main__':
    #training()
    main()