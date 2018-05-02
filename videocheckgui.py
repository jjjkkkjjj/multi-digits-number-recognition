# -*- coding: utf-8 -*-
import sys
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import cv2

class GUI(QMainWindow):
    def __init__(self):
        super(GUI, self).__init__()

        # parameter
        self.width = 960
        self.height = 640
        self.framerate = 30

        self.videopath = ''
        self.cropped_width = 64
        self.cropped_height = 64
        self.width_number_area = 28
        self.height_number_area = 28

        self.frame_num = 0
        self.now = 0
        self.images = None
        self.results = []

        self.initUI()

    def initUI(self):
        self.textEdit = QTextEdit()
        self.setCentralWidget(self.textEdit)
        self.statusBar()

        # set menubar's icon
        openFile = QAction('Open video', self)
        # set shortcut
        openFile.setShortcut('Ctrl+O')
        # set statusbar
        openFile.setStatusTip('Open new video')
        openFile.triggered.connect(self.showDialog)

        saveFile = QAction('Save video', self)
        saveFile.setShortcut('Ctrl+S')
        saveFile.setStatusTip('Save csv')
        saveFile.triggered.connect(self.savecsv)

        quitapplication = QAction('quit', self)
        quitapplication.setShortcut('Ctrl+Q')
        quitapplication.setStatusTip('Close The App')
        quitapplication.triggered.connect(self.quit)

        nextframe = QAction('next frame', self)
        nextframe.setShortcut('Ctrl+N')
        nextframe.setStatusTip('Move next frame')
        nextframe.triggered.connect(lambda: self.showimage(nextframe_num=1))

        previousframe = QAction('previous frame', self)
        previousframe.setShortcut('Ctrl+P')
        previousframe.setStatusTip('Move previous frame')
        previousframe.triggered.connect(lambda: self.showimage(nextframe_num=-1))

        # make menubar
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(openFile)
        fileMenu.addAction(saveFile)
        fileMenu.addAction(quitapplication)
        editMenu = menubar.addMenu('&Edit')
        editMenu.addAction(nextframe)
        editMenu.addAction(previousframe)

        self.leftdock_videolist = QDockWidget(self)
        self.leftdock_videolist.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.leftdock_videolist.setFloating(False)
        self.leftdock_videowidget = VideoListDockWidget(self)
        self.leftdock_videolist.setWidget(self.leftdock_videowidget)
        self.leftdock_videolist.setMinimumSize(QSize(400, self.maximumHeight()))
        # self.leftdock_videolist.setWidget(SliderTreeView())

        self.addDockWidget(Qt.LeftDockWidgetArea, self.leftdock_videolist)

        self.Videowidget = VideoDisplayWidget(self)
        # self.video = VideoCapture(self.Videowidget)
        self.setCentralWidget(self.Videowidget)

        self.showMaximized()

    def showDialog(self):
        # show file dialog
        fname = QFileDialog.getOpenFileName(self, 'Open file', '')
        # fname[0] is selected path
        if fname[0]:
            self.set_video(fname[0])

    def set_video(self, videopath):
        video = cv2.VideoCapture(videopath)
        if video.isOpened():
            frame_num = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            self.Videowidget.show_progressbar(frame_num)
            # pixcelimg = []
            images = []
            for num in range(frame_num):
                ret, img = video.read()
                images.append(cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (self.width, self.height)))
                self.Videowidget.setprogressvalue(num)
            self.images = images
            self.now = 0
            self.frame_num = frame_num
            self.Videowidget.finishprogress()
            self.leftdock_videowidget.set_item(videopath, 0)

            self.showimage()

    def showimage(self, nextframe_num=0):
        # process
        if (self.now == self.frame_num - 1 and nextframe_num > 0) or (
                self.now == 0 and nextframe_num < 0):
            self.Videowidget.timer.stop()
        else:
            self.now += nextframe_num
            self.leftdock_videowidget.next_slidervalue()
        self.draw_now()

    def draw_now(self):
        self.Videowidget.draw(self.images[self.now])

    def start_button_clicked(self):
        self.Videowidget.timer.timeout.connect(lambda: self.showimage(nextframe_num=1))
        self.Videowidget.timer.start(10)  # ミリ秒単位

    def pause_button_clicked(self):
        self.Videowidget.timer.stop()


    def savecsv(self):
        print("s")

    def quit(self):
        choice = QMessageBox.question(self, 'Message', 'Do you really want to exit?', QMessageBox.Yes | QMessageBox.No)
        if choice == QMessageBox.Yes:
            sys.exit()
        else:
            pass



class VideoDisplayWidget(QWidget):
    def __init__(self,parent):
        super(VideoDisplayWidget, self).__init__(parent)
        self.parent = parent
        #self.layout = QFormLayout(self)
        hbox = QHBoxLayout()
        hbox2 = QHBoxLayout()
        vbox = QVBoxLayout()

        # show widget
        self.startButton = QPushButton('Start', parent)
        self.startButton.clicked.connect(parent.start_button_clicked)
        hbox.addWidget(self.startButton)

        self.pauseButton = QPushButton('Pause', parent)
        self.pauseButton.clicked.connect(parent.pause_button_clicked)
        hbox.addWidget(self.pauseButton)
        vbox.addLayout(hbox)

        self.video_label = QLabel()
        vbox.addWidget(self.video_label)


        self.setLayout(vbox)

        self.timer = QTimer()

    def draw(self, img=None):
        if img is None:
            self.video_label.clear()
        else:
            qimg = QImage(img, img.shape[1], img.shape[0], QImage.Format_RGB888)
            pixcelimg = QPixmap.fromImage(qimg)
            self.video_label.setPixmap(pixcelimg)

    def setprogressvalue(self, i):
        self.progressbar.setValue(i)
        QApplication.processEvents()

    def finishprogress(self):
        self.window.close()

    def show_progressbar(self, frame_num):
        self.window = QDialog(self.parent)
        self.progressbar = QProgressBar(self.parent)
        self.progressbar.setRange(0, frame_num)
        layout = QHBoxLayout()
        layout.addWidget(self.progressbar)
        self.window.setLayout(layout)
        self.window.show()


class VideoListDockWidget(QWidget):
    def __init__(self, parent):
        super(VideoListDockWidget, self).__init__(parent)
        self.parent = parent

        self.initUI()

    def initUI(self):
        vbox = QVBoxLayout(self)
        #hbox = QHBoxLayout()
        #self.delbutton = QPushButton("削除", self)
        #self.delbutton.clicked.connect(self.remove)
        #self.detailbutton = QPushButton("拡大", self)
        #self.detailbutton.clicked.connect(self.expand)

        #hbox.addWidget(self.delbutton)
        #hbox.addWidget(self.detailbutton)
        #vbox.addLayout(hbox)

        # slider
        self.slidertreeview = QTreeView()

        self._datamodel = QStandardItemModel(0, 3)
        self._datamodel.setHeaderData(0, Qt.Horizontal, 'video name')
        self._datamodel.setHeaderData(1, Qt.Horizontal, 'frame/total')
        self._datamodel.setHeaderData(2, Qt.Horizontal, 'slider')
        self.slidertreeview.setModel(self._datamodel)

        self.item = {}
        self.slider = None

        # self.model = Model()
        # self.setModel(self.model)
        self.slidertreeview.setIndentation(0)
        self.slidertreeview.setSelectionMode(QAbstractItemView.ExtendedSelection)

        vbox.addWidget(self.slidertreeview)

        self.setLayout(vbox)

    def sliderchanged(self, value):
        self.parent.now = value
        frame = QStandardItem(str(value + 1) + '/' + str(self.item["frame_max"]))
        self._datamodel.setItem(0, 1, frame)
        self.parent.draw_now()

    def next_slidervalue(self):
        self.slider.setValue(self.parent.now)


    def set_item(self, videopath, frame_now):
        video = cv2.VideoCapture(videopath)
        tmp = videopath.split('/')
        self.item = { "name":tmp[len(tmp) - 1], "frame_max":int(video.get(cv2.CAP_PROP_FRAME_COUNT))}

        videoname = QStandardItem(self.item["name"])
        self._datamodel.setItem(0, 0, videoname)

        frame = QStandardItem(str(frame_now + 1) + '/' + str(self.item["frame_max"]))
        self._datamodel.setItem(0, 1, frame)

        slider = QSlider(Qt.Horizontal, self)
        slider.setRange(0, self.item["frame_max"] - 1)
        slider.setValue(0)
        slider.setFocusPolicy(Qt.NoFocus)
        # slider.valueChanged[int].connect(self.parent.sliderchanged)
        slider.valueChanged[int].connect(self.sliderchanged)
        index = self._datamodel.index(0, 2, QModelIndex())
        self.slidertreeview.setIndexWidget(index, slider)
        self.slider = slider

        #self.update_list()

