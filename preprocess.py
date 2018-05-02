import cv2
import numpy as np
import imutils

class PreProcess:
    def __init__(self, width, height, videopath):
        # cropped size
        self.width = width
        self.height = height
        # number's size
        self.Width = 28
        self.Height = 28

        self.path = videopath

        self.__rect = None
        self.__sx = 0
        self.__sy = 0
        self.__abs_x = 0
        self.__abs_y = 0
        self.__abs_sx = 0
        self.__abs_sy = 0
        self.__img_win = None
        self.__img = None
        self.__cropped = False
        # self.Number_Position[frame_num][digit]['xini','xfin','yini','yfin']
        self.Number_Position = []

    def read_video(self):
        print("reading video now")

        video = cv2.VideoCapture(self.path)
        ret, self.__img = video.read()
        if not ret:
            print("can't read video, check your filepath")
            exit()
        print("crop number area")
        print("f: set, r: reset, q: quit")
        self.__rect = (0, 0, self.__img.shape[1], self.__img.shape[0])
        self.__img_win = self.__img.copy()

        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("img", self.__callback)

        while (1):
            cv2.imshow("img", self.__img_win)
            k = cv2.waitKey(1)

            if k == ord('f'):
                break

            if k == ord("r"):
                self.__rect = (0, 0, self.__img.shape[1], self.__img.shape[0])
                self.__img_win = self.__img.copy()

            if k == ord("q"):
                exit()

        self.yini = self.__rect[1]
        self.yfin = self.__rect[1] + self.__rect[3]
        self.xini = self.__rect[0]
        self.xfin = self.__rect[0] + self.__rect[2]

        video.release()
        cv2.destroyAllWindows()

        self.__cropped = True

        return True

    def video2image(self):
        if not self.__cropped:
            print("you should call .read_video before video2image")
            exit()
        video = cv2.VideoCapture(self.path)
        cnt = 0

        while True:
            ret, img = video.read()
            if not ret:
                break
            cv2.imwrite('image/' + str(cnt) + '.jpg', img[self.yini:self.yfin, self.xini:self.xfin])
            cnt += 1

        video.release()
        return

    def search_number_area(self, save_image=False):
        if not self.__cropped:
            print("you should call .read_video before search_number_area")
            exit()

        video = cv2.VideoCapture(self.path)

        nums_image = 0
        while True:
            ret, img = video.read()
            if not ret:
                break
            img = cv2.cvtColor(cv2.resize(img[self.yini:self.yfin, self.xini:self.xfin], (self.height, self.width)), cv2.COLOR_RGB2GRAY)
            # make binary image
            img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            # dilation
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            # find contours
            cnts = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if imutils.is_cv2() else cnts[1]

            number_position = []
            tmp = 0

            w_area = (self.width*0.08, self.width*0.6)
            h_area = self.height*0.6
            # loop over the digit area candidates
            for c in cnts:
                # compute the bounding box of the contour
                (x, y, w, h) = cv2.boundingRect(c)
                # if the contour is sufficiently large, it must be a digit

                if (w >= w_area[0] and w <= w_area[1]) and h >= h_area:
                    number_position.append({'xini':x, 'xfin':x+w, 'yini':y, 'yfin':y + h})
                    if save_image:
                        cv2.imwrite('image-preprocessed/' + '{0}-{1}.jpg'.format(nums_image, tmp), cv2.resize(img[y:y + h, x:x + w],(self.Height, self.Width)))
                    tmp += 1
                    #cv2.imshow("image", img[y:y + h, x:x + w])
                    #cv2.waitKey()

            # rearrange left side
            index = np.argsort(np.array([number_position[i]['xini'] for i in range(len(number_position))]))
            number_position = [number_position[index[i]] for i in range(len(number_position))]

            self.Number_Position.append(number_position)
            nums_image += 1

        video.release()
        return self.Number_Position

    def __callback(self, event, x, y, flags, param):
        self.__abs_x, self.__abs_y = self.__rect[0] + x, self.__rect[1] + y

        if event == cv2.EVENT_LBUTTONDOWN:
            self.__sx, self.__sy = x, y
            self.__abs_sx, self.__abs_sy = self.__abs_x, self.__abs_y

        if flags == cv2.EVENT_FLAG_LBUTTON:
            self.__img_win = self.__img.copy()[self.__rect[1]:self.__rect[1] + self.__rect[3], self.__rect[0]:self.__rect[0] + self.__rect[2]]
            cv2.rectangle(self.__img_win, (self.__sx, self.__sy), (x, y), (0, 0, 0), 2)

        if event == cv2.EVENT_LBUTTONUP:
            rect_x = np.clip(min(self.__abs_sx, self.__abs_x), 0, self.__img.shape[1] - 2)
            rect_y = np.clip(min(self.__abs_sy, self.__abs_y), 0, self.__img.shape[0] - 2)
            rect_w = np.clip(abs(self.__abs_sx - self.__abs_x), 1, self.__img.shape[1] - rect_x)
            rect_h = np.clip(abs(self.__abs_sy - self.__abs_y), 1, self.__img.shape[0] - rect_y)
            self.__rect = (rect_x, rect_y, rect_w, rect_h)
            self.__img_win = self.__img.copy()[self.__rect[1]:self.__rect[1] + self.__rect[3], self.__rect[0]:self.__rect[0] + self.__rect[2]]