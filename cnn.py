from keras.datasets import mnist
from keras.utils import np_utils
from preprocess import PreProcess
from keras.layers import Input, Convolution2D, MaxPooling2D, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Dense
from keras.constraints import maxnorm
from keras.callbacks import ModelCheckpoint
from keras.models import Model
import numpy as np
import cv2

class CNN(PreProcess):
    def __init__(self, width, height, videopath):
        super().__init__(width, height, videopath)
        self.num_class = 10 # 0~9

        weight_num = 5
        input = Input(shape=(1, self.Height, self.Width))

        x = Convolution2D(nb_filter=20, nb_col=weight_num, nb_row=weight_num, border_mode='same', name='conv1')(input)
        x = Activation('relu', name='act1')(x)
        x = BatchNormalization(name='bnorm1')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th', name='pool1')(x)
        x = Dropout(0.2, name='drop1')(x)

        x = Convolution2D(nb_filter=50, nb_col=weight_num, nb_row=weight_num, border_mode='same', name='conv2')(x)
        x = Activation('relu', name='act2')(x)
        x = BatchNormalization(name='bnorm2')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th', name='pool2')(x)
        x = Dropout(0.2, name='drop2')(x)

        x = Flatten(name='feature_vector')(x)
        x = Dense(output_dim=500, activation='relu')(x)

        x = Dense(output_dim=self.num_class, activation='softmax')(x)

        output = x

        self.model = Model(input=input, output=output)

    def predict(self):
        super().read_video()
        super().search_number_area(save_image=False)

        self.model.load_weights('models/119-tra_0.0143-val_0.0331-.hdf5')

        video = cv2.VideoCapture(self.path)

        for frame_num in range(len(self.Number_Position)):
            ret, img = video.read()
            img = cv2.cvtColor(cv2.resize(img[self.yini:self.yfin, self.xini:self.xfin], (self.height, self.width)), cv2.COLOR_RGB2GRAY)
            cv2.imshow('cropped img', img)

            results = []
            for number_position in self.Number_Position[frame_num]:
                Img = cv2.resize(img[number_position['yini']:number_position['yfin'], number_position['xini']:number_position['xfin']], (self.Height - 6, self.Width - 6))
                # padding
                tmp = np.ones((self.Height, self.Width))*255
                tmp[3:self.Height-3, 3:self.Width -3] = Img
                Img = tmp
                # make binary image
                #Img = cv2.threshold(Img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

                # dilation
                # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
                # Img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)


                Img = Img.astype('float32')
                Img /= 255.0
                Img = np.where(Img > 0.4, 1, Img)

                Img = np.fabs(np.expand_dims(Img, axis=0)[:, np.newaxis, :, :] - 1)

                #print(Img[0][0])
                cv2.imshow('a', Img[0][0])
                cv2.waitKey(15)
                #exit()
                result = np.array(self.model.predict(x=Img, verbose=0))
                results.append(np.argmax(result))
            print(results)
            k = cv2.waitKey(100)
            if k == ord("q"):
                exit()

        video.release()

    def train(self):
        (data_train, label_train), (data_test, label_test) = mnist.load_data()


        # normilization
        data_train = data_train.astype('float32')[:, np.newaxis, :, :]
        # print(data_train.shape)
        # (60000, 1, 28, 28)
        data_train /= 255.0
        # cv2.imshow('a', data_train[0][0])
        # cv2.waitKey()
        # exit()
        data_test = data_test.astype('float32')[:, np.newaxis, :, :]
        data_test /= 255.0

        # one-hot
        label_train = np_utils.to_categorical(label_train, self.num_class)
        label_test = np_utils.to_categorical(label_test, self.num_class)

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        checkpointer = ModelCheckpoint(filepath="models/{epoch:02d}-tra_{loss:.4f}-val_{val_loss:.4f}-.hdf5", monitor='loss', verbose=1, save_best_only=True)

        self.model.fit(x=data_train, y=label_train, batch_size=32, nb_epoch=10000, verbose=0, callbacks=[checkpointer], validation_data=(data_test, label_test))