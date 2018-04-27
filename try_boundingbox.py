import cv2
import os
import imutils


files = os.listdir('image/')
for file in files:
    #img = cv2.imread('image/' + file)
    img = cv2.imread('image/0.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #img = cv2.bitwise_not(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))

    # make binary image
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    cv2.imshow('a', img)
    cv2.waitKey()
    exit()
    # find contours
    cnts = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    digitCnts = []

    # loop over the digit area candidates
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        # if the contour is sufficiently large, it must be a digit

        if (w >= 10 and w <=40) and h >= 40:
            print(w)
            digitCnts.append(c)
            cv2.imshow("image", img[y:y + h, x:x + w])
            cv2.waitKey()


    #cv2.imshow("image", img)
    #cv2.waitKey()

    exit()