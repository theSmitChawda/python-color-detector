# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 15:29:36 2022

@author: Administrator
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn import preprocessing

cap = cv.VideoCapture(0)
ret = True

while ret:
    ret, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # define range of yellow color in HSV
    lower_yellow = np.array([20,50,50])
    upper_yellow = np.array([30, 255, 255])
    mask = cv.inRange(hsv, lower_yellow, upper_yellow)

    # Bitwise-AND mask and original image
    yellow = cv.bitwise_and(frame, frame, mask = mask)
    cv.namedWindow('Original',cv.WINDOW_NORMAL)
    cv.namedWindow('Mask',cv.WINDOW_NORMAL)
    cv.namedWindow('Yellow',cv.WINDOW_NORMAL)
    cv.imshow('Original',frame)
    cv.imshow('Mask',mask)
    cv.imshow('Yellow',yellow)
    if cv.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv.destroyAllWindows()