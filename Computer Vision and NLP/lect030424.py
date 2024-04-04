import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('/home/leyla/AI-CVpractice/Computer Vision and NLP/gerry.png')
'''
FIRST EXERCISE:
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#gray_eq = cv2.equalizeHist(gray)

channels = cv2.split(img)
eq_channels = []

for chan in channels:
    eq_channels.append(cv2.equalizeHist(chan))

equalized = cv2.merge(eq_channels)

cv2.imshow('1', img)
cv2.imshow('2', equalized)
cv2.waitKey(0)'''
#HSV = Hue, Saturation, Value
#hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#h, s, v = cv2.split(hsv_img)

#CLAHE
clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize=(6,6))
eq_img = clahe.apply(img)
# eq_v = cv2.equalizeHist(v)
'''
SECOND EXERCISE
equalized = cv2.merge([h, s, eq_v])
equalized = cv2.cvtColor(equalized, cv2.COLOR_HSV2BGR)
'''
cv2.imshow('1', img)
cv2.imshow('2', eq_img)
#cv2.imshow('3', )
cv2.waitKey(0)
