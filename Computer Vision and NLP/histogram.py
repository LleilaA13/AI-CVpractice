import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('/home/leyla/AI-CVpractice/Computer Vision and NLP/gerry.')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_eq = cv2.equalizeHist(gray)
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    hist = cv2.calcHist([img], [i] ,None, [256], [0, 256])
    plt.plot(hist, color = col)
#hist = cv2.calcHist([gray], [0], None, [256], [0, 256]) 
#[0] is the channel , [0, 256] values the pixel can have, [256] num of pixels on x-axis

plt.plot(hist)
plt.show()

