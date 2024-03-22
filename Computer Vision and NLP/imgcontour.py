import cv2
import numpy as np

img = cv2.imread('/home/leyla/AI-CVpractice/Computer Vision and NLP/gerry.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# x-derivative, y-derivative (last 2 args)
'''der_x = cv2.Sobel(img_gray, -1, 1, 0)
der_y = cv2.Sobel(img_gray, -1, 0, 1)
scaled_x = cv2.convertScaleAbs(der_x)
scaled_y = cv2.convertScaleAbs(der_y)
out = cv2.addWeighted(scaled_x, 0.5, scaled_y, 0.5, 0)
'''
der = cv2.Laplacian(img_gray, -1, (3, 3))

out = cv2.convertScaleAbs(der)

cv2.imshow('laplacian', out)

#cv2.imshow('X', out)
#cv2.imshow('Y', scaled_y)
cv2.waitKey(0)
