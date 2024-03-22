import cv2
import numpy as np

img1 = np.zeros((500, 500), dtype='uint8')
img2 = np.zeros((500, 500), dtype='uint8')


cv2.rectangle(img1, (200, 100), (300, 300), (255, 255, 255), -1)
cv2.rectangle(img2, (50, 50), (350, 300), (255, 255, 255), -1)

result = cv2.bitwise_or(img1, img2)
cv2.imshow('output 1', img1)
cv2.imshow('output 2', img2)
cv2.imshow('result', result)
cv2.waitKey(0)