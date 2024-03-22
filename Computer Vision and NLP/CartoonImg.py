import cv2
import numpy as np

img = cv2.imread('/home/leyla/AI-CVpractice/Computer Vision and NLP/gerry.png')
#we need gray scale img to detect the edges
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#apply light blur for cleaning the img a bit
gray = cv2.medianBlur(gray, 5)

#using Laplacian filter to estract contours
edges = cv2.Laplacian(gray, cv2.CV_8U, ksize = 5)

#threshold the edges img to get only good edges
ret, thresholded = cv2.threshold(edges, 70, 255, cv2.THRESH_BINARY_INV)

#use the bilateral filter w/ high values for getting 
color_img= cv2.bilateralFilter(img ,10, 255, 255)

#Put together color and sketch
skt = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)

#et's do the bitwise AND for merging sketch and color
output = cv2.bitwise_and(color_img, skt)

cv2.imshow('out', output)
cv2.waitKey(0)