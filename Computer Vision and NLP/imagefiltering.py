import cv2
import numpy as np

img = cv2.imread(
    '/home/leyla/AI-CVpractice/Computer Vision and NLP/saltandpepper.png')
my_kernel = np.array([
    [66, 0, 1],
    [1, 6, 66],
    [1, 0, 1]
])/9 #use kernel that is odd and symmetric, to always have the center of the kernel

#filtered_img = cv2.filter2D(img, -1, my_kernel) #same num of channels w/ -1 apply convolution on imput img and returns output img
#blur:
filtered_blur = cv2.blur(img, (19, 19))
filtered_gauss = cv2.GaussianBlur(img, (5,5), 0)
filtered_median = cv2.medianBlur(img, 5) 
bilateral_filter = cv2.bilateralFilter(img ,9, 75, 75)
sharped_img = cv2.addWeighted(img, 0.5, filtered_gauss, 5.5, 0)

cv2.imshow('Output', filtered_blur)
cv2.imshow('g.b', filtered_gauss)
cv2.imshow('median', filtered_median)
cv2.imshow('bilateral filter', bilateral_filter)
cv2.waitKey(0)





