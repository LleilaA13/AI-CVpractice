import cv2
import numpy as np

'''
x = np.uint8([250])
y = np.uint8([50])

result_opencv = cv2.add(x, y)
result_np = x + y
print(f'Opencv: {result_opencv}')
print(f'Numpy:{result_np}')
'''
img = cv2.imread('/home/leyla/AI-CVpractice/Computer Vision and NLP/gerry.png')
img2 = cv2.imread('/home/leyla/AI-CVpractice/Computer Vision and NLP/kuromicon.png')
M = np.full(img2.shape, 50, dtype = 'uint8')
img2_res = cv2.resize(img2, (img.shape[1], img.shape[0]))
added_img = cv2.addWeighted(img, 0.5, img2_res,0.5, 1)
#cv2.imshow('output', img2)
cv2.imshow('out', added_img)
cv2.waitKey(0)