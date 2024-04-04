import cv2
import numpy as np

img = cv2.imread('/home/leyla/AI-CVpractice/Computer Vision and NLP/gerry.png')

mask = np.zeros(img.shape, dtype='uint8')

cv2.rectangle(mask, (40, 40), (150, 150), (255, 255, 255), -1)

result = cv2.bitwise_xor(img, mask)
cv2.imshow('1', result)
cv2.waitKey(0)
cv2.destroyAllWindows()