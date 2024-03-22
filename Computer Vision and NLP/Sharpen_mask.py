import cv2
import numpy as np
img = cv2.imread('')
my_kernel = np.array(
    [
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
        
    ]
)

sharp = cv2.filter2D(img, -1, my_kernel)
cv2.imshow('output', sharp)
cv2.waitKey(0)