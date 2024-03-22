import cv2
import numpy as np

# define the callback function


def onClick(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(src_points) < 4:
            src_points.append([x, y])
            cv2.circle(img_copy, (x, y), 10, (0, 0, 255), -1)
            cv2.imshow('Img', img_copy)


# read image
img = cv2.imread('/home/leyla/AI-CVpractice/Computer Vision and NLP/gerry.png')

# create a copy of the input image
img_copy = img.copy()

# defining the starting points
src_points = []

# defining destinations points
dst_points = np.array(
    [
        [0, 0],
        [0, 800],
        [600, 800],
        [600, 0]
    ], np.float32
)

# create the window
cv2.namedWindow('Img', cv2.WINDOW_KEEPRATIO)
cv2.setMouseCallback('Img', onClick)

# Show the image
cv2.imshow('Img', img_copy)
cv2.waitKey(0)

# compute the matrix M
src_float = np.array(src_points, dtype=np.float32)
M = cv2.getPerspectiveTransform(src_float, dst_points)

# get the final image
out_img = cv2.warpPerspective(img, M, (600, 800))

cv2.imshow('Result', out_img)
cv2.waitKey(0)
