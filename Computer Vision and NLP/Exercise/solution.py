# Correction of the billboard exercise

import cv2
import numpy as np

# define the onClick function (mouse call back function)


def OnClick(event, x, y, flags, pars):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(dst_points) < 4:
            dst_points.append([x, y])
            cv2.circle(img_copy, (x, y), 50, (0, 0, 255), -1)
            cv2.imshow('Base image', img_copy)


# loard the 2 images
base_img = cv2.imread(
    '/home/leyla/AI-CVpractice/Computer Vision and NLP/Exercise/billboard.jpg')
img_copy = base_img.copy()  # i am copying so we can draw the circle on that img
img2 = cv2.imread(
    '/home/leyla/AI-CVpractice/Computer Vision and NLP/Exercise/kuromicon.jpg')

# get the image data
# took first 2 values of shape of the billboard image
base_h, base_w = base_img.shape[:2]
img2_h, img2_w = img2.shape[:2]  # took first 2 values of shape of the img

# create src and destination pts (cols -> rows)
src_points = np.array(
    [
        [0, 0],
        [0, img2_h],
        [img2_w, img2_h],
        [img2_w, 0]
    ], dtype=np.float32  # specify type sennò cv2 si lamenta ~Pannnone
)

dst_points = []  # empy bc we will fill them with the clicks function

# define the window for getting the destination pts
cv2.namedWindow('Base image', cv2.WINDOW_KEEPRATIO)
# it will check if we are doing smth with the mouse, and execute the function
cv2.setMouseCallback('Base image', OnClick)

# show the image to be clicked
cv2.imshow('Base Img', base_img)
cv2.waitKey(0)

# get the homography matrix
dst_float = np.array(dst_points, dtype=np.float32)
H = cv2.getPerspectiveTransform(src_points, dst_float)

#apply the homography matric to the image angles
warped = cv2.warpPerspective(img2, H, (base_w, base_h))

#create a mask to drop black pixels from the warped img
mask = np.zeros(base_img.shape, dtype = np.uint8)

cv2.fillConvexPoly(mask, np.int32(dst_float), (255, 255, 255))

#apply the mask
mask = cv2.bitwise_not(mask) #invert the mask for removing the colored pixels
masked_bill = cv2.bitwise_and(base_img, mask)

#apply the same mask to our second img
final_img = cv2.bitwise_or(masked_bill, warped)

cv2.namedWindow('Final img', cv2.WINDOW_KEEPRATIO)
cv2.imshow('Final img', final_img)
cv2.waitKey(0)