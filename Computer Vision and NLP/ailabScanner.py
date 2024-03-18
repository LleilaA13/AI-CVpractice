import cv2
import numpy as np

#define call back function for second ex:
def OnClick(event, x, y,flags, pars):
    if event == cv2.EVENT_LBUTTONDOWN:
            if len(src) < 4: #no prospettiva
                src.append([x, y])
                cv2.circle(img_copy, (x, y), 10,(0, 0, 255), -1 ) #img , center coord of circle, radius, color, the thickness (-1 filled)
                cv2.imshow('Image', img_copy)
                
img = cv2.imread('/home/leyla/AI-CVpractice/gerry.png')

# corners coordinates:
# t1 - 28, 227
# b1 - 131, 987
# br - 730, 860
# tr - 572, 149

# defining the starting points:

src_points = np.array(
    [
        [28, 227],
        [131, 987],
        [730, 860],
        [572, 149]
    ], np.float32
)

# ims = cv2.resize(img, None, fx = 0.5, fy = 0.5)
# cv2.imshow('Original', ims)
# cv2.waitKey(0)

# destination points:
dst_pts = np.array(
    [
        [0, 0],
        [0, 800],
        [600, 800],
        [600, 0]
    ], np.float32
)

# get the transformation matrix M
M = cv2.getPerspectiveTransform(src_points, dst_pts)

# compute the output img
out_img = cv2.warpPerspective(img, M, (600, 800))
# cv2.imshow('result', out_img)
# cv2.waitKey(0)

#SECOND EXERCISE OF THE CLASS:

# now we create a copy of the input img
img_copy = img.copy()
src = np.array([]) 

dst = np.array(
    [
        [0, 0],
        [0, 800],
        [600, 800],
        [600, 0]
    ], np.float32
)

# create the window
cv2.namedWindow('Image', cv2.WINDOW_KEEPRATIO)
cv2.setMouseCallback('Image', OnClick)

#SHow the img
cv2.imshow('Img', img_copy)
cv2.waitKey(0)

#compute the matrix M
src_float = np.array(src, dtype = np.float32)
M = cv2.getPerspectiveTransform(src, dst)

#get final img
out = cv2.warpPerspective(img, M, (600,800))
out = cv2.resize(out, None, fx = 0.5, fy = 0.5)

cv2.imshow('Result', out)
cv2.waitKey(0)