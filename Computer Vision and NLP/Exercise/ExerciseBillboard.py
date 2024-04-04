import cv2
import numpy as np
'''In this exercise, you will have to use both the homography matrix and some bitwise operators in order to
place an image of your choice over a billboard image. The latter is provided by me, and you can find it in
the same folder of this document.
You have to perform two main steps:
1 - Use the homography to match the image of your choice with the billboard (we have seen this in class);
2 - Create a binary mask. In a binary mask, white pixels are the ones that we want in the final image, black
pixels are the ones that we do not want in the final image (think about the example seen in class with
white circle and t-rex);
3 - Use the binary mask to place the image you choose on the billboard.
Advice: you can use the function fillConvexPoly to create the white pixels in the mask.'''

#define the callback function that selects the image we need on the billboard
def onClick(event, x, y, flags, pars):
    if event ==  cv2.EVENT_LBUTTONDOWN:
        if len(src_pts) < 4:
            




src_pts = []