import cv2 as cv
import cv2
import numpy as np

# Load the billboard image and the image to be placed on the billboard
billboard = cv2.imread(
    '/home/leyla/AI-CVpractice/Computer Vision and NLP/Exercise/billboard.jpg')


# Read the billboard and the image to be placed on it)
img = cv.resize(billboard, None, fx=0.1, fy=0.1, interpolation=cv.INTER_LINEAR)
poster = cv.imread('/home/leyla/AI-CVpractice/Computer Vision and NLP/Exercise/kuromicon.jpg')

# Blur image
blur = cv.GaussianBlur(img, (5, 5), cv.BORDER_CONSTANT)

# Edge Cascade
canny = cv.Canny(blur, 130, 175)
dilated = cv.dilate(canny, (3, 3), iterations=3)

# Find contours in the dilated image
contours, hierarchy = cv.findContours(
    dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv.contourArea, reverse=True)

# Approximate the largest contour as a quadrilateral
epsilon = 0.1 * cv.arcLength(contours[0], True)
approx = cv.approxPolyDP(contours[0], epsilon, True)
billboard_points = np.array([[approx[0][0][0], approx[0][0][1]], [approx[3][0][0], approx[3][0][1]], [
                            approx[2][0][0], approx[2][0][1]], [approx[1][0][0], approx[1][0][1]]], dtype=np.float32)
image_points = np.array([[0, 0], [poster.shape[1], 0], [
                        poster.shape[1], poster.shape[0]], [0, poster.shape[0]]], dtype=np.float32)

# Calculate the perspective transform matrix
M = cv.getPerspectiveTransform(image_points, billboard_points)
warped_image = cv.warpPerspective(poster, M, (img.shape[1], img.shape[0]))

# Create a binary mask for the image
mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
cv.fillConvexPoly(mask, billboard_points.astype(int), (255, 255, 255))

# Use the mask to place the image on the billboard
result = cv.bitwise_and(img, img, mask=cv.bitwise_not(mask))
result = cv.add(result, warped_image)

cv.imshow('Result', result)
cv.waitKey(0)


