import cv2

img = cv2.imread('/home/leyla/Pictures/kuromicon.jpeg', 0) #we are reading the img as a gray scale img

print(f'Rows: {img.shape[0]}')
print(f'Columns: {img.shape[1]}')
#print(f'Channels: {img.shape[2]}') #Index error! why? -> we have one channel

#cv2.imshow('Loaded Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows() #close all the open windows of OpenCV

#convert the img after it's been read
cv2.imwrite('/home/leyla/Pictures/kuromicon.png', img)
img2 = cv2.imread('/home/leyla/Pictures/kuromicon.png')

#cv2.imshow('PNG', img2)

cv2.waitKey(0)

img = cv2.imread('/home/leyla/Pictures/kuromicon.jpeg')
# Images are just NumPy arrays. The top-left pixel can be
# found at (0, 0)
(b, g, r) = img[0, 0]
print(b, g, r)
print("Pixel at (0, 0) - Red: {}, Green: {}, Blue: {}".format(r, g, b))

#slicing?
img55 = cv2.imread('/home/leyla/Pictures/kuromicon.png')
# Let's make the top-left corner of the image green
img55[0:50, 0:50] = (0, 255, 0)

# Show our updated image

#cv2.imshow("updated", img55)

cv2.waitKey(0)
cv2.destroyAllWindows()  # needed only for jupyter notebook


import numpy as np
img66 = np.zeros((500, 500, 3), dtype = 'uint8') #standard data type for channels

#def a color
green = (0, 255, 0)

#draw a line
cv2.line(img66, (10, 10), (200, 200), green, 3)

#draw a rectangle
red = (0, 0, 255)
cv2.rectangle(img66,(30, 30), (70, 70), red, -1)



#draw a circle
blue = (255, 0, 0)
(centerx, centery)= (img66.shape[0] // 2, img.shape[1] //2)
cv2.circle(img66, (centerx, centery), 20, blue)

#cv2.imshow('immagine', img66)
cv2.waitKey(0)

#3 channels
img5 = cv2.imread('/home/leyla/Pictures/kuromicon.jpeg')
(b, g, r) = cv2.split(img)
img_copy = cv2.merge((r, g, b))

#cv2.imshow('Red Channel;', r)
#cv2.imshow('Green Channel;', g)
#cv2.imshow('Blue Channel;', b)
#cv2.imshow('ffffff',img_copy)
cv2.waitKey(0)

#let's draw one last rectangle: blue and filled in
canvas = np.zeros((300, 300, 3), dtype="uint8")
blu = (255, 0, 0)
cv2.rectangle(canvas, (200, 50), (225, 125), blu, -1)
cv2.imshow('canvas', canvas)
cv2.waitKey(0)


(r, g, b) = cv2.split(img)
