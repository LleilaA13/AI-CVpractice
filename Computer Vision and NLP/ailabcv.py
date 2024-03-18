import cv2
import time

img = cv2.imread('/home/leyla/Pictures/kuromicon.jpeg')

print(f'Width: {img.shape[1]} pixels')
print(f'Height: {img.shape[0]} pixels')
print(f'Channels: {img.shape[2]}')


cv2.imshow('Image', img)
cv2.waitKey(0)
time.sleep(2)
#cv2.destroyAllWindows()
