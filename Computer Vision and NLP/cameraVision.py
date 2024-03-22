import cv2
import numpy as np

cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read() #read returns boolean and the frame itself
    # we need gray scale img to detect the edges
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # apply light blur for cleaning the img a bit
    cv2.flip(frame,1, frame)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
'''
    # using Laplacian filter to estract contours
    edges = cv2.Laplacian(gray, cv2.CV_8U, ksize=5)

    # threshold the edges img to get only good edges
    ret, thresholded = cv2.threshold(edges, 70, 255, cv2.THRESH_BINARY_INV)

    # use the bilateral filter w/ high values for getting
    color_img = cv2.bilateralFilter(frame, 10, 255, 255)

    # Put together color and sketch
    skt = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)

    # et's do the bitwise AND for merging sketch and color
    output = cv2.bitwise_and(color_img, skt)
'''
