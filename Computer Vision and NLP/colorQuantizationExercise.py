#pytorch        
import numpy as np
import cv2
from sklearn.cluster import KMeans #imput data must be a table


#load the image
img = cv2.imread(
    '/home/leyla/Desktop/repos/AI-CVpractice/Computer Vision and NLP/gerry.png')

#store the shape of the image
(h, w, c) = img.shape

#create the K-means instance
model = KMeans(n_clusters=7) #specify the number of clusters (groups) we want to divide the data into

#reshape the image from 3d to 2d
img2D = img.reshape(h*w, c) #pytorch does not support 3D data

#map the colors in the image to the clusters
cluster_labels = model.fit_predict(img2D) #compute the distance from the centroids, not a real training 

#convert centroids values to good pixel values
bgr_cols = model.cluster_centers_.round(0).astype(int)

#get back the 3D image
img_quant = np.reshape(bgr_cols[cluster_labels], (h, w, c))
img_quant = img_quant.astype(np.uint8)

#show the original and quantized images
cv2.imshow('Original Image', img)
cv2.imshow('Quantized Image', img_quant)
cv2.waitKey(0)