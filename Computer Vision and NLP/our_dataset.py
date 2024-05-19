"""
Created Mer 17/05/24 - 15:54 - AI lab
author@ walver

    -
    Dataset structure example

    imgs -> folder that contains the images
    labels.csv -> file that contains the labels for the images

    labels.csv:
    img1.jpg,0
    img2.jpg,1
    img3.jpg,2
    img4.jpg,0

    0 = dog
    1 = cat
    2 = car
    3 = airplane
    -
"""
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset


class OurDataSet(Dataset):
    def __init__(self, labels_path, imgs_dir, transform=None):
        super().__init__()
        self.imgs_dir = imgs_dir
        self.transform = transform
        self.labels = pd.read_csv(labels_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # create the imgs path
        imgs_path = f'{self.imgs_dir}/{self.labels.iloc[index, 0]}'

        # read the image with the path
        image = read_image(imgs_path)  # cv2.imread(imgs.path) for opencv

        # read the label
        label = self.labels.iloc[index, 1]

        # apply transformations
        if self.transform:
            image = self.transform(image)

        # return the pair image, label
        return image, label
