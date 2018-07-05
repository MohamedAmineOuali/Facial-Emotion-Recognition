import os

import cv2
import h5py
import numpy as np


def padding_square(image):
    desired_size=max(image.shape[0],image.shape[1])
    delta_w = desired_size - image.shape[1]
    delta_h = desired_size - image.shape[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)
    return new_im


def preprocessing(dataPath="/home/Amine/dataset/SFEW_2/Train/", size=(100, 100)):
    datalabel = {"Angry": 0, "Disgust": 1, "Fear": 2, "Happy": 3, "Sad": 4, "Surprise": 5, "Neutral": 6}

    images = []
    labels = []
    for root, dirs, files in os.walk(dataPath):
        for filename in files:
            image = cv2.imread(os.path.join(root, filename), 0)
            if image is None:
                continue
            image=padding_square(image)
            images.append(cv2.resize(image, size))
            label = root.split("/")[-1]
            labels.append(datalabel[label])

    return images, labels



data,label=preprocessing()

data=np.array(data)
label=np.array(label)

h5f = h5py.File('../../dataset/SFEW_2/train.h5', 'w')
h5f.create_dataset('data', data=data)
h5f.create_dataset('labels', data=label)
h5f.close()