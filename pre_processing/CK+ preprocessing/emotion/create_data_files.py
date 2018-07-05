import numpy as np
import cv2
import os
import sklearn.model_selection

from PIL import Image

import h5py
from keras.utils.np_utils import to_categorical


def count_sequence(imageRoot):
    maxi = 0
    for dir in os.listdir(imageRoot):
        for subdir in os.listdir(os.path.join(imageRoot, dir)):
            maxi = max(maxi, len(os.listdir(os.path.join(imageRoot, dir, subdir))))
    return maxi


def count_exemple(labelRoot):
    count = 0
    for root, dirs, files in os.walk(labelRoot):
        count += len(files)

    return count


def generate_files(imageRoot, labelRoot, size):
    index_to_emotion = {0: "anger", 1: "contempt", 2: "disgust", 3: "fear", 4: "happy", 5: "sadness", 6: "surprise",
                        7: "neutral"}
    emotion_to_index = {v: k for k, v in index_to_emotion.items()}

    max_sequence = count_sequence(imageRoot)

    nbExemple = count_exemple(labelRoot)

    data = np.zeros([nbExemple, max_sequence, size[0] * size[1]])
    labels = np.zeros([nbExemple])

    curExemple = 0

    for root, dirs, files in os.walk(labelRoot):
        for filename in files:
            imagePath = root.replace(labelRoot, imageRoot)
            with open(os.path.join(root, filename))as file:
                for line in file:
                    labels[curExemple] = (int(line) - 1) % 8

            for i, filename in enumerate(os.listdir(imagePath)):
                image = Image.open(os.path.join(imagePath, filename))
                image_modifed = np.array(image).flatten()
                data[curExemple][i][:] = image_modifed
        if len(files) > 0:
            curExemple += 1

    assert curExemple == nbExemple
    labels = to_categorical(labels, num_classes=8)

    # data = np.array([data.astype(np.uint8), data.astype(np.uint8), data.astype(np.uint8)]) for having 3 channel
    # data = np.transpose(data, (1, 2, 3, 0))   for having 3 channel

    data = data.reshape(data.shape[0], data.shape[1], size[0], size[1], 1)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data, labels, test_size=0.15,random_state=1)

    X_train, X_validation, y_train, y_validation = sklearn.model_selection.train_test_split(X_train, y_train,
                                                                                            test_size=0.1,random_state=1)

    h5f = h5py.File('../../../dataset/validation.h5', 'w')
    h5f.create_dataset('data', data=X_validation)
    h5f.create_dataset('labels', data=y_validation)
    h5f.create_dataset('index_to_emotion', data=str(index_to_emotion))
    h5f.create_dataset('emotion_to_label', data=str(emotion_to_index))
    h5f.close()

    h5f = h5py.File('../../../dataset/train.h5', 'w')
    h5f.create_dataset('data', data=X_train)
    h5f.create_dataset('labels', data=y_train)
    h5f.create_dataset('index_to_emotion', data=str(index_to_emotion))
    h5f.create_dataset('emotion_to_label', data=str(emotion_to_index))
    h5f.close()

    h5f = h5py.File('../../../dataset/test.h5', 'w')
    h5f.create_dataset('data', data=X_test)
    h5f.create_dataset('labels', data=y_test)
    h5f.create_dataset('index_to_emotion', data=str(index_to_emotion))
    h5f.create_dataset('emotion_to_label', data=str(emotion_to_index))
    h5f.close()


labelRoot = "/home/Amine/dataset/www.consortium.ri.cmu.edu/data/ck/CK+/Emotions/"
imageRoot = "/home/Amine/dataset/www.consortium.ri.cmu.edu/data/ck/CK+/images/"
size = (200, 200)
generate_files(imageRoot, labelRoot, size)
