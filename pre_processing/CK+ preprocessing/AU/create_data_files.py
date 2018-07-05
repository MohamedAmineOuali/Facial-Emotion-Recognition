import numpy as np
import cv2
import os

from PIL import Image


import h5py


# def generate_bin(imageRoot,labelRoot):
#     s = map()
#     for root, dirs, files in os.walk(labelRoot):
#         for filename in files:
#             with open(os.path.join(root , filename))as file:
#                 for line in file:
#                     s.add(int(line))
#
#
#
#     data=[]
#     for root, dirs, files in os.walk(labelRoot):
#         for filename in files:
#             imagePath=root.replace(labelRoot,imageRoot)
#             actionUnits=[]
#             with open(os.path.join(root , filename))as file:
#                 for line in file:
#                         actionUnits.append(int(line))
#
#             images=[]
#             for filename in os.listdir(imagePath):
#                 image = Image.open(os.path.join(imagePath, filename))
#                 image_modifed = np.array(image)
#                 grey = image_modifed[:].flatten()
#                 images.append(grey)
#             if(len(os.listdir(imagePath))<10):
#                 print(imagePath)
#
#             data.append((images,actionUnits))
#
#     data=np.array(data)
#     np.save('data.bin',data)
#     # data.("data.txt")


def count_exemple_sequence(imageRoot):
    total = 0
    maxi = 0
    for dir in os.listdir(imageRoot):
        for subdir in os.listdir(os.path.join(imageRoot, dir)):
            total += 1
            maxi = max(maxi, len(os.listdir(os.path.join(imageRoot, dir, subdir))))
    return total, maxi


def label_map(labelRoot):
    s = set()
    for root, dirs, files in os.walk(labelRoot):
        for filename in files:
            with open(os.path.join(root, filename))as file:
                for line in file:
                    s.add(int(line))
    m = {}
    for i, actionUnit in enumerate(s):
        m[actionUnit] = i

    return m


def generate_files(imageRoot, labelRoot, size):
    AU_to_index = {0:"anger", 1:"contempt", 2:"disgust", 3:"fear", 4:"happy", 5:"sadness", 6:"surprise",-1:"neutral"}

    index_to_AU= {v: k for k, v in AU_to_index.items()}
    nbExemple, max_sequence = count_exemple_sequence(imageRoot)

    data = np.zeros([nbExemple,max_sequence, size[0] *size[1]])
    labels = np.zeros([nbExemple, len(AU_to_index)])

    curExemple = 0

    for root, dirs, files in os.walk(labelRoot):
        for filename in files:
            imagePath = root.replace(labelRoot, imageRoot)
            actionUnits = []
            with open(os.path.join(root, filename))as file:
                for line in file:
                    actionUnits.append(AU_to_index[int(line)])

            labels[curExemple][actionUnits] = 1

            for i, filename in enumerate(os.listdir(imagePath)):
                image = Image.open(os.path.join(imagePath, filename))
                image_modifed = np.array(image).flatten()
                data[curExemple][i][:]=image_modifed
        if(len(files)>0):
            curExemple += 1

    h5f = h5py.File('../dataset/data_label.h5', 'w')
    h5f.create_dataset('data', data=data)
    h5f.create_dataset('labels',data=labels)
    h5f.create_dataset('AU_to_index', data=str(AU_to_index))
    h5f.create_dataset('index_to_AU', data=str(index_to_AU))
    h5f.close()


    # data.tofile("../dataset/data.bin")
    # labels.tofile("../dataset/label.bin")
    # np.save('../dataset/index_to_AU.npy', index_to_AU)
    # np.save('../dataset/AU_to_index.npy',AU_to_index)
    # np.savetxt("data.csv", data, delimiter=",")
    # np.savetxt("labels.csv", labels, delimiter=",")



labelRoot = "/home/Amine/dataset/www.consortium.ri.cmu.edu/data/ck/CK+/AU/"
imageRoot = "/home/Amine/dataset/www.consortium.ri.cmu.edu/data/ck/CK+/images/"
size = (224, 224)
generate_files(imageRoot, labelRoot, size)
