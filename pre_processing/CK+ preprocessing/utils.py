import cv2
import os
from matplotlib import pyplot as plt


def show(img):
    cv2.imshow("tt", img)
    cv2.waitKey(20)


def show_images(imageRoot):
    for root, dirs, files in os.walk(imageRoot):
        for filename in files:
            img = cv2.imread(root + "/" + filename, cv2.IMREAD_ANYCOLOR)
            show(img)


def size_verification(imageRoot):
    total = 0

    width = []
    height = []
    for root, dirs, files in os.walk(imageRoot):
        for filename in files:
            img = cv2.imread(root + "/" + filename, cv2.IMREAD_ANYCOLOR)
            total += 1
            width.append(img.shape[0])
            height.append(img.shape[1])
    plt.figure()
    print(total)
    plt.scatter(width, height)
    plt.show(block=True)
