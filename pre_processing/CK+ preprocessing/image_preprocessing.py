import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import cv2
from utils import size_verification, show_images


def norm(img1, img2):
    return cv2.norm(img1, img2)


def region_to_select(images, face_cascade):
    i = 0
    faces = []
    for image in images:
        face = face_cascade.detectMultiScale(image, 1.3, 5)
        if len(face) > 0:
            cur = face[0]
            while i < len(face) and face[i][2] > face[i - 1][2]:
                cur = face[i]
                i += 1
            if cur[2] > 200:
                faces.append(cur)

    w = 0
    h = 0
    x = 10000
    y = 10000
    for face in faces:
        w = max(w, face[2])
        h = max(h, face[3])
        x = min(x, face[0])
        y = min(y, face[1])

    return x, y, w, h


def select_images(images, limit, area):
    (x, y, w, h) = area

    while len(images) > limit:
        distance = 100000
        index = -1
        for i in range(1, len(images)):
            cur = norm(images[i][y:y + h, x:x + w], images[i - 1][y:y + h, x:x + w])
            if distance > cur:
                distance = cur
                index = i
        if index > len(images) / 2:
            del images[index - 1]
        else:
            del images[index]

    return images


def adjust_images_delete_similar(size, root, outputRoot, face_cascade):
    files = os.listdir(root)

    images = []
    for file in files:
        images.append(cv2.imread(os.path.join(root , file), cv2.IMREAD_GRAYSCALE))

    (x, y, w, h) = region_to_select(images, face_cascade)
    images = select_images(images, 5, (x, y, w, h))

    for i in range(0, len(images)):
        filename = os.path.join(outputRoot, files[i])
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        cv2.imwrite(filename, cv2.resize(images[i][y:y + h, x:x + w], size))


def process_images(size, rootpath, rootoutputPath):
    face_cascade = cv2.CascadeClassifier('haarcascades_cuda/haarcascade_frontalface_default.xml')

    dirs = os.listdir(rootpath)

    for dir1 in dirs:
        adjust_images_delete_similar(size,os.path.join(rootpath ,dir1), os.path.join(rootoutputPath , dir1), face_cascade)


def filter_similar(size, imageRoot, outputRoot):
    executor = ThreadPoolExecutor(max_workers=8 * 3)

    for dir in os.listdir(imageRoot):
        # process_images(size, os.path.join(imageRoot , dir) ,os.path.join(outputRoot , dir))
        executor.submit(process_images, size, os.path.join(imageRoot , dir) ,os.path.join(outputRoot , dir))

    executor.shutdown(wait=True)


def resize(imageRoot, outputRoot):
    for dir in os.listdir(imageRoot):
        for dir1 in os.listdir(os.path.join(imageRoot, dir)):
            for filename in os.listdir(os.path.join(imageRoot , dir, dir1)):
                img = cv2.imread(os.path.join(imageRoot , dir , dir1 , filename), cv2.IMREAD_ANYCOLOR)
                if img.shape[1] == 720:
                    if dir == "S501" or dir == "S502":
                        img = img[:, 50:690]
                    else:
                        img = img[:, 10:650]
                elif img.shape[0] == 490:
                    img = img[10:490, :]
                newFilename = os.path.join(outputRoot , dir, dir1 , filename)
                os.makedirs(os.path.dirname(newFilename), exist_ok=True)
                cv2.imwrite(newFilename, img)


def images_preprocessing(size, imageRoot, outputRoot):
    resize(imageRoot, "tmp/")
    filter_similar(size, "tmp/", outputRoot)
    size_verification(outputRoot)
    show_images(outputRoot)
    shutil.rmtree("tmp/")


imageRoot = "/home/Amine/dataset/www.consortium.ri.cmu.edu/data/ck/CK+/cohn-kanade-images/"
outputRoot = "/home/Amine/dataset/www.consortium.ri.cmu.edu/data/ck/CK+/images/"
size = (200, 200)
images_preprocessing(size, imageRoot, outputRoot)
