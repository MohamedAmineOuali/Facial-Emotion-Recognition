import os

import cv2
import h5py
import numpy as np


def preprocessing(dataPath="/home/Amine/dataset/KDEF_and_AKDEF/",size=(100,100)):
    datalabel={"AN":0,"DI":1,"AF":2,"HA":3, "SA":4 ,"SU":5,"NE":6}

    position=["S","HL","HR"]


    patterns={}

    for key,value in datalabel.items():
        for p in position:
            patterns[key+p]=value

    images=[]
    labels=[]
    for root, dirs, files in os.walk(dataPath):
        for filename in files:
            for pattern,label in patterns.items():
                if( pattern in filename):
                    image = cv2.imread(os.path.join(root, filename),0)
                    image=image[100:-100,:]#cropping

                    if(image[0][25]==0):#if is bad example
                        break

                    images.append(cv2.resize(image,size))
                    labels.append(label)
                    break

    return images,labels

data,label=preprocessing()

data=np.array(data)
label=np.array(label)

h5f = h5py.File('../../dataset/KDEF/data.h5', 'w')
h5f.create_dataset('data', data=data)
h5f.create_dataset('labels', data=label)
h5f.close()