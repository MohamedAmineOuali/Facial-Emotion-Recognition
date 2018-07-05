import os

import cv2
import h5py
import numpy as np

from pre_processing.shared.utilities import cropping_face, process_video


def preprocessing(nbSequence,dataPath="/home/Amine/dataset/CASIA/",size=(100,100)):
    datalabel={"S_A":0,"S_D":1,"S_F":2,"S_H":3, "S_S2":4 ,"S_S1":5}

    sequences=[]
    labels=[]
    for root, dirs, files in os.walk(dataPath):
        for filename in files:
            for key,label in datalabel.items():
                if key in filename:
                    sequence=process_video(os.path.join(root,filename),nbSequence,None)
                    if(sequence is None):
                        continue
                    x, y, w, h=cropping_face(sequence)

                    for i in range(len(sequence)):
                        sequence[i]=cv2.resize(sequence[i][y:y + h, x:x + w],size)

                    sequences.append(sequence)
                    labels.append(label)
                    break

    return sequences,labels


data,label=preprocessing(5)

data=np.array(data)
label=np.array(label)

h5f = h5py.File('../../dataset/CASIA/data.h5', 'w')
h5f.create_dataset('data', data=data)
h5f.create_dataset('labels', data=label)
h5f.close()