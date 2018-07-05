import os

import cv2
import h5py
import numpy as np

from pre_processing.shared.utilities import process_video


def preprocessing(nbSequence,dataPath="/home/Amine/dataset/I. KDEF-dyn I/S4 Stimuli (Video-clips)/",size=(100,100)):
    datalabel={"Anger":0,"Disgust":1,"Fear":2,"Happiness":3, "Sadness":4 ,"Surprise":5}

    sequences=[]
    labels=[]
    for root, dirs, files in os.walk(dataPath):
        for key,label in datalabel.items():
            if key not in root:
                continue
            for filename in files:
                sequence=process_video(os.path.join(root,filename),nbSequence,size)
                if(sequence is None):
                    continue
                sequences.append(sequence)
                labels.append(label)
            break

    return sequences,labels


data,label=preprocessing(5)

data=np.array(data)
label=np.array(label)

h5f = h5py.File('../../dataset/KDEF-dyn/data.h5', 'w')
h5f.create_dataset('data', data=data)
h5f.create_dataset('labels', data=label)
h5f.close()