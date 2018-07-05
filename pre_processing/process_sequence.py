import h5py
import numpy as np
from sklearn.utils import shuffle

import sklearn.model_selection

sequence_data=["../dataset/KDEF-dyn/data.h5","../dataset/CASIA/data.h5"]


size=(100,100)
nbSequence=5
sequences=np.empty((0,nbSequence, size[0],size[1]))
labels=np.empty((0))

for filename in sequence_data:
    with h5py.File(filename, 'r') as f:
        X_train = np.copy(f['data'])
        y_train = np.copy(f['labels'])
        sequences=np.append(sequences,X_train,axis=0)
        labels=np.append(labels,y_train,axis=0)




sequences,labels=shuffle(sequences,labels)


h5f = h5py.File('../dataset/sequences/train.h5', 'w')
h5f.create_dataset('data', data=sequences)
h5f.create_dataset('labels', data=labels)
h5f.close()


sequences=np.empty((0,nbSequence, size[0],size[1]))
labels=np.empty((0))

with h5py.File("../dataset/CK+/data.h5", 'r') as f:
    X_train = np.copy(f['data'])
    y_train = np.copy(f['labels'])
    sequences = np.append(sequences, X_train, axis=0)
    labels = np.append(labels, y_train, axis=0)


sequences,labels=shuffle(sequences,labels)


h5f = h5py.File('../dataset/sequences/val.h5', 'w')
h5f.create_dataset('data', data=sequences)
h5f.create_dataset('labels', data=labels)
h5f.close()