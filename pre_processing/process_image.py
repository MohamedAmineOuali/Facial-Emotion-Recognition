import h5py
import numpy as np
from sklearn.utils import shuffle

sequence_data=["../dataset/KDEF-dyn/data.h5","../dataset/CASIA/data.h5","../dataset/CK+/data.h5"]

image_data=["../dataset/KDEF/data.h5","../dataset/SFEW_2/train.h5"]

size=(100,100)

images=np.empty((0, size[0],size[1]))
labels=np.empty((0))

for filename in sequence_data:
    with h5py.File(filename, 'r') as f:
        X_train = np.copy(f['data'])
        y_train = np.copy(f['labels'])
        images=np.append(images,X_train[:,-1,:,:],axis=0)
        labels=np.append(labels,y_train,axis=0)

for filename in image_data:
    with h5py.File(filename, 'r') as f:
        X_train = np.copy(f['data'])
        y_train = np.copy(f['labels'])
        images=np.append(images,X_train,axis=0)
        labels=np.append(labels,y_train,axis=0)



images,labels=shuffle(images,labels)

h5f = h5py.File('../dataset/train.h5', 'w')
h5f.create_dataset('data', data=images)
h5f.create_dataset('labels', data=labels)
h5f.close()