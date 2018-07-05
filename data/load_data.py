import gc
import h5py
import numpy as np
from keras.utils import to_categorical
import os

def load_training_data(path:str,sequence=True,labels=False):
    DATA_PATH = os.path.join(path,"train.h5")

    with h5py.File(DATA_PATH, 'r') as f:
        X_train = np.copy(f['data']) / 255
        y_train = np.copy(f['labels'])

    if sequence:
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3], 1)
    else:
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)

    if not labels:
        y_train = to_categorical(y_train, 7)

    DATA_PATH = os.path.join(path,'val.h5')

    with h5py.File(DATA_PATH, 'r') as f:
        X_val = np.copy(f['data']) / 255
        y_val = np.copy(f['labels'])

    if sequence:
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], X_val.shape[3], 1)
    else:
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)

    if not labels:
        y_val = to_categorical(y_val, 7)

    gc.collect()

    return X_train,y_train,X_val,y_val


def load_val_data(path:str,sequence=True,labels=False):

    DATA_PATH = os.path.join(path,'val.h5')

    with h5py.File(DATA_PATH, 'r') as f:
        X_val = np.copy(f['data']) / 255
        y_val = np.copy(f['labels'])

    if sequence:
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], X_val.shape[3], 1)
    else:
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2],  1)

    if not labels:
        y_val = to_categorical(y_val, 7)

    gc.collect()

    return X_val,y_val
