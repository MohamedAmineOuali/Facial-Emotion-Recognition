import numpy as np
import matplotlib.pyplot as plt
import os

from data.load_data import load_val_data, load_training_data
from models.simple_cnn_model import SimpleCnnModel
from models.vgg_models import Vgg16Model





model=SimpleCnnModel(False,sequence=True,stateful=True)

dataPath="weights/simple_cnn_based/sequence/statefull"

destination="confusion_matrix/simple_cnn_model/sequence/Statefull"


def dataset_matrix(path,name):
    X_train, y_train, X_val, y_val = load_training_data(path, sequence=True, labels=True)

    model.save_confusion_matrix(X_train, y_train, os.path.join(curPath, "training {}.png".format(name)),
                          ''.format(y_train.shape[0]),normalize=True)
    model.save_confusion_matrix(X_val, y_val, os.path.join(curPath, "val {}.png".format(name)),
                          ''.format(y_val.shape[0]),normalize=True)


for root, dirs, files in os.walk(dataPath):
    for filename in files:
        print(filename)

        curPath=os.path.join(root, filename)
        model.load_weights(curPath)
        curPath=curPath.replace(dataPath,destination)
        os.makedirs(curPath, exist_ok=True)

        # dataset_matrix("dataset/train_val","mix")
        # dataset_matrix("dataset/train_val2","CK+")
        # dataset_matrix("dataset/val_SFEW_2", "SFEW_2")

        dataset_matrix("dataset/sequences", "sequence")
        plt.close("all")





