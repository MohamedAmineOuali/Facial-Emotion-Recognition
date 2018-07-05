

from data.load_data import load_training_data
from models.simple_cnn_model import *

X_train,y_train,X_val,y_val=load_training_data("dataset/train_val2",sequence=False)


model=SimpleCnnModel(False, sequence=False, cnnTrainable=False)




model.train(X_train, y_train,[X_val,y_val],batch_size=4,epochs=1000,data_augmentation=False)


