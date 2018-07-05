
import keras

from models.model import IModel
import keras
from keras.layers import *

from keras.models import Sequential



class SimpleCnnModel(IModel):

    @staticmethod
    def basic_model():
        model = Sequential()
        model.add(Convolution2D(filters=16, kernel_size=(7, 7), padding='same',
                                name='image_array', input_shape=(100, 100, 1)))
        model.add(BatchNormalization())
        model.add(Convolution2D(filters=16, kernel_size=(7, 7), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(.5))

        model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
        model.add(BatchNormalization())
        model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(.5))

        model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(.5))

        model.add(Convolution2D(filters=128, kernel_size=(3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Convolution2D(filters=128, kernel_size=(3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(.5))

        model.add(Convolution2D(filters=256, kernel_size=(3, 3), padding='same'))
        model.add(BatchNormalization())

        return model


    def cnn_model(self):
        cnnModel = SimpleCnnModel.basic_model()
        cnnModel.add(Convolution2D(filters=7, kernel_size=(3, 3), padding='same'))
        cnnModel.add(GlobalAveragePooling2D())
        cnnModel.add(Activation('softmax', name='predictions'))
        return cnnModel


    def sequence_model(self,cnnTrainable, stateful,nbSequence, bach_size):
        cnnModel = SimpleCnnModel.basic_model()
        cnnModel.add(GlobalAveragePooling2D())
        if cnnTrainable == False:
            for layer in cnnModel.layers:
                layer.trainable = False

        model = keras.models.Sequential()

        if stateful:
            model.add(keras.layers.TimeDistributed(cnnModel, batch_input_shape=(bach_size, nbSequence, 100, 100, 1), name="CNN_Model"))
        else:
            model.add(keras.layers.TimeDistributed(cnnModel, input_shape=(nbSequence, 100, 100, 1), name="CNN_Model"))

        model.add(keras.layers.LSTM(256, name="lstm1", return_sequences=True, stateful=stateful))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.LSTM(128, name="lstm2", stateful=stateful))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(128))
        model.add(keras.layers.PReLU(alpha_initializer='zeros'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(64))
        model.add(keras.layers.PReLU(alpha_initializer='zeros'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(7, name='predictions'))
        model.add(keras.layers.Activation('softmax'))
        return model

    def __init__(self, trainable, sequence=False,nbSequence=5, cnnTrainable=False, stateful=False, bach_size=1):
        super().__init__(trainable, sequence,nbSequence, cnnTrainable, stateful, bach_size)

