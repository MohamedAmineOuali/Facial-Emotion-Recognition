
import keras

from models.model import IModel




class Vgg16Model(IModel):

    @staticmethod
    def basic_model():
        vggmodel = keras.applications.vgg16.VGG16(include_top=False, weights=None,
                                                  input_shape=(100, 100, 1))
        cnnModel = keras.models.Sequential()
        for layer in vggmodel.layers:
            cnnModel.add(layer)
        cnnModel.add(keras.layers.GlobalAveragePooling2D())
        cnnModel.add(keras.layers.Dense(256))
        cnnModel.add(keras.layers.PReLU(alpha_initializer='zeros'))
        cnnModel.add(keras.layers.Dropout(0.5))

        return cnnModel


    def cnn_model(self):
        cnnModel = Vgg16Model.basic_model()
        cnnModel.add(keras.layers.Dense(7, name='predictions'))
        cnnModel.add(keras.layers.Activation('softmax'))
        return cnnModel


    def sequence_model(self,cnnTrainable, stateful,nbSequence, bach_size):
        cnnModel = Vgg16Model.basic_model()

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

    def __init__(self, trainable, sequence=False, nbSequence=5,cnnTrainable=False, stateful=False, bach_size=1):
        super().__init__(trainable, sequence,nbSequence, cnnTrainable, stateful, bach_size)



