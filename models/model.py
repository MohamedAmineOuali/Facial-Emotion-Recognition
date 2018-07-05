import gc

import cv2
import tensorflow as tf
import numpy as np
import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, f1_score, precision_score

from models.utilis import plot_confusion_matrix


class SequenceGenerator:

    def __init__(self, seq_len=5):
        self.datagen = []
        for _ in range(0, seq_len):
            self.datagen.append(tf.contrib.keras.preprocessing.image.ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.06,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.06,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False))  # randomly flip images

    def flow(self, x, y, batch_size):
        flows = []
        for i, generator in enumerate(self.datagen):
            generator.fit(x[:, i], seed=1)
            flows.append(generator.flow(x[:, i],
                                        y, seed=1,
                                        batch_size=batch_size))

        while True:
            x = []
            y = None
            for flow in flows:
                cur_x, y = next(flow)
                x.append(cur_x)
            x = np.array(x)
            x = np.transpose(x, (1, 0, 2, 3, 4))
            gc.collect()
            yield x, y


class StaticGenerator:
    def __init__(self):
        self.datagen = tf.contrib.keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

    def flow(self, x, y, batch_size):
        self.datagen.fit(x)
        flow = self.datagen.flow(x, y, batch_size=batch_size)
        while True:
            yield next(flow)


class IModel:
    def sequence_model(self, cnnTrainable, stateful, nbSequence, bach_size):
        raise NotImplementedError

    def cnn_model(self):
        raise NotImplementedError

    def __init__(self, trainable, sequence, nbSequence, cnnTrainable, stateful, bach_size):
        self.model = None  # type: keras.models.Model
        if (sequence == False and stateful == True):
            raise Exception("if sequence not specified stateFull should be false")

        if (sequence == False and cnnTrainable == True):
            raise Exception("if sequence not specified cnnTrainable should be false")

        if bach_size != 1 and stateful == False:
            raise Exception("bach size should not be specified if sequence is not")

        if sequence:
            self.model = self.sequence_model(cnnTrainable, stateful, nbSequence, bach_size)
            self.class_names = ["Angry", "Disgusted", "fear", "happy", "Sad", "surprised"]
        else:
            self.model = self.cnn_model()
            self.class_names = ["Angry", "Disgusted", "fear", "happy", "Sad", "surprised", "neutral"]

        if not trainable:
            for layer in self.model.layers:
                layer.trainable = False

        ada = keras.optimizers.Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
        self.model.compile(loss='categorical_crossentropy', optimizer=ada, metrics=['accuracy'])
        self.sequence=sequence
        self.nbSequence = nbSequence
        self.bach_size = bach_size

    def evaluate(self, X_test, y_test, batch_size=30):
        scores = self.model.evaluate(X_test, y_test, verbose=1, batch_size=batch_size)
        return scores

    def predict(self, img):
        img = cv2.resize(img, (100, 100))
        if self.sequence:
            img = img.reshape((1,1, 100, 100, 1)) / 255
        else:
            img = img.reshape((1,100, 100, 1)) / 255

        return self.model.predict(img)

    def predict_classes(self, imgs):
        return self.model.predict_classes(imgs)

    def confusion_matrix(self, X_test, y_test, batch_size=1):
        y_pred = self.model.predict_classes(X_test, batch_size=batch_size)
        return confusion_matrix(y_test, y_pred)

    def evaluation_metrics(self, X_test, y_true,batch_size=1):
        y_pred = self.model.predict_classes(X_test, batch_size=batch_size)
        cnf_matrix=confusion_matrix(y_true, y_pred)

        precision = precision_score(y_true, y_pred,average='weighted')
        recall = recall_score(y_true, y_pred,average='weighted')
        accuracy = accuracy_score(y_true, y_pred)

        F1_score = f1_score(y_true, y_pred,average='weighted')

        return cnf_matrix,accuracy, precision, recall, F1_score

    def save_confusion_matrix(self, X_test, y_test, filePath, title, batch_size=1,normalize=False):

        plt.figure(figsize=(8.7, 6.2))

        cnf_matrix, accuracy, precision, recall, F1_score = self.evaluation_metrics(X_test, y_test)
        plot_confusion_matrix(cnf_matrix, classes=self.class_names,
                              title=title,normalize=normalize)


        text='accuracy: {:.2f}% \nprecision: {:.2f} \nrecall: {:.2f} \n F1_score: {:.2f}'.format(accuracy * 100, precision,
                                                                                            recall, F1_score)
        plt.figtext(0.82, 0.5,text,bbox=dict(facecolor='white'))

        plt.subplots_adjust(right=0.85)

        plt.savefig(filePath)

    def load_weights(self, filename):
        self.model.load_weights(filename)
        print("weights loaded")

    def train(self, X_train, y_train, validation_data, batch_size, epochs, callbacks=None, data_augmentation=True,
              initial_epoch=0):
        if data_augmentation:
            if self.sequence:
                generator = SequenceGenerator()
            else:
                generator = StaticGenerator()

            self.model.fit_generator(generator.flow(X_train, y_train, batch_size),
                                     steps_per_epoch=X_train.shape[0] / batch_size,
                                     epochs=epochs,
                                     validation_data=validation_data,
                                     callbacks=callbacks, initial_epoch=initial_epoch)
        else:
            self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                           validation_data=validation_data, callbacks=callbacks, initial_epoch=initial_epoch)
