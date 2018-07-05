from data.load_data import *
from models.simple_cnn_model import SimpleCnnModel
from models.vgg_models import Vgg16Model

model=Vgg16Model(False)

model.model.summary()

model.load_weights("weights/vgg_based/static/wild_validation/87-64.hdf5")


X_test,y_test=load_val_data('dataset/train_val',sequence=False)


scores=model.evaluate(X_test,y_test,batch_size=1)
print("%s: %.2f%%" % (model.model.metrics_names[1], scores[1]*100))

print("%s: %.2f%%" % (model.model.metrics_names[0], scores[0]*100))