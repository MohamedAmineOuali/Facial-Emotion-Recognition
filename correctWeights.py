import h5py
import sys

from models.simple_cnn_model import SimpleCnnModel

filepath = "82,84.hdf5"


model=SimpleCnnModel(False,sequence=True).model
layer = model
symbolic_weights = []
for l in layer.layers:
    if len(l.weights) > 0:
        for el in l.weights:
            symbolic_weights.append(el)
names = [el.name for el in symbolic_weights]

# Get the weights from the file in the same order set_weights wants them
with  h5py.File(filepath, "r")as f:
    weights = []
    if 'layer_names' not in f.attrs and 'model_weights' in f:
        f = f['model_weights']

    for name in names:
        if( name.split("/")[0] not in f.keys()):
            g=f['CNN_Model']
        else:
            g=f[name.split("/")[0]]

        weights.append(g[name].value)

    layer.set_weights(weights)


model.save_weights("corrected"+filepath)
