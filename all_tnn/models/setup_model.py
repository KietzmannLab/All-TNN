'''Selection of models to be used in training
'''
import os
import tensorflow as tf
from tensorflow.keras.layers import Input
from task_helper_functions import localdir_modulespec
from all_tnn.models import tnn as tnn_module

# All-TNNs
def tnn_model_fn(input_shape, n_classes, hparams):
    input_layer = Input(batch_shape=input_shape)
    model = tnn_module.tnn(input_layer, n_classes, hparams, conv_control_net=False)
    return model

# CNNs
def tnn_conv_control_model_fn(input_shape, n_classes, hparams):
    input_layer = Input(batch_shape=input_shape)
    model = tnn_module.tnn(input_layer, n_classes, hparams, conv_control_net=True)
    return model

def get_model_function(model_name):
    if model_name == 'tnn':
        return tnn_model_fn
    elif model_name == 'tnn_conv_control':
        return tnn_conv_control_model_fn
    else:
        raise ValueError('Model not available: {}'.format(model_name))
