'''Selection of models to be used in training
'''
import os
import tensorflow as tf
from tensorflow.keras.layers import Input
from task_helper_functions import localdir_modulespec

local_dir = os.path.dirname(os.path.realpath(__file__))

# All-TNNs
def tnn_model_fn(input_shape, n_classes, hparams):
    tnn_module = localdir_modulespec('tnn', local_dir)
    input_layer = Input(batch_shape=input_shape)
    model = tnn_module.tnn(input_layer, n_classes, hparams, conv_control_net=False)
    return model

# CNNs
def tnn_conv_control_model_fn(input_shape, n_classes, hparams):
    tnn_module = localdir_modulespec('tnn', local_dir)
    input_layer = Input(batch_shape=input_shape)
    model = tnn_module.tnn(input_layer, n_classes, hparams, conv_control_net=True)
    return model

# All-TNNs + SimCLR
def tnn_simclr_model_fn(input_shape, n_classes, hparams):
    input_layer = Input(batch_shape=input_shape)
    tnn_simclr_module = localdir_modulespec('tnn_simclr', local_dir)
    simclr_helper_module = localdir_modulespec('model_helper/simclr_helper_functions', local_dir)
    contrastive_augmenter = simclr_helper_module.get_augmenter(**simclr_helper_module.CONTRASTIVE_AUGMENTATION, hparams=hparams)  
    simclr_model = tnn_simclr_module.tnn_simclr(contrastive_augmenter, input_layer, hparams)
    return simclr_model

def get_model_function(model_name):
    if model_name == 'tnn':
        return tnn_model_fn
    elif model_name == 'tnn_conv_control':
        return tnn_conv_control_model_fn
    elif model_name == 'tnn_simclr':
        return tnn_simclr_model_fn
    else:
        raise ValueError('Model not available: {}'.format(model_name))
