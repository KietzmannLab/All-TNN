import os
import tensorflow as tf
from tensorflow.python.keras import backend as k
from tensorflow.keras import layers
from all_tnn.models.model_helper import model_helper_functions


class Hypercolumn2D(tf.keras.layers.LocallyConnected2D):
    """This layer is almost identical to its superclass, LocallyConnected2D. Only the call and get_weights methods are
    extended in case this is enabled in the hyperparameters."""

    def __init__(self, **kwargs):
        super(Hypercolumn2D, self).__init__(**kwargs)
        self.goodness_of_gradient = None # tf.ones_like(self.output_size)

    def call(self, inputs):
        # Forward pass
        output = super(Hypercolumn2D, self).call(inputs)

        return output

    def get_weights(self):
        weights = super(Hypercolumn2D, self).get_weights()

        if len(weights) == 2:
            kernels, biases = weights
        elif len(weights) == 1:
            kernels = weights
            biases = None

        return kernels, biases


def parse_local_layers(hparams, l, l_args, loss_layers, x, conv_control_net=False):

    if conv_control_net:
        layer = layers.Conv2D(name=f'conv_{l}', **l_args)
        local_layer = False
    else:
        layer = Hypercolumn2D(name=f'sheet_{l}', **l_args)
        local_layer = True

    x = layer(x)
    if local_layer:
        loss_layers.append(layer)  # do not add spatial loss for conv layers

    x = model_helper_functions.parse_normalization_and_activation(x, l, 0, hparams)

    if hparams['dropout_rate'] > 0:
        x = layers.Dropout(rate=hparams['dropout_rate'])(x)
        
    return x
