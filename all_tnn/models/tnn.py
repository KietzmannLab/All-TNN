import os
import sys 
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/../')
from task_helper_functions import localdir_modulespec

local_dir = os.path.dirname(os.path.realpath(__file__))
local_funcs = localdir_modulespec('local', local_dir+'/layers')
help_funcs = localdir_modulespec('model_helper_functions', local_dir+'/model_helper')
tnn_help_funcs = localdir_modulespec('tnn_helper_functions', local_dir+'/model_helper')
tnn_spatial_loss_helper = localdir_modulespec('tnn_helper_functions', local_dir+'/model_helper')


def tnn(input_tensor, n_classes, hparams, conv_control_net):
    """
    Creates TNN or control network with standard conv layers if conv_control_net is True.
    """
    x = input_tensor
    shared_args = {
        'use_bias': hparams['use_bias'],
        'kernel_initializer': hparams['kernel_initializer']
    }

    if hparams['layer_regularizer'].lower() == 'l2':
        shared_args['kernel_regularizer'] = tf.keras.regularizers.L2(hparams['regularize'])
    elif hparams['layer_regularizer'].lower() == 'l1':
        shared_args['kernel_regularizer'] = tf.keras.regularizers.L1(hparams['regularize'])
    else:
        raise ValueError(f"Invalid layer_regularizer: {hparams['layer_regularizer']}")

    layer_args = [
        {'filters': 64, 'kernel_size': 7, 'strides': 3, **shared_args},
        {'filters': 81, 'kernel_size': 3, 'strides': 1, **shared_args},
        {'filters': 81, 'kernel_size': 3, 'strides': 1, **shared_args},
        {'filters': 256, 'kernel_size': 3, 'strides': 1, **shared_args},
        {'filters': 256, 'kernel_size': 3, 'strides': 1, **shared_args},
        {'filters': 2500, 'kernel_size': 3, 'strides': 1, **shared_args}
    ]

    if hparams['layer_alpha_factors'] == [-1]:
        hparams['layer_alpha_factors'] = [1.0] * len(layer_args)

    loss_layers = []
    for l, l_args in enumerate(layer_args):
        x = local_funcs.parse_local_layers(hparams, l, l_args, loss_layers, x, conv_control_net)
        if l in {0, 2, 4}:
            print(f'Adding pooling after layer {l} (printed from tnn.py)')
            x = tf.keras.layers.MaxPool2D()(x)
        if hparams.get("learnable_dropout_mask", False):
            steep_sigmoid_layer = tnn_spatial_loss_helper.SteepSigmoidMultiplierLayer(
                input_shape=x.shape[1:], 
                l1_regularization=hparams["dropout_l1_regularizer_value"], 
                name=f'ssml_{l}'
            )
            x = steep_sigmoid_layer(x)

    if hparams.get("conv_readout", False):
        outputs = help_funcs.parse_conv_readout(n_classes, hparams, x)
    else:
        outputs = help_funcs.parse_readout(n_classes, hparams, x)

    model = tnn_spatial_loss_helper.SpatialLossModel(
        inputs=input_tensor, 
        outputs=outputs, 
        hparams=hparams, 
        name='tnn' if not conv_control_net else 'tnn_conv_control'
    )

    if hparams.get('spatial_loss', False) and not hparams.get('test_mode', False):
        model.configure_regularizer(loss_layers)

    model.summary()
    return model