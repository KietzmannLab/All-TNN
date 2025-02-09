from __future__ import division, absolute_import, print_function
import os
import tensorflow as tf
from tensorflow.keras import layers


def parse_normalization_and_activation(x, l, t, hparams):

    if hparams['norm_first']:
        x = parse_normalization(x, l, t, hparams)
        x = parse_activation(x, l, t, hparams['activation'])
    else:
        x = parse_activation(x, l, t, hparams['activation'])
        x = parse_normalization(x, l, t, hparams)

    return x


def parse_activation(x, l, t, activation):
    return tf.keras.layers.Activation(activation, name=f'{activation}_Layer_{l}_Time_{t}')(x)


def parse_normalization(x, l, t, hparams):

    norm_type = hparams['norm_type']
    axis = hparams['norm_axes'] # axis = [1, 2, 3] if norm_type == 'LN' else -1
    suffix = f'Layer_{l}_Time_{t}'

    if norm_type == 'LN':
        return tf.keras.layers.LayerNormalization(axis=axis, name=f'LayerNorm_{suffix}',
                                                  scale=hparams['scale_norm'], center=hparams['center_norm'])(x)
    elif norm_type == 'BN':
        return tf.keras.layers.BatchNormalization(axis, name=f'BatchNorm_{suffix}',
                                                  scale=hparams['scale_norm'], center=hparams['center_norm'])(x)
    elif norm_type == 'GN':
        return tf.keras.layers.GroupNormalization(groups=32, epsilon=1e-6, axis=axis, name=f'GroupNorm_{suffix}',
                                                  scale=hparams['scale_norm'], center=hparams['center_norm'])(x)
    elif norm_type == 'no_norm':
        return x
    else:
        raise Exception(f'norm_type not recognized, entered: {norm_type}. please use "BN", "IN", "LN", "DN", "GN", or "no_norm"')


def network_output(x, hparams, t=0):

    if hparams['model_output_activation'] == 'no_model_output_activation':
        print('WARNING: You did are not using an output activation on this network. This is good if you are '
              'training on embeddings, but otherwise just make sure this is what you want.')
        # The linear activation is an identity function. So this will simply cast to float32 (needed for
        # numerical stability when using tf's mixed_precision policy).
        activation_str = 'linear'
    else:
        activation_str = hparams['model_output_activation'] # Default is softmax for classification task
        
    return tf.keras.layers.Activation(activation_str, name=f'output_time_{t}', dtype='float32')(x)


def save_full_model(final_epoch, hparams, net, saved_model_path):
    print(f'Saving full model at {saved_model_path}.')
    try:
        net.save(os.path.join(saved_model_path, f"{hparams['model_name']}_ep{final_epoch:03d}"),
                 overwrite=True)
    except:
        print('Cannot save full model. This is probably because it contains custom layers, which need a bit of work '
              'to be saved in this format. The checkpoint format should still work.')


def load_full_model(saved_model_path, hparams, final_epoch):
    model = tf.keras.models.load_model(
        os.path.join(saved_model_path, f"{hparams['model_name']}_ep{final_epoch:03d}"),
        compile=False
    )

    return model


def parse_readout(n_classes, hparams, x, t=0):

    x = layers.Flatten()(x)

    # Add dropout if enabled
    if hparams['dropout_rate'] > 0:
        if hparams['test_mode']:
            print('We are not in training mode, disabling dropout')
        x = layers.Dropout(hparams['dropout_rate'])(x, training=hparams['test_mode'])

    x = layers.Dense(n_classes, name='ReadoutDense')(x)

    outputs = network_output(x, hparams, t)

    return outputs


def parse_conv_readout(n_classes, hparams, x, t=0):
    # Ensure the input tensor x is properly shaped. If not (1, 1), use GlobalAveragePooling2D to aggregate spatial info.
    if x.shape[1] != 1 or x.shape[2] != 1:
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Reshape x to have a shape of (batch_size, 1, 1, channels) if not already.
    # This step is crucial if the input x is not already in this shape.
    x = tf.keras.layers.Reshape((1, 1, x.shape[-1]))(x)

    # Add dropout if enabled and in non-test mode
    if hparams['dropout_rate'] > 0 and not hparams['test_mode']:
        x = tf.keras.layers.Dropout(hparams['dropout_rate'])(x, training=True)

    # Apply a Conv2D layer with kernel size (1, 1) for classification.
    # This acts as a dense layer applied to each feature map individually.
    x = tf.keras.layers.Conv2D(filters=n_classes, kernel_size=(1, 1),
                               use_bias=True, activation=None, name='ReadoutConv')(x)

    # Flatten the output to match the expected output shape for classification
    x = tf.keras.layers.Flatten()(x)

    # Optionally, apply an activation function like softmax if it's a multi-class classification problem.
    # The activation can be applied outside this function using the `network_output` or directly here.
    outputs = tf.keras.layers.Activation('softmax', dtype='float32', name=f'output_time_{t}')(x)

    return outputs
