import tensorflow as tf
from typing import Tuple
from all_tnn.task_helper_functions import load_model_from_path

def load_models(epoch, hparams, saved_model_path, pre_relu, post_norm, n_classes_override=None,) :
    '''
    If noisy=True, add noise after each layer (and adapt activities model to return the noisy activities). Used to test
    robustness of network to noisy neural activities
    '''

    hparams['analysis_epoch'] = epoch
    model = load_model_from_path(saved_model_path, epoch, hparams=hparams, n_classes=n_classes_override,
                                 test_mode=True, print_summary=True)

    activities_model = get_activities_model_tnn(model, hparams, pre_relu, post_norm)

    return activities_model, model

def get_activities_model_tnn(model, hparams, pre_relu, post_norm):
    """Get a model that returns activities."""
    print('Using get_activities_model_tnn from setup_model_and_data.py')

    model_name = hparams['model_name']
    conv_control_net = 'conv_control' in model_name
    readout_layers = []

    if 'simclr' in model_name:
        tnn_core_model_layers = model.encoder.layers if 'finetune' not in model_name else model.layers[1].layers
    else:
        tnn_core_model_layers = model.layers

    output_next_activation, output_next_next_activation = False, False
    for layer in tnn_core_model_layers:
        readout_layers, output_next_activation, output_next_next_activation = process_layer(readout_layers, layer, output_next_activation, output_next_next_activation, 
                                                                                            post_norm, pre_relu, conv_control_net)

    activities_model = tf.keras.Model(
        inputs=tnn_core_model_layers[0].input if 'simclr' in model_name else model.input, 
        outputs=readout_layers, 
        name=f'{model_name}_activities'
    )

    for layer in activities_model.layers:
        layer.trainable = False

    return activities_model

def is_target_layer(layer, conv_control_net):
    return layer.__class__.__name__ == 'Hypercolumn2D' or (conv_control_net and layer.__class__.__name__ == 'Conv2D')

def process_layer(
                  readout_layers: list,
                  layer: tf.keras.layers.Layer, 
                  output_next_activation: bool, 
                  output_next_next_activation: bool,
                  post_norm: bool, pre_relu: bool, conv_control_net: bool,
                  ) -> Tuple[bool, bool]:
    """
    Process a layer to determine if its output should be included in the readout layers.

    Args:
        readout_layers: The list of layers to readout.
        layer: The current layer being processed.
        output_next_activation: A flag indicating if the next activation layer's output should be included.
        output_next_next_activation: A flag indicating if the next next activation layer's output should be included.

    Returns:
        A tuple containing updated flags (output_next_activation, output_next_next_activation).
    """
    if not post_norm:
        if pre_relu:
            if is_target_layer(layer, conv_control_net):
                readout_layers.append(layer.output)
                print(f'\tAdding {layer.name} to returned activities')
        else:
            if is_target_layer(layer, conv_control_net):
                return readout_layers, True, False
            if output_next_activation and layer.__class__.__name__ == 'Activation':
                readout_layers.append(layer.output)
                print(f'\tAdding {layer.name} to returned activities')
                return readout_layers, False, False
    else:
        if is_target_layer(layer, conv_control_net):
            return readout_layers, False, True
        if output_next_next_activation and layer.__class__.__name__ == 'Activation':
            return readout_layers, True, False
        if output_next_activation and 'Normalization' in layer.__class__.__name__:
            readout_layers.append(layer.output)
            print(f'\tAdding {layer.name} after normalization to returned activities')
            return readout_layers, False, False
    return readout_layers, output_next_activation, output_next_next_activation



class DatasetLabelRenamer:
    def __init__(self, model, default_label_key='dense_2'):
        """
        Initializes the LabelRenamer with the model and default label key.

        Args:
            model (tf.keras.Model): The Keras model whose last layer name is used.
            default_label_key (str): The key in y_dict to extract the label tensor.
        """
        self.model = model
        self.default_label_key = default_label_key
        self.new_label_key = self.model.layers[-1]._name

    def __call__(self, x, y_dict, sample_weight):
        """
        Renames the label keys in the dataset.

        Args:
            x: Input features.
            y_dict (dict): Original label dictionary.
            sample_weight: Sample weights.

        Returns:
            Tuple containing input features, updated label dictionary, and sample weights.
        """
        label_tensor = y_dict[self.default_label_key]
        new_y_dict = {self.new_label_key: label_tensor}
        return x, new_y_dict, sample_weight