
import numpy as np
import tensorflow as tf
import os, h5py, pickle, importlib, math, glob
from tensorflow.keras import backend as k
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import math_ops
from shutil import copyfile
from all_tnn.models.model_helper.model_helper_functions import save_full_model


##################################################################################################
## ----------------------------------- Model Setup -------------------------------------------- ##
##################################################################################################

def get_model(hparams, n_classes, strategy=None, saved_model_path=None):
    """
    Get model from model function.

    Args:
        hparams (dict): Hyperparameters for the model.
        n_classes (int): Number of classes for the dataset.
        strategy (tf.distribute.Strategy, optional): Distribution strategy for parallelization.
        saved_model_path (str, optional): Path to the saved model for loading.

    Returns:
        tf.keras.Model: Compiled Keras model.
    """
    # Determine input shape
    input_shape = [hparams['batch_size'], hparams['image_size'], hparams['image_size'], 3]

    # Adjust batch_size for distribution strategy
    if strategy is not None:
        input_shape[0] = input_shape[0] // strategy.num_replicas_in_sync

    # Load model function from the saved model path if provided
    if saved_model_path is not None:
        file_path = f"{saved_model_path}/_code_used_for_training/models"
        module_name = 'setup_model'
        setup_model = localdir_modulespec(module_name, file_path)
    else:
        from models import setup_model

    # Get the model function
    model_function = setup_model.get_model_function(hparams['model_name'])

    # Handle finetuning case
    if 'finetune' in hparams['model_name'].lower():
        base_model_name = hparams['model_name'].replace('finetuned_', '')
        base_model_function = setup_model.get_model_function(base_model_name)
        simclr_encoder_model = base_model_function(input_shape, n_classes, hparams)
        net = model_function(simclr_encoder_model, input_shape, n_classes, hparams)
    else:
        net = model_function(input_shape, n_classes, hparams)

    return net


def get_losses_and_metrics(network_output_layers, spatial_loss_values, hparams):
    """
    Gets loss, metric, and loss_weights dict with optional RDL objective.
    """

    if 'simclr' in hparams['model_name'].lower() and 'finetune' not in hparams['model_name'].lower() and not hparams['finetune_flag']:
        import keras_cv
        print('Using a self-supervised simCLR model.')
        from models.model_helper.simclr_helper_functions import SimCLRLossMetric, SimCLRSpatialLossMetric, CombinedSimCLRSpatialLoss

        if hparams['simclr_temperature'] == 'none':
            raise ValueError('You are using a simCLR model but have not specified a temperature in hparams["simclr_temperature"].')

        loss_dict, metric_dict = {}, {}
        for layer in network_output_layers:
            loss_dict[layer] = keras_cv.losses.SimCLRLoss(temperature=hparams['simclr_temperature'])
            metric_dict[layer] = [SimCLRLossMetric(), SimCLRSpatialLossMetric(spatial_loss_values, hparams['alpha'], hparams)]
            if 'tnn' in hparams['model_name']:
                from models.model_helper.tnn_helper_functions import SpatialLossMetric
                metric_dict[layer].append(SpatialLossMetric(spatial_loss_values, hparams['alpha'], hparams))

        return loss_dict, metric_dict

    else:
        loss_dict, metric_dict = {}, {}

        for layer in network_output_layers:

            loss_dict[layer] = tf.keras.losses.CategoricalCrossentropy()
            metric_dict[layer] = [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.TopKCategoricalAccuracy()]

            if 'tnn' in hparams['model_name']:
                from models.model_helper.tnn_helper_functions import SpatialLossMetric
                metric_dict[layer].append(SpatialLossMetric(spatial_loss_values, hparams['alpha'], hparams))

        return loss_dict, metric_dict


def compile_model(net, hparams, loss_dict, metric_dict):
    """
    Specifies the optimizer, metrics, and loss function for training the model.

    Args:
        net (tf.keras.Model): The model to compile.
        hparams (dict): Hyperparameters for the model and optimizer.
        loss_dict (dict): Dictionary of loss functions for each output layer.
        metric_dict (dict): Dictionary of metrics for each output layer.

    Returns:
        tf.keras.Model: The compiled model.
    """
    optimizer = get_optimizer(hparams)

    if 'simclr' in hparams['model_name'].lower() and 'finetune' not in hparams['model_name'].lower() and not hparams['finetune_flag']:
        net.compile(encoder_loss=loss_dict, encoder_metrics=metric_dict, encoder_optimizer=optimizer)
        for projector in net.projectors:
            projector.build(net.encoder.output_shape)  # Build the projector shape
    else:
        net.compile(loss=loss_dict, metrics=metric_dict, optimizer=optimizer)

    return net


def get_optimizer(hparams):
    """
    Returns a TensorFlow optimizer based on hyperparameters.

    Args:
        hparams (dict): A dictionary containing hyperparameters for the optimizer.

    Returns:
        tf.keras.optimizers.Optimizer: An instance of a TensorFlow optimizer.

    Raises:
        Exception: If the optimizer specified in hparams is not implemented.
    """
    optimizer_name = hparams['optimizer'].lower()
    optimizer_args = {'learning_rate': hparams['learning_rate']}

    if 'clip_norm' in hparams and hparams['clip_norm'] is not None:
        optimizer_args['clipnorm'] = hparams['clip_norm']

    if optimizer_name == 'adam':
        optimizer_args.update({
            'epsilon': hparams['optim_epsilon'],
            'beta_1': hparams.get('adam_beta_1', 0.9),
            'beta_2': hparams.get('adam_beta_2', 0.999),
        })
        optimizer = tf.keras.optimizers.Adam(**optimizer_args)
    elif optimizer_name == 'sgd':
        optimizer = tf.keras.optimizers.SGD(**optimizer_args)
    else:
        raise Exception(f'Optimizer {hparams["optimizer"]} not implemented')

    return optimizer


def get_activities_model(model, model_name):
    """
    Get a model that returns activities.

    Args:
        model (tf.keras.Model): The original model.
        model_name (str): Name of the model.

    Returns:
        tf.keras.Model: A model that outputs the activities of specified layers.
    """
    print('Using get_activities_model from task_helper_functions.py')

    # Collect the readout layers
    readout_layers = [
        layer.output for layer in model.layers if any(f'sheet_{i}' in layer.name.lower() for i in range(len(model.layers)))
    ]

    activities_model = tf.keras.Model(inputs=model.input, outputs=readout_layers, name=f'{model_name}_activities')

    # If finetuning, then only load the encoder weights, layers[1] is the encoder
    if 'simclr' in model_name.lower() and 'finetune' in model_name.lower():
        activities_model = activities_model.layers[1]

    for layer in activities_model.layers:
        layer.trainable = False

    return activities_model

def get_model_file_from_name(model_name):
    """
    Get the model file name from the model name.

    Args:
        model_name (str): The model name.

    Returns:
        str: The corresponding model file name.
    """
    model_name_mapping = {
        'tnn_conv_control': 'tnn',
        'tnn_simclr': 'tnn_simclr',
        'tnn_conv_control_encoder_simclr': 'tnn_simclr',
        'finetuned_tnn_simclr': 'tnn_simclr',
        'simclr_blt_vNet': 'blt_vNet_simCLR'
    }

    for key in model_name_mapping:
        if key in model_name:
            return model_name_mapping[key]

    return model_name.replace("_half_channels", "")

def load_model_from_path(saved_model_path, epoch_to_load, n_classes=None, test_mode=False, print_summary=False, hparams=None):
    """
    Load a model from a specified path.

    Args:
        saved_model_path (str): Path to the saved model directory.
        epoch_to_load (int): Epoch of the weights to load.
        n_classes (int, optional): Number of classes for the model.
        test_mode (bool, optional): Whether to set the model to test mode.
        print_summary (bool, optional): Whether to print the model summary.
        hparams (dict, optional): Hyperparameters for the model.

    Returns:
        tf.keras.Model: The loaded and compiled model.
    """
    if hparams is None:
        with open(f'{saved_model_path}/hparams.pickle', 'rb') as f:
            hparams = pickle.load(f)

    try:
        n_classes = get_n_classes(hparams) if get_n_classes(hparams) else n_classes
    except:
        n_classes = hparams.get('num_classes', get_n_classes(hparams) if get_n_classes(hparams) else 0)

    net = get_model(hparams, n_classes, saved_model_path=saved_model_path, strategy=None)

    print(f'\nLoading weights for epoch {epoch_to_load}...')
    weights_filename = 'model_weights_init.h5' if epoch_to_load == 0 else f'ckpt_ep{epoch_to_load:03d}.h5'

    if 'simclr' in hparams['model_name'] and not hparams.get('finetune_flag', False):
        if 'finetune' not in hparams['model_name']:
            net.encoder.load_weights(os.path.join(saved_model_path, 'training_checkpoints', weights_filename))
        else:
            load_finetune_weights(net, saved_model_path, epoch_to_load, weights_filename, hparams)
    else:
        load_weights_with_fallback(net, saved_model_path, weights_filename, hparams)

    if test_mode:
        set_layers_trainable(net, False)
        print(f"test_mode={hparams['test_mode']}, setting trainable=False for all layers")
    else:
        print(f"test_mode={hparams['test_mode']}, all layers are trainable")

    compile_model_based_on_hparams(net, hparams, print_summary)
    
    return net

def load_finetune_weights(net, saved_model_path, epoch_to_load, weights_filename, hparams):
    if epoch_to_load > 0:
        net.load_weights(os.path.join(saved_model_path, 'training_checkpoints', weights_filename))
    else:
        net.layers[1].load_weights(os.path.join(saved_model_path, 'training_checkpoints', weights_filename))
        for i, projector in enumerate(net.projectors):
            projector.load_weights(os.path.join(saved_model_path, 'training_checkpoints', weights_filename[:-3] + f"_projector_{i}.h5"))

def load_weights_with_fallback(net, saved_model_path, weights_filename, hparams):
    try:
        net.load_weights(os.path.join(saved_model_path, 'training_checkpoints', weights_filename))
    except:
        replace_readout_layer(net, hparams)
        net.load_weights(os.path.join(saved_model_path, 'training_checkpoints', weights_filename))

def replace_readout_layer(net, hparams):
    x = net.layers[-3].output
    new_readout = tf.keras.layers.Dense(hparams['num_classes_new_readout'], activation='softmax', name='output_time_0')(x)
    if 'simclr' in hparams['model_name'].lower():
        net = tf.keras.Sequential([
            net.layers[1].input,
            net.layers[1],
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(hparams['num_classes_new_readout'], activation='softmax', name='output_time_0')
        ], name="finetuned_simclr_model")
    else:
        net = tf.keras.Model(inputs=net.input, outputs=new_readout)

def set_layers_trainable(net, trainable):
    for layer in net.layers:
        layer.trainable = trainable

def compile_model_based_on_hparams(net, hparams, print_summary):
    if 'simclr' in hparams['model_name'] and 'finetune' not in hparams['model_name']:
        if print_summary:
            net.layers[1].summary()
        local_dir = os.path.dirname(os.path.realpath(__file__))
        simclr_helper_functions = localdir_modulespec('simclr_helper_functions', local_dir + '/models/model_helper')
        net.compile(
            encoder_loss=simclr_helper_functions.CombinedSimCLRSpatialLoss(net.encoder, hparams),
            encoder_optimizer=tf.keras.optimizers.Adam()
        )
    else:
        if print_summary:
            net.summary()
        metric_dict = {layer: [tf.keras.metrics.categorical_accuracy, tf.keras.metrics.top_k_categorical_accuracy] for layer in net.output_names}
        net.compile(metrics=metric_dict)

def localdir_modulespec(module_name, dir_path):
    """
    Import functions from remote folders.

    Args:
        module_name (str): Name of the module.
        dir_path (str): Directory path of the module.

    Returns:
        module: The imported module.
    """
    file_path = os.path.join(dir_path, f'{module_name}.py')
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def get_n_classes(hparams=None, dataset_path=None, dataset_subset=None):
    """
    Get the number of classes from the dataset.

    Args:
        hparams (dict, optional): Hyperparameters dictionary.
        dataset_path (str, optional): Path to the dataset.

    Returns:
        int: Number of classes.
    """
    if hparams and 'simclr' in hparams['model_name'].lower() and 'finetune' not in hparams['model_name'].lower():
        return 0

    if not hparams and not dataset_path:
        raise ValueError('hparams or dataset_path must be provided.')

    num_classes = hparams.get('num_classes', 565) if hparams.get('num_classes', 565) else 565
    return num_classes


##################################################################################################
## ----------------------------- Saving Models code and paras  -------------------------------- ##
##################################################################################################

def make_saving_name(hparams):
    """
    Create a saving name for the model.

    Args:
        hparams (dict): Hyperparameters dictionary.

    Returns:
        str: The saving name.
    """
    return f"{hparams['model_name']}{hparams['model_name_suffix']}"

# Additional utility function
def save_full_model(epoch, hparams, net, saved_model_path):
    """
    Save the full model in SavedModel format.

    Args:
        epoch (int): Current epoch.
        hparams (dict): Hyperparameters for the model.
        net (tf.keras.Model): The model to be saved.
        saved_model_path (str): Path to save the model.
    """
    full_model_save_dir = os.path.join(saved_model_path, f'full_model_epoch_{epoch:03d}')
    os.makedirs(full_model_save_dir, exist_ok=True)
    net.save(full_model_save_dir, save_format='tf')
    print(f'Saved full model at {full_model_save_dir}')
    
def save_code_and_params(hparams, model_savedir):
    """
    Save hyperparameters and the code used for training.

    Args:
        hparams (dict): Hyperparameters dictionary.
        model_savedir (str): Directory to save the model and code.
    """
    # Save hyperparameters
    with open(f'{model_savedir}/hparams.txt', 'w') as f:
        for k, v in hparams.items():
            f.write(f'{k}: {v}\n')

    with open(f'{model_savedir}/hparams.pickle', 'wb') as f:
        pickle.dump(hparams, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save the code used for training
    code_saving_path_root = f'{model_savedir}/_code_used_for_training'
    os.makedirs(code_saving_path_root, exist_ok=True)

    files_to_copy = [
        ("task.py", "task.py"),
        ("task_helper_functions.py", "task_helper_functions.py"),
        ("run_n_epochs.py", "run_n_epochs.py")
    ]

    for src, dst in files_to_copy:
        copyfile(src, f'{code_saving_path_root}/{dst}')

    # Save model-related code
    code_saving_path_models = f'{code_saving_path_root}/models'
    os.makedirs(code_saving_path_models, exist_ok=True)
    os.makedirs(f'{code_saving_path_models}/layers', exist_ok=True)
    os.makedirs(f'{code_saving_path_models}/model_helper', exist_ok=True)

    model_file = get_model_file_from_name(hparams['model_name'])
    model_files_to_copy = [
        (f"./models/{model_file}.py", f"{model_file}.py"),
        ("./models/setup_model.py", "setup_model.py"),
        ("./models/model_helper/model_helper_functions.py", "model_helper/model_helper_functions.py"),
        ("./models/model_helper/simclr_helper_functions.py", "model_helper/simclr_helper_functions.py")
    ]

    if 'tnn' in hparams['model_name'].lower():
        model_files_to_copy.extend([
            ("./models/layers/local.py", "layers/local.py"),
            ("./models/model_helper/tnn_helper_functions.py", "model_helper/tnn_helper_functions.py")
        ])

    for src, dst in model_files_to_copy:
        copyfile(src, f'{code_saving_path_models}/{dst}')

    # Save dataset loader code
    code_saving_path_dataset = f'{code_saving_path_root}/dataset_loader'
    os.makedirs(code_saving_path_dataset, exist_ok=True)

    dataset_files_to_copy = [
        ("./dataset_loader/make_tf_dataset.py", "make_tf_dataset.py"),
        ("./dataset_loader/tf_dataset_helper_functions.py", "tf_dataset_helper_functions.py")
    ]

    for src, dst in dataset_files_to_copy:
        copyfile(src, f'{code_saving_path_dataset}/{dst}')

##################################################################################################
## ----------------------------------- Checkpoints -------------------------------------------- ##
##################################################################################################
def custom_load_checkpoint(hparams, net, ckpt_path, saved_model_path):
    """
    Custom checkpoint loading logic.

    Args:
        hparams (dict): Hyperparameters dictionary.
        net (tf.keras.Model): The model.
        ckpt_path (str): Path to the checkpoint directory.
        saved_model_path (str): Path to the saved model directory.
    """
    init_new_model = False
    if hparams['start_epoch'] == 0:
        init_new_model = True
    elif hparams['start_epoch'] == -1:
        init_new_model = load_latest_checkpoint(net, ckpt_path, hparams)
    else:
        load_specific_checkpoint(net, ckpt_path, hparams)

    if init_new_model:
        save_initial_checkpoint(net, ckpt_path, saved_model_path, hparams)

def load_latest_checkpoint(net, ckpt_path, hparams):
    if 'simclr' in hparams['model_name']:
        raise NotImplementedError('Loading the latest checkpoint of a simCLR model is not implemented yet.')
    try:
        ckpt_list = glob.glob(f'{ckpt_path}/*')
        weights_to_load = max(ckpt_list, key=os.path.getctime)
        net.load_weights(weights_to_load)
        print(f'Loading weight from {weights_to_load}')
        hparams['start_epoch'] = int(weights_to_load[-6:-3])
        hparams['n_epochs'] -= (hparams['start_epoch'] + 1)
    except:
        print(f'No weights found in {ckpt_path}. Starting a new initialized network instead.')
        return True
    return False


def load_specific_checkpoint(net, ckpt_path, hparams):
    if 'simclr' in hparams['model_name'] and 'finetune' not in hparams['model_name']:
        encoder_ckpt_path = os.path.join(ckpt_path, f'ckpt_ep{hparams["start_epoch"]:03d}.h5')
        print(f'Loading simclr weight from {encoder_ckpt_path}')
        net.encoder.load_weights(encoder_ckpt_path)
        for i, projector in enumerate(net.projectors):
            projector_ckpt_path = os.path.join(ckpt_path, f'projector_ckpt_ep{hparams["start_epoch"]:03d}_{i}.h5')
            projector.load_weights(projector_ckpt_path)
    else:
        weights_to_load = os.path.join(ckpt_path, f'ckpt_ep{hparams["start_epoch"]:03d}.h5')
        print(f'Loading pretrained weight from {weights_to_load}')
        net.load_weights(weights_to_load)

def save_initial_checkpoint(net, ckpt_path, saved_model_path, hparams):
    init_ckpt_path = os.path.join(ckpt_path, 'model_weights_init.h5')
    init_model_path = os.path.join(saved_model_path, f"{hparams['model_name']}_init")
    print(f'Saving initial checkpoint to {init_ckpt_path} and init full model to {init_model_path}')

    if 'simclr' in hparams['model_name']:
        if 'finetune' not in hparams['model_name'] and not hparams['finetune_flag']:
            net.encoder.save_weights(init_ckpt_path[:-3] + "_encoder.h5")
            for i, projector in enumerate(net.projectors):
                projector.save_weights(init_ckpt_path[:-3] + f"_projector_{i}.h5")
        else:
            save_finetune_initial_checkpoint(net, init_ckpt_path, saved_model_path, hparams)
    else:
        net.save_weights(init_ckpt_path)

    try:
        save_full_model(0, hparams, net, saved_model_path)
    except:
        print('Cannot save full model. This is probably because it contains custom layers, which need a bit of work to be saved in this format. The checkpoint format should still work.')

def save_finetune_initial_checkpoint(net, init_ckpt_path, saved_model_path, hparams):
    epoch_to_load = hparams.get('finetune_start_epoch', hparams.get('previous_simclr_encoder_trained_epoch_num', 0))
    if epoch_to_load > 0:
        try:
            net.save_weights(init_ckpt_path)
        except:
            net.layers[1].save_weights(init_ckpt_path)

        previous_simclr_encoder_trained_save_dir = hparams.get('previous_simclr_encoder_trained_save_dir', hparams.get('finetune_start_weight_path'))
        weights_filename = f'ckpt_ep{epoch_to_load:03d}.h5'

        print(f'Loading pretrained encoder weights from {previous_simclr_encoder_trained_save_dir}')
        try:
            net.layers[1].load_weights(os.path.join(previous_simclr_encoder_trained_save_dir, 'training_checkpoints', weights_filename))
            net.layers[1].trainable = hparams.get('trainable_encoder_in_finetune', False)
        except:
            net.layers[1].load_weights(os.path.join(saved_model_path, 'training_checkpoints', weights_filename))
            net.layers[1].trainable = hparams.get('trainable_encoder_in_finetune', False)
    else:
        raise Exception('You are trying to finetune the SimCLR model, but you should define the pretrained encoder path')


##################################################################################################
## ----------------------------------- Callbacks ---------------------------------------------- ##
##################################################################################################

class AlphaScheduler(tf.keras.callbacks.Callback):
    """
    Scheduler for the alpha parameter with double-edge sigmoid schedule.
    """
    def __init__(self, n_batches, hparams):
        print(f'Creating alpha parameter schedule in {hparams["alpha_schedule"]} mode')
        self.i_epoch = hparams['start_epoch']
        self.alpha_scheduler_start_epoch = hparams.get('alpha_scheduler_start_epoch', 0)
        total_epochs = hparams['total_epochs'] if hparams['total_epochs'] > 0 else hparams['n_epochs']
        assert total_epochs > hparams['n_warmup_epochs']

        self.n_batches = n_batches
        x = np.linspace(0, 1, n_batches * (total_epochs - self.alpha_scheduler_start_epoch))

        self.schedule = self.create_schedule(x, hparams, n_batches, total_epochs)
        self.schedule = tf.reshape(self.schedule, (total_epochs, n_batches))
        if self.alpha_scheduler_start_epoch > 0 and ('decreasing' in hparams['alpha_schedule'] or 'increasing' in hparams['alpha_schedule']):
            alpha_value = 0 if hparams['alpha_scheduler_mode'].startswith('increasing') else 1
            pre_schedule = np.full((self.alpha_scheduler_start_epoch, n_batches), alpha_value)
            self.schedule = np.concatenate([pre_schedule, self.schedule])

    def create_schedule(self, x, hparams, n_batches, total_epochs):
        if hparams['alpha_schedule'] == 'linear':
            return tf.linspace(0., 1., n_batches * total_epochs)
        elif hparams['alpha_schedule'] == 'cosine':
            return 0.5 * (1 + np.sin(np.pi * x))
        elif hparams['alpha_schedule'] == 'sigmoid':
            return 1 / (1 + np.exp(-hparams['sigmoid_steepness'] * (x - hparams['sigmoid_position'])))
        elif hparams['alpha_schedule'] == 'decreasing_sigmoid':
            return 1 - 1 / (1 + np.exp(-hparams['sigmoid_steepness'] * (x - hparams['sigmoid_position'])))
        elif hparams['alpha_schedule'] == 'symmetric_sigmoids':
            midpoint = n_batches * total_epochs // 2
            first_half = 1 / (1 + np.exp(-hparams['sigmoid_steepness'] * (x[:midpoint] * 2 - hparams['sigmoid_position'])))
            second_half = first_half[::-1]
            return np.concatenate([first_half, second_half])
        elif hparams['alpha_schedule'] == 'gaussian':
            return np.exp(-((x - 0.5) ** 2) / (2 * hparams['gaussian_variance'] ** 2))
        else:
            raise ValueError('hparams["alpha_schedule"] not understood.')

    def on_batch_begin(self, batch, logs=None):
        current_alpha = self.schedule[self.i_epoch, min(self.n_batches - 1, batch)]
        tf.keras.backend.set_value(self.model.alpha_multiplier, current_alpha)
        tf.summary.scalar('alpha_scheduled', data=current_alpha, step=batch)

    def on_epoch_end(self, epoch, logs=None):
        self.i_epoch += 1

class BatchLearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, n_batches, hparams):
        print(f'Creating learning rate schedule in {hparams["learning_rate_schedule"]} mode')
        self.i_epoch = hparams['start_epoch']
        self.alpha_scheduler_start_epoch = hparams.get('alpha_scheduler_start_epoch', 0)

        total_epochs = hparams['total_epochs'] if hparams['total_epochs'] > 0 else hparams['n_epochs']
        assert total_epochs > hparams['n_warmup_epochs']

        self.n_batches = n_batches
        self.schedule = self.create_schedule(n_batches, total_epochs, hparams)

    def create_schedule(self, n_batches, total_epochs, hparams):
        total_steps = np.arange(n_batches * (total_epochs - hparams['n_warmup_epochs']), dtype=np.float32) / (n_batches * total_epochs)
        warmup_schedule = tf.linspace(hparams['learning_rate'] / 10, hparams['learning_rate'], n_batches * hparams['n_warmup_epochs'])
        
        if hparams['learning_rate_schedule'] == 'cosine':
            cosine_decayed = 0.5 * (1.0 + np.cos(np.pi * total_steps))
            return tf.concat([warmup_schedule, cosine_decayed], axis=0) if hparams['n_warmup_epochs'] > 0 else cosine_decayed
        elif hparams['learning_rate_schedule'] == 'cosine_restarts':
            n_cycles = 10
            cosine_restarts = 0.5 * (1.0 + np.cos(n_cycles * np.pi * (np.sqrt(total_steps) % (1 / n_cycles))))
            return tf.concat([warmup_schedule, cosine_restarts], axis=0) if hparams['n_warmup_epochs'] > 0 else cosine_restarts
        elif hparams['learning_rate_schedule'] == 'none':
            return tf.concat([warmup_schedule, tf.ones_like(total_steps)], axis=0) if hparams['n_warmup_epochs'] > 0 else tf.ones_like(total_steps)
        elif hparams['learning_rate_schedule'] == 'fixed_after_warmup':
            stable_schedule = tf.ones_like(total_steps) * hparams['learning_rate']
            return tf.concat([warmup_schedule, stable_schedule], axis=0) if hparams['n_warmup_epochs'] > 0 else stable_schedule
        else:
            raise ValueError('Unknown learning rate schedule.')

    def on_batch_begin(self, batch, logs=None):
        current_lr = self.schedule[self.i_epoch - self.alpha_scheduler_start_epoch, min(self.n_batches - 1, batch)]
        tf.keras.backend.set_value(self.model.optimizer.lr, current_lr)    
        tf.summary.scalar('lr', data=current_lr, step=batch)

    def on_epoch_end(self, epoch, logs=None):
        self.i_epoch += 1

class TensorBoardFix(tf.keras.callbacks.TensorBoard):
    """
    This fixes incorrect step values when using the TensorBoard callback.
    """
    def on_train_begin(self, *args, **kwargs):
        super().on_train_begin(*args, **kwargs)
        tf.summary.experimental.set_step(self._train_step)

    def on_test_begin(self, *args, **kwargs):
        super().on_test_begin(*args, **kwargs)
        tf.summary.experimental.set_step(self._val_step)

class SimCLRModelCheckpoint(tf.keras.callbacks.Callback):
    """
    Custom callback to save SimCLR model checkpoints.

    Args:
        encoder_save_dir (str): Directory to save encoder weights.
        save_freq (int, optional): Frequency of saving checkpoints.
        projectors_save_dir (str, optional): Directory to save projector weights.
    """
    def __init__(self, encoder_save_dir, save_freq=10, projectors_save_dir=None):
        super().__init__()
        self.encoder_save_dir = encoder_save_dir
        self.projectors_save_dir = projectors_save_dir
        self.save_freq = save_freq
        self.epoch_count = 0

        os.makedirs(self.encoder_save_dir, exist_ok=True)
        if self.projectors_save_dir is not None:
            os.makedirs(self.projectors_save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_count += 1
        if self.epoch_count % self.save_freq == 0:
            encoder_path = os.path.join(self.encoder_save_dir, f"ckpt_ep{self.epoch_count:03d}.h5")
            self.model.encoder.save_weights(encoder_path)

            if self.projectors_save_dir is not None:
                projector_path_0 = os.path.join(self.projectors_save_dir, f"projector_ckpt_ep{self.epoch_count:03d}_0.h5")
                projector_path_1 = os.path.join(self.projectors_save_dir, f"projector_ckpt_ep{self.epoch_count:03d}_1.h5")
                self.model.projectors[0].save_weights(projector_path_0)
                self.model.projectors[1].save_weights(projector_path_1)




##################################################################################################
## --------------------------- Data Processing for None h5 dataset ---------------------------- ##
# The functions below are used for testing on images that are not stored in .h5 format.
# During training and in most testing cases, images are in our standard .h5 format, and
# processing is done automatically. But for the orientation selectivity analysis, we
# create grating stimuli that are not in this standard .h5 formal. These functions can
# be used to make sure these non-h5 images are preprocessed correctly.
##################################################################################################

def preprocess_batch(batch, hparams):
    """
    Preprocess a batch of images.

    Args:
        batch (list): A batch of images.
        hparams (dict): Hyperparameters for preprocessing.

    Returns:
        tf.Tensor: Preprocessed batch of images.
    """
    for i, img in enumerate(batch):
        batch[i] = tf_preprocess_image(img, hparams=hparams)
    return tf.cast(batch, tf.float32)  # Ensure batch is of type float32 to avoid TensorFlow warnings

def tf_preprocess_image(image, hparams):
    """
    Preprocess a single image.

    Args:
        image (tf.Tensor): The input image.
        hparams (dict): Hyperparameters for preprocessing.

    Returns:
        tf.Tensor: Preprocessed image.
    """
    image = tf.cast(image, tf.float32)
    image = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1/255.)(image)  # Scale image to [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    return tf_normalize(image, hparams['image_normalization'])

def tf_normalize(image, img_normalization):
    """
    Normalize an image.

    Args:
        image (tf.Tensor): The input image.
        img_normalization (str or None): Normalization method.

    Returns:
        tf.Tensor: Normalized image.

    Raises:
        ValueError: If an unknown normalization method is provided.
    """
    if img_normalization == 'z_scoring':
        image = tf.image.per_image_standardization(image)
    elif img_normalization == '[-1,1]':
        image = tf.keras.layers.experimental.preprocessing.Rescaling(scale=2, offset=-1)(image)  # Assumes input is [0, 1]
    elif img_normalization is None:
        pass
    else:
        raise ValueError("Please use '[-1,1]' or 'z_scoring' for the normalization argument")
    return image
