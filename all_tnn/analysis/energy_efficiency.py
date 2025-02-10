import os, pickle, csv
from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from scipy.ndimage import convolve
from numpy.random import default_rng
from scipy.spatial.distance import pdist, squareform
from all_tnn.analysis.util.analysis_help_funcs import get_activations_on_dataset
from all_tnn.analysis.visualization.layer_visualization import lower_tri, remove_spines, save, wrapped_distance
from all_tnn.models.model_helper.tnn_helper_functions import channels_to_sheet


def get_feature_responses(activities_model, data_generators_dict,dataset_names, hparams=None, layer=-1, pre_relu=False, post_norm = False, plot=True):
    """Loading the activites model to get the features
        weight shape : shape=(n_rows * c_dim[0],
                                n_cols * c_dim[1],
                                in_channels * kernel_size[0] * kernel_size[1]))
        Feature shape will be : (n_rows * c_dim[0],
                                n_cols * c_dim[1],
                                in_channels * kernel_size[0] * kernel_size[1]))
        1. compute features with activiti models --> but adapt to all vs all
        2. compute_weights_cosine_distance (see tnn_helper) --> but adapt to all vs all 
        # --> which means, in array with shape (rows, cols, channels), every postion in row and col will be compared with each other
        # e.g. (0,0, channells) compare to (0,1, channels), (1,0, channels), (1,1, channels) and so on to get correlation similarity
        # --> Go through all rows and cols, create a dict mapping cortical distance to correlation: {distance: correlation}
    """
    
    activities,  labels = {}, {}
    mean_activities, var_activities = {}, {}
    n_samples = {}
    n_layers = len(data_generators_dict['dummy_activities_batch'])

    for this_dataset_name in dataset_names:
        print(f'Gathering activities for {this_dataset_name}')
        activities[this_dataset_name], labels[this_dataset_name] = \
            get_activations_on_dataset(activities_model, data_generators_dict[this_dataset_name], this_dataset_name,
                                       one_hot=False, hparams=hparams)

        mean_activities[this_dataset_name] = [np.mean(a, axis=0) for a in activities[this_dataset_name]]
        var_activities[this_dataset_name] = [np.var(a, axis=0) for a in activities[this_dataset_name]]
        n_samples[this_dataset_name] = labels[this_dataset_name].shape[0]

    # mean and var of activities excluding one dataset
    # for no_leave_cat_out in dataset_names: if c != leave_cat_out:
    concat_acts = [[] for _ in range(n_layers)]
    for layer in range(n_layers):
        for c in dataset_names:
            concat_acts[layer].append(activities[c][layer])

    out_activities = [np.concatenate(concat_acts[l]) for l in range(n_layers)]
    mean_activities[f'np_out'] = [np.mean(a, axis=0) for a in out_activities]
    var_activities[f'no_out'] = [np.var(a, axis=0) for a in out_activities]
    n_samples[f'no_out'] = out_activities[0].shape[0] #! out_activities[0] is the activities of the first layer

    feature_responses = {
        'out_activities': out_activities,
        'mean_activities': mean_activities,
        'var_activities': var_activities,
        'n_samples': n_samples}
    
    # save the feature responses
    os.makedirs(os.path.join(hparams['analysis_dir'],'feature_responses'), exist_ok = True)
    for layer_id in range(1, len(out_activities)+1):
        np.save(os.path.join(hparams['analysis_dir'],'feature_responses', f'out_activities_{layer_id}.npy'), out_activities[layer_id-1])
    for act_type_name in mean_activities.keys():
        for layer_id in range(1, len(mean_activities[act_type_name])+1):
            np.save(os.path.join(hparams['analysis_dir'],'feature_responses', f'mean_activities_{act_type_name}_{layer_id}.npy'), mean_activities[act_type_name][layer_id-1])
            
    for var_act_type_name in var_activities.keys():
        for layer_id in range(1, len(var_activities[var_act_type_name])+1):
            np.save(os.path.join(hparams['analysis_dir'],'feature_responses', f'var_activities_{var_act_type_name}_{layer_id}.npy'), var_activities[var_act_type_name][layer_id-1])
    # save n_samples
    with open(os.path.join(hparams['analysis_dir'],'feature_responses', 'n_samples.pkl'), 'wb') as f:
        pickle.dump(n_samples, f)

    del activities, labels
    return feature_responses

def load_feature_responses(hparams):
    # Initialize the dictionary to collect feature responses (load above saved feature responses)
    feature_responses = {
        'out_activities': [],
        'mean_activities': {},
        'var_activities': {},
        'n_samples': None
    }

    # Directory where feature responses are stored
    feature_dir = os.path.join(hparams['analysis_dir'], 'feature_responses')

    # Load out_activities
    layer_id = 1
    while os.path.exists(os.path.join(feature_dir, f'out_activities_{layer_id}.npy')):
        feature_responses['out_activities'].append(
            np.load(os.path.join(feature_dir, f'out_activities_{layer_id}.npy'))
        )
        layer_id += 1

    # Load mean_activities and var_activities
    activity_types = [('mean_activities', feature_responses['mean_activities']),
                      ('var_activities', feature_responses['var_activities'])]

    for act_type_name, act_dict in activity_types:
        layer_id = 1
        # Assuming there's at least one file for each type to determine keys
        while True:
            try:
                with np.load(os.path.join(feature_dir, f'{act_type_name}_type1_{layer_id}.npy')) as data:
                    for key in data.files:
                        if key not in act_dict:
                            act_dict[key] = []
                        act_dict[key].append(data[key])
                layer_id += 1
            except FileNotFoundError:
                break

    # Load n_samples
    with open(os.path.join(feature_dir, 'n_samples.pkl'), 'rb') as f:
        feature_responses['n_samples'] = pickle.load(f)


    return feature_responses


def stack_energy_maps(n_energy_consumption_map_across_layers, interp_order=1):
    """
    Stack multiple energy consumption maps into one by interpolating to the size of the largest map and summing them.

    Args:
    n_energy_consumption_map_across_layers (list of numpy.ndarray): List of 2D arrays with different sizes.
    interp_order (int): Order of the spline interpolation (1=linear, 3=cubic, etc.)

    Returns:
    numpy.ndarray: A single stacked energy map.
    """
    from scipy.ndimage import zoom

    # Determine the target size (the size of the largest layer)
    target_size = max((layer.shape for layer in n_energy_consumption_map_across_layers), key=lambda x: x[0] * x[1])

    # Function to interpolate layers
    def interpolate_layer(layer, target_size):
        factor = [n / o for n, o in zip(target_size, layer.shape)]
        return zoom(layer, factor, order=interp_order)  # Adjust interpolation order here

    # Interpolate all layers to the common resolution
    interpolated_layers = [interpolate_layer(layer, target_size) for layer in n_energy_consumption_map_across_layers]

    # Stack the layers by summing them up
    stacked_energy_map = np.sum(np.array(interpolated_layers), axis=0)

    return stacked_energy_map



def analyze_activation_transmission_efficiency(
    model_layers,
    input_datasets,
    epoch,
    feature_responses,
    model_name,
    hparams,
    multi_models_neural_dict,
    PRE_RELU,
    NORM_PREV_LAYER=True,
    NORM_LAYER_OUT=True,
    grating_w_entropies=None
):
    """
    Iterate through layers (Hypercolumn2D or Conv2D in a control net) 
    to compute and analyze activation transmission efficiency (L1-based energy).

    Args:
        model_layers (list): List of model layers.
        input_datasets (tf.data.Dataset or np.array-like): Input data in batches or array form.
        epoch (int): Current epoch number for directory saving.
        feature_responses (dict): Layer-wise activations collected from the model.
        model_name (str): Model identifier/name.
        hparams (dict): Hyperparameters and related config.
        multi_models_neural_dict (dict): Dictionary aggregating results across models.
        PRE_RELU (bool): Whether analysis is done pre-ReLU or post-ReLU layers.
        NORM_PREV_LAYER (bool): If True, normalizes previous layer's outputs.
        NORM_LAYER_OUT (bool): If True, normalizes current layer outputs.
        grating_w_entropies (np.array): Optional, used if entropy-based analysis is also required.

    Returns:
        None. Results are saved to disk and appended to multi_models_neural_dict.
    """
    analysis_dir = hparams['analysis_dir']
    conv_control_net = 'conv_control' in hparams['model_name']

    # Identify relevant layers
    if conv_control_net:
        hypercolumn_layers = [ly for ly in model_layers if ly.__class__.__name__ == 'Conv2D']
        using_conv_readout = hparams.get('conv_readout', False)
        if using_conv_readout:
            hypercolumn_layers = hypercolumn_layers[:-1]
    else:
        hypercolumn_layers = [ly for ly in model_layers if ly.__class__.__name__ == 'Hypercolumn2D']

    norm_layers = [ly for ly in model_layers if 'Normalization' in ly.__class__.__name__]

    # Prepare structures to store results
    total_energy_consumption_map_across_layers = []
    mean_energy_consumption_map_across_layers = []
    sum_total_energy_consumption_across_layers = []
    average_total_energy_consumption_across_layers = []

    # Process each layer
    with tqdm(total=len(hypercolumn_layers) + 1) as pbar:
        for i, layer in enumerate(hypercolumn_layers):
            # Confirm the matching normalization layer if it exists
            assert layer.name.split('_')[-1] == norm_layers[i].name.split('_')[2], \
                "Layer and Norm layer mismatch."

            # Calculate energy consumption
            total_energy_consumption_per_unit = calculate_layer_energy_consumption(
                feature_responses=feature_responses,
                input_datasets=input_datasets,
                hparams=hparams,
                layer_i=i,
                layer=layer,
                norm_layers=norm_layers,
                pbar=pbar,
                epoch=epoch,
                NORM_LAYER_OUT=NORM_LAYER_OUT,
                NORM_PREV_LAYER=NORM_PREV_LAYER
            )


            # Total across stimuli
            layer_total_map = channels_to_sheet(
                np.expand_dims(total_energy_consumption_per_unit, axis=0), 
                return_np=True
            )
            layer_total_map = np.squeeze(layer_total_map)
            total_energy_consumption_map_across_layers.append(layer_total_map)
            sum_total_energy_consumption_across_layers.append(np.sum(total_energy_consumption_per_unit))
            average_total_energy_consumption_across_layers.append(
                np.mean(total_energy_consumption_per_unit)
            )

            pbar.update(1)

    # Make directories for analysis outputs
    analysis_dir_path = os.path.join(
        hparams['analysis_dir'],
        'energy_efficiency',
        f'ep{epoch}'
    )
    os.makedirs(analysis_dir_path, exist_ok=True)

    # Stack multi-layer maps (except the very last if single receptive field)
    stacked_total_energy_map = stack_energy_maps(total_energy_consumption_map_across_layers[:-1])

    # print(f'mean of stacked_mean_energy_map: {np.mean(stacked_mean_energy_map)}')
    print(f'mean of stacked_total_energy_map: {np.mean(stacked_total_energy_map)}')

    # Save stacked maps
    energy_map_folder = os.path.join(analysis_dir_path, 'energy_maps')
    os.makedirs(energy_map_folder, exist_ok=True)

    total_map_fname = (
        f"ali_5layers_NORM_LAYER_OUT_{NORM_LAYER_OUT}_"
        f"NORM_PREV_LAYER_{NORM_PREV_LAYER}_{model_name}_alpha{hparams['alpha']}"
        f"_drop{hparams['dropout_rate']}_pre_relu_{PRE_RELU}_stacked_total_energy_map.npy"
    )
    # np.save(os.path.join(energy_map_folder, mean_map_fname), stacked_mean_energy_map)
    np.save(os.path.join(energy_map_folder, total_map_fname), stacked_total_energy_map)

    # Plot per-layer mean energy consumption maps + radial distribution
    layer_maps_dir = os.path.join(analysis_dir_path, 'energy_maps_and_trend_from_center')
    os.makedirs(layer_maps_dir, exist_ok=True)

    for layer_i, mean_energy_map in tqdm(
        enumerate(mean_energy_consumption_map_across_layers),
        desc='Plotting energy maps and radial energy distribution'
    ):
        fig, axs = plt.subplots(2, 1, figsize=(10, 12))
        im_m = axs[0].imshow(
            mean_energy_map, cmap='viridis', 
            vmin=0, vmax=0.3 * np.max(mean_energy_map)
        )
        axs[0].set_title(f'Energy Map for Layer {layer_i + 1}')
        axs[0].set_axis_off()

        # Save array
        map_save_name = (
            f"ali_way_energy_map_layer{layer_i + 1}_NORM_LAYER_OUT_{NORM_LAYER_OUT}_"
            f"NORM_PREV_LAYER_{NORM_PREV_LAYER}.npy"
        )
        np.save(os.path.join(analysis_dir_path, map_save_name), mean_energy_map)


def calculate_layer_energy_consumption(
    feature_responses,
    input_datasets,
    hparams,
    layer_i,
    layer,
    norm_layers,
    pbar,
    epoch,
    NORM_LAYER_OUT=True,
    NORM_PREV_LAYER=True
):
    """
    Calculate the presynaptic and postsynaptic energy for a given layer. 
    Weights are set to absolute values for the computation of L1-based energy.

    Args:
        feature_responses (dict): Contains 'out_activities' with layer-wise activations.
        input_datasets (tf.data.Dataset or np.array-like): Batches or data array.
        hparams (dict): Hyperparameters including batch_size, analysis_dir, etc.
        layer_i (int): Index of the current layer.
        layer (tf.keras.layers.Layer): Current layer in the model.
        norm_layers (list): List of normalization layers in the model.
        pbar (tqdm.tqdm): TQDM progress bar for logging.
        epoch (int): Current epoch for directory naming.
        NORM_LAYER_OUT (bool): Whether to apply normalization to the current layer output.
        NORM_PREV_LAYER (bool): Whether to apply normalization to the previous layer output.

    Returns:
        tuple:
          (overall_sum_energy_consumption_per_unit, 
           overall_mean_energy_consumption_per_unit)
    """
    pbar.set_description(f"Preparing data for layer {layer_i}")

    # Identify current and previous norm layers
    cur_norm_layer = norm_layers[layer_i]
    prev_norm_layer = norm_layers[layer_i - 1] if layer_i > 0 else None
    assert cur_norm_layer.name.split('_')[2] == layer.name.split('_')[-1], \
        "Norm layer name does not match layer name."
    if layer_i > 0:
        assert int(prev_norm_layer.name.split('_')[2]) == int(layer.name.split('_')[-1]) - 1, \
            "Previous norm layer does not match."

    # Extract the layer weights and bias
    layer_weights = layer.get_weights()
    if len(layer_weights) == 2:
        kernel_weights, bias_weights = layer_weights
    else:
        kernel_weights = layer_weights
        bias_weights = None

    # Convert weights to absolute values
    kernel_weights_abs = np.abs(kernel_weights)
    if bias_weights is not None:
        bias_weights_abs = np.zeros(bias_weights.shape)
    else:
        bias_weights_abs = None

    layer.set_weights([kernel_weights_abs, bias_weights_abs])

    # Check no negative values
    abs_w, abs_b = layer.get_weights()
    assert tf.reduce_min(abs_w) >= 0, "Negative values in absolute kernel weights."
    assert tf.reduce_min(abs_b) >= 0, "Negative values in absolute bias weights."

    presynaptic_energy_batches = []
    postsynaptic_energy_batches = []

    # Batch iteration
    batch_size = hparams['batch_size']
    batch_id = 0

    # If input_datasets is a tf.data.Dataset, we iterate differently vs. an array
    # The code remains the same from original, as we preserve logic.
    for inputs_and_labels in input_datasets:
        if isinstance(inputs_and_labels, tuple):
            inputs_batch, *_ = inputs_and_labels
        else:
            inputs_batch = inputs_and_labels

        layer_activations = feature_responses['out_activities'][layer_i][
            batch_id * batch_size:(batch_id + 1) * batch_size
        ]

        # Postsynaptic energy (layer output)
        if NORM_LAYER_OUT:
            layer_activations_abs = np.abs(cur_norm_layer(layer_activations))
        else:
            layer_activations_abs = np.abs(layer_activations)
        postsynaptic_energy = layer_activations_abs

        # Previous layer's activations
        if layer_i > 0:
            prev_activations = feature_responses['out_activities'][layer_i - 1][
                batch_id * batch_size:(batch_id + 1) * batch_size
            ]
        else:
            prev_activations = inputs_batch

        # Normalize previous layer if required
        if NORM_PREV_LAYER and layer_i > 0:
            prev_activations = prev_norm_layer(prev_activations)

        # The original code forcibly uses a MaxPool2D after certain layers:
        if (layer_i - 1) in [0, 2, 4]:
            print(f'Adding pooling after layer {layer_i + 1} (printed from tnn.py)')
            prev_activations = tf.keras.layers.MaxPool2D()(prev_activations)

        prev_activations_abs = np.abs(prev_activations)
        batch_id += 1

        # Presynaptic energy: we rely on the layer with absolute weights
        presynaptic_energy = layer(prev_activations_abs)
        assert tf.reduce_min(presynaptic_energy) >= 0, "Negative energy values found."

        presynaptic_energy_batches.append(presynaptic_energy)
        postsynaptic_energy_batches.append(postsynaptic_energy)

    presynaptic_energy = np.concatenate(presynaptic_energy_batches, axis=0)
    postsynaptic_energy = np.concatenate(postsynaptic_energy_batches, axis=0)

    # Summaries
    mean_post = np.mean(postsynaptic_energy, axis=0)
    mean_pre = np.mean(presynaptic_energy, axis=0)
    sum_post = np.sum(postsynaptic_energy, axis=0)
    sum_pre = np.sum(presynaptic_energy, axis=0)

    # Save to disk
    save_dir = os.path.join(hparams['analysis_dir'], 'energy_efficiency', f'ep{epoch}')
    os.makedirs(save_dir, exist_ok=True)

   
    overall_sum_energy = (1/3) * sum_post + (2/3) * sum_pre
    np.save(
        os.path.join(
            save_dir,
            f"ali_way_overall_sum_energy_consumption_layer{layer_i+1}_"
            f"NORM_LAYER_OUT_{NORM_LAYER_OUT}_NORM_PREV_LAYER_{NORM_PREV_LAYER}.npy"
        ),
        overall_sum_energy
    )

    return overall_sum_energy 


def analyze_energy_efficiency(
    feature_responses,
    model_name,
    hparams,
    all_l1_norm_data,
    multi_models_neural_dict,
    PRE_RELU,
    POST_NORM,
    grating_w_entropies=None,
    epoch=None
):
    """
    Compute/plot L1 norm of activations for each layer as a measure of energy 
    efficiency, optionally correlating with an entropy map.

    Args:
        feature_responses (dict): Contains 'out_activities' for each layer.
        model_name (str): Name/identifier of the model.
        hparams (dict): Hyperparameter and path configs.
        all_l1_norm_data (list): Aggregator list for L1 norms from multiple models.
        multi_models_neural_dict (dict): Summary dictionary storing results for multiple models.
        PRE_RELU (bool): Whether analysis is on pre- or post-ReLU outputs.
        POST_NORM (bool): If model outputs are normalized post-layers.
        grating_w_entropies (np.array): If provided, used for comparison with layer-based energy maps.
        epoch (int): Current epoch number for naming.

    Returns:
        None. Plots and CSV outputs are saved, and dictionaries updated in-place.
    """
    activities = feature_responses['out_activities']

    l1_norms_average_stimuli_then_units = []
    l1_norms_average_units_then_stimuli = []
    l1_norms_indivduals = []
    l1_norms_total = []

    energy_maps = []
    window_sized_energy_maps = []

    # L1 norms for each layer
    for layer_activations in activities:
        layer_activations = np.abs(layer_activations)
        flatten_act = layer_activations.reshape(layer_activations.shape[0], -1)

        sum_by_sample = np.sum(np.abs(flatten_act), axis=1)
        l1_norms_average_units_then_stimuli.append(np.mean(sum_by_sample))

        sum_by_unit = np.sum(np.abs(flatten_act), axis=0)
        l1_norms_average_stimuli_then_units.append(np.mean(sum_by_unit))

        mean_layer_act = np.mean(layer_activations, axis=0)  # average across batch
        l1_norms_total.append(np.sum(np.abs(mean_layer_act)))

        indiv = np.sum(np.abs(mean_layer_act)) / (
            mean_layer_act.shape[0] * mean_layer_act.shape[1] * mean_layer_act.shape[2]
        )
        l1_norms_indivduals.append(indiv)

        e_map = np.squeeze(
            channels_to_sheet(np.expand_dims(mean_layer_act, 0), return_np=True)
        )
        # 8x8 average filter
        kernel = np.ones((8, 8)) / 64.0
        window_map = convolve(e_map, kernel, mode='wrap')

        energy_maps.append(np.abs(e_map))
        window_sized_energy_maps.append(window_map)



## ----------------- Utils Functions----------------- ##
# Function to create a circular mask
def create_circular_mask(h_w, center=None, radius=None):
    if center is None:  # use the middle of the image
        center = (int(h_w[0]/2), int(h_w[1]/2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], h_w[0]-center[0], h_w[1]-center[1])

    Y, X = np.ogrid[:h_w[0], :h_w[1]]
    dist_from_center = np.sqrt((X - center[1])**2 + (Y-center[0])**2)

    mask = dist_from_center <= radius
    return mask

def plot_radial_energy_distribution(layer_i, energy_map, save_dir):
    center_point = (energy_map.shape[0] // 2, energy_map.shape[1] // 2)
    max_radius = min(center_point)
    radial_energy = []

    for radius in range(max_radius):
        mask = create_circular_mask(energy_map.shape, center=center_point, radius=radius)
        masked_energy = np.ma.masked_array(energy_map, ~mask)
        average_energy = np.ma.mean(masked_energy)
        radial_energy.append(average_energy)

    plt.plot(range(max_radius), radial_energy, c='blue')
    plt.title(f'Energy Drop from Center to Border for Layer {layer_i+1}')
    plt.xlabel('Distance from Center')
    plt.ylabel('Average Energy')
    plt.grid(True)
    file_base = f"ali_way_energy_drop_layer{layer_i+1}"
    os.makedirs(os.path.join(save_dir, 'radial_energy_distribution'), exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'radial_energy_distribution', f"{file_base}.pdf"), dpi=450)
    plt.savefig(os.path.join(save_dir, 'radial_energy_distribution',  f"{file_base}.png"), dpi=450)
    plt.close()

def plot_energy_map(layer_i, energy_map, save_dir):
    plt.figure(figsize=(10, 6))
    plt.imshow(energy_map, cmap='viridis')
    plt.colorbar()
    plt.title(f'Energy Map for Layer {layer_i+1}')
    plt.xticks([])
    plt.yticks([])
    file_base = f"ali_way_energy_map_layer{layer_i+1}"
    os.makedirs(os.path.join(save_dir, 'energy_maps'), exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'energy_maps', f"{file_base}.png"), dpi=450)
    plt.savefig(os.path.join(save_dir, 'energy_maps', f"{file_base}.pdf"), dpi=450)
    plt.close()

def calculate_radial_energy(energy_map, center_point, max_radius):
    radial_energy = []
    for radius in range(max_radius):
        mask = create_circular_mask(energy_map.shape, center=center_point, radius=radius)
        masked_energy = np.ma.masked_array(energy_map, ~mask)
        average_energy = np.ma.mean(masked_energy)
        radial_energy.append(average_energy)
    return radial_energy

def create_circular_mask(shape, center, radius):
    Y, X = np.ogrid[:shape[0], :shape[1]]
    dist_from_center = np.sqrt((X - center[1])**2 + (Y - center[0])**2)
    mask = dist_from_center <= radius
    return mask
