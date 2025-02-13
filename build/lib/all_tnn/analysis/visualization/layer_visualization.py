import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb
from matplotlib.colors import ListedColormap
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from typing import List, Tuple, Union, Any
import tensorflow as tf
from tensorflow.keras import backend 

from all_tnn.models.model_helper.tnn_helper_functions import compute_weights_cosine_distance, channels_to_sheet, extract_weight_dims
from all_tnn.analysis.config import *
from all_tnn.analysis.util.analysis_help_funcs import set_gridlines, mask_array_by_value, format_weights

Iterable = Union[List, Tuple, np.ndarray]

def visualize_layer(output_dict, epoch, layer_i, analysis_dir, model_name, layer=None, save=True, show=False):
    """
    Main visualisation function (inclueds Category Selectivity, Orientation Selectivity, Distance vs Weight Similarity)
    Collect relevant data for the specified layer(s) and make all plots
    """
    data_dict = dict()
    print(CATEGORY_STATS, output_dict.keys())
    if CATEGORY_STATS:
        if 'category_selectivities' in output_dict:
            print(f'Getting category selectivity maps for layer {layer_i}')
            mean_maps = {k: output_dict['category_selectivities']['selectivity']['mean_activities'][k][layer_i] for
                         k in output_dict['category_selectivities']['selectivity']['mean_activities'].keys()}
            variance_maps = {k: output_dict['category_selectivities']['selectivity']['var_activities'][k][layer_i] for
                             k in output_dict['category_selectivities']['selectivity']['var_activities'].keys()}

            # Calculate dprime from mean&variance selectivity maps
            dprime_maps = {}
            for k in mean_maps.keys():
                if not 'out' in k:
                    in_means = mean_maps[k]
                    out_means = mean_maps[f'out_{k}']
                    in_stds = np.sqrt(variance_maps[k])
                    out_stds = np.sqrt(variance_maps[f'out_{k}'])
                    dprime_maps[k] = (in_means-out_means)/((in_stds+out_stds)/2 + 1e-10)

            data_dict = plot_category_selectivity_maps(data_dict, dprime_maps, epoch, layer_i, analysis_dir, save=save, show=show)

    if ORIENTATION_SELECTIVITY:
        if 'grating_w_tuning_curves' in output_dict:
            print(f'Getting orientation selectivity and entropy maps for layer {layer_i}')
            for i_w, f in enumerate(WAVELENGTHS):

                hue = output_dict['grating_w_tuning_curves'][0][layer_i][i_w]
                s = output_dict['grating_w_tuning_curves'][1][layer_i][i_w]

                # These are the orientation selectivity maps
                data_dict[f'activations_w{WAVELENGTHS[i_w]}'] = {
                    'data': np.expand_dims(hue, axis=0),  # add batch dims, since [b,h,w,c] is expected
                    'shape': 'hsv',
                    'cm': plt.cm.hsv
                }

                # Entropy maps
                for i_w, entropies in enumerate(output_dict['grating_w_entropies'][layer_i]):
                    data_dict[f'entropies_w{i_w}'] = {
                        'data': np.expand_dims(entropies, axis=0),
                        'cm': plt.cm.gray
                    }
        # Iterate over plot types and plot
        for p_type, data in data_dict.items():
            plot_orientation_entropy_maps(p_type, data, epoch, layer_i, analysis_dir, save=save, show=show)

    if GET_SPATIAL_LOSS and layer is not None:
        if 'mean_cosdist_loss' not in data_dict.keys():
            print(f'Computing spatial loss for layer {layer_i}')
            # Get kernel weights and biases, and format them
            layer_weights = layer.get_weights()
            if len(layer_weights) == 2:
                kernel_weights, bias_weights = layer_weights
            else:
                bias_weights = None
            kernel_size = layer.kernel_size
            if 'conv_control' in model_name:
                # if we are analyzing a convolutional control network, we get CNN weights, which need different formatting
                n_row, n_col, in_channels, channel_dim = kernel_weights.shape
                out_channels = np.prod(channel_dim)
                # Get weights in different formats
                row_col_weights, row_col_weights_without_bias_norm, row_col_weights_with_bias_norm, kernel_weights, bias_weights = \
                    format_weights(kernel_weights, bias_weights, n_row, n_col, kernel_size, in_channels, out_channels, conv_control_net=True)
            else:
                # "standard" non-convolutional locally connected case
                n_row, n_col, in_channels, channel_dim = extract_weight_dims(kernel_weights, kernel_size)
                out_channels = np.prod(channel_dim)
                # Get weights in different formats
                row_col_weights, row_col_weights_without_bias_norm, row_col_weights_with_bias_norm = \
                    format_weights(kernel_weights, bias_weights, n_row, n_col, kernel_size, in_channels, out_channels)

        # Cosine distances between neighbouring units
        spatial_loss_path = os.makedirs(f'{analysis_dir}/spatial_loss_layer/ep{epoch:03}/{layer_i}', exist_ok=True)
        mean_cosdist = compute_weights_cosine_distance(kernel_weights, bias_weights, kernel_size, circular=True, save_path=spatial_loss_path)
        data_dict['mean_cosdist_loss'] = mean_cosdist.numpy()
    return data_dict

def plot_category_selectivity_maps(data_dict, dprime_maps, epoch, layer_i, analysis_dir, save=True, show=False):
    # plot with showing the max dprime for each neuron in one different color per category
    fig, ax = plt.subplots()
    dprime_cmaps = [make_linear_cm_extend(217, 95, 2), make_linear_cm_extend(27, 158, 119), make_linear_cm_extend(117, 112, 179)]
    dprime_cmaps_dict = {k: i for k, i in zip(dprime_maps.keys(), dprime_cmaps)}
    dprime_sheets = {k: channels_to_sheet(v, return_np=True) for k, v in dprime_maps.items()}

    for k in dprime_sheets.keys():
        for k2 in dprime_sheets.keys():
            if k2 != k:
                for i in range(dprime_sheets[k].shape[0]):
                    for j in range(dprime_sheets[k].shape[1]):
                        # inefficient double for loop 
                        if dprime_sheets[k][i,j] < dprime_sheets[k2][i,j]:
                            dprime_sheets[k][i,j] = 0
        masked_dprime_sheet = mask_array_by_value(dprime_sheets[k], 0)
        data_dict[f'masked_dprime_sheet_layer{layer_i}_{k}'] = {}
        data_dict[f'masked_dprime_sheet_layer{layer_i}_{k}']['data'] = masked_dprime_sheet
        data_dict[f'masked_dprime_sheet_layer{layer_i}_{k}']['shape'] = 'skip_plotting'
        this_plt = ax.imshow(masked_dprime_sheet, interpolation='nearest', cmap=dprime_cmaps_dict[k])
        this_cb = plt.colorbar(this_plt, shrink=0.25)
        this_cb.set_label(k)
    ax.set_title(f'Epoch {epoch}')
    
    if show:
        plt.show()
    if save: 
        os.makedirs(f'{analysis_dir}/category_selectivity/', exist_ok=True)
        plt.savefig(f'{analysis_dir}/category_selectivity/dprime_combined_layer{layer_i}_ep{epoch}.png',dpi=300)

        # pickle d' sheet  
        print(f'Saving dprime sheets for layer {layer_i}')
        with open(f'{analysis_dir}/category_selectivity/dprime_sheets_layer{layer_i}.pkl', 'wb') as f:                
            pickle.dump(dprime_sheets, f)
    
    plt.close('all')
    return data_dict

def plot_orientation_entropy_maps(plot_type, data, epoch, layer_i, analysis_dir, mask_value=0, save=True, show=False):
    data_grid = data['data']

    # Plot HSV value per neuron
    if data.get('shape') == 'hsv':
        data_grid = channels_to_sheet(data_grid, return_np=True)
        data_grid = np.stack([data_grid, np.ones(data_grid.shape), np.ones(data_grid.shape)], axis=-1)
        data_grid = hsv_to_rgb(data_grid)

    elif data.get('shape') == 'skip_plotting':
        return

    # Plot 1 value per neuron
    else:
        data_grid = channels_to_sheet(data_grid, return_np=True)

    # Plot
    plt.figure()
    plt.imshow(mask_array_by_value(data_grid, mask_value), norm=data.get('norm'), cmap=data.get('cm', plt.cm.viridis))
    cb = plt.colorbar()
    ax = plt.gca()

    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    if save:
        try:
            dir = os.path.join(analysis_dir, 'orientation_selectivity',  f'ep{epoch:03}', f"{layer_i}",)
        except:
            import pdb; pdb.set_trace()
        os.makedirs(dir, exist_ok=True)
        plt.savefig(fname=os.path.join(dir, f"layer{layer_i}_{plot_type}.png",),
                    bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()

    plt.close('all')


def visualize_model(model_layers, epoch, output_dict, hparams, get_spatial_loss=False):
    """Iterate through the (Hypercolumn) layers and visualize corresponding data"""
    if 'conv_control' in hparams['model_name']:
        hypercolumn_layers = [l for l in model_layers if l.__class__.__name__ == 'Conv2D']
        using_conv_readout = hparams.get('conv_readout', False)
        if using_conv_readout:
            hypercolumn_layers = hypercolumn_layers[:-1]
    else:
        hypercolumn_layers = [l for l in model_layers if l.__class__.__name__ == 'Hypercolumn2D']

    # For each model layer
    with tqdm(total=len(hypercolumn_layers) + 1) as pbar:
        spatial_losses = []
        for i, layer in enumerate(hypercolumn_layers):

            # Visualize the layer and save the data
            data_dict = visualize_layer(output_dict, epoch, i, hparams['analysis_dir'], hparams['model_name'], layer)
            if get_spatial_loss and GET_SPATIAL_LOSS:
                spatial_losses.append(data_dict['mean_cosdist_loss'])
            pbar.update(1)
    if get_spatial_loss:
        return spatial_losses
    else:
        return None


########################################################################################################################
## Utils
########################################################################################################################
def lower_tri(x, keep_diagonal: bool = False):
    """return lower triangle of x, excluding diagonal"""
    assert len(x.shape) == 2
    return x[np.tril_indices_from(x, k=-1 if not keep_diagonal else 0)]

def remove_spines(axes, to_remove=None):
    """
    Removes spines from pyplot axis: import from plot_utils

    Inputs
        axes (list or np.ndarray): list of pyplot axes
        to_remove (str list): can be any combo of "top", "right", "left", "bottom"
    """

    # if axes is a 2d array, flatten
    if isinstance(axes, np.ndarray):
        axes = axes.ravel()

    # enforce inputs as list
    axes = make_iterable(axes)

    if to_remove == "all":
        to_remove = ["top", "right", "left", "bottom"]
    elif to_remove is None:
        to_remove = ["top", "right"]

    for ax in axes:
        for spine in to_remove:
            ax.spines[spine].set_visible(False)

def make_iterable(x) -> Iterable:
    """
    If x is not already array-like, turn it into a list or np.array

    Inputs
        x: either array_like (in which case nothing happens) or non-iterable,
            in which case it gets wrapped by a list
    """

    if not isinstance(x, (list, tuple, np.ndarray)):
        return [x]
    return x

def cosine_similarity(vec1, vec2):
    # Normalize each vector to unit length
    vec1_norm = vec1 / np.linalg.norm(vec1)
    vec2_norm = vec2 / np.linalg.norm(vec2)

    # Compute the dot product between the normalized vectors
    sim = np.dot(vec1_norm, vec2_norm)
    return sim

def save(fig, name: str, ext: str = "pdf", dpi: int = 300, close_after: bool = False):
    '''from analysis.util import plot_utils'''
    import matplotlib
    from pathlib import Path

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    SAVE_DIR = Path("save_dir") 
    path = SAVE_DIR / f"{name}.{ext}"
    path.parent.mkdir(exist_ok=True, parents=True)
    # os.makedirs(path, exist_ok=True)

    fig.savefig(path, dpi=dpi, bbox_inches="tight", transparent=True)
    if close_after:
        plt.close(fig)            

## Plotting Distance vs Weight Similarity
def analyze_distance_vs_weight_similarity(layer_i, row_col_weights, hparams, PRE_RELU, layers_to_analyze=[5]):
    """
    Analyzes and visualizes the distance versus weight similarity for a given layer.

    Args:
        layer_i (int): The current layer index.
        row_col_weights (ndarray): Weights of the convolutional layers.
        hparams (dict): Hyperparameters and analysis directory information.
        PRE_RELU (bool): Indicates whether to consider pre ReLU activations.
    """

    # Check if the current layer should be analyzed
    if layer_i not in layers_to_analyze:
        print(f"Skip layer {layer_i} for distance_vs_weight_similarity analysis")
        return

    # Reshape the weights for similarity computation
    n_rows, n_cols, n_channels = row_col_weights.shape[:3]
    c_dim = (int(np.sqrt(n_channels)), int(np.sqrt(n_channels)))
    kernel_size = row_col_weights.shape[3:5]
    in_channels = row_col_weights.shape[-1]
    row_col_weights = backend.reshape(row_col_weights, shape=(n_rows * c_dim[0] * n_cols * c_dim[1], in_channels * kernel_size[0] * kernel_size[1]))

    # Compute cosine similarity
    corr = lower_tri(1 - squareform(pdist(row_col_weights, metric='cosine')))

    # Compute coordinates and distances
    coordinates = np.indices((n_rows * c_dim[0], n_cols * c_dim[1])).transpose(1, 2, 0).reshape((n_rows * c_dim[0] * n_cols * c_dim[1], 2))
    dist = lower_tri(squareform(pdist(coordinates, wrapped_distance)))

    # Plotting
    plot_similarity_vs_distance(dist, corr, layer_i, hparams, PRE_RELU)

def plot_similarity_vs_distance(dist, corr, layer_i, hparams, PRE_RELU):
    """
    Plots the similarity versus distance scatter plot.

    Args:
        dist (ndarray): The distance values.
        corr (ndarray): The correlation/similarity values.
        layer_i (int): The current layer index.
        hparams (dict): Hyperparameters and analysis directory information.
        PRE_RELU (bool): Indicates whether to consider pre ReLU activations.
    """
    fig, ax = plt.subplots(figsize=(1.5, 1.5))
    ax.scatter(dist, corr, s=0.1, c="k", linewidths=0, alpha=0.5, rasterized=True)
    remove_spines(ax)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_ylim(-1, 1) # limit y range -1 to 1

    # Prepare directory and file paths
    analysis_dir = f"{hparams['analysis_dir']}/distance_vs_weight_similarity"
    os.makedirs(analysis_dir, exist_ok=True)
    base_filename = f"{analysis_dir}/{hparams['model_name']}_alpha{hparams['alpha']}_drop{hparams['dropout_rate']}_pre_relu_{PRE_RELU}_layer{layer_i}_metric_cosine_respones_similarity_vs_weight_distance_ecoset_weight"

    # Save the plots
    plt.savefig(f"{base_filename}.png", dpi=600)
    plt.savefig(f"{base_filename}.pdf", dpi=600)

def wrapped_distance(pt1, pt2, boundary=(50, 50)):
    """
    Compute the wrapped distance between two points in a 2D space with circular boundaries.
    
    :param pt1: First point as a tuple (x, y).
    :param pt2: Second point as a tuple (x, y).
    :param boundary: Tuple representing the boundary size (width, height).
    :return: Wrapped distance between the two points.
    """
    dx = min(abs(pt1[0] - pt2[0]), boundary[0] - abs(pt1[0] - pt2[0]))
    dy = min(abs(pt1[1] - pt2[1]), boundary[1] - abs(pt1[1] - pt2[1]))
    return np.sqrt(dx**2 + dy**2)

def make_linear_cm_extend(r, g, b, plot=False):
    """
    Helper function:
    Used to increase the saturation of a color (used for the category selectivity maps plots)
    """

    N = 256
    extension = 50
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(1, r / 256, N)
    vals[:, 1] = np.linspace(1, g / 256, N)
    vals[:, 2] = np.linspace(1, b / 256, N)
    end = np.array([r / 256, g / 256, b / 256, 1])
    for x in range(extension):
        vals = np.concatenate((vals, np.reshape(end, (1, 4))))
    newcmp = ListedColormap(vals)
    return newcmp