import os, pickle, yaml
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
from sklearn.preprocessing import scale
from numpy.ma import masked_array
from all_tnn.models.model_helper.tnn_helper_functions import channels_to_sheet


def get_activations_on_dataset(activities_model, tf_dataset, this_dataset_name, one_hot, hparams=None, plot_path=None,
                               row_shifts=None, col_shifts=None):
    """
    Collects and optionally plots the activities from a given model on a TensorFlow dataset.

    Args:
        activities_model: The model to extract activities from.
        tf_dataset: The TensorFlow dataset to process.
        this_dataset_name: The name of the dataset being processed.
        one_hot: Whether the labels are one-hot encoded.
        hparams: Hyperparameters dictionary, defaults to None.
        plot_path: Path to save plots, defaults to None.
        row_shifts: List of row shifts to apply, defaults to None.
        col_shifts: List of column shifts to apply, defaults to None.

    Returns:
        activities: List of numpy arrays containing model activities.
        labels: Numpy array of labels corresponding to the dataset.
    """

    print('Gathering activities...')

    if plot_path:
        os.makedirs(plot_path, exist_ok=True)
        print(f'\tPlotting images and activities to {plot_path} while collecting activities...')

    # Determine the number of layers in the model
    for x in tf_dataset:
        dummy_batch = x[:1]
        break
    n_layers = len(activities_model(dummy_batch))
    
    # Determined finetuned or not
    finetune_flag = hparams.get('finetune', False)

    # Initialize storage for activities and labels
    temp_activities = []
    temp_labels = []

    # Set default row and column shifts if not provided
    row_shifts = [0] if row_shifts is None else row_shifts
    if 0 not in row_shifts:
        row_shifts.append(0)
    col_shifts = [0] if col_shifts is None else col_shifts
    if 0 not in col_shifts:
        col_shifts.append(0)
    print(f'Applying row_shifts: {row_shifts}, col_shifts: {col_shifts}')


    for i, x in enumerate(tf_dataset):
        if i % 100 == 0:
            print(f'\rBatch {i}', end='')

        for r_s in row_shifts:
            for c_s in col_shifts:
                if r_s == 0 and c_s == 0:
                    # No shift
                    activities = [a.numpy() for a in activities_model(x[:1])]
                    temp_activities.append(activities)
                    temp_labels.append(x[1]['dense_2'].numpy())  
                else:
                    # Apply shifts
                    shifted_img_input = np.roll(x[0], shift=r_s, axis=1)
                    shifted_img_input = np.roll(shifted_img_input, shift=c_s, axis=2)
                    activities = [a.numpy() for a in activities_model(shifted_img_input)]
                    temp_activities.append(activities)
                    temp_labels.append(x[1]['dense_2'].numpy())
                    print(r_s, c_s, shifted_img_input[0, 0, 0, 0])

        if plot_path and i % 100 == 0:
            fig, ax = plt.subplots(10, n_layers + 1)
            for j in range(10):
                ax[j][0].imshow(((x[0][j].numpy() + 1) * 255 / 2).astype(np.uint8))
                for n in range(n_layers):
                    ax[j][n + 1].imshow(crop_center(channels_to_sheet(temp_activities[-1][n][j], return_np=True), 10, 10))
                for n in range(n_layers + 1):
                    ax[j][n].set_axis_off()
            plt.savefig(f'{plot_path}/imgs_acts_{this_dataset_name}_{i}.png')
            plt.close()

    # Concatenate activities
    activities = [np.concatenate([temp_activities[i][l] for i in range(len(temp_activities))], axis=0) for l in range(n_layers)]

    # Handle labels for SimCLR models
    if hparams and 'simclr' in hparams['model_name'].lower() and 'finetune' not in hparams['model_name'] and not finetune_flag:
        labels = np.ones(x.shape[0], dtype=np.int32) * -1  # Dummy labels for the SimCLR model
    else:
        labels = np.concatenate(temp_labels, axis=0)
        if not one_hot:
            labels = np.argmax(labels, axis=1)

    if plot_path:
        print(f'\tSaving activity RDM to {plot_path}')
        rdms = make_rdm(activities)
        plot_rdms(rdms, plot_path, f'rdms_{this_dataset_name}')

    del temp_activities, temp_labels

    return activities, labels


def get_mean_activations_per_class_on_dataset(activities_model, tf_dataset):
    '''Returns mean activities of each layer of activities_model for each class in tf_dataset.
    Note: the n_classes is inferred automatically from the dimension of the 1-hot vectors of tf_dataset'''

    print('Computing mean activities per class...')

    for x in tf_dataset:
        dummy_labels = x[1]
        dummy_activities = [a.numpy() for a in activities_model(x[0])]
        break

    n_layers = len(dummy_activities)
    n_classes = dummy_labels['dense_2'].shape[-1]
    layer_sizes = [dummy_activities[l].shape[1:] for l in range(n_layers)]

    sums = [np.zeros(((n_classes,) + layer_sizes[l])) for l in range(n_layers)]
    means = [np.zeros_like(sums[l]) for l in range(n_layers)]
    counters = np.zeros((n_classes,))
    for i, x in enumerate(tf_dataset):
        if i % 100 == 0:
            print(f'\rBatch {i}', end='')
        batch_activities = [a.numpy() for a in activities_model(x[0])]
        batch_labels = np.argmax(x[1]['dense_2'].numpy(), axis=1)  # 1hot -> scalar
        for i in np.unique(batch_labels):
            counters[i] += np.count_nonzero(batch_labels == i)
            for l in range(n_layers):
                sums[l][i] += np.sum(batch_activities[l][batch_labels == i], axis=0)

    for l in range(n_layers):
        for c in range(n_classes):
            means[l][c] = sums[l][c] / counters[c] if counters[c] > 0 else sums[l][c]

    del sums

    return means, counters


def crop_center(img, cropx, cropy):
    y, x = img.shape
    startx = max(0, x // 2 - (cropx // 2))
    starty = max(0, y // 2 - (cropy // 2))
    endx = min(x, startx + cropx)
    endy = min(y, starty + cropy)
    return img[starty:endy, startx:endx]


def make_rdm(activities, distance='correlation'):
    dist = []
    n_activ = len(activities) if isinstance(activities, list) else 1
    for l in range(n_activ):
        dist.append(pdist(np.reshape(activities[l], [activities[l].shape[0], -1]), distance))
    return dist[0] if n_activ == 1 else dist


def plot_rdms(rdm_list, path, name):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    n_rdms = len(rdm_list) if isinstance(rdm_list, list) else 1

    fig, ax = plt.subplots(1,n_rdms , figsize=(n_rdms * 7, 7))
    [ax[n].set_axis_off() for n in range(n_rdms)]

    for n in range(n_rdms):
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        img = ax[n].imshow(squareform(rdm_list[n]), cmap='viridis')
        divider = make_axes_locatable(ax[n])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(img, cax=cax)

    plt.savefig(f'{path}/{name}.png')
    plt.close()


def plot_mds(activities, categories, cat_names, save_path):
    '''activities: [array([n_inputs, n_units])]*n_layers,
    categories: will be used for coloring the plot dots (e.g. uses 0 for objects and 1 for faces)'''

    n_layers, n_imgs = len(activities), activities[0].shape[0]
    activities = [np.reshape(l, [n_imgs, -1]) for l in activities]  # flatten all layer activities
    unique_categories = np.unique(categories)
    mds_embedding = MDS(n_components=2)
    fig, ax = plt.subplots(n_layers, figsize=(7, 7*n_layers))
    for l in range(n_layers):
        print(f'\tComputing MDS for layer {l}')
        these_activities = scale(activities[l])
        mds_data = mds_embedding.fit_transform(these_activities)
        for group in unique_categories:
            group_data = mds_data[categories == group, :]
            ax[l].scatter(group_data[:, 0], group_data[:, 1], s=100, label=group)
        ax[l].set_xlabel(f'layer {l}')
        ax[l].xaxis.set_label_position('top')
        ax[l].tick_params(axis="x", bottom=False, top=False, labelbottom=False, labeltop=False)
        ax[l].tick_params(axis="y", left=False, right=False, labelleft=False, labelright=False)
        ax[l].legend(cat_names)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close('all')


def load_and_override_hparams(saved_model_path, **kwargs):
    # load hparams from a model's save_path. You can override params using the kwargs.
 
    with open(f'{saved_model_path}/hparams.pickle', 'rb') as f:
        hparams = pickle.load(f)
    
    hparams['saved_model_path'] = saved_model_path

    for k, v in kwargs.items():
        hparams[k] = v
    
    return hparams

def convert_numpy_arrays_to_lists(data):
    if isinstance(data, dict):
        # If the data is a dictionary, recursively process its values
        return {key: convert_numpy_arrays_to_lists(value) if isinstance(value, np.ndarray) else value for key, value in data.items()}
    else:
        # Otherwise, return the data as is
        return data

def results_to_disk(cache_dir, output_dict):

    os.makedirs(cache_dir, exist_ok=True)
    print(f'Saving output_dict to {cache_dir}')
    pickle.dump(output_dict, open(os.path.join(cache_dir, f'output_dict.pickle'), 'wb'))

    numeric_output = {k: v for k, v in output_dict.items() if k in ["dropout_analysis", "subnetworks_analysis"]}

    yaml.dump(numeric_output,
              open(os.path.join(cache_dir, f'output_dict_{datetime.today().strftime("%y%m%d")}.yaml'), 'w'))


def results_from_disk(cache_dir):
    output_dict_path = os.path.join(cache_dir, f'output_dict.p')
    if os.path.exists(output_dict_path):
        print(f'Found existing output_dict in {output_dict_path}. Loading...')
        return pickle.load(open(output_dict_path, 'rb'))
    else:
        print(f'Did not find an existing output_dict in {output_dict_path}. Starting from scratch...')
        return dict()


def bar_plot(data, tick_labels=None, group_names=None, save_path=None, x_label=None, xticks_rotation=0, y_label=None, title=None):
    '''Make and save a bar plot from a data vector
    data: nxm array with n groups of m datapoints
    tick_labels: list of m strings, to label the x-axis
    group_names: list of n strings, to label each group of datapoints
    output_path: str indicating where to save the image
    x_label & y_label: str writes what data are shown on the x & y axes'''

    # Data to plot
    data = np.array(data)
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)
    n_groups = data.shape[0]
    n_ticks = data.shape[1]

    if tick_labels is not None:
        if not n_ticks == len(tick_labels):
            if n_groups == len(tick_labels):
                print('You likely have provided the data in the wrong format: [n_item_per_group, n_groups],'
                      'instead of [n_groups, n_items_per_group], since n_groups does not match len(tick_labels).'
                      'Transposing your data matrix')
                data = data.T
                n_groups = data.shape[0]
                n_ticks = data.shape[1]
            else:
                raise Exception(f'Data of shape {data.shape} is incompatible with len(tick_labels) = {len(tick_labels)}')

    # Create plot
    plt.subplots()
    index = np.arange(n_ticks)
    bar_width = 0.5/n_groups
    opacity = 1

    for i in range(n_groups):
        plt.bar(index+i*bar_width, data[i, :], bar_width, alpha=opacity*(1-i/20))

    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)

    if tick_labels is not None:
        plt.xticks(index+(n_groups//2)*(bar_width/n_groups), tick_labels, rotation=xticks_rotation)
    else:
        plt.xticks(index+(n_groups//2)*(bar_width/n_groups))
    if group_names is not None:
        plt.legend(group_names)
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.show() if save_path is None else plt.savefig(save_path)
    plt.close('all')


def mask_array_by_value(img, mask_value=0):
    '''Masks all zeros in an array so they are ignored. Good for detecting 0s in plots'''
    if mask_value is None:
        return img
    else:
        return masked_array(img, img==mask_value)


def set_gridlines(ax, channel_dim, data_grid, hc_transform, kernel_plot_minor_elems=0, kernel_plot_major_elems=0):
    """Set double or single gridlines, dependent on the type of plot"""

    biggest_dim = np.max(data_grid.shape)

    if kernel_plot_minor_elems > 0:
        minor_x = np.arange(-0.5, data_grid.shape[1], kernel_plot_minor_elems)
        minor_y = np.arange(-0.5, data_grid.shape[0], kernel_plot_minor_elems)
        ax.set_xticks(minor_x, minor=True)
        ax.set_yticks(minor_y, minor=True)
        plt.grid(b=True, color='black', which='minor', linestyle='-', linewidth=np.min([0.25, 50/biggest_dim]))
        major_x = np.arange(-0.5, data_grid.shape[1], kernel_plot_major_elems)
        major_y = np.arange(-0.5, data_grid.shape[0], kernel_plot_major_elems)
        ax.set_xticks(major_x)
        ax.set_yticks(major_y)
    else:
        if hc_transform:
            major_x = np.arange(-0.5, data_grid.shape[1], channel_dim[1])
            major_y = np.arange(-0.5, data_grid.shape[0], channel_dim[0])
            ax.set_xticks(major_x)
            ax.set_yticks(major_y)
        else:
            major_x = np.arange(-0.5, data_grid.shape[1], data_grid.shape[1] // channel_dim[1])
            major_y = np.arange(-0.5, data_grid.shape[0], data_grid.shape[0] // channel_dim[0])
            ax.set_xticks(major_x)
            ax.set_yticks(major_y)

    plt.grid(b=True, color='black', which='major', linewidth=np.min([1, 200 / biggest_dim]))


def format_weights(kernel_weights, bias_weights, n_row, n_col, kernel_size, in_channels, out_channels, conv_control_net=False):

    if conv_control_net:
        # in the case of a convolutional network (used as a control in the lc_net project), weights need a different
        # kind of formatting. We start with shape [row_ker_size, col_ker_size, in_channel, out_channels]
        row_col_weights = np.tile(np.expand_dims(kernel_weights, [0, 1]), [n_row, n_col, 1, 1, 1, 1])  # add row & col dimensions (absent from CNNs because fo weight sharing)
        lc_like_weights = np.reshape(row_col_weights, [n_row*n_col, np.prod(kernel_size)*in_channels, out_channels])
        bias_weights = np.tile(np.expand_dims(bias_weights, [0, 1]), [n_row, n_col, 1])  # add row & col dimensions (absent from CNNs because fo weight sharing)

    else:
        # lc_net case
        # kernel weights is [n_rows*n_cols, rows_kernel_size*cols_kernel_size*in_channels, out_channels]
        row_col_weights = kernel_weights.reshape((n_row, n_col, kernel_size[0], kernel_size[1], in_channels, out_channels))

    # Change to im_row, im_col, out_channels, rows_kernel_size, cols_kernel_size, in_channel
    row_col_weights = np.transpose(row_col_weights, (0,1,-1,2,3,4))

    # Reshape to get 1 vector of weights for each row/col [row, col, out_channels, n_ker_dims]
    row_col_weights_without_bias = row_col_weights.reshape((*row_col_weights.shape[:3], -1))
    row_col_weights_with_bias = np.concatenate([row_col_weights_without_bias, np.expand_dims(bias_weights,-1)], axis=-1)

    # Get norm of each kernel [row, col, out_channels]
    row_col_weights_without_bias_norm = np.linalg.norm(row_col_weights_without_bias, axis=-1)
    row_col_weights_with_bias_norm = np.linalg.norm(row_col_weights_with_bias, axis=-1)

    if conv_control_net:
        return row_col_weights, row_col_weights_without_bias_norm, row_col_weights_with_bias_norm, lc_like_weights, bias_weights
    else:
        return row_col_weights, row_col_weights_without_bias_norm, row_col_weights_with_bias_norm
    

def get_epochs_from_dir(model_dir, epoch):
    checkpoints_path = os.path.join(model_dir, "training_checkpoints")
    if epoch == 'most_recent':
        return int(sorted(os.listdir(checkpoints_path))[-1].split("_")[-1][2:5])
    elif epoch == 'all':
        return [int(item.split("_")[-1][2:5]) for item in os.listdir(checkpoints_path)]
    else:
        return int(epoch)

