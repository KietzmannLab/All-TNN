import math, os
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import entropy
import scipy.signal as signal
from all_tnn.models.model_helper.tnn_helper_functions import channels_to_sheet, sheet_to_channels, extract_channel_dim
from all_tnn.task_helper_functions import preprocess_batch

def create_grating(w, ori, phase, wave, imsize):
    """
    Function to create oriented gratings for the orientation selectivity analysis
    :param w: spatial wavelength (in pixels)
    :param ori: wave orientation (in degrees, [0-360])
    :param phase: wave phase (in degrees, [0-360])
    :param wave: type of wave ('sqr' or 'sin')
    :param imsize: image size (integer)
    :return: numpy array of shape (imsize, imsize)
    """
    # Get x and y coordinates
    x, y = np.meshgrid(np.arange(imsize), np.arange(imsize))

    # Get the appropriate gradient
    gradient = np.sin(ori * math.pi / 180) * x - np.cos(ori * math.pi / 180) * y

    # Plug gradient into wave function
    if wave == 'sin':
        grating = np.sin((2 * math.pi * gradient) / w + (phase * math.pi) / 180)
    elif wave == 'sqr':
        grating = signal.square((2 * math.pi * gradient) / w + (phase * math.pi) / 180)
    else:
        raise NotImplementedError

    grating = (255*(grating+1)/2) # to match what networks expect

    return grating


def get_gratings_stimuli(wavelengths, hparams, n_angles):
    """
    Function used to generate oriented grating for the orientation selectivity analysis.
    Returns a two-dimensional array with grating stimuli, where the first axis is for the frequency and the second
    axis is for the phase.
    """

    x, y, z = hparams['image_size'], hparams['image_size'], 3
    gratings = []

    with tqdm(total=sum(n_angles * np.array(wavelengths)), desc='Generating grating stimuli') as pbar:
        for w in wavelengths:
            g_batch = np.zeros((w, n_angles, x, y, z)) # , dtype=np.uint8

            for a in range(n_angles):
                for phase in range(w):
                    img = create_grating(w=w,
                                         phase=360 * phase / w,
                                         ori=180 * a / n_angles,
                                         imsize=hparams['image_size'],
                                         wave='sin')
                    img = np.repeat(img[:, :, np.newaxis], repeats=z, axis=2)

                    # Use model preprocessing (in this case, turn image from float64 to float32)
                    g_batch[phase][a] = img

                    pbar.update(1)

            g_batch = g_batch.reshape((w, n_angles, x, y, z))
            gratings.append(g_batch)

    return gratings


def get_tuning_curves(activities_model, w_p_batches, n_angles, hparams):
    """Find the orientation selectivity angle and corresponding spread for each wavelength/phase
    """

    n_layers, n_wavelengths = len(activities_model.output_shape), len(w_p_batches)
    
    #* Exclude last conv readout layer
    using_conv_readout = hparams.get('conv_readout', False)
    if using_conv_readout:
        n_layers -= 1
        # n_wavelengths -= 1

    wavelength_tuning_curve_peaks = np.empty((n_layers, n_wavelengths), dtype=object)
    wavelength_tuning_curve_spreads = np.empty((n_layers, n_wavelengths), dtype=object)
    wavelength_activities = np.empty((n_layers, n_wavelengths), dtype=object)

    with tqdm(total=n_layers * n_wavelengths, desc='Getting tuning curves') as pbar:
        # For each spatial wavelength
        for i_w, p_batches in enumerate(w_p_batches):
            w_activities_raw = []

            # For each phase
            for batch in p_batches:
                batch = preprocess_batch(batch, hparams)

                # Get the activities
                grating_activities = activities_model(batch)
                grating_activities = [g.numpy() for g in grating_activities]
                w_activities_raw.append(grating_activities)

            # Reorder from layer-list of phase-lists to phase-list of layer-lists
            w_activities_raw = group_by_phase(w_activities_raw)

            for layer_i in range(n_layers):

                layer_activities = w_activities_raw[layer_i]

                max_phase_activities = get_max_phase_activation(layer_activities, n_angles)
                wavelength_activities[layer_i][i_w] = max_phase_activities

                # Compute the peak, magnitude and std of the angle distribution for each RF
                grating_curve_peaks = mean_angle(max_phase_activities)
                grating_curve_std = np.nan_to_num(np.std(max_phase_activities, axis=0))/(np.sum(max_phase_activities, axis=0))

                # Scale the spread and peak between 0 and 1
                grating_curve_peaks /= (2 * math.pi)
                grating_curve_std /= np.std([1] + [0] * (n_angles - 1))

                wavelength_tuning_curve_peaks[layer_i][i_w] = grating_curve_peaks
                wavelength_tuning_curve_spreads[layer_i][i_w] = grating_curve_std

                pbar.update(1)

    return (wavelength_tuning_curve_peaks, wavelength_tuning_curve_spreads), wavelength_activities


def get_max_phase_activation(layer_activities, n_angles):

    original_shape = layer_activities.shape[2:]
    layer_activities = layer_activities.reshape(*layer_activities.shape[:2], -1)

    # Index the phase of the activities by the argument that maximizes the activation
    max_activation_phase = np.argmax(layer_activities, axis=0)

    # Flatten for easier indexing
    layer_activities = layer_activities.reshape(layer_activities.shape[0], -1)
    max_activation_phase = max_activation_phase.flatten()

    # Index max phase
    layer_activities = layer_activities[max_activation_phase, np.arange(len(max_activation_phase))]

    # Unflatten and reshape back to original
    layer_activities = layer_activities.reshape((n_angles, -1))
    layer_activities = layer_activities.reshape((n_angles, *original_shape))

    return layer_activities


def mean_angle(activations):
    """Get the meanangle of the total activation by projecting activations onto a circle"""

    # Get the orientations in linear space in radians
    orientations = np.linspace(0, 2 * math.pi, activations.shape[0] + 1)[:-1]
    orientations = orientations[:, np.newaxis, np.newaxis, np.newaxis]

    # Get xy coordinates for the orientations and scale by their activations
    x = np.sum(np.cos(orientations) * activations, axis=0)
    y = np.sum(np.sin(orientations) * activations, axis=0)

    # Compute the arctangent to get an angle
    at = np.arctan2(y, x)

    # Rescale from [-pi, pi] to [0, 2pi]
    at = (at + 2 * math.pi) % (2 * math.pi)

    return at


def group_by_phase(activations):
    """Re-arrange the list of activations into a phase-grouped list"""

    concatenate_base = [array[np.newaxis] for array in activations[0]]

    for i in np.arange(len(activations))[1:]:
        for j in np.arange(len(activations[i])):
            concatenate_base[j] = np.concatenate((concatenate_base[j], activations[i][j][np.newaxis]))

    return concatenate_base


def compute_spatial_entropies(grating_w_activities, hparams, n_bins, window_size=8):

    print(f'Computing orientation selectivity entropies with window_size {window_size}')

    entropies = np.empty(grating_w_activities[0].shape, dtype=object)
    radial_entropies = np.empty(grating_w_activities[0].shape, dtype=object)

    orientations, spread = grating_w_activities

    # Iterate over all layer/spatial wavelength combinations
    with tqdm(total=np.prod(grating_w_activities[0].shape), desc='Computing spatial entropies') as pbar:

        for layer_i, (layer_ori, layer_spread) in enumerate(list(zip(orientations, spread))):
            for wavelength_i, (layer_ori_w, layer_spread_w) in enumerate(list(zip(layer_ori, layer_spread))):

                # Get the orientation/spread maps for each spatial wavelength
                o = layer_ori_w
                s = layer_spread_w

                # Bin the orientations
                o_bins = (o * n_bins).astype(int)

                # Exclude occurrences that do not enter the ReLU (negative bin value is ignored)
                o_bins[s == 0] = -1
                n_row, n_col, n_channels = o_bins.shape
                channel_dim = extract_channel_dim(n_channels)

                # Unfold channel dim
                o_bins = np.squeeze(channels_to_sheet(o_bins, return_np=True))

                # Setup for sliding window spatial entropy calculation
                x_range, y_range = np.arange(o_bins.shape[0]), np.arange(o_bins.shape[1])
                layer_entropy = np.zeros(o_bins.shape)

                # Double for-loop for sliding window
                for x in x_range:
                    for y in y_range:

                        # Take values from the map
                        x_indices = np.take(x_range, range(x - window_size // 2, x + 1 + window_size // 2), mode='wrap')
                        y_indices = np.take(y_range, range(y - window_size // 2, y + 1 + window_size // 2), mode='wrap')
                        window = o_bins[x_indices][:, y_indices]

                        # Count the wavelength for each orientation bin
                        # (note that the negative value assigned above means it is ignored here)
                        bin_wavelengths = [(window == b).sum() for b in range(n_bins)]

                        # Compute entropy
                        layer_entropy[x][y] = entropy(bin_wavelengths)
                # Reshape back to conventional 3D shape and store in array
                layer_entropy = sheet_to_channels(layer_entropy, channel_dim, return_np=True)
                entropies[layer_i][wavelength_i] = layer_entropy

                pbar.update(1)

    return entropies, radial_entropies


def orientation_selectivity(activities_model, wavelengths, n_angles, hparams, entropy_sliding_window_size, output_dict):

    w_p_batches = get_gratings_stimuli(wavelengths, hparams, n_angles)
    grating_w_tuning_curves, grating_w_activities = get_tuning_curves(activities_model, w_p_batches, n_angles, hparams)
    grating_w_entropies, radial_entropies = compute_spatial_entropies(grating_w_tuning_curves, hparams, n_angles, entropy_sliding_window_size)

    fig, ax = plt.subplots(max(len(grating_w_entropies),2), max(len(wavelengths),2))  # max(X,2) is because you can't index axes if subplots gets a 1, so we use this dirty hack in the case where wevelengths has just 1 entry
    for l in range(len(grating_w_entropies)):
        for w in range(len(wavelengths)):
            try:
                ax[l][w].imshow(np.squeeze(channels_to_sheet(np.expand_dims(grating_w_entropies[l][w], 0), return_np=True)))
            except:
                print(f'Could not plot layer {l} and wavelength {w}')
                import pdb; pdb.set_trace()
            ax[l][w].axis('off')
            if w == 0:
                ax[l][0].set_ylabel(f'Layer {l}')
            if l == 0:
                ax[0][w].set_title(f'Wavelength {w}')

    os.makedirs(f'{hparams["analysis_dir"]}/entropy_graphs', exist_ok=True)
    plt.savefig(f'{hparams["analysis_dir"]}/entropy_graphs/entropy_maps_slidingWindowSize{entropy_sliding_window_size}.png', dpi=300)
    plt.close()

    orientation_selectivity_dict = {'grating_w_activities': grating_w_activities,
            'grating_w_entropies': grating_w_entropies,
            'grating_w_radial_entropies': radial_entropies,
            'grating_w_tuning_curves': grating_w_tuning_curves}

    return orientation_selectivity_dict

