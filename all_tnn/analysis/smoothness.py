import os, pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from all_tnn.models.model_helper.tnn_helper_functions import channels_to_sheet

##### Fourier analysis #####
def fft_smoothness(wavelength_tuning_curve_peaks, category_selectivities):
    print(f'Calculating smoothness with FFT')
    smooth_dict = {}

    # Orientation selectivity (layer 0)
    wav_os_sheet = channels_to_sheet(wavelength_tuning_curve_peaks[0][0], return_np=True)
    fft = np.fft.fftshift(np.fft.fft2(wav_os_sheet)) 
    smooth_dict['os_sheet'] = fft

    # Category selectivity (layer 5)
    for i, category in enumerate(category_selectivities.keys()):
        fft = np.fft.fftshift(np.fft.fft2(category_selectivities[category])) # already is a sheet        
        smooth_dict[f'{category}_selectivity'] = fft
    return smooth_dict

def fft_plot_2d(fft, ax, name=None):
    if name=='os_sheet': vmax = 7000
    else: vmax = 400
    im = ax.imshow(np.abs(fft), norm=mcolors.LogNorm(vmax=vmax))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f'FTT of {name}')
    

def fft_plot_power_spectrum(fft, ax, name=None):
    power_spectrum = np.abs(fft)**2
    center = np.array(power_spectrum.shape) // 2
    radial_profile = get_radial_profile(power_spectrum, center)
    ax.plot(radial_profile)
    ax.set_yscale('log')
    ax.set_title(f'Radial profile of power spectrum {name}')
    ax.set_xlabel('Radius')
    ax.set_ylabel('Mean Power Spectrum Value')

def get_radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int32)
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile 

### Cluster size ###
def get_avg_cluster_size(wavelength_tuning_curve_peaks, cat_selectivity):
    print(f'Calculating average cluster size')
    radius_dicts = {}

    # Orientation selectivity (layer 0)
    wav_os_sheet = channels_to_sheet(wavelength_tuning_curve_peaks[0][0], return_np=True)
    os_bins = np.linspace(0, 1, num=9) # 8 orientations
    wav_os_sheet_binned = np.digitize(wav_os_sheet, os_bins) - 1 # -1 to get zero-index bins
    radius_dicts['os_sheet'] = get_radius_dict(wav_os_sheet_binned, len(os_bins)-1)
    
    # Category selectivity (layer 5)
    # Merge the category selectivity maps into one array where each position gets a value 
    # # representing the category with the highest selectivity
    new_joined_map = np.stack([cat_selectivity[category] for category in cat_selectivity.keys()], axis=2)
    merged_d_map = np.argmax(new_joined_map, axis=-1)
    radius_dicts['d_prime'] = get_radius_dict(merged_d_map, 3)

    return radius_dicts    

def get_radius_dict(sheet, bin_size):
    # For each unit, calculate the radius to the point where the area contains all 8 orientations
    radius_dict = {}
    unit_coords = np.argwhere(sheet >= 0)
    for i, (x, y) in enumerate(unit_coords):
        if i % 1000 == 0: print(f'Unit {i} out of {len(unit_coords)}')
        radius = 1
        # Expand the radius step by step
        while True:
            # Shift the array such that the current unit is at the center
            shifted_sheet = np.roll(sheet, shift=(-x, -y), axis=(0, 1))

            # Get the coordinates of the units within the current radius
            center_x, center_y = shifted_sheet.shape[0] // 2, shifted_sheet.shape[1] // 2
            x_min = center_x - radius
            x_max = center_x + radius + 1
            y_min = center_y - radius
            y_max = center_y + radius + 1

            # Check the selectivity of the units within the current radius
            selectivities = set(shifted_sheet[x_min:x_max, y_min:y_max].flatten())
            if len(selectivities) == bin_size:
                radius_dict[(x, y)] = radius
                break
            # If not all orientations are found, increase the radius by 1
            radius += 1
            if radius >= min(sheet.shape) / 2:
                radius_dict[(x, y)] = np.nan
                break
    return radius_dict


def smoothness_main(output_dict, epoch=600, analysis_path=None):
    """
    Analyze smoothness metrics by loading category selectivity data and computing 
    cluster size and FFT smoothness on wavelength tuning curves.

    Parameters:
        output_dict (dict): A dictionary containing 'grating_w_tuning_curves' data.
                            Expects a tuple (wavelength_tuning_curve_peaks, <unused>).
        epoch (int, optional): Epoch number used to construct the path for loading
                               selectivity data. Defaults to 600.
        analysis_path (str, optional): Base directory for the 'category_selectivity' folder.
                                       Defaults to None.
    
    Returns:
        dict: A dictionary with a single key 'smoothness_analysis', which contains:
              - 'cluster_size': Average cluster size of wavelength tuning curve peaks.
              - 'fft': FFT smoothness of wavelength tuning curve peaks.
    """
    # Construct the first path (epoch-specific)
    epoch_path = os.path.join(analysis_path, 'category_selectivity', f'epoch{epoch}')
    file_name = 'dprime_sheets_layer5.pkl'
    file_path = os.path.join(epoch_path, file_name)
    
    try:
        # Attempt to load from the epoch-specific directory
        with open(file_path, 'rb') as f:
            cat_selectivity = pickle.load(f)
    except FileNotFoundError:
        # Fallback to the default directory if the epoch-specific file doesn't exist
        default_path = os.path.join(analysis_path, 'category_selectivity', file_name)
        with open(default_path, 'rb') as f:
            cat_selectivity = pickle.load(f)
    
    # Extract the wavelength tuning curve peaks
    wavelength_tuning_curve_peaks, _ = output_dict['grating_w_tuning_curves']
    
    # Compute smoothness metrics
    out_dict = {
        'cluster_size': get_avg_cluster_size(wavelength_tuning_curve_peaks, cat_selectivity),
        'fft': fft_smoothness(wavelength_tuning_curve_peaks, cat_selectivity)
    }
    
    return {'smoothness_analysis': out_dict}