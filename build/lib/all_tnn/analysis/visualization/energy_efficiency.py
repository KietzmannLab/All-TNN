import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict
import matplotlib
from tqdm import tqdm
from all_tnn.analysis.visualization.smoothness_entropy_visualization import get_radial_profile

# Set plotting style
import scienceplots  
plt.style.use(['science', 'nature', "ieee", 'no-latex'])
from .colors import COLOR_THEME_WITH_ALPHA_SWEEP
color_palette = COLOR_THEME_WITH_ALPHA_SWEEP[1:] 

def preload_energy_maps(model_name_path_dict, 
                        fixed_epochs, 
                        seed_range, 
                        pre_or_postrelu, 
                        NORM_PREV_LAYER, 
                        NORM_LAYER_OUT,
                        max_layer=6):
    """
    Pre-load all npy files into a dictionary to avoid repeated I/O.
    Returns a dict with keys (model_name, epoch, seed, layer).
    """
    energy_data = {}
    for model_name, base_path in tqdm(model_name_path_dict.items()):
        for epoch in fixed_epochs:
            for seed in seed_range:
                for layer in range(1, max_layer):
                    # Possibly disable in-loop modifications of NORM_LAYER_OUT 
                    # or handle them beforehand if 'simclr' appears in model_name.
                    if 'shift' in model_name.lower():
                        continue

                    suffix = ''
                    if 'simclr' in model_name.lower():
                        # You might fix or store this condition once outside.
                        actual_norm_layer_out = False
                    else:
                        actual_norm_layer_out = NORM_LAYER_OUT

                    filename = (
                        f"ali_way_overall_sum_energy_consumption_layer{layer}"
                        f"_NORM_LAYER_OUT_{actual_norm_layer_out}"
                        f"_NORM_PREV_LAYER_{NORM_PREV_LAYER}{suffix}.npy"
                    )
                    file_path = os.path.join(
                        base_path.replace('seed1', f'seed{seed}'), 
                        f'analysis_{pre_or_postrelu}/energy_efficiency/ep{epoch}/', 
                        filename
                    )
                    # Load data once
                    energy_data[(model_name, epoch, seed, layer)] = np.load(file_path)
    
    return energy_data


def plot_energy_consumption_across_epochs_lineplot(
    model_name_path_dict, 
    alphas, 
    fixed_epochs,
    save_fig_path=None, 
    pre_or_postrelu='prerelu', 
    prefix_list=['', 'ali_'], 
    energy_consumption_types=['total', 'sum'], 
    seed_range=range(1, 6), 
    models_epochs_dict=None, 
    NORM_PREV_LAYER=True, 
    NORM_LAYER_OUT=False,
    show_plot=False,
):
    plt.rcParams['font.family'] = 'sans-serif'
    
    # Pre-load everything to avoid repeated I/O
    max_layer = 6  # only have layers 0..5
    all_energy_data = preload_energy_maps(
        model_name_path_dict,
        fixed_epochs,
        seed_range,
        pre_or_postrelu,
        NORM_PREV_LAYER,
        NORM_LAYER_OUT,
        max_layer=max_layer,
    )
    
    # Prepare figure
    fig, ax = plt.subplots(figsize=(3.54, 2))
    
    model_data = defaultdict(lambda: defaultdict(tuple))  # to store (mean, ci)
    seeds_model_data = defaultdict(lambda: defaultdict(list))
    
    # If you really want a color palette
    color_palette = plt.cm.viridis(np.linspace(0, 1, len(model_name_path_dict)))
    
    # Now compute sums from the pre-loaded data
    for model_idx, (model_name, _) in enumerate(model_name_path_dict.items()):
        if 'finetune' in model_name.lower(): #* finetune and non-finetune models are the same in representation level
            continue
        if 'shift' in model_name.lower(): #* shift and non-shift models are the same in representation level
            continue
        for epoch in fixed_epochs:
            epoch_energy = []
            for seed in seed_range:
                # Summation across layers
                total_energy = 0
                for layer in range(1, max_layer):
                    energy_map = all_energy_data[(model_name, epoch, seed, layer)]
                    total_energy += np.sum(np.abs(energy_map))
                epoch_energy.append(total_energy)
            
            # Calculate statistics
            means = np.mean(epoch_energy)
            std_err = np.std(epoch_energy, ddof=1) / np.sqrt(len(epoch_energy))
            ci = stats.t.ppf(0.975, len(epoch_energy)-1) * std_err
            
            model_data[model_name][epoch] = (means, ci)
            seeds_model_data[model_name][epoch] = epoch_energy
    
    # Plotting
    for model_idx, (model_name, epochs_data) in enumerate(model_data.items()):
        epochs = sorted(epochs_data.keys())
        means, cis = zip(*[epochs_data[e] for e in epochs])
        ax.errorbar(
            epochs, 
            means, 
            yerr=cis, 
            label=model_name, 
            marker='o', 
            linestyle='-', 
            color=color_palette[model_idx]
        )
    
    ax.set_yscale('log')
    ax.set_xlim(left=0, right=max(fixed_epochs) + 5)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Total Energy Consumption')
    ax.set_title("Total Energy Consumption in Ali' Way across Epochs")
    ax.set_xticks(fixed_epochs)
    ax.set_xticklabels([str(epoch) for epoch in fixed_epochs])
    ax.tick_params(which='both', top=False, right=False)  # Disable top and right ticks #  bottom=False,
    ax.legend()
    sns.despine(trim=True, left=False)  # Remove top and right spines
    
    if save_fig_path:
        plt.savefig(os.path.join(save_fig_path, 'energy_consumption_across_epochs_line.png'), dpi=300)
        plt.close()
    if show_plot: plt.show()
    else: plt.close() 


def plot_stacked_energy_map_energy_vs_eccentricity(model_name_path_dict, alphas, save_fig_path=None, pre_or_postrelu='prerelu', prefix_list=['', 'ali_'], energy_consumption_types=['total', 'sum'], seed_range=range(1, 6), models_epochs_dict=None, NORM_PREV_LAYER=True,NORM_LAYER_OUT=False, show_plot=False):
    # Dictionary to hold model data
    # model_data = defaultdict(lambda: defaultdict(dict))

    # Collect data
    for prefix, energy_type in zip(prefix_list, energy_consumption_types):

        rad_energy_dict = defaultdict(dict)
        for model_name, base_path in model_name_path_dict.items():
            if 'finetune' in model_name.lower(): #* finetune and non-finetune models are the same in representation level
                continue

            alpha = alphas[model_name]
            for seed in seed_range:
                epoch = models_epochs_dict[model_name][seed-1]
                file_path = os.path.join(base_path.replace('seed1', f'seed{seed}'), f'analysis_{pre_or_postrelu}/energy_efficiency/ep{epoch}/', 
                                            f"{prefix}5layers_NORM_LAYER_OUT_{NORM_LAYER_OUT}_NORM_PREV_LAYER_{NORM_PREV_LAYER}_{model_name}_alpha{alpha}_drop0.0_pre_relu_{True if pre_or_postrelu=='prerelu' else False}_stacked_total_energy_map.npy")
                stacked_energy_map = np.load(file_path)

                # Get the radial profile
                center = np.array(stacked_energy_map.shape) / 2
                rad_energy_dict[model_name][seed] = get_radial_profile(stacked_energy_map, center)
            

        cmap = COLOR_THEME_WITH_ALPHA_SWEEP[1:]
        plot_radial_energy(rad_energy_dict, cmap, list(model_name_path_dict.keys()), list(model_name_path_dict.keys()), save_fig_path, show_plot=show_plot)



def plot_radial_energy(rad_energy_dict, cmap, model_names, legend_names, save_dir, show_plot=False):
    plt.style.use(['nature', 'science', "ieee", 'no-latex'])
    fm = matplotlib.font_manager
    fm._get_fontconfig_fonts.cache_clear()
    plt.rcParams['font.family'] = 'sans-serif'

    fig, ax = plt.subplots(figsize=(3.54, 2)) # half width of NHB

    for model in model_names:
        if 'finetune' in model.lower(): #* finetune and non-finetune models are the same in representation level
            continue
        energy_profiles_layer = [seed for seed in rad_energy_dict[model].values()]
        e_layer_1D_mean = np.mean(energy_profiles_layer, axis=0)
        # Normalize the energy values to start from 100% at the center and decrease
        max_energy = e_layer_1D_mean[0]  # assuming maximum energy is at the center
        normalized_energy = 100 * e_layer_1D_mean / max_energy  # normalize to percentage

        e_layer_1D_std = np.std(energy_profiles_layer, axis=0)
        normalized_std = 100 * e_layer_1D_std / max_energy  # normalize standard deviation
        ci = 1.96 * normalized_std / np.sqrt(len(energy_profiles_layer))  # confidence interval
        ax.plot(normalized_energy, color=cmap[model_names.index(model)], label=legend_names[model_names.index(model)])
        ax.fill_between(range(len(normalized_energy)), (normalized_energy) - ci, (normalized_energy) + ci, color=cmap[model_names.index(model)], alpha=0.2)

    ax.set_xlabel('Eccentricity')
    ax.set_ylabel('Relative Energy Consumption [%]')
    ax.set_ylim([0, 120])  # Set y-axis from 100% to 0%
    ax.set_xlim([0, 200])  # Set x-axis from 0 to 20
    ax.tick_params(which='both', top=False, right=False)  
    ax.legend()
    sns.despine(trim=True, left=False)  
    os.makedirs(save_dir, exist_ok=True)
    print(os.path.join(save_dir, 'radial_energy_profile.pdf'))
    plt.savefig(os.path.join(save_dir, 'radial_energy_profile.pdf'), dpi=300)

    if show_plot: plt.show()
    else: plt.close()


def plot_stacked_energy_maps_normalized(model_name_path_dict, alphas, save_fig_path=None, pre_or_postrelu='prerelu', prefix_list=['', 'ali_'], energy_consumption_types=['total', 'sum'], seed_range=range(1, 6), models_epochs_dict=None, NORM_PREV_LAYER=True, NORM_LAYER_OUT=False, show_plot=False):

    # Set common color scale limits
    vmin, vmax = 0, 1  # Adjust these based on known or expected data range for normalization

    for seed in seed_range:
        plt.figure(figsize=(15, 3 * len(model_name_path_dict) * len(prefix_list) * len(energy_consumption_types)))
        plt.subplots_adjust(hspace=0.4, wspace=0.4)  # Adjust space between plots

        # Plotting index
        plot_index = 1

        for prefix, energy_type in zip(prefix_list, energy_consumption_types):
            for model_name, base_path in model_name_path_dict.items():
                alpha = alphas[model_name]
                epoch = models_epochs_dict[model_name][seed - 1]
                
                # Shift models have only one seed
                if 'shift' in model_name.lower() and seed > 1:
                    continue

                if 'cnn' not in model_name.lower():
                    file_name = (
                        f"{prefix}5layers_NORM_LAYER_OUT_{NORM_LAYER_OUT}_"
                        f"NORM_PREV_LAYER_{NORM_PREV_LAYER}_{model_name}_alpha{alpha}_drop0.0_"
                        f"pre_relu_{True if pre_or_postrelu=='prerelu' else False}_"
                        f"stacked_total_energy_map.npy"
                    )
                    file_path = os.path.join(
                        base_path.replace('seed1', f'seed{seed}'),
                        f'analysis_{pre_or_postrelu}',
                        'energy_efficiency',
                        f'ep{epoch}',
                        file_name
                        )
                    # Load the energy map
                    stacked_energy_map = np.load(file_path)
                    
                    
                else:
                    # import pdb; pdb.set_trace()
                    cnn_file_paths = [os.path.join(base_path.replace('seed1', f'seed{seed}'), f'analysis_{pre_or_postrelu}/energy_efficiency/ep{epoch}/', 
                                            f"ali_way_overall_sum_energy_consumption_layer{layer_i}_NORM_LAYER_OUT_{NORM_LAYER_OUT}_NORM_PREV_LAYER_{NORM_PREV_LAYER}.npy") for layer_i in range(1, 6)]
                    # import pdb; pdb.set_trace()
                    acc_energy_maps = [np.mean(np.load(file_path),axis=-1) for file_path in cnn_file_paths] #* (48, 48, 64)  to (48, 48) for CNN, not to sheet to keep the spatial info
                    stacked_energy_map = stack_energy_maps(acc_energy_maps)

                # Normalize each map between 0 and 1 if the max value is greater than zero
                if np.max(stacked_energy_map) > 0:
                    stacked_energy_map = stacked_energy_map / np.max(stacked_energy_map)

                # Plotting
                ax = plt.subplot(len(model_name_path_dict) * len(prefix_list) * len(energy_consumption_types), 1, plot_index)
                cax = ax.imshow(stacked_energy_map, cmap='viridis', vmin=vmin, vmax=vmax) #! default
                # cax = ax.imshow(stacked_energy_map, cmap='viridis') #! supplementary
                # if plot_index % len(model_name_path_dict) == 1:
                plt.colorbar(cax, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04)  # Adjust colorbar size and padding
                ax.set_title(f"{model_name} - {energy_type} - Alpha: {alpha} - Seed: {seed}", fontsize=10)
                # ax.set_xlabel('Layers', fontsize=8)
                # ax.set_ylabel('Channels', fontsize=8)
                
                # Remove x and y ticks
                ax.set_xticks([])
                ax.set_yticks([])
                
                plot_index += 1

        # Adjust layout and save/show the figure
        if save_fig_path:
            os.makedirs(save_fig_path, exist_ok=True)
            file_name = os.path.join(save_fig_path, f'Seed_{seed}_with_ori_cnn_energy_maps.pdf')
            plt.savefig(file_name)
            print(f"Saved to {file_name}")
        
        if show_plot: plt.show()
        else: plt.close()

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