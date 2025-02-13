import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import scienceplots
from all_tnn.models.model_helper.tnn_helper_functions import channels_to_sheet
plt.style.use([ 'nature','science',"ieee",'no-latex'])

def get_fft_data(output_dict, model_names):
    categories = output_dict[0]['LCN_lr_0.05']['smoothness_analysis']['fft'].keys()
    fft_dict = {cat: [] for cat in categories}
    for c, cat in enumerate(categories):
        for i, model in enumerate(model_names):
            model_ffts = []
            for seed in output_dict:
                fft = seed[model]['smoothness_analysis']['fft'][cat]
                power_spectrum = np.abs(fft)**2
                center = np.array(power_spectrum.shape) // 2
                radial_profile = get_radial_profile(power_spectrum, center)
                model_ffts.append(radial_profile)  
            fft_dict[cat].append(model_ffts)
    return fft_dict

def plot_fft(output_dict, cmap, model_names, base_path):
    """ Plot the average power spectrum of the orientation selectivity and 
    the average power spectrum of the faces, scenes and tools selectivity.""" 

    fft_dict = get_fft_data(output_dict, model_names)
    selectivity_cats = [cat for cat in fft_dict.keys() if cat != 'os_sheet']    
    fft_dict['avg_category'] = [np.mean([fft_dict[cat][i] for cat in selectivity_cats], axis=0) for i in range(len(fft_dict[selectivity_cats[0]]))]

    with plt.style.context([ 'nature','science',"ieee",'no-latex']):
        plt.rcParams['font.family'] = 'sans-serif'
        fig, axs = plt.subplots(2, 1, figsize=(6, 12))

        for i, cat in enumerate(['os_sheet', 'avg_category']):
            for j, model in enumerate(fft_dict[cat]):
                means = np.mean(model, axis=0) 
                slope, _ = np.polyfit(np.log(range(1, len(means)+1)), np.log(means), 1)
                ci_low, ci_high = stats.t.interval(0.95, len(model)-1, loc=means, scale=stats.sem(model))
                axs[i].fill_between(range(len(means)), ci_low, ci_high, color=cmap[j], alpha=0.2)
                axs[i].plot(means, label=f'{model_names[j]}, slope: {slope:.2f}', color=cmap[j], linewidth=1.5)
                
            cat_title = 'orientation selectivity (layer 1)' if cat == 'os_sheet' else cat + '(faces, scenes, tools) (layer 6)'
            
            axs[i].set_xscale('log')
            axs[i].set_yscale('log')
            ylim = 10**8 if cat == 'os_sheet' else 10**4
            axs[i].set_ylim([None, ylim])
            axs[i].set_title(cat_title)
            axs[i].set_xlabel('Spatial frequency [Hz, log scale]')
            axs[i].set_ylabel('Power [log scale]')
            axs[i].tick_params(which='both', right=False, top=False)#
            axs[i].spines['top'].set_visible(False)
            axs[i].spines['right'].set_visible(False)
            axs[i].legend()

        plt.savefig(f'{base_path}/fft_avg_cat_{len(output_dict)}seeds.pdf')
        plt.close()

def get_radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int32)
    mask = r <= center[0]
    tbin = np.bincount(r[mask].ravel(), data[mask].ravel())
    nr = np.bincount(r[mask].ravel())
    radialprofile = tbin / nr
    return radialprofile 
 
def plot_pinwheel_vs_eccentricity(output_dict, cmap, base_path):
    """ Plot the radial profile of the pinwheel density normalized by the entropy."""
    model_names = list(output_dict[0].keys())
    plt.figure(figsize=(6,3))
    for model in model_names:
        radial_profile = output_dict[0][model]['radial_profile']
        x = range(1, len(radial_profile)+1)
        slope, intercept = np.polyfit(x, radial_profile, 1)
        plt.plot(radial_profile, label=model.split('_lr_0.05')[0] + f', slope: {slope:.2f}', color=cmap(model_names.index(model)))
        plt.plot(x, slope*np.array(x) + intercept, color=cmap(model_names.index(model)), linestyle='--')

    plt.title(f'Radial profile of pinwheel density (normalized by entropy)')
    plt.ylabel('Pinwheel density  / entropy')
    plt.xlabel('Eccentricity')
    plt.legend() 
    plt.savefig(f'{base_path}/radial_profile_pw_entr_div.pdf')
    plt.close()


def plot_cluster_size(output_dict, cmap, model_names, base_path, stats=False, save=True, show=False):
    """ Bar plot of the average cluster size of the orientation selectivity and the average category selectivity."""
    n_models = len(model_names)
    categories = ['os_sheet', 'd_prime']
    category_titles = ['orientation selectivity', 'category selectivity']
    bar_width = 0.8

    with plt.style.context([ 'nature','science',"ieee",'no-latex']):
        plt.rcParams['font.family'] = 'sans-serif'
        fig, axs = plt.subplots(2, figsize=(3.5, 6))

        for subplot, category_type in enumerate(categories):
            cluster_sizes = {model: compute_cluster_sizes(output_dict, model, category_type) for model in model_names}
            for i, model in enumerate(model_names):
                means = np.mean(cluster_sizes[model]) 
                sd = np.std(cluster_sizes[model]) 
                axs[subplot].bar(i, means, yerr=sd, label=model_names[i], color=cmap[i], width=bar_width)
            axs[subplot].set_ylabel('Average cluster size [units]')
            axs[subplot].set_xticks(range(n_models))
            axs[subplot].set_xticklabels(model_names)
            axs[subplot].set_title(f'{category_titles[subplot]} cluster size in layer {[1, 6][subplot]}')
            axs[subplot].spines['top'].set_visible(False)
            axs[subplot].spines['right'].set_visible(False)
            axs[subplot].tick_params(which='both', right=False, top=False, bottom=False)

            if stats: 
                import statsmodels.stats.multitest
                significance_dict = {}
                for i in range(n_models):
                    for j in range(i + 1, n_models):
                        t_stat, p_value = stats.ttest_ind(cluster_sizes[model_names[i]], cluster_sizes[model_names[j]])
                        fdr_corrected_p = statsmodels.stats.multitest.multipletests(p_value, alpha=0.05, method='fdr_bh')[1]
                        print(f"T-test between {model_names[i]} and {model_names[j]}: p = {p_value}, fdr p = {fdr_corrected_p}")
                        significance_dict[(i, j)] = fdr_corrected_p         
        
        plt.tight_layout()
        if save: plt.savefig(f'{base_path}/cluster_size_{len(output_dict)}seeds.pdf', dpi=200)
        if show: plt.show()
        plt.close()

def compute_cluster_sizes(output_dict, model, category):
    return [np.mean(list(seed[model]['smoothness_analysis']['cluster_size'][category].values())) for seed in output_dict]


def cluster_size_vs_eccentricity(output_dict, cmap, model_names, base_path, save=True, show=False): 
    """ Plot the radial profile of cluster size of the orientation selectivity in the first layer."""
    model_radial_dict = {}
    
    with plt.style.context([ 'nature','science',"ieee",'no-latex']):
        plt.rcParams['font.family'] = 'sans-serif'
        fig, ax = plt.subplots(figsize=(8, 3.5))

        for i, model in enumerate(model_names):
            model_radial_dict[model] = []
            for seed_dict in output_dict:
                layer_size = int(np.sqrt(len(seed_dict[model]['smoothness_analysis']['cluster_size']['os_sheet'])))
                matrix_cluster_sizes = np.zeros((layer_size, layer_size))
                for key, value in seed_dict[model]['smoothness_analysis']['cluster_size']['os_sheet'].items():
                    matrix_cluster_sizes[key] = value
                
                radial_profile = get_radial_profile(matrix_cluster_sizes, np.array(matrix_cluster_sizes.shape) // 2)
                model_radial_dict[model].append(radial_profile)
            try:
                means = np.mean(model_radial_dict[model], axis=0)
            except:
                import pdb; pdb.set_trace()
            ci_low, ci_high = stats.t.interval(0.95, len(model_radial_dict[model])-1, loc=means, scale=stats.sem(model_radial_dict[model]))
            
            ax.plot(means, label=model_names[i], color=cmap[model_names.index(model)])
            ax.fill_between(range(len(means)), ci_low, ci_high, color=cmap[model_names.index(model)], alpha=0.2)
        
        ax.set_title('Cluster size vs eccentricity of orientation selectivity')
        ax.set_ylabel('Cluster size')
        ax.set_xlabel('Eccentricity')
        ax.set_xlim([0, 190])
        ax.set_ylim([1,27])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(which='both', right=False, top=False)#
        plt.legend()
        plt.tight_layout()
        if save: plt.savefig(f'{base_path}/cluster_size_vs_eccentricity.pdf')
        if show: plt.show()
        plt.close()

def calculate_radial_entropy(output_dict, model_names):
    rad_entr_dict = {}
    for i, model in enumerate(model_names):
            rad_entr_dict[model] = {}
            for seed_idx, seed in enumerate(output_dict):
                if model not in seed.keys(): 
                    continue
                else:
                    try:
                        layer_entropy_layer = seed[model]['grating_w_entropies'][0][0]  # for pickle dict
                    except: 
                        layer_entropy_layer = seed[model]['grating_w_entropies']['layer_0'] # for h5
                    layer_entropy = np.squeeze(channels_to_sheet(layer_entropy_layer, return_np=True))
                    center = np.array(layer_entropy.shape) / 2
                    if 'shifted' in model:
                        center = np.array([center[0] + 77, center[1] + 77])
                    radial_profile = get_radial_profile(layer_entropy, center)
                    rad_entr_dict[model][seed_idx] = {'e_layer_1D':radial_profile}
    return rad_entr_dict

def plot_radial_entropy(rad_energy_dict, cmap, model_names, save_dir, save=True, show=False):
    plt.style.context([ 'nature','science',"ieee",'no-latex'])
    plt.rcParams['font.family'] = 'sans-serif'
    fig, ax = plt.subplots(figsize=(3.9, 1.7)) 
    for model in model_names:
        e_layer_1D_values = [seed['e_layer_1D'] for seed in rad_energy_dict[model].values()]
        e_layer_1D_mean = np.mean(e_layer_1D_values, axis=0)
        e_layer_1D_std = np.std(e_layer_1D_values, axis=0)
        ci = 1.96 * e_layer_1D_std / np.sqrt(len(e_layer_1D_values))  
        ax.plot(e_layer_1D_mean, color=cmap[model_names.index(model)], label=model)
        ax.fill_between(range(len(e_layer_1D_mean)), (e_layer_1D_mean) - ci, (e_layer_1D_mean) + ci, color=cmap[model_names.index(model)], alpha=0.2)

    ax.set_xlabel('Eccentricity')
    ax.set_ylabel('Entropy [nats]')
    ax.set_ylim([0, 2.2])  
    ax.set_xlim([0, 180])  
    ax.tick_params(which='both', top=False, right=False)  
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='lower right')
    if save: plt.savefig(os.path.join(save_dir, 'radial_entropy_profile.pdf'), dpi=200)
    if show: plt.show()
    plt.close()

def visualize_smoothness(output_dicts, seeds, model_names, cmap, plot_path, save=True, show=False):
    # plot_fft(output_dict, cmap, model_names, plot_path, save, show)
    # plot_pinwheel_vs_eccentricity(output_dict, cmap,  plot_path, save, show)
    plot_cluster_size(output_dicts, cmap, model_names, plot_path, save, show)
    cluster_size_vs_eccentricity(output_dicts, cmap, model_names, plot_path, save, show)
    entropy_dict = calculate_radial_entropy(output_dicts, model_names)
    plot_radial_entropy(entropy_dict, cmap, model_names, plot_path, save, show)
