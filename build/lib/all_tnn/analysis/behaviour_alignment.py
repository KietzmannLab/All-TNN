import scipy
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.stats
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform, correlation, cosine, cdist
from sklearn.metrics import mean_squared_error
from pyemd import emd
from sklearn.utils import resample
from all_tnn.analysis.util.statistic_funcs import perform_non_parametric_statistical_analysis, perform_statistical_analysis, perform_permutation_statistical_analysis, perform_sign_permutation_test_statistical_analysis, perform_permutation_statistical_analysis

## --------------------------------------------------------------- ##
## -----------------  Visual Behaviour Alignment ----------------- ##
## --------------------------------------------------------------- ## 
def visual_behaviour_alignment_analysis(
    acc_maps_data,
    alignment_metric,
    model_names,
    comparison_modes,
    columns,
    categories_num=16,
    map_norm_mode='max',
    alignment_mode='mean',
    sampling_num=100,
    model_names_to_plot=None,
    verbose=False
):
    """
    Analyze the alignment between individual participants' accuracy maps and seed models.

    Parameters:
        acc_maps_data (list): 
            - acc_maps_data[0]: Individual participants' accuracy maps.
            - acc_maps_data[1:]: Seed models' accuracy maps.
        alignment_metric (str): Metric used for alignment calculation.
        comparison_modes (list): Modes for comparison.
        model_names (list): List of all model names, excluding human if not plotted.
        columns (list): Column names for the resulting DataFrame.
        categories_num (int, optional): Number of categories. Defaults to 16.
        map_norm_mode (str, optional): Normalization mode for maps. Defaults to 'max'.
        alignment_mode (str, optional): Mode for calculating normal alignment. Defaults to 'mean'.
        sampling_num (int, optional): Number of bootstrap samples. Defaults to 100.
        model_names_to_plot (dict, optional): Mapping of model names to plot. Defaults to None.
        verbose (bool, optional): If True, prints debug information. Defaults to False.

    Returns:
        tuple: 
            - DataFrame containing alignment data.
            - Noise ceiling value.
            - Dictionary with statistical significance results.
    """
    
    # Separate individual and model accuracy maps
    individual_acc_maps = np.array(acc_maps_data[0])
    model_acc_maps = np.array(acc_maps_data[1:])
    
    # Normalize individual and model accuracy maps
    individual_acc_maps = np.array([
        normalize_maps(map_set, map_norm_mode, map_norm_func) for map_set in individual_acc_maps
    ])
    model_acc_maps = np.array([
        [normalize_maps(map_set, map_norm_mode, map_norm_func) for map_set in model_maps]
        for model_maps in model_acc_maps
    ])
    
    # Update model names, excluding the first one (assumed to be human) unless specified
    updated_model_names = update_model_names(model_names, model_names_to_plot)
    
    num_participants = individual_acc_maps.shape[0]
    num_models = model_acc_maps.shape[0]
    num_seeds = model_acc_maps.shape[1]
    
    # Initialize alignment matrices for each model
    alignment_matrices = initialize_alignment_matrices(
        updated_model_names,
        num_seeds,
        num_participants,
        categories_num,
        sampling_num
    )
    
    visual_behaviour_data = []
    
    # Compute noise ceiling
    noise_ceiling, _ = compute_noise_ceiling(individual_acc_maps, alignment_metric)
    
    # Compute alignment matrices
    for model_idx, seeds_model_maps in enumerate(model_acc_maps):
        model_name = updated_model_names[model_idx]
        
        for seed_idx, seed_map in enumerate(seeds_model_maps):
            average_human_map = np.mean(individual_acc_maps, axis=0)
            
            # Align with individual participants
            align_with_individuals(
                alignment_matrices,
                model_name,
                seed_idx,
                seed_map,
                individual_acc_maps,
                alignment_metric,
                categories_num
            )
            
            # Align with average human map
            align_with_average(
                alignment_matrices,
                model_name,
                seed_idx,
                seed_map,
                average_human_map,
                alignment_metric,
                categories_num
            )
            
            # Bootstrap alignment with resampled average human maps
            perform_bootstrapping(
                alignment_matrices,
                model_name,
                seed_idx,
                seed_map,
                individual_acc_maps,
                alignment_metric,
                categories_num,
                sampling_num,
                verbose
            )
    
    if verbose:
        print(f"Alignment matrices: {alignment_matrices}\n")
    
    # Handle specified comparison modes and collect alignment data
    handle_comparison_modes(
        alignment_matrices,
        comparison_modes,
        visual_behaviour_data,
        alignment_mode,
        alignment_metric
    )
    
    # Create DataFrame and perform statistical analysis
    df_alignment, significance = create_dataframe_and_analyze(
        visual_behaviour_data,
        columns,
        alignment_metric
    )
    
    if verbose:
        print(f"Alignment DataFrame:\n{df_alignment}")
    
    return df_alignment, noise_ceiling, significance


## -------------------------------------------------- ##
## -----------------  ADM Alignment ----------------- ##
## -------------------------------------------------- ##
def adm_alignment_analysis(
    raw_individuals_vs_seeds_model_acc_maps,
    model_names,
    seeds_list,  # Note: 'seeds_list' is not used in the original function; it will be removed if redundant
    comparison_modes,
    create_adm_metric,
    rsa_metric,
    columns,
    categories_num=16,
    map_norm_mode='max',
    alignment_mode='mean',
    sampling_num=100,
    model_names_to_plot=None,
    verbose=False
):
    """
    Analyze the alignment between individual participants' ADM maps and seed models.

    Parameters:
        raw_individuals_vs_seeds_model_acc_maps (list): 
            - raw_individuals_vs_seeds_model_acc_maps[0]: Individual participants' accuracy maps.
            - raw_individuals_vs_seeds_model_acc_maps[:]: Seed models' accuracy maps.
        model_names (list): List of all model names, excluding human if not plotted.
        seeds_list (list): List of seed identifiers (Unused in the refactored function).
        comparison_modes (list): Modes for comparison.
        create_adm_metric (str): Metric used to create ADM.
        rsa_metric (str): Metric used for RSA calculation.
        columns (list): Column names for the resulting DataFrame.
        categories_num (int, optional): Number of categories. Defaults to 16.
        map_norm_mode (str, optional): Normalization mode for maps. Defaults to 'max'.
        alignment_mode (str, optional): Mode for calculating normal alignment. Defaults to 'mean'.
        sampling_num (int, optional): Number of bootstrap samples. Defaults to 100.
        model_names_to_plot (dict, optional): Mapping of model names to plot. Defaults to None.
        verbose (bool, optional): If True, prints debug information. Defaults to False.

    Returns:
        tuple: 
            - DataFrame containing ADM alignment data.
            - Noise ceiling value.
            - Dictionary with statistical significance results.
            - ADM dictionary for saving later analysis.
    """
    
    # Separate individual and model accuracy maps
    raw_individual_maps = np.array(raw_individuals_vs_seeds_model_acc_maps[0])
    raw_seeds_maps = np.array(raw_individuals_vs_seeds_model_acc_maps[1:])
    
    # Normalize individual and model accuracy maps
    # raw_individual_maps = np.array([
    #     normalize_adm_maps(map_set, map_norm_mode, map_norm_func) for map_set in raw_individual_maps
    # ])
    # individual_acc_maps = np.array([
    #     normalize_maps(map_set, map_norm_mode, map_norm_func) for map_set in individual_acc_maps
    # ])
    individual_acc_maps = np.array([normalize_maps(map_set, map_norm_mode, map_norm_func) for map_set in raw_individual_maps ])
    raw_seeds_maps = np.array([
        [normalize_adm_maps(map_set, map_norm_mode, map_norm_func) for map_set in model_maps]
        for model_maps in raw_seeds_maps
    ])
    
    # Update model names, excluding the first one (assumed to be human) unless specified
    updated_model_names = update_adm_model_names(model_names, model_names_to_plot)
    
    num_participants = raw_individual_maps.shape[0]
    num_models = raw_seeds_maps.shape[0]
    num_seeds = raw_seeds_maps.shape[1]
    categories_comb = int(categories_num * (categories_num - 1) / 2)
    
    # Initialize ADM matrices for each model
    adm_matrices = initialize_adm_matrices(
        updated_model_names,
        num_seeds,
        num_participants,
        categories_comb,
        sampling_num
    )
    
    adm_alignment_data = []
    # Compute noise ceiling
    noise_ceiling, _ = compute_adm_noise_ceiling(individual_acc_maps.reshape(num_participants, -1), rsa_metric)

    # Create average human ADM
    average_human_adm, _ = create_average_adm(individual_acc_maps, create_adm_metric)
    # Compute individual human ADMs
    human_adms = compute_individual_adms(individual_acc_maps, create_adm_metric)
    # Compute model ADMs
    model_adms, average_model_adms = compute_model_adms(raw_seeds_maps, create_adm_metric, categories_num, updated_model_names)
    # Perform bootstrapping on human ADMs
    bootstrap_human_adms = perform_adm_bootstrapping(individual_acc_maps, create_adm_metric, sampling_num)
    # Compute model-human ADM agreements
    model_human_agreement = compute_model_human_agreement(model_adms, human_adms, rsa_metric)
    
    # Compute model-average human ADM agreements
    model_average_human_agreement, model_bootstrap_human_agreement = compute_model_average_human_agreement(
        model_adms,
        average_human_adm,
        bootstrap_human_adms,
        rsa_metric,
        sampling_num
    )
    
    if verbose:
        print(f"Debug human_adms: {human_adms}\n --> human_adms shape: {human_adms.shape}")
        print(f"Debug model_adms: {model_adms} \n --> model_adms shapes: {[model_adms[model_name].shape for model_name in updated_model_names]}")
        print(f"Debug model_human_agreement: {model_human_agreement} \n --> model_human_agreement shapes: {[model_human_agreement[model_name].shape for model_name in updated_model_names]}")
    
    # Handle specified comparison modes and collect ADM alignment data
    handle_adm_comparison_modes(
        model_bootstrap_human_agreement,
        comparison_modes,
        adm_alignment_data,
        rsa_metric
    )
    
    if verbose:
        print(f"Debug adm_alignment_data: {pd.DataFrame(adm_alignment_data, columns=columns)}")
    
    # Create DataFrame and perform statistical analysis
    df_adm_agreement, significance_dict = create_adm_dataframe_and_analyze(
        adm_alignment_data,
        columns,
        rsa_metric
    )
    
    if verbose:
        print(f"Debug df_adm_agreement:\n{df_adm_agreement}")
    
    # Create ADM dictionary for saving later analysis
    adm_dict = {
        'average_human_adm': squareform(average_human_adm),
        'average_model_adms': {model_name: squareform(average_model_adms[model_name]) for model_name in updated_model_names},
        'human_adms': [squareform(human_adm) for human_adm in human_adms],
        'model_adms': {model_name: [squareform(model_adms[model_name][i]) for i in range(model_adms[model_name].shape[0])] for model_name in updated_model_names},
    }
    
    return df_adm_agreement, noise_ceiling, significance_dict, adm_dict



## ------------------------------------------------ ##
## -----------------  utils funcs ----------------- ##
## ------------------------------------------------ ##
# Behavioural agreement utils funcs
def normalize_maps(maps, mode, norm_func):
    """
    Normalize accuracy maps based on the specified mode.

    Parameters:
        maps (numpy.ndarray): The accuracy maps to normalize.
        mode (str): The normalization mode.
        norm_func (callable): Function to compute normalization factor.

    Returns:
        numpy.ndarray: Normalized accuracy maps.
    """
    return maps / norm_func(maps, mode)

def update_model_names(model_names, model_names_to_plot):
    """
    Update model names by excluding the first one or mapping them if provided.

    Parameters:
        model_names (list): Original list of model names.
        model_names_to_plot (dict or None): Mapping of model names to plot.

    Returns:
        list: Updated list of model names.
    """
    if model_names_to_plot is None:
        return model_names[:]
    return [model_names_to_plot[name] for name in model_names[:]]

def initialize_alignment_matrices(model_names, num_seeds, num_participants, categories_num, sampling_num):
    """
    Initialize alignment matrices for each model.

    Parameters:
        model_names (list): List of model names.
        num_seeds (int): Number of seeds per model.
        num_participants (int): Number of participants.
        categories_num (int): Number of categories.
        sampling_num (int): Number of bootstrap samples.

    Returns:
        dict: Nested dictionary containing alignment matrices.
    """
    return {
        model: {
            'individual': np.zeros((num_seeds, num_participants, categories_num)),
            'average': np.zeros((num_seeds, categories_num)),
            'bootstrap': np.zeros((num_seeds, sampling_num, categories_num))
        }
        for model in model_names
    }

def compute_noise_ceiling(individual_acc_maps, alignment_metric):
    """
    Compute the noise ceiling from individual accuracy maps.

    Parameters:
        individual_acc_maps (numpy.ndarray): Individual participants' accuracy maps.
        alignment_metric (str): Metric used for alignment calculation.

    Returns:
        tuple: Noise ceiling value and additional info.
    """
    return get_noise_ceiling_from_individual_acc_maps(individual_acc_maps, alignment_metric)

def calculate_alignment(seed_map, comparison_map, alignment_metric):
    """
    Calculate alignment between seed map and comparison map.

    Parameters:
        seed_map (numpy.ndarray): Seed model's accuracy map for a category.
        comparison_map (numpy.ndarray): Comparison accuracy map for a category.
        alignment_metric (str): Metric used for alignment calculation.

    Returns:
        float: Alignment score.
    """
    return get_2vectors_alignment(seed_map.flatten(), comparison_map.flatten(), metric=alignment_metric)

def align_with_individuals(alignment_matrices, model_name, seed_idx, seed_map, individual_acc_maps, alignment_metric, categories_num):
    """
    Align seed maps with individual participants' accuracy maps.

    Parameters:
        alignment_matrices (dict): Alignment matrices to update.
        model_name (str): Current model name.
        seed_idx (int): Index of the current seed.
        seed_map (numpy.ndarray): Current seed's accuracy map.
        individual_acc_maps (numpy.ndarray): Individual participants' accuracy maps.
        alignment_metric (str): Metric used for alignment calculation.
        categories_num (int): Number of categories.

    Returns:
        None
    """
    for participant_idx, individual_map in enumerate(individual_acc_maps):
        for category in range(categories_num):
            alignment = calculate_alignment(seed_map[category], individual_map[category], alignment_metric)
            alignment_matrices[model_name]['individual'][seed_idx, participant_idx, category] = alignment

def align_with_average(alignment_matrices, model_name, seed_idx, seed_map, average_human_map, alignment_metric, categories_num):
    """
    Align seed maps with the average human accuracy map.

    Parameters:
        alignment_matrices (dict): Alignment matrices to update.
        model_name (str): Current model name.
        seed_idx (int): Index of the current seed.
        seed_map (numpy.ndarray): Current seed's accuracy map.
        average_human_map (numpy.ndarray): Average human accuracy map.
        alignment_metric (str): Metric used for alignment calculation.
        categories_num (int): Number of categories.

    Returns:
        None
    """
    for category in range(categories_num):
        alignment = calculate_alignment(seed_map[category], average_human_map[category], alignment_metric)
        alignment_matrices[model_name]['average'][seed_idx, category] = alignment

def perform_bootstrapping(alignment_matrices, model_name, seed_idx, seed_map, individual_acc_maps, alignment_metric, categories_num, sampling_num, verbose):
    """
    Perform bootstrapping to align seed maps with resampled average human maps.

    Parameters:
        alignment_matrices (dict): Alignment matrices to update.
        model_name (str): Current model name.
        seed_idx (int): Index of the current seed.
        seed_map (numpy.ndarray): Current seed's accuracy map.
        individual_acc_maps (numpy.ndarray): Individual participants' accuracy maps.
        alignment_metric (str): Metric used for alignment calculation.
        categories_num (int): Number of categories.
        sampling_num (int): Number of bootstrap samples.
        verbose (bool): Flag to display progress.

    Returns:
        None
    """
    for sample_id in tqdm(range(sampling_num), desc='Bootstrapping', disable=not verbose):
        bootstrap_sample = resample(individual_acc_maps, n_samples=individual_acc_maps.shape[0])
        bootstrap_average_map = np.mean(bootstrap_sample, axis=0)
        
        for category in range(categories_num):
            alignment = calculate_alignment(seed_map[category], bootstrap_average_map[category], alignment_metric)
            alignment_matrices[model_name]['bootstrap'][seed_idx, sample_id, category] = alignment

def collect_alignment_data(visual_behaviour_data, comparison_mode, alignment_data, labels, alignment_metric):
    """
    Aggregate alignment data with corresponding labels and metadata.

    Parameters:
        visual_behaviour_data (list): List to store aggregated data.
        comparison_mode (str): Current comparison mode.
        alignment_data (numpy.ndarray): Alignment data to collect.
        labels (list): Labels corresponding to the alignment data.
        alignment_metric (str): Metric used for alignment calculation.

    Returns:
        None
    """
    visual_behaviour_data.extend(zip(
        labels,
        alignment_data,
        [str(alignment_metric)] * len(alignment_data),
        [comparison_mode] * len(alignment_data)
    ))

def handle_comparison_modes(alignment_matrices, comparison_modes, visual_behaviour_data, alignment_mode, alignment_metric):
    """
    Handle specified comparison modes and collect alignment data.

    Parameters:
        alignment_matrices (dict): Alignment matrices for all models.
        comparison_modes (list): Modes for comparison.
        visual_behaviour_data (list): List to store aggregated data.
        alignment_mode (str): Mode for calculating normal alignment.

    Returns:
        None
    """
    if 'individual_model_vs_average_human' in comparison_modes:
        for model_name, matrices in alignment_matrices.items():
            normalized_alignment = calculate_normal_alignment(
                matrices['bootstrap'],
                alignment_mode,
                axis=(0, 2)
            ).flatten()
            collect_alignment_data(
                visual_behaviour_data,
                'individual_model_vs_average_human_bootstrap',
                normalized_alignment,
                [model_name] * len(normalized_alignment),
                alignment_metric
            )
    else:
        raise ValueError(f"Invalid comparison mode: {comparison_modes}")

def create_dataframe_and_analyze(visual_behaviour_data, columns, alignment_metric):
    """
    Create a DataFrame from alignment data and perform statistical analysis.

    Parameters:
        visual_behaviour_data (list): Aggregated alignment data.
        columns (list): Column names for the resulting DataFrame.
        alignment_metric (str): Metric used for alignment calculation.

    Returns:
        tuple: DataFrame and statistical significance results.
    """
    df_alignment = pd.DataFrame(visual_behaviour_data, columns=columns)
    significance = perform_permutation_statistical_analysis(
        df_alignment,
        alignment_metric,
        num_permutations=10000,
        permutation_type='independent'
    )
    return df_alignment, significance


# ADM utils funcs
def normalize_adm_maps(maps, mode, norm_func):
    """
    Normalize ADM maps based on the specified mode.

    Parameters:
        maps (numpy.ndarray): The ADM maps to normalize.
        mode (str): The normalization mode.
        norm_func (callable): Function to compute normalization factor.

    Returns:
        numpy.ndarray: Normalized ADM maps.
    """
    return maps / norm_func(maps, mode)

def update_adm_model_names(model_names, model_names_to_plot):
    """
    Update ADM model names by excluding the first one or mapping them if provided.

    Parameters:
        model_names (list): Original list of model names.
        model_names_to_plot (dict or None): Mapping of model names to plot.

    Returns:
        list: Updated list of model names.
    """
    if model_names_to_plot is None:
        return model_names[:]
    return [model_names_to_plot[name] for name in model_names[:]]

def initialize_adm_matrices(model_names, num_seeds, num_participants, categories_comb, sampling_num):
    """
    Initialize ADM matrices for each model.

    Parameters:
        model_names (list): List of model names.
        num_seeds (int): Number of seeds per model.
        num_participants (int): Number of participants.
        categories_comb (int): Number of category combinations.
        sampling_num (int): Number of bootstrap samples.

    Returns:
        dict: Nested dictionary containing ADM matrices.
    """
    return {
        model: {
            'adms': np.zeros((num_seeds, categories_comb)),
            'human_agreement': np.zeros((num_seeds, num_participants)),
            'average_human_agreement': np.zeros(num_seeds),
            'bootstrap_human_agreement': np.zeros((num_seeds, sampling_num))
        }
        for model in model_names
    }

def compute_adm_noise_ceiling(human_adms, rsa_metric):
    """
    Compute the noise ceiling from individual ADM maps.

    Parameters:
        human_adms (numpy.ndarray): Individual human ADM maps.
        rsa_metric (str): Metric used for RSA calculation.

    Returns:
        tuple: Noise ceiling value and additional info.
    """
    return get_noise_ceiling_from_individual_adms(human_adms, rsa_metric)

def calculate_adm_alignment(adm1, adm2, rsa_metric):
    """
    Calculate alignment between two ADM vectors.

    Parameters:
        adm1 (numpy.ndarray): First ADM vector.
        adm2 (numpy.ndarray): Second ADM vector.
        rsa_metric (str): Metric used for alignment calculation.

    Returns:
        float: Alignment score.
    """
    return get_2vectors_alignment(adm1, adm2, metric=rsa_metric)

def create_average_adm(raw_maps, create_adm_metric):
    """
    Create an average ADM map from raw maps using a specified metric.

    Parameters:
        raw_maps (numpy.ndarray): Raw accuracy maps.
        create_adm_metric (str): Metric used to create ADM.

    Returns:
        tuple: Average ADM and additional info.
    """
    return create_adm_by_metric(np.nanmean(raw_maps, axis=0), metric=create_adm_metric)

def compute_individual_adms(raw_individual_maps, create_adm_metric):
    """
    Compute individual human ADM maps.

    Parameters:
        raw_individual_maps (numpy.ndarray): Individual participants' normalized accuracy maps.
        create_adm_metric (str): Metric used to create ADM.

    Returns:
        numpy.ndarray: Array of individual ADM maps.
    """
    human_adms = np.zeros((raw_individual_maps.shape[0], int(raw_individual_maps.shape[1] * (raw_individual_maps.shape[1] - 1) / 2)))
    for i, individual_map in enumerate(raw_individual_maps):
        human_adms[i], _ = create_adm_by_metric(individual_map, metric=create_adm_metric)
    return human_adms

def compute_model_adms(raw_seeds_maps, create_adm_metric, categories_num, updated_model_names ):
    """
    Compute ADM maps for each model.

    Parameters:
        raw_seeds_maps (numpy.ndarray): Seed models' normalized accuracy maps.
        create_adm_metric (str): Metric used to create ADM.
        categories_num (int): Number of categories.

    Returns:
        dict: Dictionary of model ADMs.
    """
    categories_comb = int(categories_num * (categories_num - 1) / 2)
    # import pdb; pdb.set_trace()
    model_adms = {model_name: np.zeros((raw_seeds_maps.shape[1], categories_comb)) for model_name in updated_model_names}
    average_model_adms = {model_name: np.zeros(categories_comb) for model_name in updated_model_names}

    for model_idx, seeds_model_maps in enumerate(raw_seeds_maps):
        model_name = list(updated_model_names)[model_idx]
        for j, seed_model_map in enumerate(seeds_model_maps):
            normalized_seed_map = seed_model_map / np.max(seed_model_map, axis=0)
            model_adms[model_name][j], _ = create_adm_by_metric(seed_model_map, metric=create_adm_metric)
        average_model_adms[model_name], _ = create_average_adm(seeds_model_maps, create_adm_metric)

    return model_adms, average_model_adms

def perform_adm_bootstrapping(raw_individual_maps, create_adm_metric, sampling_num):
    """
    Perform bootstrapping on human ADM maps.

    Parameters:
        raw_individual_maps (numpy.ndarray): Individual participants' normalized accuracy maps.
        create_adm_metric (str): Metric used to create ADM.
        sampling_num (int): Number of bootstrap samples.

    Returns:
        list: List of bootstrapped ADM maps.
    """
    bootstrap_human_adms = []
    for _ in tqdm(range(sampling_num), desc='Bootstrapping ADM'):
        resampled_maps = resample(raw_individual_maps, n_samples=raw_individual_maps.shape[0])
        bootstrap_adm, _ = create_adm_by_metric(np.nanmean(resampled_maps, axis=0), metric=create_adm_metric)
        bootstrap_human_adms.append(bootstrap_adm)
    return bootstrap_human_adms

def compute_model_human_agreement(model_adms, human_adms, rsa_metric):
    """
    Compute agreement between model ADMs and human ADMs.

    Parameters:
        model_adms (dict): Dictionary of model ADM maps.
        human_adms (numpy.ndarray): Individual human ADM maps.
        rsa_metric (str): Metric used for RSA calculation.

    Returns:
        dict: Dictionary of model-human ADM agreement matrices.
    """
    model_human_agreement = {model_name: np.zeros((adms.shape[0], human_adms.shape[0])) for model_name, adms in model_adms.items()}

    for model_name, adms in model_adms.items():
        for i, model_adm in enumerate(adms):
            for j, human_adm in enumerate(human_adms):
                alignment = calculate_adm_alignment(model_adm, human_adm, rsa_metric)
                model_human_agreement[model_name][i, j] = alignment

    return model_human_agreement

def compute_model_average_human_agreement(model_adms, average_human_adm, bootstrap_human_adms, rsa_metric, sampling_num):
    """
    Compute agreement between model ADMs and average human ADM maps.

    Parameters:
        model_adms (dict): Dictionary of model ADM maps.
        average_human_adm (numpy.ndarray): Average human ADM map.
        bootstrap_human_adms (list): List of bootstrapped human ADM maps.
        rsa_metric (str): Metric used for RSA calculation.
        sampling_num (int): Number of bootstrap samples.

    Returns:
        dict: Dictionary of model-average human ADM agreement matrices.
    """
    model_average_human_agreement = {model_name: np.zeros(model_adms[model_name].shape[0]) for model_name in model_adms.keys()}
    model_bootstrap_human_agreement = {model_name: np.zeros((model_adms[model_name].shape[0], sampling_num)) for model_name in model_adms.keys()}

    for model_name, adms in model_adms.items():
        for i, model_adm in enumerate(adms):
            # Alignment with average human ADM
            alignment = calculate_adm_alignment(model_adm, average_human_adm, rsa_metric)
            model_average_human_agreement[model_name][i] = alignment

            # Alignment with bootstrapped human ADMs
            for sample_id, bootstrap_adm in enumerate(bootstrap_human_adms):
                mask = ~np.isnan(model_adm) & ~np.isnan(bootstrap_adm)
                adm_alignment = calculate_adm_alignment(model_adm[mask], bootstrap_adm[mask], rsa_metric)
                model_bootstrap_human_agreement[model_name][i, sample_id] = adm_alignment

    return model_average_human_agreement, model_bootstrap_human_agreement

def collect_adm_alignment_data(adm_alignment_data, comparison_mode, alignment_values, model_names):
    """
    Aggregate ADM alignment data with corresponding labels and metadata.

    Parameters:
        adm_alignment_data (list): List to store aggregated data.
        comparison_mode (str): Current comparison mode.
        alignment_values (numpy.ndarray): Alignment data to collect.
        model_names (list): List of model names corresponding to the alignment data.

    Returns:
        None
    """
    for model_name, alignment in zip(model_names, alignment_values):
        adm_alignment_data.append([model_name, alignment, comparison_mode])

def handle_adm_comparison_modes(model_bootstrap_human_agreement, comparison_modes, adm_alignment_data, rsa_metric):
    """
    Handle specified ADM comparison modes and collect alignment data accordingly.

    Parameters:
        model_bootstrap_human_agreement (dict): Model-bootstrap human ADM agreement matrices.
        comparison_modes (list): Modes for comparison.
        adm_alignment_data (list): List to store aggregated data.
        rsa_metric (str): Metric used for RSA calculation.

    Returns:
        None
    """
    if 'individual_model_vs_average_human' in comparison_modes:
        for model_name, agreement_matrix in model_bootstrap_human_agreement.items():
            normalized_alignment = calculate_normal_alignment(
                agreement_matrix,
                mode='mean',  # Assuming alignment_mode is 'mean' as per original function
                axis=0
            ).flatten()
            collect_adm_alignment_data(
                adm_alignment_data,
                'individual_model_vs_average_human_bootstrap',
                normalized_alignment,
                [model_name] * len(normalized_alignment)
            )
    else:
        raise ValueError(f"Invalid comparison mode: {comparison_modes}")

def create_adm_dataframe_and_analyze(adm_alignment_data, columns, rsa_metric):
    """
    Create a DataFrame from ADM alignment data and perform statistical analysis.

    Parameters:
        adm_alignment_data (list): Aggregated ADM alignment data.
        columns (list): Column names for the resulting DataFrame.
        rsa_metric (str): Metric used for RSA calculation.

    Returns:
        tuple: DataFrame and statistical significance results.
    """

    df_adm_agreement = pd.DataFrame(adm_alignment_data, columns=columns)
    significance = perform_permutation_statistical_analysis(
        df_adm_agreement,
        rsa_metric,
        num_permutations=10000,
        permutation_type='independent'
    )
    return df_adm_agreement, significance




# -------------------------------------------------------------------------
# Noise ceiling calculation from individual participants' 5x5 acc maps
# -------------------------------------------------------------------------
def get_noise_ceiling_from_individual_acc_maps(individual_participants_5x5_acc_maps, metric):
    """
    Compute the noise ceiling and its upper bound from 5x5 accuracy maps of individual participants.
    
    Args:
        individual_participants_5x5_acc_maps (list or np.array): 
            A list/array of shape (num_participants, num_conditions, 5, 5).
        metric (str): 
            The similarity metric to use (e.g., 'cosine', 'pearsonr', etc.).
            
    Returns:
        tuple: (noise_ceiling, noise_ceiling_upper)
    """
    # Convert the list of participant maps into a NumPy array
    acc_maps_array = np.array(individual_participants_5x5_acc_maps)
    num_participants = acc_maps_array.shape[0]
    
    # Precompute mean accuracy maps, excluding each participant in turn
    mean_acc_maps_excl = np.array([
        np.mean(np.delete(acc_maps_array, idx, axis=0), axis=0) 
        for idx in range(num_participants)
    ])  # shape: (num_participants, num_conditions, 5, 5)

    # Lists to store participant-wise similarities
    nc_similarity_list = []
    nc_upper_similarity_list = []

    # Calculate similarities for each participant
    for p_idx in range(num_participants):
        current_p_maps = acc_maps_array[p_idx]
        excl_mean_maps  = mean_acc_maps_excl[p_idx]

        # Category-wise similarities
        nc_scores = [
            get_2vectors_alignment(
                current_p_maps[k].flatten(), 
                excl_mean_maps[k].flatten(), 
                metric
            ) 
            for k in range(current_p_maps.shape[0])
        ]
        nc_upper_scores = [
            get_2vectors_alignment(
                current_p_maps[k].flatten(), 
                np.mean(acc_maps_array[:, k], axis=0).flatten(), 
                metric
            ) 
            for k in range(current_p_maps.shape[0])
        ]

        # Average out the similarities for this participant
        nc_similarity_list.append(np.mean(nc_scores))
        nc_upper_similarity_list.append(np.mean(nc_upper_scores))

    # Final noise ceiling and upper noise ceiling
    noise_ceiling = np.mean(nc_similarity_list)
    noise_ceiling_upper = np.mean(nc_upper_similarity_list)
    return noise_ceiling, noise_ceiling_upper


# -------------------------------------------------------------------------
# Noise ceiling calculation from individual participants' ADMs
# -------------------------------------------------------------------------
def get_noise_ceiling_from_individual_adms(model_adm, metric):
    """
    Compute noise ceiling and its upper bound from participants' ADM (Representational Dissimilarity Matrices).
    
    Args:
        model_adm (list or np.array): 
            A list/array of participant ADMs, shape (num_participants, num_conditions, num_conditions) or similar.
        metric (str): 
            The similarity metric to use.

    Returns:
        tuple: (noise_ceiling, noise_ceiling_upper)
    """
    # Ensure input is a NumPy array
    adm_results_array = np.array(model_adm)
    num_participants = adm_results_array.shape[0]

    # Precompute mean ADM excluding each participant
    mean_adm_results_excl = np.array([
        np.mean(np.delete(adm_results_array, idx, axis=0), axis=0) 
        for idx in range(num_participants)
    ])
    overall_mean_adm = np.mean(adm_results_array, axis=0)

    # Lists for participant-wise noise ceiling metrics
    nc_similarity_list = []
    nc_upper_similarity_list = []

    # Calculate similarity for each participant
    for p_idx in range(num_participants):
        participant_adm = adm_results_array[p_idx]
        excl_participant_adm = mean_adm_results_excl[p_idx]

        # Compare participant vs. exclude-mean and participant vs. overall mean
        noise_ceiling_scores = [
            get_2vectors_alignment(
                participant_adm.flatten(), 
                excl_participant_adm.flatten(), 
                metric
            )
        ]
        noise_ceiling_upper_scores = [
            get_2vectors_alignment(
                overall_mean_adm, 
                excl_participant_adm.flatten(), 
                metric
            )
        ]

        nc_similarity_list.append(np.mean(noise_ceiling_scores))
        nc_upper_similarity_list.append(np.mean(noise_ceiling_upper_scores))

    # Compute final noise ceiling values
    noise_ceiling = np.mean(nc_similarity_list)
    noise_ceiling_upper = np.mean(nc_upper_similarity_list)
    return noise_ceiling, noise_ceiling_upper


# -------------------------------------------------------------------------
# Calculate a normal alignment measure with different aggregation modes
# -------------------------------------------------------------------------
def calculate_normal_alignment(x, mode='mean', axis=None):
    """
    Aggregate a given array using different strategies (mean, max, min).

    Args:
        x (np.array): Array to be aggregated.
        mode (str): The aggregation mode ('mean', 'max', 'min').
        axis (int or None): Axis over which to aggregate.
    
    Returns:
        float or np.array: Result of the aggregation.
    """
    if mode == 'mean':
        return np.nanmean(x, axis=axis)
    elif mode == 'mean':  # This is redundant, but kept as in original code
        return np.nanmean(x, axis=axis)
    elif mode == 'max':
        return np.max(x, axis=axis)
    elif mode == 'min':
        return np.min(x, axis=axis)
    else:
        raise ValueError(f"Invalid mode: {mode} or {axis}")


# -------------------------------------------------------------------------
# A simple statistic function (difference of means)
# -------------------------------------------------------------------------
def statistic(x, y, axis):
    """
    Compute the difference of means between two arrays along a specified axis.
    
    Args:
        x (np.array): First array.
        y (np.array): Second array.
        axis (int): Axis along which the means are computed.

    Returns:
        float or np.array: Difference in means.
    """
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)


# -------------------------------------------------------------------------
# Create an ADM (distance matrix) by a chosen metric
# -------------------------------------------------------------------------
def create_adm_by_metric(pred_activations, metric="correlation_distance", ranked=False):
    """
    Create a Representational Dissimilarity Matrix (ADM) from predicted activations using a chosen metric.

    Args:
        pred_activations (list or np.array): Activation data across conditions.
        metric (str): Distance metric to use (e.g. 'correlation_distance', 'l2', 'mse', etc.).
        ranked (bool): Whether to return the ranked version of the distance matrix.

    Returns:
        np.array or tuple: 
            - If ranked=True: (squareform of ranked distances, ranked distances).
            - Otherwise: (flat_triangle_adm, squareform of flat_triangle_adm).
    """
    # Flatten activations for pairwise distance
    reshaped_activations = np.reshape(pred_activations, [len(pred_activations), -1])

    # Choose the appropriate pairwise distance operation
    if metric == "spearmanr":
        flat_triangle_adm = pdist(
            reshaped_activations, 
            lambda u, v: 1 - spearmanr_pdist(np.array(u).flatten(), np.array(v).flatten())
        )
    elif metric == "pearsonr":
        flat_triangle_adm = pdist(
            reshaped_activations, 
            lambda u, v: 1 - pearsonr_pdist(np.array(u).flatten(), np.array(v).flatten())
        )
    elif metric == "cosine":
        flat_triangle_adm = pdist(
            reshaped_activations, 
            lambda u, v: 1 - cosine(np.array(u).flatten(), np.array(v).flatten())
        )
    else:
        raise NotImplementedError(f"No metric for {metric}")

    # Rank the distances if required
    ranked_adm = scipy.stats.rankdata(flat_triangle_adm)

    # Return according to the 'ranked' option
    if ranked:
        return squareform(ranked_adm), ranked_adm
    else:
        return flat_triangle_adm, squareform(flat_triangle_adm)


# -------------------------------------------------------------------------
# Generic function to get similarity between two 1D arrays
# -------------------------------------------------------------------------
def get_2vectors_alignment(arr1, arr2, metric="cosine"):
    """
    Compute a similarity/alignment measure between two 1D arrays using a chosen metric.

    Args:
        arr1 (np.array or list): First array.
        arr2 (np.array or list): Second array.
        metric (str): The similarity/distance metric to use.
    
    Returns:
        float: Computed similarity/distance value.
    """
    assert np.array(arr1).shape == np.array(arr2).shape, (
        f"Shape mismatch: {np.array(arr1).shape} vs {np.array(arr2).shape}"
    )

    if metric == "pearsonr":
        similarity = pearsonr_pdist(arr1, arr2)
    elif metric == "spearmanr":
        similarity = spearmanr_pdist(arr1, arr2)
    elif metric == "cosine":
        similarity = 1 - cosine(arr1, arr2)
    else:
        raise NotImplementedError(f"No metric for {metric}")

    return similarity


# -------------------------------------------------------------------------
# Pearson correlation helper
# -------------------------------------------------------------------------
def pearsonr_pdist(u, v):
    """
    Compute the Pearson correlation coefficient between two 1D arrays (no NaN filtering).
    
    Args:
        u (np.array): First array.
        v (np.array): Second array.
    
    Returns:
        float: Pearson correlation coefficient.
    """
    r, _ = scipy.stats.pearsonr(u, v)
    return r


# -------------------------------------------------------------------------
# Spearman correlation helper
# -------------------------------------------------------------------------
def spearmanr_pdist(u, v):
    """
    Compute the Spearman correlation coefficient between two 1D arrays.
    
    Args:
        u (np.array): First array.
        v (np.array): Second array.
    
    Returns:
        float: Spearman correlation coefficient.
    """
    rs, _ = scipy.stats.spearmanr(u, v)
    return rs


# -------------------------------------------------------------------------
# L2 distance between two 1D arrays
# -------------------------------------------------------------------------
def l2_distance(p, q):
    """
    Compute the Euclidean distance (L2) between two 1D arrays.
    
    Args:
        p (np.array): First array.
        q (np.array): Second array.
    
    Returns:
        float: L2 distance.
    """
    return np.linalg.norm(p - q)


# -------------------------------------------------------------------------
# Map normalization utility
# -------------------------------------------------------------------------
def map_norm_func(x, map_norm_mode):
    """
    Normalize a map either by its maximum value, by sum, or not at all.

    Args:
        x (np.array): 2D/1D array to be normalized.
        map_norm_mode (str): One of {'max', 'sum', 'original'}.

    Returns:
        float: The denominator used for normalization.
    """
    if map_norm_mode == 'max':
        denom = np.max(x)
    elif map_norm_mode == 'sum':
        denom = np.sum(x)
    elif map_norm_mode == 'original':
        denom = 1
    else:
        raise NotImplementedError(f"{map_norm_mode} is not a supported normalization mode.")
    return denom

