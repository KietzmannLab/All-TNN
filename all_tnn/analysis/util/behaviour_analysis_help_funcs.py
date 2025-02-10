import os
import numpy as np
import matplotlib.pyplot as plt

from all_tnn.analysis.util.annotation_help_funcs import COCO_ID_NAME_MAP, COCO_NAME_ID_MAP, BEHAVIOUR_CATEGORIES_ORDERED_COCO_ID, CATEGORY_NAMES
from all_tnn.analysis.util.behaviour_mat_data_func import  get_all_participants_results_to_acc_maps_dict

def get_behaviour_data(
            model_names,
            test_epochs,
            map_norm_mode,
            alphas = None, 
            seeds_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            result_source_dir="./save_dir/", 
            size_factor=1,
            experiment_categories_order=CATEGORY_NAMES, 
            SUBJ_START=0,
            SUBJ_END=30,
            decimals=2,
            model_epoch_map = None,
            analysis_datasource_type =''
            ):

    individual_participants_5x5_acc_maps_array, raw_individual_participants_5x5_acc_maps_array = get_individual_participants_5x5_acc_maps(result_source_dir, experiment_categories_order, map_norm_mode=map_norm_mode,
                                                                                                            SUBJ_START=SUBJ_START, SUBJ_END=SUBJ_END)
    seeds_model_data_normed, seeds_model_data, model_data = get_acc_maps(
                                            result_source_dir, experiment_categories_order, 
                                            map_norm_mode=map_norm_mode,
                                            epochs=test_epochs,  seeds_list= seeds_list,
                                            model_names= model_names,
                                            alphas=alphas if alphas else [None]*len(model_names),
                                            size_factor=size_factor,
                                            model_epoch_map = model_epoch_map,
                                            analysis_datasource_type =analysis_datasource_type)
                                  

    return {
        'individual_participants_5x5_acc_maps_array': np.around(individual_participants_5x5_acc_maps_array, decimals),
        'raw_individual_participants_5x5_acc_maps_array': np.around(raw_individual_participants_5x5_acc_maps_array, decimals),
        'seeds_model_data_normed': {model_name: np.around(seed_model_data, decimals) for model_name, seed_model_data in seeds_model_data_normed.items()},
        'seeds_model_data': {model_name: np.around(seeds_model_data, decimals) for model_name, seeds_model_data in seeds_model_data.items()},
        'model_data': {model_name: np.around(model_data, decimals) for model_name, model_data in model_data.items()},
    }


def get_acc_maps(
    result_source_dir,
    experiment_categories_order,
    epochs,
    map_norm_mode,
    subj_start=0,
    subj_end=30,
    seeds_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    model_names = ["CNN", "LCN", "Alpha_0.01", 'Alpha_0.1', 'Alpha_1.0', 'Alpha_5.0', 'Alpha_10.0', 'Alpha_20.0', 'Alpha_50.0', 'Alpha_100.0'],
    alphas=[None, 0.0, 0.01, 0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0], # Corresponding to model_names
    size_factor=1,
    model_epoch_map = None,
    analysis_datasource_type ='',
):
    """Retrieve object distribution maps, positional uncertainties for 16 categories, and 5x5 accuracy maps data for various systems.
    Args:
        result_source_dir (str): Base directory of all source data.
        experiment_categories_order (list(int)): Category names in behavioral experimental order.
        epoch (int, optional): Testing epoch. 
        subj_start (int, optional): First participant's ID. Defaults to 0.
        subj_end (int, optional): Last participant's ID. Defaults to 30.
    Returns:
        dict: A dictionary containing object distribution maps, positional uncertainties, and 5x5 accuracy maps data for different models.
    """
    # Initialize data structures
    model_data = {model_name: [] for model_name in model_names}
    seeds_model_data = {model_name: [] for model_name in model_names}
    seeds_model_data_normed = {model_name: [] for model_name in model_names}

    # Convert seeds_list to list if it's a single integer
    if isinstance(seeds_list, int):
        seeds_list = [seeds_list]

    # Load and normalize model data
    for model_name, epoch in zip(model_names, epochs):
        for seed in seeds_list:
            alpha = alphas[model_names.index(model_name)]

            model_accuracy_maps = get_model_accuracy_maps(result_source_dir+f'/acc_maps_saved/scale_factor_{size_factor}_acc_maps/', model_name, seed, epoch, alpha, model_epoch_map, size_factor) #* Model accuracy maps saved in acc_maps_saved
                                                            
            normed_model_accuracy_maps = np.array([map_/map_norm_func(map_,map_norm_mode) for map_ in model_accuracy_maps]) # normed
            seeds_model_data[model_name].append(model_accuracy_maps)
            seeds_model_data_normed[model_name].append(normed_model_accuracy_maps)

    
    for model, data in seeds_model_data.items():
        if data:
            model_data[model] = np.mean(data, axis=0)
    

    return seeds_model_data_normed,seeds_model_data, model_data


##################################################################################################
## Utils
##################################################################################################
def get_individual_participants_5x5_acc_maps(result_source_dir, experiment_categories_order, map_norm_mode, SUBJ_START=0, SUBJ_END=30,):
    """
    Function to get individual participants' 5x5 accuracy maps.

    Parameters:
    result_source_dir : str
        Directory where the result sources are located.
    experiment_categories_order : list
        List of experiment name orders.
    SUBJ_START : int, optional
        Starting index for subjects.
    SUBJ_END : int, optional
        Ending index for subjects.
    """
    # Set human behavior results path
    human_behavior_results_path = os.path.join(result_source_dir,"position_experiments_results/")

    # Get individual participants accuracy maps
    _, individual_participants_acc_maps = get_human_acc_accuracy_maps_dict(human_behavior_results_path, SUBJ_ID_RANGE=[SUBJ_START, SUBJ_END])
    
    # Processing individual participants' accuracy maps
    individual_participants_5x5_acc_maps_array = []
    raw_individual_participants_5x5_acc_maps_array = []
    for subj in range(SUBJ_START, SUBJ_END):
        participant_categorical_acc_matrices = []
        raw_participant_categorical_acc_matrices = []
        for cat in experiment_categories_order:
            participant_categorical_acc_matrices.append(individual_participants_acc_maps[cat][subj] / map_norm_func(individual_participants_acc_maps[cat][subj], map_norm_mode))
            raw_participant_categorical_acc_matrices.append(individual_participants_acc_maps[cat][subj])
        individual_participants_5x5_acc_maps_array.append(np.array(participant_categorical_acc_matrices))
        raw_individual_participants_5x5_acc_maps_array.append(np.array(raw_participant_categorical_acc_matrices))

    individual_participants_5x5_acc_maps_array = np.array(individual_participants_5x5_acc_maps_array)
    raw_individual_participants_5x5_acc_maps_array = np.array(raw_individual_participants_5x5_acc_maps_array)

    return individual_participants_5x5_acc_maps_array, raw_individual_participants_5x5_acc_maps_array


def get_human_acc_accuracy_maps_dict(human_behavior_results_path, SUBJ_ID_RANGE = [0,31], percentage_range = [0,1]):
    """get human behavior accuracy matrices from path

    Args:
        human_behavior_results_path (str): path
        SUBJ_ID_RANGE (list, optional): _description_. Defaults to [0,31].
        normed (str, optional): "normed" or "weighted normed" or ""(Do nothig)  . Defaults to "".
        percentage_range (list, optional):  Use 100% for now. Defaults to [0,1].
    
    Return:
        dicts: cat-to-average acc map in dict, cat-to-individual acc maps in dict
    """
    human_behavior_results_path_list = os.listdir(human_behavior_results_path)
    human_behavior_results_path_list = [os.path.join(human_behavior_results_path,i) for i in human_behavior_results_path_list]
    len_results_mat =len(human_behavior_results_path_list)
    
    print(f"- Sorting {len_results_mat} paticipants in Name order")
    human_behavior_results_path_list = sorted(human_behavior_results_path_list, key=lambda pstr: int(pstr.split("_")[-2].lstrip("subj")))[SUBJ_ID_RANGE[0]:SUBJ_ID_RANGE[1]]
    
    print(f"- Loading all Human acc results to matrix data...\n")
    average_participants_acc_maps, individual_participants_acc_maps = choose_mode_to_get_human_accuracy_maps(human_behavior_results_path_list, 
                                                                if_cat_name_in_number = False, threshold = 0.1, 
                                                                percentage_range = percentage_range)

    return average_participants_acc_maps, individual_participants_acc_maps

def choose_mode_to_get_human_accuracy_maps( path_list, if_cat_name_in_number = False,threshold = 0.1, percentage_range=[0,1]):
    """Get accuracy matrices accoding to categories, in 4 different nomalization modes :  Normalized, or filtered with threshold,  or both,  or both none

    Args:
        path_list (list): path list of results(accuracy matrices)
        if_cat_name_in_number (bool, optional): whether indexing categories names in number? or string. Defaults to False (in str).
        threshold (float, optional): Recognition Performance below this threshold (under chance level) will be considered as a failure of recognition. Defaults to 0.1.
        percentage_range (list, optional): how many percentage of Data  want to use, [0,1] means use all collected data from start 0% to end 100%. Defaults to [0,1].

    Returns:
        average_accuracy_maps_across_participants(dict):  a dict store cat to (16,5,5) the average accuracy maps of all paticipants in 16 categoies
        individual_accuracy_maps(dict):  a dict store particiapnts_id & cat to (16,30,5,5),  contains accuracy maps on each paticipant in 16 categoies
    """
    if not type(path_list) == list:
        path_list = [path_list] # If only 1 categories
        
    average_accuracy_maps_across_participants, individual_accuracy_maps = get_all_participants_results_to_acc_maps_dict(path_list, if_cat_name_in_number = if_cat_name_in_number, percentage_range=percentage_range) # True
    return average_accuracy_maps_across_participants, individual_accuracy_maps


def from_object_distribution_maps_to_5x5_matrices(object_distribution_list,cat_num=16, map_norm_mode="max"):
    """Input object distributions in scatter for 16 categories in 10x10,
       Return 5x5 object distribution maps
    
    Args:
        object_distribution_list(list):  object distribution maps aross 16 cats, defauly is 100x100 pixels maps
    
    Return:
        np.array: 5x5 object distribution matrices across 16 cats
    """
    data_matrices = []
    for p in range(cat_num):
        object_distribution_maps = object_distribution_list[p]/map_norm_func(object_distribution_list[p], map_norm_mode)
        arr = []
        for i in range(5):
            for j in range(5):
                arr.append(np.sum(object_distribution_maps[20*i:20*(i+1), 20*j:20*(j+1)]))
        arr = np.reshape(arr/map_norm_func(arr, map_norm_mode),(5,5))
        data_matrices.append(arr)

    return np.array(data_matrices)


def get_model_accuracy_maps(result_source_dir, model_name, seed, epoch, alpha, model_epoch_map, size_factor=1):
    '''Return 16x5x5 for 16 categories '''

    
    if model_name not in model_epoch_map:
        raise NotImplementedError(f"{model_name} model is not supported")
    if model_name in ["8_neighbours_TNN_alpha_10_lr_0.05", 'shifted_TNN_alpha_10_lr_0.05']:
        seed = 1

    epoch = model_epoch_map[model_name][seed-1]
    model_path = f"{result_source_dir}acc_maps_{model_name}_{size_factor}_model_id_{seed}_ep{epoch}.npy"

    model_accuracy_maps = load_and_normalize_model_data(model_path)
    
    if len(model_accuracy_maps) == 0:
        raise FileNotFoundError('No model accuracy maps loaded.')
    
    return model_accuracy_maps

def map_norm_func(x, map_norm_mode):
    if map_norm_mode == 'max':
        map_denominator_val = np.max(x)
    elif map_norm_mode == 'sum':
        map_denominator_val = np.sum(x)
    elif map_norm_mode == 'original':
        map_denominator_val = 1
    else:
        raise NotImplementedError(f"{map_norm_mode} map_norm_mode is not supported")
    return map_denominator_val
    

def load_and_normalize_model_data(model_path):
    
    model_data = np.load(model_path)

    # Replace 0 with the small constant to avoid division problem
    mask = (model_data == 0) | np.isnan(model_data) 
    model_data[mask] = 1e-10

    return model_data

def plot_average_human_maps(human_acc_maps, save_fig_path=None, show_plot=False):
    normed_human_acc_maps = np.array([maps/np.max(maps) for maps in human_acc_maps])
    avg_human_acc_maps = np.mean(normed_human_acc_maps, axis=0)


    # The 16 category labels in the desired order:
    categories = [
        'motorcycle', 'train', 'bus', 'airplane',
        'kite', 'cat', 'pizza', 'broccoli',
        'bear', 'elephant', 'zebra', 'giraffe',
        'laptop', 'scissors', 'refrigerator', 'toilet'
    ]

    # Create a 4x4 grid of subplots
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(8,8))

    # Plot each 5x5 map in the proper position
    for i, ax in enumerate(axes.flat):
        # Show the data with a consistent colormap and normalization
        im = ax.imshow(avg_human_acc_maps[i]/np.max(avg_human_acc_maps[i]), cmap='magma', vmin=0, vmax=1)
        
        # Add title (category name) above each subplot
        ax.set_title(categories[i])
        
        # Remove axis ticks for a cleaner look
        ax.axis('off')

    # Make space for and add a single colorbar for the entire figure
    fig.subplots_adjust(right=0.85)  # Shrink main plot area to fit colorbar
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Classification accuracy [normalized]')

    plt.suptitle('Average human accuracy maps', fontsize=16)
    if save_fig_path is not None:
        print(f"Saving figure to {save_fig_path}")
        plt.savefig(save_fig_path, bbox_inches='tight')
    if show_plot:
        plt.show()