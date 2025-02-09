from mat4py import loadmat
import numpy as np


def get_all_participants_results_to_acc_maps_dict(path_list, if_cat_name_in_number = True, percentage_range = [0,1]):
    """Get category-accuracy matrix from mat-format results for all participants

    Args:
        path_list (list): path list of results
        if_cat_name_in_number (bool, optional): if categories names in number? or string. Defaults to True.
        percentage_range (list, optional): how many percentage of Data you want to use, [0,1] means use all collected data from start to end. Defaults to [0,1].

    Returns:
        all_cat_name_acc_matrix(np.array): mean accuracy matrix of all paticipants in 16 categoies
        list_all_cat_name_to_acc_matrix(list(np.array,...,np.array)): list contains accuracy matrix of each paticipant in 16 categoies
    """

    average_accuracy_maps_across_participants = {} 
    individual_accuracy_maps = {}

    # Traverse, path_list = ["./position_experiment_subj9_run0.mat","./position_experiment_subj9_run2.mat",...,"./position_experiment_subj30_run0.mat" ]
    for path in path_list:
        data = loadmat(path)
        cat_name_to_acc_matrix = get_mat_data_to_acc_maps_dict(data, percentage_range)

        for cat_name in cat_name_to_acc_matrix:

            if if_cat_name_in_number:
                if cat_name.split()[1] in average_accuracy_maps_across_participants:
                    average_accuracy_maps_across_participants[cat_name.split()[1]] = np.add(cat_name_to_acc_matrix[cat_name],average_accuracy_maps_across_participants[cat_name.split()[1] ])
                    individual_accuracy_maps [cat_name.split()[1]].append(cat_name_to_acc_matrix[cat_name])                
                else:
                    average_accuracy_maps_across_participants[cat_name.split()[1]]  = cat_name_to_acc_matrix[cat_name]
                    individual_accuracy_maps [cat_name.split()[1]] = [cat_name_to_acc_matrix[cat_name]]

            else:
                if cat_name in average_accuracy_maps_across_participants:
                    average_accuracy_maps_across_participants[cat_name] = np.add(cat_name_to_acc_matrix[cat_name],average_accuracy_maps_across_participants[cat_name])
                    individual_accuracy_maps [cat_name].append(cat_name_to_acc_matrix[cat_name])  
                else:
                    average_accuracy_maps_across_participants[cat_name]  = cat_name_to_acc_matrix[cat_name]
                    individual_accuracy_maps [cat_name] = [cat_name_to_acc_matrix[cat_name]]

    return average_accuracy_maps_across_participants, individual_accuracy_maps
    

##################################################################################################
## Utils
##################################################################################################
def get_id_cat_name_mapping_dict_from_mat_file(data):
    """
    Get id-name mapping dict from mat format dataset

    Args:
        data: mat-format data, which save collect data from participants behaviour experiment

    Returns:
        id_to_cat_name[dict]: id mapping to name 
    """    

    cat_name_list = data["cfg"]["categories"]['name']
    id_to_cat_name= {}
    for id in range(len(cat_name_list)):
        id_to_cat_name[id+1] = cat_name_list[id]

    return id_to_cat_name

def get_acc_matrices_from_mat(data, percentage_range = [0,1]):
    """
    Get category-location to accuracy matrix dict({[category][location] : accuracy}) from mat-format data

    Args:
        data({[category][location] : accuracy}): mat-format data
        percentage_range (list, optional): how many percentage of Data  want to use, [0,1] means use all collected data from start 0% to end 100%. Defaults to [0,1].

    Returns:
        cat_loc_mean_acc_dict (dict): {[category][location] : accuracy}
    """ 

    original_trails_data = data["dat"]["random_trials"]
    trails_data = original_trails_data[int(len(original_trails_data)*percentage_range[0]):int(len(original_trails_data)*percentage_range[1])]
    categories_list = [i[0] for i in trails_data]
    exempler_list = [i[1] for i in trails_data]
    vert_pos_list = [i[2] for i in trails_data]
    horz_pos_list = [i[3] for i in trails_data]

    accuracy = [i[0] for i in data["dat"]["accuracy"]]

    # Create dict mapping category locations to accuracy
    cat_loc_acc_dict = {}
    for i in range(len(trails_data)):
        cat = categories_list[i]
        cat_loc_acc_dict[cat] = {}

    for i in range(len(trails_data)):
        cat = categories_list[i]
        loc = (vert_pos_list[i],horz_pos_list[i])
        if loc not in cat_loc_acc_dict[cat]:
            cat_loc_acc_dict[cat][loc] = [accuracy[i]]
        else:
            cat_loc_acc_dict[cat][loc].append(accuracy[i])

    # Get mean acc of loc for every cat
    cat_loc_mean_acc_dict= {}
    for i in range(len(trails_data)):
        cat = categories_list[i]
        cat_loc_mean_acc_dict[cat] = {}


    for cat in cat_loc_acc_dict:
        for loc in cat_loc_acc_dict[cat]:
            cat_loc_mean_acc_dict[cat][loc] = np.mean(cat_loc_acc_dict[cat][loc])

    return cat_loc_mean_acc_dict


def get_mat_data_to_acc_maps_dict(data, percentage_range = [0,1]):
    """Get category to accuracy matrix dict from mat-format data

    Args:
        data: mat-format data
        percentage_range (list, optional): how many percentage of Data you want to use, [0,1] means use all collected data from start to end. Defaults to [0,1].

    Returns:
        cat_name_to_acc_matrix_dict (np.array):  category to accuracy matrix matrix dict
    """

    
    id_to_cat_name = get_id_cat_name_mapping_dict_from_mat_file(data)
    cat_name_loc_mean_acc_dict = {id_to_cat_name[k]:v for k,v in get_acc_matrices_from_mat(data, percentage_range).items()}
    cat_name_to_acc_matrix_dict= {id_to_cat_name[k]:cat_loc_mean_acc_dict2plot(cat_name_loc_mean_acc_dict,id_to_cat_name[k]) for k,v in get_acc_matrices_from_mat(data).items()}

    return cat_name_to_acc_matrix_dict


def cat_loc_mean_acc_dict2plot(cat_loc_mean_acc_dict, cat, loc_num = 5):
    """
    Convert category-location-accuracy 2-layer dict to a loc_num x loc_num matrix

    Args:
        cat_loc_mean_acc_dict (dict): category-location-accuracy 2-layer mapping dict : {[category][location] : accuracy}
        cat (str): categories name
        loc_num (int, optional): gird side length . Defaults to 5 (5x5 grids).

    Returns:
        acc_matrix: accuracy matrix
    """    
    acc_matrix = np.empty((loc_num,loc_num))
    for loc in cat_loc_mean_acc_dict[cat]:
        acc_matrix[loc[0]-1][loc[1]-1] = cat_loc_mean_acc_dict[cat][loc]

    return acc_matrix
