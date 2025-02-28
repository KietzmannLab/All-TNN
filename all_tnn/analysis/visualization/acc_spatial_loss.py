import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from all_tnn.analysis.util.convert_dict2h5 import read_h52dict

def generate_analysis_df(
    base_src_dir_path,
    MODEL_NAMES,
    seeds_range,
    models_epochs_dict,
    MODEL_NAMES_TO_PLOT,
    model_results_dict_filename = 'multi_models_neural_dict.h5',
):
    """
    Generate a pandas DataFrame containing model accuracies, losses, top-k accuracies,
    spatial smoothness, and categorical losses from pickled/h5 data files.

    Parameters
    ----------
    base_src_dir_path : str
        The base directory path to save or load data from.
    MODEL_NAMES : list
        A list of model names to iterate over.
    seeds_range : list or range
        A list or range of seed values.
    models_epochs_dict : dict
        A dictionary mapping each model_name to a list of epoch directory names,
        indexed by (seed-1).
    MODEL_NAMES_TO_PLOT : dict
        A dictionary mapping each model_name to a label (string) to be used in the DataFrame.
    results_dict : str
        Results file in dict

    Returns
    -------
    df : pandas.DataFrame
        A DataFrame with columns:
        [ "Model",
          "Accuracy",
          "ecoset Loss",
          "ecoset Topk Accuracy",
          "Spatial Smoothness",
          "ecoset Categorical Loss" ]
    """

    # Create the base directory if it doesn't exist
    os.makedirs(base_src_dir_path, exist_ok=True)

    # Initialize dictionaries to store data
    model_acc_dict = defaultdict(list)
    model_test_topk_accuracy = defaultdict(list)
    model_test_loss_dict = defaultdict(list)
    model_mean_cosdist_dict = defaultdict(list)

    # Loop over each model and seed to collect data
    for model_name in MODEL_NAMES:
        print(f"Loading model_name = {model_name}")
        for seed in seeds_range:
            print(f"Loading seed {seed}/5", end="\r")
            src_path = os.path.join(
                base_src_dir_path,
                f'seed{seed}/{model_results_dict_filename}'
            )
            model_results_dict = read_h52dict(src_path)
            if 'TNN_simclr' == model_name:
                model_name = 'TNN_simclr_finetune' #* only finetuned model is used for analysis

            # print(f"key: model_results_dict.keys() = {model_results_dict.keys()}")
            model_acc_dict[model_name].append(model_results_dict[model_name]['ecoset_test_accuracy'])
            model_test_loss_dict[model_name].append(model_results_dict[model_name]['ecoset_test_loss'])
            model_test_topk_accuracy[model_name].append(model_results_dict[model_name]['ecoset_test_topk_accuracy'])
            model_mean_cosdist_dict[model_name].append(np.sum(model_results_dict[model_name]['mean_cosdist']))

    # Create a list of rows for the final DataFrame
    data = []
    for model_index, model_name in enumerate(MODEL_NAMES):
        for i in range(len(model_acc_dict[model_name])):
            data.append([
                MODEL_NAMES_TO_PLOT[model_name],                 # "Model"
                model_acc_dict[model_name][i],                   # "Accuracy"
                model_test_loss_dict[model_name][i],             # "ecoset Loss"
                model_test_topk_accuracy[model_name][i],         # "ecoset Topk Accuracy"
                1 / model_mean_cosdist_dict[model_name][i],      # "Spatial Smoothness" = inverse of mean_cosdist
            ])

    # Build the DataFrame
    df = pd.DataFrame(
        data,
        columns=[
            "Model",
            "Accuracy",
            "ecoset Loss",
            "ecoset Topk Accuracy",
            "Spatial Smoothness",
        ]
    )

    return df
