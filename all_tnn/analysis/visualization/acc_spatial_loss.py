import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict

def generate_analysis_df(
    base_src_dir_path,
    MODEL_NAMES,
    seeds_range,
    models_epochs_dict,
    MODEL_NAMES_TO_PLOT,
    results_file_name='acc_smoothness_loss.pickle',
):
    """
    Generate a pandas DataFrame containing model accuracies, losses, top-k accuracies,
    spatial smoothness, and categorical losses from pickled data files.

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
    results_file_name : str
        The name of the results file to load from the source directory.

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
    model_test_catergrization_loss_dict = defaultdict(list)
    model_mean_cosdist_dict = defaultdict(list)

    # Loop over each model and seed to collect data
    for model_name in MODEL_NAMES:
        for seed in seeds_range:
            src_path = os.path.join(
                base_src_dir_path,
                f'{models_epochs_dict[model_name][seed-1]}/seed{seed}/{results_file_name}'
            )


            # Load data from pickle
            with open(src_path, 'rb') as handle:
                model_seeds_dict = pickle.load(handle)

                if 'TNN_simclr' == model_name:
                    model_name = 'TNN_simclr_finetune' #* only finetuned model is used for analysis
                # print(f"key: model_seeds_dict.keys() = {model_seeds_dict.keys()}")
                model_acc_dict[model_name].append(model_seeds_dict[model_name]['ecoset_test_accuracy'])
                model_test_loss_dict[model_name].append(model_seeds_dict[model_name]['ecoset_test_loss'])
                model_test_topk_accuracy[model_name].append(model_seeds_dict[model_name]['ecoset_test_topk_accuracy'])
                model_mean_cosdist_dict[model_name].append(np.sum(model_seeds_dict[model_name]['mean_cosdist']))

                # Categorical loss logic
                if model_name not in ['CNN_lr_0.05', 'LCN_lr_0.05']:
                    cat_loss = (
                        model_seeds_dict[model_name]['ecoset_test_accuracy']
                        - model_seeds_dict[model_name]['ecoset_test_loss']
                    )
                else:
                    cat_loss = model_seeds_dict[model_name]['ecoset_test_accuracy']

                model_test_catergrization_loss_dict[model_name].append(cat_loss)

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
                model_test_catergrization_loss_dict[model_name][i]  # "ecoset Categorical Loss"
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
            "ecoset Categorical Loss"
        ]
    )

    return df