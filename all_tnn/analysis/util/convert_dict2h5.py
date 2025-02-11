

import pickle
import h5py
import os
import json
import numpy as np
from collections import defaultdict  # Used by nested_dict()


def save_dict_to_h5(h5_group, dictionary):
    """
    Recursively walk through `dictionary` and store its contents in the provided HDF5 group.
    """
    for key, value in dictionary.items():
        if isinstance(value, dict):
            # Create a subgroup for a nested dictionary.
            subgroup = h5_group.create_group(key)
            save_dict_to_h5(subgroup, value)
        else:
            # Try to store the value directly as a dataset.
            try:
                h5_group.create_dataset(key, data=np.array(value))
            except Exception:
                
                # First fallback: attempt to squeeze the value (assuming it's a numpy-like array, e.g. different shape in each layer).
                try:
                    if len(value.squeeze().shape) == 1:
                        squeezed_value = value.squeeze()
                        for i in range(len(squeezed_value)):
                            item = squeezed_value[i]
                            if not isinstance(item, dict):
                                # Save non-dict items directly.
                                h5_group.create_dataset(f'{key}/layer_{i}', data=item)
                            else:
                                # For dictionary items, iterate and save their content.
                                for sub_key, sub_value in item.items():
                                    if isinstance(sub_value, (int, float, np.ndarray, list, tuple)):
                                        try:
                                            h5_group.create_dataset(f'{key}/layer_{i}/{sub_key}', data=np.array(sub_value))
                                        except Exception:
                                            raise ValueError(f'Cannot store {sub_key} with shape {sub_value.shape}')
                                    elif isinstance(sub_value, dict):
                                        for sub_sub_key, sub_sub_value in sub_value.items():
                                            if isinstance(sub_sub_value, (int, float, np.ndarray, list, tuple)):
                                                try:
                                                    h5_group.create_dataset(f'{key}/layer_{i}/{sub_key}/{sub_sub_key}', data=np.array(sub_sub_value))
                                                except Exception:
                                                    raise ValueError(f'Cannot store {sub_sub_key} with shape {sub_sub_value.shape}')
                
                except Exception:
                    # Second fallback: if value is a tuple, iterate over its elements.
                    if isinstance(value, tuple):
                        for order, value_item in enumerate(value):
                                if len(value_item.shape) == 1:
                                    squeezed_value = value_item.squeeze()
                                    for i in range(len(squeezed_value)):
                                        item = squeezed_value[i]
                                        if not isinstance(item, dict):
                                            h5_group.create_dataset(f'{key}/{order}/layer_{i}', data=item)
                                        else:
                                            for sub_key, sub_value in item.items():
                                                if isinstance(sub_value, (int, float, np.ndarray, list, tuple)):
                                                    try:
                                                        h5_group.create_dataset(f'{key}/{order}/layer_{i}/{sub_key}', data=np.array(sub_value))
                                                    except Exception:
                                                        raise ValueError(f'Cannot store {sub_key} with shape {sub_value.shape}')
                                                elif isinstance(sub_value, dict):
                                                    for sub_sub_key, sub_sub_value in sub_value.items():
                                                        if isinstance(sub_sub_value, (int, float, np.ndarray, list, tuple)):
                                                            try:
                                                                h5_group.create_dataset(f'{key}/{order}/layer_{i}/{sub_key}/{sub_sub_key}', data=np.array(sub_sub_value))
                                                            except Exception:
                                                                raise ValueError(f'Cannot store {sub_sub_key} with shape {sub_sub_value.shape}')



def nested_dict():
    """Generates a nested dictionary of unlimited depth."""
    return defaultdict(nested_dict)


def convert_dict2h5(dict_file, h5_file=None):
    """
    Load a pickled dictionary from `dict_file` and save its contents to an HDF5 file.
    If `h5_file` is not provided, the output file is placed in the same directory as `dict_file`.
    """
    # Load the dictionary from the pickle file.
    with open(dict_file, 'rb') as f:
        data_dict = pickle.load(f)
    model_names = list(data_dict.keys())

    # Ensure the base directory exists.
    base_dir = os.path.dirname(dict_file)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Determine the output HDF5 file path.
    if h5_file:
        save_h5_path = h5_file
    else:
        file_base = os.path.splitext(os.path.basename(dict_file))[0]
        save_h5_path = os.path.join(base_dir, f'{file_base}.h5')

    # Ensure the directory for the HDF5 file exists.
    os.makedirs(os.path.dirname(save_h5_path), exist_ok=True)

    # Write the data to the HDF5 file.
    with h5py.File(save_h5_path, 'w') as hf:
        for model_name in model_names:
            model_group = hf.create_group(model_name)
            print(f'Saving {model_name} to {save_h5_path}')
            save_dict_to_h5(model_group, data_dict[model_name])


if __name__ == '__main__':

    for seed in range(1,6):
        dict_path = f'./save_dir/_analyses_data/neural_level_analysis/300/seed{seed}/all_multi_models_neural_dict.pickle'
        convert_dict2h5(dict_path, h5_file= f'h5/seed{seed}/all_multi_models_neural_dict.h5')

