import os
import numpy as np
from all_tnn.analysis.util.analysis_help_funcs import get_activations_on_dataset

def compute_selectivities(activities_model, dataset_names, data_generators_dict,hparams=None):
    activities,  labels = {}, {}
    mean_activities, var_activities = {}, {}
    n_samples = {}
    n_layers = len(data_generators_dict['dummy_activities_batch'])

    for this_dataset_name in dataset_names:
        print(f'Gathering activities for {this_dataset_name}')
        activities[this_dataset_name], labels[this_dataset_name] = \
            get_activations_on_dataset(activities_model, data_generators_dict[this_dataset_name], this_dataset_name,
                                       one_hot=False, hparams=hparams)

        mean_activities[this_dataset_name] = [np.mean(a, axis=0) for a in activities[this_dataset_name]]
        var_activities[this_dataset_name] = [np.var(a, axis=0) for a in activities[this_dataset_name]]
        n_samples[this_dataset_name] = labels[this_dataset_name].shape[0]

    # mean and var of activities excluding one dataset
    for leave_cat_out in dataset_names:
        concat_acts = [[] for _ in range(n_layers)]
        for layer in range(n_layers):
            for c in dataset_names:
                if c != leave_cat_out:
                    concat_acts[layer].append(activities[c][layer])
        out_activities = [np.concatenate(concat_acts[l]) for l in range(n_layers)]
        mean_activities[f'out_{leave_cat_out}'] = [np.mean(a, axis=0) for a in out_activities]
        var_activities[f'out_{leave_cat_out}'] = [np.var(a, axis=0) for a in out_activities]
        n_samples[f'out_{leave_cat_out}'] = out_activities[0].shape[0] #! out_activities[0] is the activities of the first layer

    category_selectivities = {
        'mean_activities': mean_activities,
        'var_activities': var_activities,
        'n_samples': n_samples}

    del activities, labels

    return category_selectivities

def compute_category_selectivities(activities_model, hparams, data_generators_dict, dataset_names):
    print('Computing category selectivities')
    selectivity_maps = compute_selectivities(activities_model, dataset_names, data_generators_dict, hparams=hparams)
    category_maps = {'selectivity': selectivity_maps}

    return {'category_selectivities': category_maps}
