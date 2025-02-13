import os
import os.path as op
import h5py
import scipy
from scipy.spatial.distance import squareform
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def make_predictors():
    ''' Returns a dict of predictor matrices to be used for the GLM. Based on the 16 object categories'''
    predictor_names = ['animate', 'transport', 'size', 'constant', 'food', 'tools', 'spiky']
    predictors = {key: np.ones((16, 16)) for key in predictor_names}

    predictors['animate'][4:9, 4:9] = 0
    predictors['food'][10:12, 10:12] = 0
    predictors['transport'][:4, :4] = 0
    
    for i in [10, 12, 13, 14, 15]:
        for j in [10, 12, 13, 14, 15]:
            predictors['tools'][i, j] = 0
    predictors['tools'][-4:, -4:] = 0
    
    # Spikiness is calculated using the aspect ratio formula from Bao et al. (2020)
    spike_dict = {"bus": 1.505953838195587, "airplane": 5.621822938468574,  "train": 1.6399792078761604, "motorcycle": 3.911098371621816, "bear": 2.496906278481293,   "elephant": 3.803867655242841, "giraffe": 3.2244429619466417,"zebra": 6.357984662231756,  "cat": 2.004848308509678, "kite": 3.9165081213258297,  "pizza": 1.1945420195535317,  "broccoli": 2.4247137336111093, "laptop": 1.4844661192911894,"refrigerator": 1.30857189172803, "scissors": 4.442855102749422, "toilet": 2.2997254168176817}
    spike_list = [s for s in spike_dict.values()]
    spike_list = (np.array(spike_list) - np.min(spike_list)) / (np.max(spike_list) - np.min(spike_list))
    for i in range(16):
        for j in range(16):
            predictors['spiky'][i, j] = abs(spike_list[i] - spike_list[j])
    
    # Size ranks are based on Konkle & Oliva (2011). 
    sizes = [8, 8, 8, 7, 7, 7, 7, 6, 4, 4, 3, 2, 3, 6, 2, 5] 
    sizes = (np.array(sizes) - 1) / 7   
    for i in range(16):
        for j in range(16):
            predictors['size'][i, j] = abs(sizes[i] - sizes[j])

    # Fill diagonal of each key of predictors with 0s
    for key in predictors.keys():
        np.fill_diagonal(predictors[key], 0)

    return {name: predictors[name] for name in predictor_names}

def predictors_from_names(predictors_dict, names):
    # return a [n_preds,n_pred_features] np array of predictors
    return np.array([predictors_dict[name] for name in names])

def cluster_rdm(rdm, categories, method, metric):
    row_link = scipy.cluster.hierarchy.linkage(
        rdm[np.triu_indices(rdm.shape[0], 1)],
        method=method,
        metric=metric,
    )
    row_order = scipy.cluster.hierarchy.leaves_list(row_link)
    ordered_categories = categories[row_order]

    return row_link, ordered_categories

def plot_predictors(plot_path, predictors_dict, predictor_names, names, reordered=False):
    predictors = predictors_from_names(predictors_dict, predictor_names)

    reordered_str = '_reordered' if reordered else ''
    fig, axs = plt.subplots(len(predictors), 1, figsize=(8, 6.5*len(predictors)))
    for i, predictor in enumerate(predictors):
        sns.heatmap(predictor, ax=axs[i], cmap='inferno', xticklabels=names, yticklabels=names, vmin=0, vmax=1)
        axs[i].set_title(predictor_names[i], fontsize=20)
        axs[i].tick_params(left=False, bottom=False, right=False, top=False)
    plt.tight_layout()
    plt.savefig(op.join(plot_path, f'predictors{reordered_str}.png'))
    plt.close()

def hierarchical_clustering(plot_path, data, names, method='average', plot=1):
    """ Perform hierarchical clustering on the ADM and plot. Label names are automatically reordered"""
    for model_type in ['average_human_adm', 'CNN', 'LCN', 'All-TNN\n($\\alpha=1$)', 'All-TNN\n($\\alpha=10$)', 'All-TNN\n($\\alpha=100$)', 'All-TNN\nSimCLR\n($\\alpha=10$)']:
        if model_type == 'average_human_adm': 
            this_data = data[model_type]
            row_link, ordered_categories = cluster_rdm(this_data, np.arange(this_data.shape[0]), method, "correlation")
            label_names = names
        else: 
            this_data = data['average_model_adms'][model_type]
        if plot: 
            c_map = sns.clustermap(
                this_data,
                row_linkage=row_link,
                col_linkage=row_link,
                cmap="inferno",
                figsize=(3.54*2,3.54*2),
                xticklabels=label_names,  
                yticklabels=label_names,
            )
            c_map.ax_heatmap.yaxis.tick_left()
            c_map.ax_row_dendrogram.set_visible(False)
            plt.setp(c_map.ax_heatmap.get_yticklabels(), weight='bold')
            plt.setp(c_map.ax_heatmap.get_xticklabels(), rotation=45, ha="right", weight='bold')
            c_map.ax_heatmap.tick_params(axis='both', which='both', length=0)
            plt.savefig(op.join(plot_path, f'{model_type}_{method}_cluster_map_human_corr_matrix_human_order.png'))    
            plt.close()

    return row_link, ordered_categories

def unique_variances(y, predictors, predictor_names, z_score=True):
    '''y: target variable (brain RDM)
    predictors_names: list of predictors names (one per model you want to do variance decomposition on)
    predictors_features: array of predictors features (one per model you want to do variance decomposition on)'''
    
    predictors = predictors_from_names(predictors, predictor_names)
    # get upper triangle of the matrix if needed
    if len(y.shape) == 2:
        y = squareform(y)
    predictors_features = []
    for x in predictors:
        if len(x.shape) == 2:
            predictors_features.append(squareform(x))
        else:
            predictors_features.append(x)
    if z_score:
        y = (y - np.mean(y))/np.std(y)
        predictors_features = [(x - np.mean(x))/np.std(x) for x in predictors_features]

    if len(y.shape) == 1:
        y = y[:, np.newaxis]
    for f in predictors_features:
        if len(f.shape) == 1:
            f = f[:, np.newaxis]
        assert f.shape[0] == y.shape[0], \
        f"Predictors features must have the same length as the target variable, got {f.shape[0]} and {y.shape[0]}"

    all_combinations = ['full_model'] + [f'without_{p}' for p in predictor_names]

    # Fit models for each combination of predictors
    models = {}
    var_components = {}
    for combo in all_combinations:
        if combo == 'full_model':
            X = np.column_stack(predictors_features)
        else:
            this_name = combo.split('_')[1]
            idx = predictor_names.index(this_name)
            X = np.column_stack([predictors_features[i] for i in range(len(predictor_names)) if i != idx])
        model = LinearRegression(fit_intercept=False, positive=True).fit(X, y)
        models[combo] = model
        var_components[combo] = model.score(X,y)

    for this_name in predictor_names:
        var_components[f'unique_{this_name}'] = var_components['full_model'] - var_components[f'without_{this_name}']

    return var_components, all_combinations

def bar_plot_vars(model_vars, predictor_names, plot_path):
    with plt.style.context([ 'nature','science',"ieee",'no-latex']):
        plt.rcParams['font.family'] = 'sans-serif'
        fig, ax = plt.subplots(figsize=(3, 2.5))
        bar_width = 0.5
        plot_index = np.arange(len(predictor_names))

        for i, var in enumerate(model_vars):
            ax.bar(i, var, bar_width, color='Orange', edgecolor='black')
        
        ax.set_xticks(plot_index + bar_width / 2)
        ax.set_xticklabels(predictor_names, rotation=45, ha='right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(which='both', right=False, top=False, bottom=False)
        ax.set_ylabel('Variance explained')
        plt.tight_layout()
        plt.savefig(op.join(plot_path, f'var_explained_glm_avg.png'))
        plt.close()

def bar_indiv_humans(human_indiv_unique_vars, predictor_names, plot_path):
    with plt.style.context([ 'nature','science',"ieee",'no-latex']):
        plt.rcParams['font.family'] = 'sans-serif'
        fig, ax = plt.subplots(figsize=(3, 2.5))
        bar_width = 0.5
        plot_index = np.arange(len(predictor_names))
        for i in range(len(predictor_names)):
            ax.bar(i, np.mean(human_indiv_unique_vars, axis=0)[i], yerr=np.std(human_indiv_unique_vars, axis=0)[i]/np.sqrt(30), width=bar_width, color='Orange')
        
        ax.set_xticks(plot_index + bar_width / 2)
        ax.set_xticklabels(predictor_names, rotation=45, ha='right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(which='both', right=False, top=False, bottom=False)
        ax.set_ylabel('Variance explained')
        plt.tight_layout()
        plt.savefig(op.join(plot_path, f'var_explained_glm_indiv.png'))
        plt.close()

def run_full_GLM_analysis(
    base_path='save_dir',
    data_filename='adm_dict_600_1_pearsonr_spearmanr.pkl',
    plot_path='figure4/GLM',
    model_type='average_human_adm',
    predictor_names=None,
    num_permutations=1000,
    verbose=True
):
    """
    This function loads ADM data, sets up predictors, calculates unique variances. 
    It also runs permutation tests to determine significance and produces summary bar plots.

    Parameters
    ----------
    base_path : str
        Path to the base directory containing data and where plots will be stored.
    data_filename : str
        The name of the file containing the ADM data dictionary.
    plot_path : str
    model_type : str
        Key in the data dictionary corresponding to the target ADM (e.g., 'average_human_adm').
    predictor_names : list of str
        Names of the predictors to use in the analyses (default: ['animate', 'size', 'spiky']).
    num_permutations : int
        Number of permutations used in significance testing.
    verbose : bool
        Whether to print intermediate results or be silent.

    Returns
    -------
    results : dict
        A dictionary containing analysis results, including:
        - "unique_vars" : list of unique variance values for the main model
        - "unique_vars_significance" : list of (p-value, bool) tuples for each bar in unique variance
        - "human_indiv_unique_vars" : list of lists with unique variance values for each subject
        - "human_indiv_significance" : list of lists with significance booleans for each subject
    """

    if predictor_names is None:
        predictor_names = ['animate', 'size', 'spiky']

    # -----------------------------
    # 1. Paths and directories
    # -----------------------------

    # For individual subject analysis
    indiv_path = op.join(plot_path, 'individual')
    os.makedirs(indiv_path, exist_ok=True)

    # Load data
    data = h5py.File(op.join(base_path, data_filename), 'r')

    # Category names
    names = np.array([
        "bus", "airplane", "train", "motorcycle", "bear", "elephant", "giraffe", "zebra",
        "cat", "kite", "pizza", "broccoli", "laptop", "refrigerator", "scissors", "toilet"
    ])

    # -----------------------------
    # 2. Create and plot predictors
    # -----------------------------
    if verbose:
        print("Creating and plotting predictors...")
    predictors = make_predictors()
    plot_predictors(plot_path, predictors, predictor_names, names, reordered=False)

    # -----------------------------
    # 3. Prepare target data
    # -----------------------------
    target_data = data[model_type]

    # -----------------------------
    # 4. Unique Variance (Group)
    # -----------------------------
    if verbose:
        print("______ Unique variance (Group) ______")

    # Re-create predictors for unique variance step (often the same, but separated in case of modifications)
    predictors_unique = make_predictors()
    all_vars, all_combinations_unique = unique_variances(
        target_data, predictors_unique, predictor_names, z_score=True
    )
    all_plot_names = ['full_model'] + [f'unique_{name}' for name in predictor_names]
    unique_vars = [all_vars[name] for name in all_plot_names]

    # Permutation test for significance
    permuted_partitions = []
    for n in range(num_permutations):
        shuffle_idx = np.random.permutation(target_data.shape[0])
        shuffled_target = target_data[shuffle_idx][:, shuffle_idx]
        these_vars, _ = unique_variances(
            shuffled_target, predictors_unique, predictor_names, z_score=True
        )
        these_unique_vars = [these_vars[name] for name in all_plot_names]
        permuted_partitions.append(these_unique_vars)

    permuted_partitions = np.array(permuted_partitions)
    significance_unique = []
    # significance_unique will store tuples: (p_value, is_sig_boolean)
    for p in range(permuted_partitions.shape[1]):
        distribution = permuted_partitions[:, p]
        value = unique_vars[p]
        percentile_95 = np.percentile(distribution, 95)
        p_value = np.sum(distribution >= value) / len(distribution)
        significance_unique.append((p_value, value >= percentile_95))

    # Print or log results
    if verbose:
        for x, y, z in zip(all_plot_names, unique_vars, significance_unique):
            print(f"{x}: {y:.4f}, p={z[0]:.4f}, sig={z[1]}")

    # Bar plot of unique variances
    bar_plot_vars(unique_vars, all_plot_names, plot_path=plot_path)

    # -----------------------------
    # 5. Unique Variance (Individual Humans)
    # -----------------------------
    if verbose:
        print("______ Unique variance - individual humans ______")

    human_adms_30 = data['human_adms']  # list/array of ADMs
    human_indiv_unique_vars = []
    human_indiv_significance = []

    for i, adm in enumerate(human_adms_30):
        if verbose:
            print(f"---- Processing subject {i}/30 ----", end="\r")

        all_vars_subj, _ = unique_variances(
            adm, predictors_unique, predictor_names, z_score=True
        )
        # same order of plotting
        unique_vars_subj = [all_vars_subj[name] for name in all_plot_names]
        human_indiv_unique_vars.append(unique_vars_subj)

        # Permutation for single subject
        permuted_partitions_subj = []
        for n in range(num_permutations):
            shuffle_idx = np.random.permutation(adm.shape[0])
            shuffled_target = adm[shuffle_idx][:, shuffle_idx]
            these_vars_subj, _ = unique_variances(
                shuffled_target, predictors_unique, predictor_names, z_score=True
            )
            these_unique_vars_subj = [these_vars_subj[name] for name in all_plot_names]
            permuted_partitions_subj.append(these_unique_vars_subj)

        permuted_partitions_subj = np.array(permuted_partitions_subj)
        significance_unique_subj = []
        for p in range(permuted_partitions_subj.shape[1]):
            distribution = permuted_partitions_subj[:, p]
            value = unique_vars_subj[p]
            percentile_95 = np.percentile(distribution, 95)
            significance_unique_subj.append(value >= percentile_95)

        # if verbose:
        #     for x, y, z in zip(all_plot_names, unique_vars_subj, significance_unique_subj):
        #         print(f"{x}: {y:.4f}, sig={z}")

        human_indiv_significance.append(significance_unique_subj)

    # Bar plot for all individual humans
    bar_indiv_humans(human_indiv_unique_vars, all_plot_names, plot_path=indiv_path)

    # -----------------------------
    # 6. Collect and Return Results
    # -----------------------------
    results = {
        "unique_vars": unique_vars,
        "unique_vars_significance": significance_unique,
        "human_indiv_unique_vars": human_indiv_unique_vars,
        "human_indiv_significance": human_indiv_significance,
    }

    if verbose:
        print("Analysis complete.")

    return results