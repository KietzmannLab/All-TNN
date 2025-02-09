import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import (f_oneway, ttest_ind, kruskal, pearsonr)
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from itertools import combinations
from scikit_posthocs import posthoc_dunn
from tqdm import tqdm
from scipy import stats
from scipy.stats import permutation_test

import scienceplots
plt.style.use(['nature', 'science', 'ieee', 'no-latex'])

def plot_significance_matrix(
    significance_dict, 
    roi_labels_to_plot, 
    model_names, 
    model_names_to_plot, 
    save_dir, 
    num_cols=1
):
    """
    Plot a matrix of significance (p-values) for model comparisons for each ROI.

    Args:
        significance_dict (dict):
            A dictionary containing comparison results with p-values for each model pair.
        roi_labels_to_plot (list):
            A list of strings representing the Regions of Interest (ROIs).
        model_names (list):
            A list of model names (keys) as they appear in significance_dict.
        model_names_to_plot (dict):
            A dictionary mapping model names to their display names.
        save_dir (str):
            Directory to save the output plots.
        num_cols (int):
            Number of columns in the subplot grid.
    """
    # Make sure to use sans-serif fonts
    fm = matplotlib.font_manager
    fm._get_fontconfig_fonts.cache_clear()
    plt.rcParams['font.family'] = 'sans-serif'
    
    # Determine subplot arrangement
    num_plots = len(roi_labels_to_plot)
    num_rows = (num_plots + num_cols - 1) // num_cols  # ceiling division

    # Create subplots
    scale_factor = 1
    fig, axes = plt.subplots(
        num_rows, 
        num_cols, 
        figsize=(3.54 * num_cols * scale_factor, 3.54 * num_rows * scale_factor)
    )
    plt.rcParams['font.family'] = 'sans-serif'

    axes = np.array(axes).flatten()  # Flatten axes array
    model_names_mapped = [model_names_to_plot[m] for m in model_names]

    # Iterate over each ROI
    for i, (this_roi, this_ax) in enumerate(zip(roi_labels_to_plot, axes)):
        # Initialize a DataFrame to store p-values
        df = pd.DataFrame(index=model_names_mapped, columns=model_names_mapped)

        # Populate the DataFrame with p-values from significance_dict
        for comparison in significance_dict.keys():
            try:
                model1, model2 = comparison
            except:
                model1, model2 = int(comparison[0]), int(comparison[1])
            if model1 == model2:
                try:
                    df.loc[model_names_to_plot[model1], model_names_to_plot[model2]] = None
                except:
                    df.loc[model1, model2] = None
            else:
                try:
                    df.loc[model_names_to_plot[model1], model_names_to_plot[model2]] = \
                        significance_dict[comparison]['p_value']
                    df.loc[model_names_to_plot[model2], model_names_to_plot[model1]] = \
                        significance_dict[comparison]['p_value']
                except:
                    try:
                        df.loc[model1, model2] = significance_dict[comparison]['p_value']
                        df.loc[model2, model1] = significance_dict[comparison]['p_value']
                    except:
                        df.loc[model1, model2] = significance_dict[comparison][-1]
                        df.loc[model2, model1] = significance_dict[comparison][-1]

        # Reorder the DataFrame
        df_reordered = df.reindex(index=model_names_mapped, columns=model_names_mapped)

        # Extract p-values and create a mask
        p_values = np.around(df_reordered.values.astype(float), 35)
        tick_labels = model_names_mapped

        # Create a mask for p-values > 0.05
        mask = p_values > 0.05

        # Define colormap and normalization
        cmap = sns.color_palette("Blues_r", as_cmap=True)
        norm = matplotlib.colors.Normalize(vmin=0, vmax=0.05)

        # Plot the heatmap
        ax = sns.heatmap(
            p_values, 
            cmap=cmap, 
            norm=norm, 
            annot=True, 
            cbar=True,
            linewidths=1.5, 
            linecolor="lightgray", 
            ax=this_ax, 
            square=True, 
            fmt='.1e', 
            annot_kws={"fontsize": 5}
        )

        # Style the axes
        ax.set_xticklabels(tick_labels, rotation=45, fontweight='bold', fontsize=5)
        ax.set_yticklabels(tick_labels, rotation=0, fontweight='bold', fontsize=5)

        # Style the colorbar
        cbar = ax.collections[0].colorbar
        cbar.set_label("p-value")
        cbar.ax.tick_params()
        cbar.outline.set_linewidth(1)

        # Remove boundary strokes and tick marks
        ax.tick_params(axis='both', which='both', length=0)
        ax.tick_params(axis='x', bottom=False, top=False)
        ax.tick_params(axis='y', left=False, right=False)

        # Add a thick border around the heatmap
        ax.add_patch(
            Rectangle((0, 0), len(p_values[0]), len(p_values), 
                      fill=False, edgecolor="black", lw=1, clip_on=False)
        )

        # Set plot title for the ROI
        this_ax.set_title(f'{this_roi}', fontweight='bold')

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt_suffix = "significance_matrix"
    os.makedirs(save_dir, exist_ok=True)
    print(
        f"Saving plot to {save_dir}/{plt_suffix}_{'_'.join(roi_labels_to_plot)}.png"
    )
    plt.savefig(
        f'{save_dir}/{plt_suffix}_{"_".join(roi_labels_to_plot)}.png', dpi=300
    )
    plt.savefig(
        f'{save_dir}/{plt_suffix}_{"_".join(roi_labels_to_plot)}.pdf', dpi=300
    )

    # Show the plot
    plt.show()


def sign_permutation_test(data1, data2, num_permutations=10000):
    """
    Perform a non-parametric statistical analysis using sign tests with random 
    flipping between paired values.

    Parameters:
        data1 (list or np.array): Numerical values for the first group.
        data2 (list or np.array): Numerical values for the second group.
        num_permutations (int): Number of permutations to perform.

    Returns:
        tuple:
            (observed_diff, p_value)
    """
    observed_diff = np.mean(data1) - np.mean(data2)
    count = 0

    for _ in range(num_permutations):
        flip_mask = np.random.rand(len(data1)) < 0.5
        permuted_data1 = np.where(flip_mask, data2, data1)
        permuted_data2 = np.where(flip_mask, data1, data2)
        permuted_diff = np.mean(permuted_data1) - np.mean(permuted_data2)
        if abs(permuted_diff) >= abs(observed_diff):
            count += 1

    p_value = count / num_permutations
    return observed_diff, p_value


def perform_sign_permutation_test_statistical_analysis(df, metric):
    """
    Perform a non-parametric statistical analysis using sign tests with 
    random flipping between paired values. Includes FDR correction.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        metric (str): The column name for the metric to analyze.

    Returns:
        dict: significance_dict containing pairwise comparison results with
              FDR-adjusted p-values.
    """
    models = df['Model'].unique()
    data_groups = [df[df['Model'] == model][metric].values for model in models]

    np.random.seed(42)
    num_permutations = 10000
    significance_dict = {}
    p_values = []
    comparisons = []

    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            if i >= j:
                continue
            data1 = df[df['Model'] == model1][metric].values
            data2 = df[df['Model'] == model2][metric].values
            observed_diff = np.mean(data1) - np.mean(data2)

            count = 0
            for _ in range(num_permutations):
                flip_mask = np.random.rand(len(data1)) < 0.5
                permuted_data1 = np.where(flip_mask, data2, data1)
                permuted_data2 = np.where(flip_mask, data1, data2)
                permuted_diff = np.mean(permuted_data1) - np.mean(permuted_data2)
                if abs(permuted_diff) >= abs(observed_diff):
                    count += 1

            p_value = count / num_permutations
            p_values.append(p_value)
            comparisons.append((i, j))

    # FDR correction
    reject, corrected_p_values, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

    # Store results
    for (i, j), corrected_p_value in zip(comparisons, corrected_p_values):
        sig = '***' if corrected_p_value < 0.001 else '**' if corrected_p_value < 0.01 else '*' if corrected_p_value < 0.05 else ''
        model1, model2 = models[i], models[j]
        observed_diff = np.mean(data_groups[i]) - np.mean(data_groups[j])
        significance_dict[(i, j)] = [observed_diff, None, corrected_p_value]

        print(
            f"Comparison between {model1} and {model2}\n"
            f" mean: {df[df['Model'] == model1][metric].mean()} vs "
            f"{df[df['Model'] == model2][metric].mean()},\n"
            f" 95%CI of {model1}: {np.percentile(data_groups[i], [2.5, 97.5])}, "
            f"95%CI of {model2}: {np.percentile(data_groups[j], [2.5, 97.5])},\n"
            f"mean difference = {observed_diff:.4f}, p={corrected_p_value:.1e} {sig}\n"
        )

    return significance_dict


def perform_permutation_statistical_analysis(
    df, 
    metric, 
    num_permutations=10000, 
    permutation_type='independent'
):
    """
    Perform a non-parametric statistical analysis using permutation tests 
    and apply FDR correction.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        metric (str): The column name for the metric to analyze.
        num_permutations (int): Number of permutations to perform.
        permutation_type (str): Type of permutation test 
            ('independent' or 'samples').

    Returns:
        dict: significance_dict with pairwise comparison results.
    """
    models = df['Model'].unique()
    data_groups = [df[df['Model'] == model][metric].values for model in models]

    significance_dict = {}
    p_values = []

    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            if i >= j:
                continue
            data1 = df[df['Model'] == model1][metric].values
            data2 = df[df['Model'] == model2][metric].values
            observed_diff = np.mean(data1) - np.mean(data2)

            def statistic(d1, d2):
                return np.mean(d1) - np.mean(d2)

            result = permutation_test(
                (data1, data2), 
                statistic, 
                permutation_type=permutation_type,  
                n_resamples=num_permutations, 
                alternative='two-sided'
            )
            p_value = result.pvalue
            p_values.append(p_value)
            sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''
            significance_dict[(model1, model2)] = {
                'mean_difference': observed_diff,
                'p_value': p_value,
                'significance': sig,
            }

    # FDR correction
    _, corrected_p_values, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

    k = 0
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            if i >= j:
                continue
            corrected_p_value = corrected_p_values[k]
            k += 1
            sig = '***' if corrected_p_value < 0.001 else '**' if corrected_p_value < 0.01 else '*' if corrected_p_value < 0.05 else ''
            significance_dict[(model1, model2)]['p_value'] = corrected_p_value
            significance_dict[(model1, model2)]['significance'] = sig

            # Print detailed info
            def compute_95_ci(dat):
                mean_val = np.mean(dat)
                sem_val = stats.sem(dat)
                ci = stats.t.interval(0.95, len(dat)-1, loc=mean_val, scale=sem_val)
                return mean_val, ci

            model1_data = df[df['Model'] == model1][metric]
            model1_mean, model1_ci = compute_95_ci(model1_data)
            print(
                f"{model1} performance: {model1_mean:.3f}, "
                f"95% CI: ({model1_ci[0]:.3f}, {model1_ci[1]:.3f})"
            )

            model2_data = df[df['Model'] == model2][metric]
            model2_mean, model2_ci = compute_95_ci(model2_data)
            print(
                f"{model2} performance: {model2_mean:.3f}, "
                f"95% CI: ({model2_ci[0]:.3f}, {model2_ci[1]:.3f})"
            )

            print(
                f"Comparison between {model1} and {model2}: "
                f"mean difference = {observed_diff:.3f}; "
                f"two-sided permutation test, n={num_permutations}, "
                f"p <= {corrected_p_value:.3e}; {sig}\n"
            )

    return significance_dict


def perform_statistical_analysis(df, metric):
    """
    Perform a robust non-parametric statistical analysis using Kruskal-Wallis
    followed by Dunn's post hoc test with Bonferroni correction.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        metric (str): The column name for the metric to analyze.

    Returns:
        dict: significance_dict containing pairwise comparison results.
    """
    models = df['Model'].unique()
    data_groups = [df[df['Model'] == model][metric].values for model in models]

    # Kruskal-Wallis H-test
    stat, p_val_kw = kruskal(*data_groups)
    print(f"Kruskal-Wallis test: H={stat:.2f}, p={p_val_kw:.3f}")
    y_max = max([vals.max() for vals in data_groups])

    significance_dict = {}
    if p_val_kw < 0.05:
        # Dunn's post hoc if Kruskal-Wallis is significant
        pairwise_comparisons = posthoc_dunn(
            df, 
            val_col=metric, 
            group_col='Model', 
            p_adjust='bonferroni'
        )
        for m1 in pairwise_comparisons.columns:
            for m2 in pairwise_comparisons.index:
                p_val = pairwise_comparisons.loc[m2, m1]
                if m1 != m2:
                    mean_diff = (
                        df[df['Model'] == m1][metric].mean() 
                        - df[df['Model'] == m2][metric].mean()
                    )
                    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
                    height = y_max + (y_max * 0.05 * (1 + len(significance_dict)))
                    significance_dict[
                        (list(models).index(m1), list(models).index(m2), height, sig)
                    ] = [mean_diff, None, p_val]

                    print(
                        f"Comparison between {m1} and {m2}\n mean: "
                        f"{df[df['Model'] == m1][metric].mean()} vs "
                        f"{df[df['Model'] == m2][metric].mean()},\n "
                        f"mean difference = {mean_diff:.4f}, p={p_val:.1e} {sig}\n"
                    )
    else:
        print("No significant differences found by Kruskal-Wallis test.")
        significance_dict = {}

    return significance_dict


def perform_ANOVA_statistical_analysis(df, alignment_metric):
    """
    Perform a one-way ANOVA for overall differences, then conduct pairwise t-tests 
    for post hoc comparisons, optionally followed by Tukey's test.

    Args:
        df (pd.DataFrame): Input data with 'Model' and alignment_metric columns.
        alignment_metric (str): The column name for the metric to analyze.
    """
    models = df['Model'].unique()
    alignment_values = {m: df[df['Model'] == m][alignment_metric].values for m in models}
    # Perform one-way ANOVA
    f_val, p_val = f_oneway(*alignment_values.values())
    print(f"ANOVA results: F={f_val}, p={p_val}")

    # Pairwise comparisons
    for (model1, vals1), (model2, vals2) in combinations(alignment_values.items(), 2):
        mean_diff = vals1.mean() - vals2.mean()
        t_stat, p_val_test = ttest_ind(vals1, vals2)
        sig = '***' if p_val_test < 0.001 else '**' if p_val_test < 0.01 else '*' if p_val_test < 0.05 else ''
        print(
            f"Comparison between {model1} and {model2}\n mean: "
            f"{vals1.mean()} vs {vals2.mean()},\n mean difference = "
            f"{mean_diff:.4f}, p={p_val_test:.1e} {sig}\n"
        )

    # Post-hoc (Tukey) if ANOVA is significant
    if p_val < 0.05:
        posthoc = pairwise_tukeyhsd(df[alignment_metric], df['Model'])
        print("Post-hoc analysis results:")
        print(posthoc)


def perform_non_parametric_statistical_analysis(df, alignment_metric):
    """
    Perform a non-parametric test (Mann-Whitney U) for each pair of models 
    and return significance details.

    Args:
        df (pd.DataFrame): The input data with 'Model' column.
        alignment_metric (str): The metric to analyze.

    Returns:
        dict: A dictionary mapping (bar1, bar2, height, sig) to 
              (mean_diff, stat, p_value).
    """
    from scipy.stats import mannwhitneyu
    from itertools import combinations

    models = df['Model'].unique()
    model_positions = {model: pos for pos, model in enumerate(models)}
    alignment_values = {model: df[df['Model'] == model][alignment_metric].values for model in models}
    significance_dict = {}

    y_max = max([vals.max() for vals in alignment_values.values()])

    for (model1, values1), (model2, values2) in combinations(alignment_values.items(), 2):
        mean_diff = values1.mean() - values2.mean()
        stat, p_val = mannwhitneyu(values1, values2, alternative='two-sided')

        if p_val < 0.001:
            sig = '***'
        elif p_val < 0.01:
            sig = '**'
        elif p_val < 0.05:
            sig = '*'
        else:
            sig = ''

        bar1 = model_positions[model1]
        bar2 = model_positions[model2]
        height = y_max + (y_max * 0.05 * (1 + len(significance_dict)))
        significance_dict[(bar1, bar2, height, sig)] = (mean_diff, stat, p_val)

        print(
            f"Comparison between {model1} and {model2}\n mean: "
            f"{values1.mean()} vs {values2.mean()},\n mean difference = "
            f"{mean_diff:.4f}, p={p_val:.1e} {sig}\n"
        )

    return significance_dict


def calculate_category_correlations_with_permutation(data, num_permutations=10000):
    """
    Calculate the Pearson correlation coefficients and p-values between categories 
    for each participant's 5x5 accuracy maps using permutation tests.

    Args:
        data (np.array): Shape (participants_num, category, 5, 5).
        num_permutations (int): Number of permutations for p-value calculation.

    Returns:
        tuple: (correlation_matrix, p_value_matrix).
    """
    num_participants, num_categories, map_length, map_width = data.shape
    data_reshaped = data.reshape(num_participants, num_categories, map_length * map_width)

    correlation_matrix = np.zeros((num_categories, num_categories))
    p_value_matrix = np.zeros((num_categories, num_categories))

    average_human_acc_maps = np.mean(data_reshaped, axis=0)

    for i in tqdm(range(num_categories), desc="Computing correlations across categories"):
        for j in range(i + 1, num_categories):
            permuted_corrs = np.zeros(num_permutations)
            combined_data = np.vstack((data_reshaped[:, i], data_reshaped[:, j]))
            num_combined = combined_data.shape[0]

            for n in tqdm(range(num_permutations)):
                shuffled_indices = np.random.permutation(num_combined)
                group1_indices = shuffled_indices[:num_combined // 2]
                group2_indices = shuffled_indices[num_combined // 2:]
                permuted_group1 = combined_data[group1_indices]
                permuted_group2 = combined_data[group2_indices]
                perm_group1_mean = np.mean(permuted_group1, axis=0)
                perm_group2_mean = np.mean(permuted_group2, axis=0)
                permuted_corrs[n] = pearsonr(perm_group1_mean, perm_group2_mean)[0]

            empirical_similarity = pearsonr(average_human_acc_maps[i], average_human_acc_maps[j])[0]
            p_value = (np.sum((1 - permuted_corrs) >= (1 - empirical_similarity)) + 1) / num_permutations

            correlation_matrix[i, j] = correlation_matrix[j, i] = empirical_similarity
            p_value_matrix[i, j] = p_value_matrix[j, i] = p_value

        correlation_matrix[i, i] = 1
        p_value_matrix[i, i] = None

    return correlation_matrix, p_value_matrix


def plot_correlation_significance_matrices(correlation_matrix, p_value_matrix, category_labels, save_dir):
    """
    Plot correlation matrices and significance matrices with uniform styling.

    Args:
        correlation_matrix (np.array): Matrix of correlation coefficients.
        p_value_matrix (np.array): Matrix of p-values.
        category_labels (list): Category labels for axes.
        save_dir (str): Directory to save plots.
    """
    num_categories = len(category_labels)
    fm = matplotlib.font_manager
    fm._get_fontconfig_fonts.cache_clear()
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 5

    p_value_matrix = np.where(p_value_matrix == None, np.nan, p_value_matrix)

    font_scale = 2
    fig, axes = plt.subplots(1, 2, figsize=(3.54 * 2 * font_scale, 3.54 * font_scale))

    heatmap_settings = {
        "annot": True,
        "linewidths": 1.5,
        "linecolor": 'lightgray',
        "square": True,
        "xticklabels": category_labels,
        "yticklabels": category_labels,
        "annot_kws": {'fontsize': 5},
        "cbar_kws": {'label': 'Scale', 'aspect': 20},
    }

    divider0 = make_axes_locatable(axes[0])
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)

    sns.heatmap(
        correlation_matrix, 
        ax=axes[0], 
        cbar_ax=cax0, 
        cmap='magma',  
        fmt='.2f', 
        **heatmap_settings
    )
    axes[0].set_title('Pearson Correlation Matrix', fontweight='bold', fontsize=7 * font_scale)
    axes[0].collections[0].colorbar.set_label('Correlation Coefficient')
    axes[0].collections[0].colorbar.outline.set_linewidth(1)
    axes[0].set_xticklabels(category_labels, rotation=45, fontweight='bold', fontsize=5 * font_scale)
    axes[0].set_yticklabels(category_labels, rotation=0, fontweight='bold', fontsize=5 * font_scale)
    axes[0].add_patch(
        Rectangle((0, 0), len(correlation_matrix[0]), len(correlation_matrix),
                  fill=False, edgecolor="black", lw=1, clip_on=False)
    )

    flatten_indices = np.tril_indices_from(p_value_matrix, k=-1)
    flattened_p_vals = p_value_matrix[flatten_indices]
    _, p_adjusted, _, _ = multipletests(flattened_p_vals, method='fdr_bh')
    p_adjusted = sns.utils.axis_ticklabels_from_transform(sns.matrix_functions.squareform(p_adjusted))
    # The above step is typically done via "p_adjusted = squareform(p_adjusted)", 
    # but a direct scikit-learn or custom approach might also be used if needed.

    # Because the user code is using 'squareform' from scipy.spatial.distance:
    from scipy.spatial.distance import squareform
    p_adjusted = squareform(p_adjusted)
    np.fill_diagonal(p_adjusted, np.nan)

    mask = p_adjusted > 0.05

    heatmap_settings['norm'] = matplotlib.colors.Normalize(vmin=0, vmax=0.05)
    heatmap_settings['cmap'] = 'Blues_r'

    divider1 = make_axes_locatable(axes[1])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)

    sns.heatmap(
        p_adjusted, 
        mask=None,  
        ax=axes[1], 
        cbar_ax=cax1, 
        fmt='.1e', 
        **heatmap_settings
    )
    axes[1].set_title('P-value Matrix', fontweight='bold', fontsize=7 * font_scale)
    axes[1].collections[0].colorbar.set_label('p-value')
    axes[1].collections[0].colorbar.outline.set_linewidth(1)
    axes[1].set_xticklabels(category_labels, rotation=45, fontweight='bold', fontsize=5 * font_scale)
    axes[1].set_yticklabels(category_labels, rotation=0, fontweight='bold', fontsize=5 * font_scale)
    axes[1].add_patch(
        Rectangle((0, 0), len(p_adjusted[0]), len(p_adjusted),
                  fill=False, edgecolor="black", lw=1, clip_on=False)
    )

    for ax in axes:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.tick_params(axis='both', which='both', length=0)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt_suffix = "correlation_and_significance_matrices"
    plt.savefig(f'{save_dir}/{plt_suffix}.png', dpi=300)
    plt.savefig(f'{save_dir}/{plt_suffix}.pdf', dpi=300)
    plt.close()