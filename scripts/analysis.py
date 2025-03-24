import os
from collections import defaultdict
import numpy as np
import tensorflow as tf
import h5py
import pickle

# ==============================
# Local Imports
# ==============================
from all_tnn.analysis.config import *
from all_tnn.dataset_loader.make_tf_dataset import get_dataset
from all_tnn.analysis.orientation_selectivity import orientation_selectivity
from all_tnn.analysis.category_selectivity import compute_category_selectivities
from all_tnn.analysis.energy_efficiency import (
    analyze_energy_efficiency,
    analyze_activation_transmission_efficiency,
    load_feature_responses,
    get_feature_responses,
)
from all_tnn.analysis.visualization.layer_visualization import visualize_model
from all_tnn.analysis.smoothness import smoothness_main
from all_tnn.analysis.util.behaviour_analysis_help_funcs import get_behaviour_data
from all_tnn.analysis.behaviour_alignment import visual_behaviour_alignment_analysis, adm_alignment_analysis
from all_tnn.analysis.util.statistic_funcs import plot_significance_matrix
from all_tnn.analysis.util.setup_model_and_data import load_models, DatasetLabelRenamer
from all_tnn.analysis.util.analysis_help_funcs import load_and_override_hparams, results_to_disk
from all_tnn.analysis.util.convert_dict2h5 import convert_dict2h5
from all_tnn.analysis.visualization.acc_maps_visualization import plot_bar_plot_from_df

from all_tnn.analysis.config import Config

def setup_analysis_directories_and_data_generators(
    saved_model_path,
    hparams,
    epoch,
    PRE_RELU,
    POST_NORM,
    ANALYSIS_DIR_SUFFIX,
    ECOSET_PATH,
    STF_PATH,
    activities_model
):
    """
    Set up analysis directories and data generators for the model all_tnn.analysis.

    Args:
        saved_model_path (str): The path where the model is saved.
        hparams (dict): Hyperparameters for the model.
        epoch (int): The current epoch of the all_tnn.analysis.
        PRE_RELU (bool): Flag to determine if analysis is pre-ReLU or post-ReLU.
        POST_NORM (bool): Flag to determine if analysis is after normalization.
        ANALYSIS_DIR_SUFFIX (str): Suffix for the analysis directory.
        ECOSET_PATH (str): Path to the Ecoset dataset.
        STF_PATH (str): Path to the STF dataset.
        activities_model (callable): Function to compute activities batch.

    Returns:
        tuple: A tuple containing updated hyperparameters and a dictionary of data generators.
    """
    config = Config(base_path = '/share/klab/datasets/',   analysis_dir = "/share/klab/datasets/TNN_paper_save_dir/All-TNN_share/")

    # Determine analysis subdirectory based on flags
    analysis_subdir = 'analysis_prerelu' if config.PRE_RELU else 'analysis_postrelu'
    analysis_subdir = 'analysis_postnorm' if config.POST_NORM else analysis_subdir

    # Construct and create directories
    hparams['analysis_dir'] = os.path.join(
        saved_model_path, f"{analysis_subdir}{config.ANALYSIS_DIR_SUFFIX}"
    )
    print(f"Saving results to {hparams['analysis_dir']}")
    os.makedirs(hparams['analysis_dir'], exist_ok=True)

    hparams['cache_dir'] = os.path.join(hparams['analysis_dir'], 'cache', f"ep_{epoch}")
    os.makedirs(hparams['cache_dir'], exist_ok=True)

    # Build data generators
    data_generators_dict = {
        'full_test': get_dataset(hparams, 'test', dataset_path=config.ECOSET_PATH),
        'scenes_test': get_dataset(hparams, 'test', dataset_path=config.STF_PATH, dataset_subset='scenes'),
        'tools_test': get_dataset(hparams, 'test', dataset_path=config.STF_PATH, dataset_subset='tools'),
        'vgg_faces_test': get_dataset(hparams, 'test', dataset_path=config.STF_PATH, dataset_subset='faces'),
    }

    # Acquire a dummy batch to determine data shapes
    for x in data_generators_dict[next(iter(data_generators_dict))]:
        data_generators_dict['dummy_batch'] = x[0]
        break

    # Expand dimensions if needed
    if len(data_generators_dict['dummy_batch'].shape) == 3:
        data_generators_dict['dummy_batch'] = np.expand_dims(data_generators_dict['dummy_batch'], axis=0)

    data_generators_dict['dummy_activities_batch'] = activities_model(data_generators_dict['dummy_batch'])

    return hparams, data_generators_dict


def add_epoch_to_save_dir_name(save_path, epoch, part_place=-2):
    parts = save_path.split('/')
    parts[part_place] = parts[part_place] + f'ep_{epoch}'
    return '/'.join(parts)


def nested_dict():
    """Generates a nested dictionary of unlimited depth."""
    return defaultdict(nested_dict)


def save_all_neural_level_analyses_results(
    directory,
    epoch,
    seed,
    model_names,
    multi_models_neural_dict,
    all_l1_norm_data,
    save_name_preffix=''
):
    directory_path = os.path.join(
        directory,  str(epoch), f'seed{seed}' # 'neural_level_analysis'
    )
    os.makedirs(directory_path, exist_ok=True)

    print(f"Saving neural level results to {directory_path}")
    # Save all results as h5py (recommand) or pickle
    try:
        convert_dict2h5(dict_path, h5_file= f'{save_name_preffix}multi_models_neural_dict.h5')
    except:
        pickle.dump(
            multi_models_neural_dict,
            open(os.path.join(directory_path, f'{save_name_preffix}multi_models_neural_dict.pickle'), 'wb+')
        )

    results_to_disk(directory, multi_models_neural_dict)

def analysis_multi_models():
    """
    Sequentially run all implemented analyses (if enabled) for multiple models
    in one specific epoch (Comparison Benchmark).
    """
    config = Config(base_path = '/share/klab/datasets/',   analysis_dir = "/share/klab/datasets/TNN_paper_save_dir/All-TNN_share/")
    print(f"Running analysis for models: {config.MODEL_NAME_PATH_DICT.keys()}")

    model_names = list(config.MODEL_NAME_PATH_DICT.keys())[:]  # MODELS[1:], skip 'Human'
    saved_model_paths = [config.MODEL_NAME_PATH_DICT[mn] for mn in model_names]

    # Handle epochs to analyze
    epochs_to_analyze = config.TEST_EPOCH
    if isinstance(epochs_to_analyze, int) and not config.TRAVES_EPOCHS:
        checkpoints = [epochs_to_analyze]
    else:
        checkpoints = config.TRAVES_EPOCHS

    # ==============================
    # Neural-Level Analyses
    # ==============================
    if config.ANALYSIS_ON_NEURAL_LEVEL:
        for seed in config.SEEDS_RANGE:
            for epoch in checkpoints:
                
                # nested dictionary to store all neural-level analyses
                multi_models_neural_dict = nested_dict()
                all_l1_norm_data = []

                for model_name, saved_model_path in zip(model_names, saved_model_paths):
                    # Replace 'seed1' with the current seed
                    if 'shift' not in model_name:
                        #* only 1 seed for shifted_TNN_alpha_10_lr_0.05, so no need to replace
                        saved_model_path = saved_model_path.replace('seed1', f'seed{seed}')

                    # Possibly override epoch if early stopping is enabled
                    epoch = config.MODELS_EPOCHS_DICT[model_name][seed - 1] if config.EARLY_STOPPING_FLAG else epoch
                    print(f'Analyzing epoch {epoch} for model {model_name} in {saved_model_path}')

                    # Load hyperparameters
                    hparams = load_and_override_hparams(saved_model_path, batch_size=config.BATCH_SIZE)
                    hparams['dataset'] = Econfig.COSET_PATH
                    hparams['num_classes'] = 565

                    # Load the model
                    activities_model, model = load_models(epoch, hparams, saved_model_path, config.PRE_RELU, config.POST_NORM)

                    # Set up analysis directories and data generators
                    hparams, data_generators_dict = setup_analysis_directories_and_data_generators(
                        saved_model_path, hparams, epoch,
                        config.PRE_RELU, config.POST_NORM,
                        config.ANALYSIS_DIR_SUFFIX, config.ECOSET_PATH, config.STF_PATH,
                        activities_model
                    )

                    output_dict = {}


                    # Figure 1B & 5A: Categorization performance
                    if (config.CATEGORIZATION_PERFORMANCE and not ('simclr' in model_name.lower() and 'finetune' not in model_name.lower())):
                       # Whether dataset labels need be renamed if not matching the model's output layer
                        if model.layers[-1]._name not in ['dense_2']:
                            dataset_label_renamer = DatasetLabelRenamer(model, default_label_key='dense_2')
                            new_dataset = data_generators_dict['full_test'].map(dataset_label_renamer, num_parallel_calls=tf.data.AUTOTUNE)
                            loss, acc, topk_acc = model.evaluate(new_dataset)
                        else:
                            loss, acc, topk_acc = model.evaluate(data_generators_dict['full_test'])
                        
                        output_dict.update({
                            'ecoset_test_accuracy': acc,
                            'ecoset_test_loss': loss,
                            'ecoset_test_topk_accuracy': topk_acc
                        })

                    # Figure 2A & 2C & 5B: Orientation selectivity
                    if ORIENTATION_SELECTIVITY:
                        output = orientation_selectivity(
                            activities_model, config.WAVELENGTHS, config.N_ANGLES, hparams,
                            config.ENTROPY_SLIDING_WINDOW_SIZE, output_dict
                        )
                        output_dict.update(output)

                    # Figure 2D & 5C: Category selectivity
                    if config.CATEGORY_STATS:
                        output = compute_category_selectivities(
                            activities_model,
                            hparams,
                            data_generators_dict,
                            config.SELECTIVITY_DATASETS,
                        )
                        output_dict.update(output)

                    # Figure 3A & 5D/5E/5F: Energy efficiency
                    if config.ENERGY_EFFICIENCY:
                        try:
                            feature_responses = load_feature_responses(hparams)
                        except:
                            feature_responses= get_feature_responses(activities_model, 
                                                                    data_generators_dict,
                                                                    dataset_names= ['full_test'],  # ['vgg_faces_test', 'full_test']
                                                                    hparams=hparams,
                                                                    pre_relu = config.PRE_RELU,
                                                                    post_norm = config.POST_NORM,
                                                                    )
                        analyze_energy_efficiency(
                            feature_responses,
                            model_name,
                            hparams,
                            all_l1_norm_data,
                            multi_models_neural_dict,
                            config.PRE_RELU,
                            config.POST_NORM,
                            output_dict['grating_w_entropies'],
                            epoch=epoch
                        )

                        analyze_activation_transmission_efficiency(
                            model.layers[1].layers if 'simclr' in model_name.lower() else model.layers,
                            data_generators_dict['full_test'],
                            epoch,
                            feature_responses,
                            model_name,
                            hparams,
                            multi_models_neural_dict,
                            PRE_RELU,
                            NORM_PREV_LAYER=config.NORM_PREV_LAYER,
                            NORM_LAYER_OUT=config.NORM_LAYER_OUT
                        )

                    # Visualizing model, computing spatial smoothness losses
                    spatial_losses = visualize_model(
                        model.layers[1].layers if 'simclr' in model_name.lower() else model.layers,
                        epoch,
                        output_dict,
                        hparams, 
                        get_spatial_loss=True,
                    )
                    output_dict['mean_cosdist'] = spatial_losses

                    # Figure 1C & 5B: Spatial smoothness
                    if config.SMOOTHNESS:
                        output = smoothness_main(
                            output_dict,
                            epoch=epoch, 
                            analysis_path=hparams['analysis_dir']
                        )
                        output_dict.update(output)

                    # Cleanup
                    del model, activities_model
                    multi_models_neural_dict[model_name] = output_dict

                # Save summarized analyses
                if config.SUMMARY_of_NEURAL_LEVEL_ANALYSES:
                    save_all_neural_level_analyses_results(
                        NEURAL_LEVEL_RESULT_DIR,
                        epoch,
                        seed,
                        model_names,
                        multi_models_neural_dict,
                        all_l1_norm_data,
                        save_name_preffix=config.SAVE_NEURAL_RESULTS_DICT_NAME_PREFIX
                    )

    # ==============================
    # Behaviour-Level Analyses
    # ==============================
    if config.ANALYSIS_ON_BEHAVIOUR_LEVEL:
        MODELS = list(config.MODEL_NAME_PATH_DICT.keys())
        if len(checkpoints) > 1:
            models_epochs = [[ep] * len(MODELS) for ep in checkpoints]
        else:
            models_epochs = [checkpoints * len(MODELS)]

        multi_models_behaviour_result_dict = {}
        multi_models_across_epochs_behaviour_result_dict = {}
        all_epochs_models_overal_all_normed_average_raw_acc_maps_adms = []
        all_epochs_models_overal_all_normed_average_acc_maps_normed_by_individual_map_adms = []
        all_epochs_models_overal_all_normed_average_acc_maps_normed_by_seeds_performance_adms = []

        for this_epoch in models_epochs:
            epoch = this_epoch[0]
            multi_models_behaviour_result_dict[epoch] = {}
            multi_models_across_epochs_behaviour_result_dict[epoch] = {}

            for size_factor in config.SIZE_FACTORS:
                multi_models_behaviour_result_dict[epoch][size_factor] = {}
                multi_models_across_epochs_behaviour_result_dict[epoch][size_factor] = {}

                for alignment_mode in config.ALIGNMENT_MODES:
                    multi_models_behaviour_result_dict[epoch][size_factor][alignment_mode] = {}
                    multi_models_across_epochs_behaviour_result_dict[epoch][size_factor][alignment_mode] = {}

                    for map_norm_mode in config.MAP_NORM_MODES:
                        multi_models_behaviour_result_dict[epoch][size_factor][alignment_mode][map_norm_mode] = {}
                        multi_models_across_epochs_behaviour_result_dict[epoch][size_factor][alignment_mode][map_norm_mode] = {}

                        save_path_suffix = (
                            f"{this_epoch[1]}_{map_norm_mode}_{size_factor}_"
                            f"{alignment_mode}_{config.ALIGNMENT_METRIC}_{config.RSA_METRIC}_{config.SEEDS_RANGE}"
                        )

                        if config.GET_BEHAVIOURAL_DATA:
                            behaviour_data = get_behaviour_data(
                                model_names=MODELS[:],
                                test_epochs=this_epoch[:],
                                map_norm_mode=map_norm_mode,
                                alphas=list(config.ALPHAS.values()), 
                                seeds_list=config.SEEDS_RANGE,
                                result_source_dir=config.BEHAVIOUR_RESULT_SOURCE_DIR,
                                size_factor=size_factor,
                                SUBJ_START=config.SUBJ_START,
                                SUBJ_END=config.SUBJ_END,
                                decimals=15,
                                model_epoch_map = config.MODELS_EPOCHS_DICT,
                                analysis_datasource_type='',
                            )
                            output_behaviour_dict = {'behaviour_data': behaviour_data}

                        # --------------------------
                        # Figure 4E & 5G: Behavioural alignment
                        # --------------------------
                        if config.BEHAVIOUR_ALIGNMENT:
                            config.COLUMNS[1] = config.ALIGNMENT_METRIC
                            behaviour_alignment_columns = config.COLUMNS + ["condition"]

                            df_behaviour_agreements, noise_ceiling, significance_dict = visual_behaviour_alignment_analysis(
                                acc_maps_data=[
                                    behaviour_data['raw_individual_participants_5x5_acc_maps_array']
                                ] + [
                                    behaviour_data['seeds_model_data'].get(mn, [])
                                    for mn in MODELS[:]
                                ],
                                alignment_metric=config.ALIGNMENT_METRIC,
                                model_names=MODELS,
                                comparison_modes=config.BEHAVIOUR_ANALYSIS_MODES,
                                columns=behaviour_alignment_columns,
                                # seeds_list=SEEDS_RANGE,
                                categories_num=config.CATEGORIES_NUM,
                                map_norm_mode=map_norm_mode,
                                sampling_num=config.SAMPLING_NUM,
                                alignment_mode = alignment_mode,
                                model_names_to_plot=config.MODEL_NAMES_TO_PLOT,
                                verbose=False,
                            )

                            plot_significance_matrix(
                                significance_dict,
                                roi_labels_to_plot=['Accuracy Maps Agreement'],
                                model_names=MODELS[:],
                                model_names_to_plot=config.MODEL_NAMES_TO_PLOT,
                                save_dir=os.path.join(config.BEHAVIOUR_ANALYSIS_RESULT_DIR, f'ep_{save_path_suffix}'),
                                num_cols=1
                            )

                            df_behaviour_agreements.to_csv(f"{config.BEHAVIOUR_ANALYSIS_RESULT_DIR}/df_behaviour_agreement.csv")

                            # Noise ceiling line
                            hline = {'value': noise_ceiling, 'color': 'black', 'linestyle': 'dashed', 'linewidth': 1} if config.PLOT_NC else None

                            
                            plot_bar_plot_from_df(
                                df_behaviour_agreements,
                                add_epoch_to_save_dir_name(config.BEHAVIOUR_AGREEMENT_ANALYSIS_PATH, save_path_suffix),
                                x="Model",
                                y=config.ALIGNMENT_METRIC,
                                title="Behaviour Agreement Analysis",
                                show_plot=False,
                                color3_start_id=1,
                                hline=hline,
                                figsize=(3.54, 2),
                                y_breaks = [(0, 0.2),  (0.40,  0.44)],

                                log_scale=False, significance_dict=None, 
                                show_barplot=True, bar_width=0.6, bar_alpha=0.4,error_bar_width=2,
                                # Hybrid
                                show_boxplot=True, box_width=0.2, box_linewidth=1, box_alpha=0.2, box_whis=(5, 95),
                                point_plot=None, point_plot_kwargs={"size": 2, "alpha": 0.5}, # "strip", "swarm", or None
                                verbose=True,
                            )

                        # --------------------------
                        # Figure 4E & 5H: Behavioural ADM alignment
                        # --------------------------
                        if config.BEHAVIOURAL_ADM_ALIGNMENT:
                            config.COLUMNS[1] = config.RSA_METRIC
                            df_adm_agreement, noise_ceiling, significance_dict, adm_dict = adm_alignment_analysis(
                                raw_individuals_vs_seeds_model_acc_maps=[
                                    behaviour_data['raw_individual_participants_5x5_acc_maps_array']
                                ] + [
                                    behaviour_data['seeds_model_data'].get(mn, [])
                                    for mn in MODELS[:]
                                ],
                                model_names=MODELS,
                                seeds_list=config.SEEDS_RANGE,
                                comparison_modes=config.BEHAVIOUR_ANALYSIS_MODES,
                                create_adm_metric=config.ADM_METRIC,
                                rsa_metric=config.RSA_METRIC,
                                columns=config.COLUMNS ,
                                categories_num=config.CATEGORIES_NUM,
                                map_norm_mode=map_norm_mode,
                                sampling_num=config.SAMPLING_NUM,
                                alignment_mode = alignment_mode,
                                model_names_to_plot=config.MODEL_NAMES_TO_PLOT,
                                verbose=False,
                            )

                            os.makedirs(
                                os.path.join(config.BEHAVIOUR_ANALYSIS_RESULT_DIR, f'ep_{save_path_suffix}'),
                                exist_ok=True
                            )
                            adm_dict_file_path = os.path.join(
                                config.BEHAVIOUR_ANALYSIS_RESULT_DIR,
                                f'ep_{save_path_suffix}',
                                f'adm_dict_{this_epoch[1]}_{size_factor}_{config.ALIGNMENT_METRIC}_{config.RSA_METRIC}.h5'
                            )
                            # with h5py.File(adm_dict_file_path, 'w') as hf:
                            #     for k, v in adm_dict.items():
                            #         hf.create_dataset(k, data=v)

                            plot_significance_matrix(
                                significance_dict,
                                roi_labels_to_plot=['ADM Agreement'],
                                model_names=MODELS[:],
                                model_names_to_plot=config.MODEL_NAMES_TO_PLOT,
                                save_dir=os.path.join(config.BEHAVIOUR_ANALYSIS_RESULT_DIR, f'ep_{save_path_suffix}'),
                                num_cols=1
                            )

                            df_adm_agreement.to_csv(f"{config.BEHAVIOUR_ANALYSIS_RESULT_DIR}/df_adm_agreement.csv")

                            # Noise ceiling line
                            hline = {'value': noise_ceiling, 'color': 'black', 'linestyle': 'dashed', 'linewidth': 1} if config.PLOT_NC else None

                            # TODO try no boostrap? 
                            # import pdb; pdb.set_trace()
                            plot_bar_plot_from_df(
                                df_adm_agreement,
                                add_epoch_to_save_dir_name(config.ADM_AGREEMENT_ANALYSIS_PATH, save_path_suffix),
                                x="Model",
                                y=config.RSA_METRIC,
                                title="Noise Ceiling Corrected ADM Agreement Analysis",
                                show_plot=False,
                                color3_start_id=1,
                                hline=hline,
                                figsize=(3.54, 2),
                                y_breaks = [(0, 0.15),  (0.30,  0.34)], 

                                log_scale=False, significance_dict=None, 
                                show_barplot=True, bar_width=0.6, bar_alpha=0.4,error_bar_width=2,
                                # Hybrid
                                show_boxplot=True, box_width=0.2, box_linewidth=1, box_alpha=0.2, box_whis=(5, 95),
                                point_plot=None, point_plot_kwargs={"size": 2, "alpha": 0.5}, 
                                verbose=True, 
                            )


if __name__ == '__main__':
    analysis_multi_models()