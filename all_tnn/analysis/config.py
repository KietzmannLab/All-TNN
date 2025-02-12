import os

# -------------------------------------------------------------------
# -----------------------  Dataset paths setup  ----------------------
# -------------------------------------------------------------------
ANALYSIS_DIR_SUFFIX = ''  # Optionally add a custom suffix at the end of the results directory
BASE_PATH = '/share/klab/datasets/' # where the ecoset datasets are stored
ECOSET_PATH = BASE_PATH + 'ecoset_square256_proper_chunks.h5' # default dataset
ANALYSIS_DIR = '/share/klab/datasets/TNN_paper_save_dir/All-TNN_share/' # where the analysis related data are stored
STF_PATH = ANALYSIS_DIR + '/datasets/category_selectivity_stimuli.h5'  # Scenes, tools, faces datasets for category selectivity analysis

# -------------------------------------------------------------------
# -----------------------  Model paths setup  ------------------------
# -------------------------------------------------------------------
BASE_MODEL_DIR = ANALYSIS_DIR + 'shared_weights/'  # Base directory for all models 
NEURAL_LEVEL_RESULT_DIR = ANALYSIS_DIR + 'neural_level_src/' + 'neural_analysis_results/' # "./save_dir/_analyses_data/"
BEHAVIOUR_RESULT_SOURCE_DIR  = ANALYSIS_DIR + 'behaviour_src/'
VERBOSE = 1              # Whether to display all warnings
TEST_EPOCH = None        # If specified, use a specific epoch (e.g. 300). Otherwise, traverse the list below.
TRAVES_EPOCHS = [600]    # e.g. [35,50,100,150,200,250,300,350,400,450,500,550,600]
EARLY_STOPPING_FLAG = [False, True][1]  # Whether to use early stopping

# -------------------------------------------------------------------
# ---------------  Range of seeds and model definitions  -------------
# -------------------------------------------------------------------
SEEDS_RANGE = [1, 2, 3, 4, 5][:]

#* Model names and their corresponding paths
MODEL_NAME_PATH_DICT = {
    'Human': None,

    'CNN': 'tnn_conv_control_ecoset_l2_no_flip_seed1_drop0.0_learnable_False_1e-05_alpha0.0_constant_20_0.1Factors_adam0.05_L21e-06_ecoset_square256_chunked',
    'LCN': 'tnn_ecoset_l2_no_flip_seed1_drop0.0_learnable_False_1e-05_alpha0.0_constant_20_0.1Factors_adam0.05_L21e-06_ecoset_square256_proper_chunks',
    'TNN_alpha_1': 'tnn_ecoset_l2_no_flip_seed1_drop0.0_learnable_False_1e-05_alpha1.0_constant_20_0.1Factors_adam0.05_L21e-06_ecoset_square256_proper_chunks',
    'TNN_alpha_10': 'tnn_ecoset_l2_no_flip_seed1_drop0.0_learnable_False_1e-05_alpha10.0_constant_20_0.1Factors_adam0.05_L21e-06_ecoset_square256_proper_chunks',
    'TNN_alpha_100': 'tnn_ecoset_l2_no_flip_seed1_drop0.0_learnable_False_1e-05_alpha100.0_constant_20_0.1Factors_adam0.05_L21e-06_ecoset_square256_proper_chunks',
    
    'shifted_TNN_alpha_10': 'tnn_ecoset_l2_no_flip_seed1_drop0.0_learnable_False_2024.0_alpha10.0_constant_20_0.1Factors_adam0.05_L21e-06_ecoset_shifted_square256_proper_chunks',
    'TNN_simclr_finetune': 'finetuned_tnn_simclr_no_flip_ecoset_seed1_drop0.0_learnable_False_1e-05_alpha10.0_constant_20_0.1Factors_adam0.05_L21e-06_ecoset_square256_proper_chunks', 
}
MAIN_MODEL_NAMES = list(MODEL_NAME_PATH_DICT.keys())
MODELS_EPOCHS_DICT = {
    "CNN": [35] * 5,
    "LCN": [35] * 5,
    "TNN_alpha_1": [35] * 5,
    "TNN_alpha_10": [300, 270, 260, 300, 230],
    "TNN_alpha_100": [600] * 5,
    "TNN_simclr_finetune": [600] * 5,  
    "shifted_TNN_alpha_10": [300] * 5,
}
# Prepend BASE_MODEL_DIR to every path in MODEL_NAME_PATH_DICT (when not None)
MODEL_NAME_PATH_DICT = {
    model: os.path.join(BASE_MODEL_DIR, subdir)
    for model, subdir in MODEL_NAME_PATH_DICT.items() 
    if subdir is not None
}

ALPHAS = {
    "CNN": 0.0,
    "LCN": 0.0,
    "TNN_alpha_1": 1.0,
    "TNN_alpha_10": 10.0,
    "TNN_alpha_100": 100.0,
    "shifted_TNN_alpha_10": 10.0,
    "TNN_simclr": 10.0,
    "TNN_simclr_finetune": 10.0,
}

MODEL_NAMES_TO_PLOT = {
    'CNN': 'CNN',
    'LCN': 'LCN',
    'TNN_alpha_1': 'All-TNN\n($\\alpha=1$)',
    'TNN_alpha_10': 'All-TNN\n($\\alpha=10$)',
    'TNN_alpha_100': 'All-TNN\n($\\alpha=100$)',
    'TNN_simclr_finetune': 'All-TNN\nSimCLR\n($\\alpha=10$)',
    'shifted_TNN_alpha_10': 'Shifted All-TNN\n($\\alpha=10$)',
}

# -------------------------------------------------------------------
# ---------------  Neural-level analysis toggles  -------------------
# -------------------------------------------------------------------
ANALYSIS_ON_NEURAL_LEVEL = [False, True][1]           #* Whether to run neural-level analyses
SAVE_NEURAL_RESULTS_DICT_NAME_PREFIX = ['', 'debug', 'all_'][-1] #* name for saving nerual level dict
SUMMARY_of_NEURAL_LEVEL_ANALYSES = [False, True][1]  # Summarize all neural-level analyses at the end

CATEGORIZATION_PERFORMANCE = [False, True][1]  # Accuracy analysis (with Spatial loss)
GET_SPATIAL_LOSS = [False, True][1]           # Spatial loss
ORIENTATION_SELECTIVITY = [False, True][1] # need turn on if analyze energy efficiency
CATEGORY_STATS = [False, True][1]
SMOOTHNESS = [False, True][1]                 # Whether to calculate model smoothness
ENERGY_EFFICIENCY = [False, True][1]
DISTANCE_VS_WEIGHT_SIMILARITY = [False, True][0]

PRE_RELU = [False, True][0]                # Take pre-ReLU activations?
POST_NORM = [False, True][0]               # Take activations after normalization?
NORM_PREV_LAYER = [False, True][1]         # Normalize activations of the previous layer? (default True)
NORM_LAYER_OUT = [False, True][0]          # Normalize activations of the current layer? (default False)

BATCH_SIZE = 500
OVERWRITE = [False, True][0]               # Overwrite category selectivities & searchlight maps?

# Orientation selectivity analysis
WAVELENGTHS = [3] 
N_ANGLES = 8
ENTROPY_SLIDING_WINDOW_SIZE = 8  

# Category selectivity analysis
D_PRIME_THRESHOLD = 0.85
N_PERMUTATIONS = 10000
CLUSTER_ANALYSIS = True
SELECTIVITY_DATASETS = ['vgg_faces_test', 'scenes_test', 'tools_test']
SEARCHLIGHT_RADIUS = 6
SEARCHLIGHT_STRIDE = 2
SEARCHLIGHT_N_SPLITS = 3


# -------------------------------------------------------------------
# ---------------  Behaviour-level analysis toggles  ----------------
# -------------------------------------------------------------------
ANALYSIS_ON_BEHAVIOUR_LEVEL = [False, True][0]  #* Whether to run behavioural-level analyses
PLOT_NC = False # Plot noise ceiling
GET_BEHAVIOURAL_DATA = [False, True][1]
BEHAVIOUR_ALIGNMENT = [False, True][1]
BEHAVIOURAL_ADM_ALIGNMENT = [False, True][1]


SUBJ_START = 0
SUBJ_END = 30  # Number of subjects
CATEGORIES_NUM = 16                     # Number of categories in the behavioural analysis
USING_PREDEFINED_CATEGORIES = [False, True][0]
USING_CLUSTERED_CATEGORY_ORDER = True
MAP_NORM_MODES = ["max"]                # For relative comparison, divide by the maximum value
ALIGNMENT_MODES = ['mean']
BEHAVIOUR_ANALYSIS_MODES = 'individual_model_vs_average_human'
SIZE_FACTORS = ['relative_white_controlled_original_largest205_min20px_dataset_filtered_num500_grid_fixed_back255',]

# Metrics for behavioural analysis
COLUMNS = ["Model", "Correlation", 'Condition']
SAMPLING_NUM = 100
ALIGNMENT_ANALYSIS_MODE = "bootstrap_average_human"
ALIGNMENT_METRIC = "pearsonr"
ADM_ANALYSIS_MODE = "bootstrap_average_human"
ADM_METRIC = "pearsonr"
RSA_METRIC = "spearmanr"

# -------------------------------------------------------------------
# -----  Saving directories and filenames for behaviour  -----------
# -------------------------------------------------------------------
BEHAVIOUR_ANALYSIS_RESULT_DIR = BEHAVIOUR_RESULT_SOURCE_DIR + 'behaviour_analysis_results/'
BEHAVIOUR_AGREEMENT_ANALYSIS_PATH = BEHAVIOUR_ANALYSIS_RESULT_DIR + "/behaviour_agreements.pdf"
ADM_AGREEMENT_ANALYSIS_PATH =  BEHAVIOUR_ANALYSIS_RESULT_DIR + "/adm_agreements.pdf"