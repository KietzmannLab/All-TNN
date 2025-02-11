import os

# -------------------------------------------------------------------
# -----------------------  Dataset paths setup  ----------------------
# -------------------------------------------------------------------
ANALYSIS_DIR_SUFFIX = ''  # Optionally add a custom suffix at the end of the results directory
base_path = '/share/klab/datasets/'

STF_PATH = base_path + 'TNN_paper_datasets/analysis_dataset10.h5'  # Scenes, tools, faces datasets for category selectivity analysis
ECOSET_PATH = base_path + 'ecoset_square256_proper_chunks.h5' # default dataset

# -------------------------------------------------------------------
# -----------------------  Model paths setup  ------------------------
# -------------------------------------------------------------------
BASE_DIR = '/share//klab/datasets/TNN_paper_save_dir/All-TNN_public/shared_weights'  #'./save_dir' #* Base directory for all models | for shared weights need run energy analysis from 35 to 600
BEHAVIOUR_RESULT_SOURCE_DIR  = '/share/klab/datasets/TNN_paper_save_dir/All-TNN_public/behaviour_src/'
VERBOSE = 1              # Whether to display all warnings
TEST_EPOCH = None        # If specified, use a specific epoch (e.g. 300). Otherwise, traverse the list below.
TRAVES_EPOCHS = [300]    # e.g. [35,50,100,150,200,250,300,350,400,450,500,550,600]
EARLY_STOPPING_FLAG = [False, True][1]  # Whether to use early stopping

# -------------------------------------------------------------------
# ---------------  Range of seeds and model definitions  -------------
# -------------------------------------------------------------------
SEEDS_RANGE = [1, 2, 3, 4, 5][:1]

MODEL_NAMES = [

    "CNN_lr_0.05",
    "LCN_lr_0.05",
    "TNN_alpha_1_lr_0.05",
    "TNN_alpha_10_lr_0.05",
    "TNN_alpha_100_lr_0.05",

    "TNN_simclr_finetune",
    "shifted_TNN_alpha_10_lr_0.05"
]

MODELS_EPOCHS_DICT = {
    
    "CNN_lr_0.05": [35] * 5,
    "LCN_lr_0.05": [35] * 5,
    "TNN_alpha_1_lr_0.05": [35] * 5,
    "TNN_alpha_10_lr_0.05": [300, 270, 260, 300, 230],
    "TNN_alpha_100_lr_0.05": [600] * 5,
    
    "TNN_simclr_finetune": [600] * 5,  
    "shifted_TNN_alpha_10_lr_0.05": [300] * 5,
}


#* Should align with the model names in MODELS_EPOCHS_DICT
MODEL_NAME_PATH_DICT = {
    'Human': None,
    'TNN_simclr_finetune': 'finetuned_tnn_simclr_no_flip_ecoset_seed1_drop0.0_learnable_False_1e-05_alpha10.0_constant_20_0.1Factors_adam0.05_L21e-06_ecoset_square256_proper_chunks', 
    'shifted_TNN_alpha_10_lr_0.05': 'tnn_ecoset_l2_no_flip_seed1_drop0.0_learnable_False_2024.0_alpha10.0_constant_20_0.1Factors_adam0.05_L21e-06_ecoset_shifted_square256_proper_chunks',

    'CNN_lr_0.05': 'tnn_conv_control_ecoset_l2_no_flip_seed1_drop0.0_learnable_False_1e-05_alpha0.0_constant_20_0.1Factors_adam0.05_L21e-06_ecoset_square256_chunked',
    'LCN_lr_0.05': 'tnn_ecoset_l2_no_flip_seed1_drop0.0_learnable_False_1e-05_alpha0.0_constant_20_0.1Factors_adam0.05_L21e-06_ecoset_square256_proper_chunks',
    'TNN_alpha_1_lr_0.05': 'tnn_ecoset_l2_no_flip_seed1_drop0.0_learnable_False_1e-05_alpha1.0_constant_20_0.1Factors_adam0.05_L21e-06_ecoset_square256_proper_chunks',
    'TNN_alpha_10_lr_0.05': 'tnn_ecoset_l2_no_flip_seed1_drop0.0_learnable_False_1e-05_alpha10.0_constant_20_0.1Factors_adam0.05_L21e-06_ecoset_square256_proper_chunks',
    'TNN_alpha_100_lr_0.05': 'tnn_ecoset_l2_no_flip_seed1_drop0.0_learnable_False_1e-05_alpha100.0_constant_20_0.1Factors_adam0.05_L21e-06_ecoset_square256_proper_chunks',


}

# Prepend BASE_DIR to every path in MODEL_NAME_PATH_DICT (when not None)
MODEL_NAME_PATH_DICT = {
    model: os.path.join(BASE_DIR, subdir)
    for model, subdir in MODEL_NAME_PATH_DICT.items() 
    if subdir is not None
}

ALPHAS = {
    "CNN_lr_0.05": 0.0,
    "LCN_lr_0.05": 0.0,
    "TNN_alpha_1_lr_0.05": 1.0,
    "TNN_alpha_10_lr_0.05": 10.0,
    "TNN_alpha_100_lr_0.05": 100.0,
    "shifted_TNN_alpha_10_lr_0.05": 10.0,
    "8_neighbours_TNN_alpha_10_lr_0.05": 10.0,
    "TNN_simclr": 10.0,
    "TNN_simclr_finetune": 10.0,
}

MODEL_NAMES_TO_PLOT = {
    'CNN_lr_0.05': 'CNN',
    'LCN_lr_0.05': 'LCN',
    'TNN_alpha_1_lr_0.05': 'All-TNN\n($\\alpha=1$)',
    'TNN_alpha_10_lr_0.05': 'All-TNN\n($\\alpha=10$)',
    'TNN_alpha_100_lr_0.05': 'All-TNN\n($\\alpha=100$)',
    'TNN_simclr_finetune': 'All-TNN\nSimCLR\n($\\alpha=10$)',
    'shifted_TNN_alpha_10_lr_0.05': 'Shifted All-TNN\n($\\alpha=10$)',
    '8_neighbours_TNN_alpha_10_lr_0.05': '8-Neighbours All-TNN\n($\\alpha=10$)',
    "TDANN_imagenet_supervised": 'TDANN\nSupervised',
    "TDANN_imagenet_self_supervised": 'TDANN\nSelf-Supervised',
    'TDANN\nSupervised': 'TDANN\nSupervised',
    'TDANN\nSelf-Supervised': 'TDANN\nSelf-Supervised',
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
# ---------------  Saving directories and filenames  ----------------
# -------------------------------------------------------------------
ANALYSIS_SAVE_DIR = "./save_dir/_analyses_data/"
NEURAL_ANALYSIS_RESULT_DIR = ANALYSIS_SAVE_DIR + 'neural_analysis_results/'
BEHAVIOUR_ANALYSIS_RESULT_DIR = ANALYSIS_SAVE_DIR + 'behaviour_analysis_results/'
BEHAVIOUR_AGREEMENT_ANALYSIS_PATH = BEHAVIOUR_ANALYSIS_RESULT_DIR + "/behaviour_agreements.pdf"
ADM_AGREEMENT_ANALYSIS_PATH =  BEHAVIOUR_ANALYSIS_RESULT_DIR + "/adm_agreements.pdf"