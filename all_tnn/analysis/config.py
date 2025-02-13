import os


class Config:

    def __init__(
        self,
        base_path="/share/klab/datasets/",
        analysis_dir="/share/klab/datasets/TNN_paper_save_dir/All-TNN_share-test-varun/",
    ):
        # -------------------------------------------------------------------
        # -----------------------  Dataset paths setup  ----------------------
        # -------------------------------------------------------------------
        self.ANALYSIS_DIR_SUFFIX = ""  # Optionally add a custom suffix at the end of the results directory
        self.BASE_PATH = base_path  # Where the ecoset datasets are stored
        self.ECOSET_PATH = self.BASE_PATH + "ecoset.h5"  # Default dataset
        self.ANALYSIS_DIR = (
            analysis_dir  # Where the analysis related data are stored
        )
        self.STF_PATH = (
            self.ANALYSIS_DIR + "/datasets/category_selectivity_stimuli.h5"
        )  # Scenes, tools, faces datasets for category selectivity analysis

        # -------------------------------------------------------------------
        # -----------------------  Model paths setup  ------------------------
        # -------------------------------------------------------------------
        self.BASE_MODEL_DIR = (
            self.ANALYSIS_DIR + "shared_weights/"
        )  # Base directory for all models
        self.NEURAL_LEVEL_RESULT_DIR = (
            self.ANALYSIS_DIR
            + "neural_level_src/"
            + "neural_analysis_results/"
        )  # Neural-level results
        self.BEHAVIOUR_RESULT_SOURCE_DIR = self.ANALYSIS_DIR + "behaviour_src/"
        self.VERBOSE = 1  # Whether to display all warnings
        self.TEST_EPOCH = None  # If specified, use a specific epoch (e.g., 300). Otherwise, traverse the list below.
        self.TRAVES_EPOCHS = [600]  # List of epochs
        self.EARLY_STOPPING_FLAG = [False, True][
            1
        ]  # Whether to use early stopping

        # -------------------------------------------------------------------
        # ---------------  Range of seeds and model definitions  -------------
        # -------------------------------------------------------------------
        self.SEEDS_RANGE = [1, 2, 3, 4, 5][:]  # List of seeds to use

        # * Model names and their corresponding paths
        self.MODEL_NAME_PATH_DICT = {
            "Human": None,
            "CNN": "tnn_conv_control_ecoset_l2_no_flip_seed1_drop0.0_learnable_False_1e-05_alpha0.0_constant_20_0.1Factors_adam0.05_L21e-06_ecoset_square256_chunked",
            "LCN": "tnn_ecoset_l2_no_flip_seed1_drop0.0_learnable_False_1e-05_alpha0.0_constant_20_0.1Factors_adam0.05_L21e-06_ecoset_square256_proper_chunks",
            "TNN_alpha_1": "tnn_ecoset_l2_no_flip_seed1_drop0.0_learnable_False_1e-05_alpha1.0_constant_20_0.1Factors_adam0.05_L21e-06_ecoset_square256_proper_chunks",
            "TNN_alpha_10": "tnn_ecoset_l2_no_flip_seed1_drop0.0_learnable_False_1e-05_alpha10.0_constant_20_0.1Factors_adam0.05_L21e-06_ecoset_square256_proper_chunks",
            "TNN_alpha_100": "tnn_ecoset_l2_no_flip_seed1_drop0.0_learnable_False_1e-05_alpha100.0_constant_20_0.1Factors_adam0.05_L21e-06_ecoset_square256_proper_chunks",
            "shifted_TNN_alpha_10": "tnn_ecoset_l2_no_flip_seed1_drop0.0_learnable_False_2024.0_alpha10.0_constant_20_0.1Factors_adam0.05_L21e-06_ecoset_shifted_square256_proper_chunks",
            "TNN_simclr_finetune": "finetuned_tnn_simclr_no_flip_ecoset_seed1_drop0.0_learnable_False_1e-05_alpha10.0_constant_20_0.1Factors_adam0.05_L21e-06_ecoset_square256_proper_chunks",
        }
        self.MODEL_NAMES = list(self.MODEL_NAME_PATH_DICT.keys())[1:]

        self.MODELS_EPOCHS_DICT = {
            "CNN": [35] * 5,
            "LCN": [35] * 5,
            "TNN_alpha_1": [35] * 5,
            "TNN_alpha_10": [300, 270, 260, 300, 230],
            "TNN_alpha_100": [600] * 5,
            "TNN_simclr_finetune": [600] * 5,
            "shifted_TNN_alpha_10": [300] * 5,
        }

        # Prepend BASE_MODEL_DIR to every path in MODEL_NAME_PATH_DICT (when not None)

        self.MODEL_NAME_PATH_DICT = {
            model: os.path.join(self.BASE_MODEL_DIR, subdir)
            for model, subdir in self.MODEL_NAME_PATH_DICT.items()
            if subdir is not None
        }

        self.ALPHAS = {
            "CNN": 0.0,
            "LCN": 0.0,
            "TNN_alpha_1": 1.0,
            "TNN_alpha_10": 10.0,
            "TNN_alpha_100": 100.0,
            "shifted_TNN_alpha_10": 10.0,
            "TNN_simclr": 10.0,
            "TNN_simclr_finetune": 10.0,
        }

        self.MODEL_NAMES_TO_PLOT = {
            "CNN": "CNN",
            "LCN": "LCN",
            "TNN_alpha_1": "All-TNN\n($\\alpha=1$)",
            "TNN_alpha_10": "All-TNN\n($\\alpha=10$)",
            "TNN_alpha_100": "All-TNN\n($\\alpha=100$)",
            "TNN_simclr_finetune": "All-TNN\nSimCLR\n($\\alpha=10$)",
            "shifted_TNN_alpha_10": "Shifted All-TNN\n($\\alpha=10$)",
        }

        # -------------------------------------------------------------------
        # ---------------  Neural-level analysis toggles  -------------------
        # -------------------------------------------------------------------
        self.ANALYSIS_ON_NEURAL_LEVEL = [False, True][1]
        self.SAVE_NEURAL_RESULTS_DICT_NAME_PREFIX = ["", "debug", "all_"][-1]
        self.SUMMARY_of_NEURAL_LEVEL_ANALYSES = [False, True][
            1
        ]  # Summarize neural-level analyses at the end

        self.CATEGORIZATION_PERFORMANCE = [False, True][1]  # Accuracy analysis
        self.GET_SPATIAL_LOSS = [False, True][1]  # Spatial loss
        self.ORIENTATION_SELECTIVITY = [False, True][1]  # Analyze energy efficiency
        self.CATEGORY_STATS = [False, True][1]
        self.SMOOTHNESS = [False, True][1]  # Calculate model smoothness
        self.ENERGY_EFFICIENCY = [False, True][1]
        self.DISTANCE_VS_WEIGHT_SIMILARITY = [False, True][0]

        self.PRE_RELU = [False, True][0]
        self.POST_NORM = [False, True][0]
        self.NORM_PREV_LAYER = [False, True][1]
        self.NORM_LAYER_OUT = [False, True][0]

        self.BATCH_SIZE = 500
        self.OVERWRITE = [False, True][0]

        # Orientation selectivity analysis
        self.WAVELENGTHS = [3]
        self.N_ANGLES = 8
        self.ENTROPY_SLIDING_WINDOW_SIZE = 8

        # Category selectivity analysis
        self.D_PRIME_THRESHOLD = 0.85
        self.N_PERMUTATIONS = 10000
        self.CLUSTER_ANALYSIS = True
        self.SELECTIVITY_DATASETS = [
            "vgg_faces_test",
            "scenes_test",
            "tools_test",
        ]
        self.SEARCHLIGHT_RADIUS = 6
        self.SEARCHLIGHT_STRIDE = 2
        self.SEARCHLIGHT_N_SPLITS = 3

        # -------------------------------------------------------------------
        # ---------------  Behaviour-level analysis toggles  ----------------
        # -------------------------------------------------------------------
        self.ANALYSIS_ON_BEHAVIOUR_LEVEL = [False, True][0]
        self.PLOT_NC = False
        self.GET_BEHAVIOURAL_DATA = [False, True][1]
        self.BEHAVIOUR_ALIGNMENT = [False, True][1]
        self.BEHAVIOURAL_ADM_ALIGNMENT = [False, True][1]

        self.SUBJ_START = 0
        self.SUBJ_END = 30  # Number of subjects
        self.CATEGORIES_NUM = 16
        self.USING_PREDEFINED_CATEGORIES = [False, True][0]
        self.USING_CLUSTERED_CATEGORY_ORDER = True
        self.MAP_NORM_MODES = ["max"]
        self.ALIGNMENT_MODES = ["mean"]
        self.BEHAVIOUR_ANALYSIS_MODES = "individual_model_vs_average_human"
        self.SIZE_FACTORS = [
            "relative_white_controlled_original_largest205_min20px_dataset_filtered_num500_grid_fixed_back255"
        ]

        # Metrics for behavioural analysis
        self.COLUMNS = ["Model", "Correlation", "Condition"]
        self.SAMPLING_NUM = 100
        self.ALIGNMENT_ANALYSIS_MODE = "bootstrap_average_human"
        self.ALIGNMENT_METRIC = "pearsonr"
        self.ADM_ANALYSIS_MODE = "bootstrap_average_human"
        self.ADM_METRIC = "pearsonr"
        self.RSA_METRIC = "spearmanr"

        # -------------------------------------------------------------------
        # -----  Saving directories and filenames for behaviour  -----------
        # -------------------------------------------------------------------
        self.BEHAVIOUR_ANALYSIS_RESULT_DIR = (
            self.BEHAVIOUR_RESULT_SOURCE_DIR + "behaviour_analysis_results/"
        )
        self.BEHAVIOUR_AGREEMENT_ANALYSIS_PATH = (
            self.BEHAVIOUR_ANALYSIS_RESULT_DIR + "/behaviour_agreements.pdf"
        )
        self.ADM_AGREEMENT_ANALYSIS_PATH = (
            self.BEHAVIOUR_ANALYSIS_RESULT_DIR + "/adm_agreements.pdf"
        )
