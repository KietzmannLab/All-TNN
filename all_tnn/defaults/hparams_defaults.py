import argparse
import distutils.util as dist

def get_hparams():
     """Get and parse command-line arguments."""
     parser = argparse.ArgumentParser()

     # --- Model Options ---
     parser.add_argument('--model_name', type=str, required=True, help='See models/setup_model.py for available models')
     parser.add_argument('--num_classes', type=int, default=0, help='Number of classes for the dataset')
     parser.add_argument('--layer_regularizer', type=str, default='L2', help='Regularizer for layers')
     parser.add_argument('--add_regularizer_loss', type=str, default='None', help='Add regularizer loss')

     # --- Dataset & Training Options ---
     parser.add_argument('--dataset', type=str, required=True, help='Path to dataset h5 file')
     parser.add_argument('--fixation_heatmaps_path', type=str, default=None, help='Path to fixation heatmap to mimic human fixations')
     parser.add_argument('--embedding_target', type=dist.strtobool, default=False, help='Target is an embedding vector instead of a 1-HOT vector')
     parser.add_argument('--embedding_loss', type=str, default='cosine', choices=['MSE', 'cosine'], help='Loss for embedding target')
     parser.add_argument('--target_dataset_name', type=str, default='labels', help='Dataset type for training')
     parser.add_argument('--model_output_activation', type=str, default='softmax', help='Activation function for output layer')
    
     parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
     parser.add_argument('--image_size', type=int, default=150, help='Size of square images for network input')
     parser.add_argument('--image_normalization', type=str, default='[-1,1]', help='Normalization type for images')
     parser.add_argument('--use_mixed_precision', type=dist.strtobool, default=True, help="Use mixed precision for training")
     parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamW', 'adaBelief', 'sgd', 'lamb'], help='Optimizer type')
     parser.add_argument('--adam_beta_1', type=float, default=0.9, help='Beta_1 parameter for Adam-like optimizers')
     parser.add_argument('--adam_beta_2', type=float, default=0.999, help='Beta_2 parameter for Adam-like optimizers')
     parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
     parser.add_argument('--optim_epsilon', type=float, default=1e-07, help='Epsilon parameter for Adam-like optimizers')
     parser.add_argument('--clip_norm', type=float, default=500, help='Gradient clipping value')
     parser.add_argument('--calculate_class_weights', type=dist.strtobool, default=False, help='Calculate class weights for loss')

     parser.add_argument('--start_epoch', type=int, default=-1, help='Epoch to start training from')
     parser.add_argument('--n_epochs', type=int, default=1, help='Number of epochs for training')
     parser.add_argument('--total_epochs', type=int, default=0, help='Total number of epochs for learning rate scheduler')
     parser.add_argument('--learning_rate_schedule', type=str, default='none', choices=['cosine', 'cosine_restarts', 'none'], help='Learning rate decay schedule')
     parser.add_argument('--simclr_temperature', type=float, default=0.1, help='Temperature for simclr loss')

     # --- Dropout Options ---
     parser.add_argument('--dropout_rate', type=float, default=0, help='Dropout rate')
     parser.add_argument('--learnable_dropout_mask', type=dist.strtobool, default=False, help='Use learnable dropout mask')
     parser.add_argument('--dropout_l1_regularizer_value', type=float, default=1e-5, help='L1 regularizer value for learnable dropout mask')

     # --- Normalization & Activation Options ---
     parser.add_argument('--norm_type', type=str, default='no_norm', choices=['BN', 'IN', 'LN', 'GN', 'GDN', 'DN', 'no_norm'], help='Normalization type')
     parser.add_argument('--norm_first', type=dist.strtobool, default=False, help='Normalize before activation')
     parser.add_argument('--center_norm', type=dist.strtobool, default=True, help='Center normalization parameter')
     parser.add_argument('--scale_norm', type=dist.strtobool, default=True, help='Scale normalization parameter')
     parser.add_argument('--norm_axes', nargs='+', type=int, default=[-1], help='Normalization axes')
     parser.add_argument('--activation', type=str, default='relu', help='Activation function')
     parser.add_argument('--regularize', type=float, default=1e-6, help='L2 regularizer value')

     # --- Model Options for Locally Connected, Non-Conv Nets ---
     parser.add_argument('--circular', type=dist.strtobool, default=True, help='Circular boundary constraints for spatial regularization')
     parser.add_argument('--alpha', type=float, default=0.0, help='Alpha value for loss')
     parser.add_argument('--using_eight_neighbourhood_flag', type=dist.strtobool, default=False, help='Use eight neighbourhood for spatial regularization')
     parser.add_argument('--use_bias', type=dist.strtobool, default=True, help='Use bias in the non-conv net for regularization')
     parser.add_argument('--kernel_initializer', type=str, default='glorot_uniform', help='Kernel initializer for locally connected layers')
     parser.add_argument('--spatial_loss', type=dist.strtobool, default=False, help='Apply spatial smoothing loss on kernel weights')
     parser.add_argument('--loss_filtered_by_relu', type=str, default='filter_larger_than_threshold', help='Apply ReLU on the spatial loss')
     parser.add_argument('--layer_alpha_factors', nargs='+', type=float, default=[-1], help='Factors to multiply alpha for each layer')
     parser.add_argument('--alpha_schedule', type=str, default='constant', choices=['constant', 'cosine', 'sigmoid', 'symmetric_sigmoids', 'gaussian'], help='Alpha schedule')
     parser.add_argument('--alpha_scheduler_start_epoch', type=int, default=0, help='Epoch to start alpha scheduler')
     parser.add_argument('--alpha_scheduler_mode', type=str, default='constant', help='Mode of alpha scheduler')
     parser.add_argument('--beta', type=float, default=1, help='Beta value for loss')
     parser.add_argument('--beta_schedule', type=str, default='constant', help='Beta schedule')
     parser.add_argument('--beta_scheduler_start_epoch', type=int, default=9999, help='Epoch to start beta scheduler')
     parser.add_argument('--beta_scheduler_mode', type=str, default='constant', help='Mode of beta scheduler')
     parser.add_argument('--sigmoid_steepness', type=float, default=20, help='Steepness of sigmoid function for alpha schedule')
     parser.add_argument('--sigmoid_position', type=float, default=0.5, help='Position of sigmoid function for alpha schedule')
     parser.add_argument('--gaussian_variance', type=float, default=0.1, help='Variance of Gaussian function for alpha schedule')

     # --- Saving, Loading, Reproducibility & Logging Options ---
     parser.add_argument('--save_dir', type=str, default='./save_dir', help='Directory to save checkpoints and models')
     parser.add_argument('--full_model_save_epoch', type=int, default=-1, help='Epoch to save the full model')
     parser.add_argument('--numpy_seed', type=int, default=1, help='Random seed for numpy')
     parser.add_argument('--tensorflow_seed', type=int, default=1, help='Random seed for TensorFlow')
     parser.add_argument('--verbosity', type=int, default=1, help='Verbosity level')
     parser.add_argument('--n_warmup_epochs', type=int, default=0, help='Number of epochs for linear LR warmup')

     # --- Debug/Test Mode ---
     parser.add_argument('--test_mode', type=dist.strtobool, default=False, help='Enable test mode')

     # --- Wandb Options ---
     parser.add_argument('--save_and_visualize_in_wandb', type=dist.strtobool, default=True, help='Save and visualize in Wandb')
     parser.add_argument('--save_name_in_wandb', type=str, default='All-TNN', help='Run name in Wandb')
     parser.add_argument('--save_epochs_frequency', type=int, default=10, help='Frequency of saving model in Wandb')

     # --- Hardware Options ---
     parser.add_argument('--gpu_ids', nargs='+', type=int, default=[-1], help='GPU IDs for each worker, -1 for CPU usage')

     args = parser.parse_args()
     print(args)

     return vars(args)

# Example usage
if __name__ == "__main__":
     hparams = get_hparams()