import h5py
from defaults.hparams_defaults import get_hparams
from all_tnn.analysis.util.analysis_help_funcs import load_and_override_hparams
from all_tnn.analysis.util.setup_model_and_data import load_models
from all_tnn.analysis.util.annotation_help_funcs import ecoset_ID2Name
from all_tnn.models.model_helper.model_helper_functions import save_full_model
from all_tnn.task_helper_functions import *

def all_tnn(model_name, epoch, hparams, model_path, pre_relu=False, post_norm=False, n_classes_override=565, batch_size=144):
    """
    Load the model and return the activities model and main model.

    Args:
        model_name (str): Name of the model.
        epoch (int): Epoch number to load.
        hparams (dict): Hyperparameters.
        model_path (str): Path to the model.
        pre_relu (bool, optional): Pre-ReLU flag. Defaults to False.
        post_norm (bool, optional): Post-normalization flag. Defaults to False.
        n_classes_override (int, optional): Number of classes to override. Defaults to 565.
        batch_size (int, optional): Batch size. Defaults to 144.

    Returns:
        tuple: activities_model and main model.
    """
    return load_models(epoch, hparams, model_path, pre_relu, post_norm, n_classes_override)

def generate_random_stimuli(batch_size, image_size):
    """Generate a batch of random images."""
    return np.random.rand(batch_size, image_size, image_size, 3) * 255

if __name__ == '__main__':
    model_name = 'tnn'
    model_path = './save_dir/tnn_ecoset_l2_no_flip_seed1_drop0.0_learnable_False_1e-05_alpha10.0_constant_20_0.1Factors_adam0.05_L21e-06_ecoset_square256_proper_chunks/'
    epoch = 300

    hparams = load_and_override_hparams(model_path)
    activities_model, model = all_tnn(model_name, epoch, hparams, model_path)

    # Usage example
    hparams['batch_size'] = 144 # update batch size if you want
    hparams['image_size'] = 150

    # Replace with your stimuli generation function
    random_stimuli = generate_random_stimuli(batch_size=hparams['batch_size'], image_size=hparams['image_size'])
    preprocessed_stimuli = preprocess_batch(random_stimuli, hparams)

    # Get model activations and top-1 predictions
    activations = activities_model.predict(preprocessed_stimuli) # Get model activations across all layers
    
    probabilities = model.predict(preprocessed_stimuli)
    predictions = np.argmax(probabilities, axis=1) # (batch_size,n_classes) --> (batch_size,)

    # Print predictions IDs
    # print("Predictions IDs:", predictions)
    
    # # Map predictions to ecoset names
    # ecoset_names = [ecoset_ID2Name[str(pred)] for pred in predictions]
    # print("Predictions:", ecoset_names)

    # # Save activations and prediction ids in a dict
    activations_dict = {}
    for i, layer in enumerate(activations):
        activations_dict[f'layer_{i}'] = layer
    activations_dict['predictions'] = predictions
    # activations_dict['ecoset_names'] = ecoset_names
    
    with h5py.File('activations.h5', 'w') as hf:
        for key in activations_dict:
            hf.create_dataset(key, data=activations_dict[key])
         