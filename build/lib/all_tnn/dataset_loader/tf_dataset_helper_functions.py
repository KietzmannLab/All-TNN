import os
import time
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Rescaling, RandomContrast, RandomFlip
import keras_cv  # Ensure keras_cv is installed
from all_tnn.task_helper_functions import get_n_classes

def preprocess_batch(
    data,
    labels,
    sample_weights,
    hparams,
    dataset_path=None,
    dataset_subset=None,
    fixation_heatmap=None,
    no_labels=False
):
    """
    Preprocesses a batch of data and formats labels based on the provided hyperparameters.

    Args:
        data: Input data batch.
        labels: Corresponding labels for the data.
        sample_weights: Weights for each sample in the batch.
        hparams: Hyperparameters for preprocessing and model configuration.
        dataset_path (optional): Path to the dataset.
        dataset_subset (optional): Subset of the dataset to use.
        fixation_heatmap (optional): Fixation heatmap data.
        no_labels (bool, optional): If True, labels are not processed.

    Returns:
        Tuple containing preprocessed inputs and formatted labels.
        Optionally includes sample_weights based on model configuration.
    """
    n_timesteps = 1  # Placeholder for potential recurrent pipeline
    n_classes = get_n_classes(
        hparams=hparams,
        dataset_path=dataset_path,
        dataset_subset=dataset_subset
    )

    # Preprocess images: reshape, crop, resize, and scale
    preprocessed_inputs = preprocess_batch_imgs(data, fixation_heatmap, hparams)
    if no_labels:
        return preprocessed_inputs

    model_name = hparams.get('model_name', '')
    embedding_target = hparams.get('embedding_target', False) # Whether the model is trained for embedding instead of labes
    is_simclr = 'simclr' in model_name
    is_finetune = 'finetune' in model_name or hparams.get('finetune_flag', False)

    def format_label(key, use_one_hot):
        value = tf.one_hot(labels['output'], depth=n_classes) if use_one_hot else labels['output']
        return {key: value}

    key = 'dense_2'  # or f'output_time_n' for recurrent models
    formatted_labels = format_label(key, use_one_hot=True if not embedding_target else False)
    print(f"formatted_labels: {formatted_labels}")

    # Sample weights are not needed for self-supervised learning
    if is_simclr and not is_finetune:
        return preprocessed_inputs, formatted_labels

    return preprocessed_inputs, formatted_labels, sample_weights


def preprocess_batch_imgs(images, fixation_heatmap, hparams):
    """
    Preprocesses a batch of images by reshaping, cropping, resizing, and scaling.

    Args:
        images (tf.Tensor): Batch of images with shape [batch_size, height, width, channels].
        fixation_heatmap (tf.Tensor or None): Fixation heatmaps for the images. Required if cropping based on fixation.
        hparams (dict): Hyperparameters containing preprocessing configurations.

    Returns:
        tf.Tensor: Preprocessed images with shape [batch_size, target_image_size, target_image_size, channels].
    """
    img_height, img_width = images.shape[1], images.shape[2]
    image_area = tf.cast(img_height * img_width, tf.float32)
    target_size = hparams['image_size']
    batch_size = hparams['batch_size']

    # Random cropping 
    max_crop = tf.cast(tf.reduce_min([img_height, img_width]), tf.float32)
    min_crop = tf.math.minimum(max_crop, tf.math.ceil(tf.sqrt(image_area * 0.33)))
    crop_sizes = tf.cast(
        tf.random.uniform([batch_size], minval=min_crop, maxval=max_crop),
        tf.int32
    )

    def random_resize(image, crop_size):
        cropped = tf.image.random_crop(image, size=[crop_size, crop_size, 3])
        resized = tf.image.resize(cropped, [target_size, target_size], antialias=True)
        return resized

    images = tf.map_fn(
        lambda inp: random_resize(inp[0], inp[1]),
        elems=(images, crop_sizes),
        fn_output_signature=tf.float32,
        name='random_cropped_resized_images'
    )
    
    # Scale images to [0, 1] | later might be scaled to [-1, 1] or z-scored when create the dataset
    rescaled_images = tf.keras.layers.Rescaling(scale=1/255., dtype=tf.float32)(images)

    return rescaled_images


def augment_and_normalize(images, labels, sample_weights, augment, hparams, no_labels=False):
    """
    Applies data augmentation and normalization to a batch of images.

    Args:
        images (tf.Tensor): Batch of images with shape [batch_size, height, width, channels].
        labels (tf.Tensor): Corresponding labels for the images.
        sample_weights (tf.Tensor): Weights for each sample in the batch.
        augment (bool): Flag indicating whether to apply augmentation.
        hparams (dict): Hyperparameters containing augmentation and normalization settings.
        no_labels (bool, optional): If True, labels and sample_weights are not returned. Defaults to False.

    Returns:
        tuple: Tuple containing augmented and normalized images. If `no_labels` is False, also returns labels and sample_weights.
    """
    RANDOM_SEED = 170591  # Seed for reproducibility

    if augment:
        supervised_data_aug = hparams.get('supervised_data_aug', True)

        if not supervised_data_aug:
            # Self-supervised learning data augmentation for SimCLR
            CONTRASTIVE_AUGMENTATION = {
                "crop_area_factor": (0.08, 1.0),
                "aspect_ratio_factor": (3 / 4, 4 / 3),
                "color_jitter_rate": 0.8,
                "brightness_factor": 0.2,
                "contrast_factor": 0.8,
                "saturation_factor": (0.3, 0.7),
                "hue_factor": 0.2,
            }

            # Apply random horizontal flip
            images = RandomFlip('horizontal', seed=RANDOM_SEED)(images)

            # Apply random color jitter with a specified rate
            color_jitter = keras_cv.layers.RandomApply(
                keras_cv.layers.RandomColorJitter(
                    value_range=(0, 1),
                    brightness_factor=CONTRASTIVE_AUGMENTATION['brightness_factor'],
                    contrast_factor=CONTRASTIVE_AUGMENTATION['contrast_factor'],
                    saturation_factor=CONTRASTIVE_AUGMENTATION['saturation_factor'],
                    hue_factor=CONTRASTIVE_AUGMENTATION['hue_factor'],
                ),
                rate=CONTRASTIVE_AUGMENTATION['color_jitter_rate'],
            )
            images = color_jitter(images)
        else:
            # Supervised data augmentation for classification
            images = tf.image.random_brightness(images, max_delta=32.0 / 255.0, seed=RANDOM_SEED)
            images = tf.image.random_saturation(images, lower=0.5, upper=1.5, seed=RANDOM_SEED)
            images = RandomContrast(factor=0.5, seed=RANDOM_SEED)(images)

    # Apply normalization
    images = normalize(images, hparams.get('image_normalization'))

    if no_labels:
        return images
    else:
        return images, labels, sample_weights


def normalize(image, img_normalization):
    """
    Normalizes an image tensor based on the specified normalization method.

    Args:
        image (tf.Tensor): Image tensor with values typically in [0, 1].
        img_normalization (str or None): Normalization strategy.
            - '[-1,1]': Scales image to [-1, 1].
            - 'z_scoring': Applies z-score normalization (mean=0, std=1).
            - '[-1-1]+noise': Scales to [-1, 1] and adds Gaussian noise.
            - None: No normalization.

    Returns:
        tf.Tensor: Normalized image tensor.
    """
    # Ensure image values are within [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    if img_normalization == 'z_scoring':
        tf.print('Applying z-score normalization to each image.')
        image = tf.image.per_image_standardization(image)
    elif img_normalization == '[-1,1]':
        tf.print('Normalizing images to [-1, 1].')
        rescale = Rescaling(scale=2.0, offset=-1.0)
        image = rescale(image)
    elif img_normalization == '[-1-1]+noise':
        tf.print('Normalizing images to [-1, 1] and adding Gaussian noise.')
        rescale = Rescaling(scale=2.0, offset=-1.0)
        image = rescale(image)
        noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=0.1)
        image = tf.clip_by_value(image + noise, -1.0, 1.0)
    elif img_normalization is None:
        tf.print('No normalization applied to images.')
    else:
        raise ValueError("Invalid normalization method. Choose from '[-1,1]', 'z_scoring', '[-1-1]+noise', or None.")

    return image


def assess_data_generation_speed(tf_dataset):
    """
    Assesses the data generation speed of a TensorFlow dataset by iterating through it.

    Args:
        tf_dataset (tf.data.Dataset): The TensorFlow dataset to assess.

    Raises:
        StopIteration: Halts the script after assessment.
    """
    print('Assessing data generator compute time for 2 epochs.')
    for epoch in range(2):
        start_time = time.perf_counter()
        batch_count = 0
        for _ in tf_dataset:
            batch_count += 1
        elapsed_time = time.perf_counter() - start_time
        print(f'Epoch {epoch + 1} -- Batches produced: {batch_count} -- Time elapsed: {elapsed_time:.2f} seconds')
    raise StopIteration("Data generation speed assessment complete. Script halted.")


def plot_generated_images(tf_dataset, hparams, dataset, dataset_path, fixation_heatmaps_path=None,
                         dataset_subset=None, n_epochs_to_show=1, max_n_imgs=10000, imgs_per_batch=1):
    """
    Plots and saves generated images from a TensorFlow dataset for visualization.

    Args:
        tf_dataset (tf.data.Dataset): The dataset containing images, labels, and sample weights.
        hparams (dict): Hyperparameters including batch_size and other configurations.
        dataset (str): Name of the dataset.
        dataset_path (str): Path to the dataset file.
        fixation_heatmaps_path (str, optional): Path to fixation heatmaps. Defaults to None.
        dataset_subset (str, optional): Specific subset of the dataset to visualize. Defaults to None.
        n_epochs_to_show (int, optional): Number of epochs to visualize. Defaults to 1.
        max_n_imgs (int, optional): Maximum number of images to plot. Defaults to 10000.
        imgs_per_batch (int, optional): Number of images to plot per batch. Defaults to 1.
    """
    os.makedirs('./tf_generated_images', exist_ok=True)

    dataset_name = dataset_subset if dataset_subset is not None else os.path.splitext(os.path.basename(dataset_path))[0]
    img_counter = 0

    for epoch in range(n_epochs_to_show):
        print(f'Visualizing dataset - Epoch {epoch + 1}')
        epoch_counter = 0
        for batch in tf_dataset:
            images, labels_batch, sample_weights_batch = batch[:3]  # Assuming batch structure
            batch_size = hparams.get('batch_size', images.shape[0])

            for i in range(batch_size):
                if i % (batch_size // imgs_per_batch) == 0:
                    plt.figure()

                    # Handling single images or sequences of images | No fixation heatmaps
                    if images.ndim == 4:
                        plt.imshow(images[i])
                        try:
                            label = np.argmax(labels_batch['output_time_0'][i])
                            sw = sample_weights_batch[i]
                            plt.title(f'l={label}, sw={sw}, min={np.min(images[i]):.2f}, max={np.max(images[i]):.2f}')
                        except KeyError:
                            label = np.argmax(labels_batch['output_time_0'][i])
                            plt.title(f'l={label}')
                    elif images.ndim == 5:
                        # Sequences of images
                        fig, axes = plt.subplots(1, images.shape[1], figsize=(15, 5))
                        for t in range(images.shape[1]):
                            axes[t].imshow(images[i, t])
                            label = np.argmax(labels_batch[f'output_time_{t}'][i])
                            axes[t].set_title(f'l={label}')
                            axes[t].axis('off')
                        plt.tight_layout()
                    else:
                        raise ValueError(f'Unexpected image dimensions: {images.ndim}')
                    
                    # Save the plotted figure
                    img_filename = f'./tf_generated_images/{dataset_name}_{dataset}_{img_counter}.png'
                    plt.savefig(img_filename)
                    plt.close()

                    img_counter += 1
                    if img_counter >= max_n_imgs:
                        print(f'Maximum number of images ({max_n_imgs}) reached. Stopping visualization.')
                        return
                epoch_counter += 1

    print(f'Total images saved: {img_counter}')