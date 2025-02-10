from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as k
from tensorflow.python.eager import backprop
from tensorflow.python.keras.engine import data_adapter
from tensorflow.keras import layers, models, datasets

from all_tnn.models.model_helper import tnn_helper_functions


class SpatialLoss:
    def __init__(self, n_layers, alpha, loss_filtered_by_relu=False, using_goodness_gradient_bias=False, add_regularizer_loss=False):
        self.spatial_loss_values = [0] * n_layers
        self.alpha = alpha
        self.loss_filtered_by_relu = loss_filtered_by_relu
        self.using_goodness_gradient_bias = using_goodness_gradient_bias
        self.add_regularizer_loss = add_regularizer_loss

    def __call__(self, w, b, output_shape, kernel_size, circular, layer_alpha_factors, goodness_of_gradients=None, using_eight_neighbourhood_flag=False, save_path=None):
        if goodness_of_gradients is None:
            goodness_of_gradients = [None] * len(w)
            
        loss_terms = 0.0

        for i, (weight, bias, out_shape, k_size, layer_idx) in enumerate(zip(w, b, output_shape, kernel_size, range(len(w)))):
            goodness_of_gradient = goodness_of_gradients[layer_idx]
            mean_cos_dist = tnn_helper_functions.compute_weights_cosine_distance(
                weight, bias, k_size, circular, return_maps=False,
                using_goodness_gradient_bias=self.using_goodness_gradient_bias,
                layer_gs=goodness_of_gradient, eight_neighbourhood=using_eight_neighbourhood_flag, save_path=save_path
            )

            if not self.loss_filtered_by_relu:
                unweighted_loss = mean_cos_dist
            elif self.loss_filtered_by_relu == 'filter_larger_than_threshold':
                unweighted_loss = tf.nn.relu(mean_cos_dist)
            elif self.loss_filtered_by_relu == 'filter_smaller_than_threshold':
                unweighted_loss = tf.nn.relu(-mean_cos_dist)
            else:
                raise NotImplementedError(f'loss_filtered_by_relu={self.loss_filtered_by_relu} not implemented')

            final_loss = tf.multiply(unweighted_loss, self.alpha * layer_alpha_factors[layer_idx], name=f'spatial_loss_layer{layer_idx}')
            self.spatial_loss_values[layer_idx] = final_loss
            loss_terms = tf.add(loss_terms, final_loss, name='spatial_loss_sum')

            if self.add_regularizer_loss in ['L1', 'l1']:
                l1_loss = tf.reduce_sum(tf.abs(weight))
                loss_terms = tf.add(loss_terms, l1_loss, name='spatial_loss_sum_add_L1')

        return loss_terms


class SpatialLossModel(tf.keras.Model):
    def __init__(self, inputs, outputs, hparams, name):
        super().__init__(inputs=inputs, outputs=outputs, name=name)
        self.hparams = hparams
        beta_schedule = hparams.get('beta_schedule', 'constant')
        
        self.alpha_multiplier = tf.Variable(1 if beta_schedule == 'constant' else 0, dtype=tf.float32, trainable=False)
        self.using_blurring_weight_learning = hparams['using_blurring_weight_learning']
        self.using_goodness_gradient_bias = hparams.get('using_goodness_gradient_bias', False)
        self.goodness_gradient_metric = hparams.get('goodness_gradient_metric', 'L2')

    def test_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        y_pred = self(x, training=False)
        if self.using_blurring_weight_learning:
            self.apply_gaussian_blur_to_weights(size=3, sigma=0.4)

        if self.using_goodness_gradient_bias:
            self._update_spatial_loss()

        self._update_loss_and_metrics(y, y_pred, sample_weight)

        return self._get_metrics_output()

    def train_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        if self.using_blurring_weight_learning:
            self.apply_gaussian_blur_to_weights(size=3, sigma=0.4)

        with backprop.GradientTape(persistent=True) as tape:
            y_pred = self(x, training=True)
            loss = self._compute_loss(y, y_pred, sample_weight)

            if self.using_goodness_gradient_bias:
                self._compute_goodness_of_gradients(tape, y, y_pred, sample_weight)

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y, y_pred, sample_weight)

        return self._get_metrics_output()

    def configure_regularizer(self, loss_layers):
        kernels = [l.trainable_weights[0] for l in loss_layers]
        biases = [l.trainable_variables[1] if self.hparams['use_bias'] else None for l in loss_layers]
        output_shapes = [l.output_shape for l in loss_layers]

        self.spatial_loss_args = {
            'w': kernels,
            'b': biases,
            'output_shape': output_shapes,
            'kernel_size': [l.kernel_size for l in loss_layers],
            'circular': self.hparams['circular'],
            'layer_alpha_factors': tf.constant(self.hparams['layer_alpha_factors']),
            'goodness_of_gradients': [None] * 6,
            'using_eight_neighbourhood_flag': self.hparams['using_eight_neighbourhood_flag'],
        }

        self.loss_obj = SpatialLoss(
            n_layers=len(loss_layers),
            alpha=tf.constant(self.hparams['alpha']),
            loss_filtered_by_relu=self.hparams['loss_filtered_by_relu'],
            using_goodness_gradient_bias=self.hparams['using_goodness_gradient_bias'],
            add_regularizer_loss=self.hparams['add_regularizer_loss']
        )

        if self.hparams['spatial_loss']:
            self.add_loss(lambda: self.loss_obj(**self.spatial_loss_args))

        self.spatial_loss_values = self.loss_obj.spatial_loss_values
        
    def _compute_loss(self, y, y_pred, sample_weight):
        return self.compiled_loss(
            y, y_pred, sample_weight,
            regularization_losses=[
                l if 'spatial_loss' in l.name else l
                for l in self.losses
            ]
        )

    def _compute_goodness_of_gradients(self, tape, y, y_pred, sample_weight):
        classification_loss = self.compiled_loss(y, y_pred, sample_weight)
        gradients = tape.gradient(classification_loss, self.trainable_variables)

        gradients_hypercolumns = [
            g for g in gradients if ('Reshape' in g.name or ':2' in g.name) and len(g.shape) == 3
        ]
        gradients_hypercolumn_2d_sheets = [
            tf.squeeze(channels_to_sheet(g)) for g in gradients_hypercolumns
        ]

        if self.goodness_gradient_metric == 'L2':
            goodness_of_gradients = [tf.square(g) for g in gradients_hypercolumn_2d_sheets]
        elif self.goodness_gradient_metric == 'L1':
            goodness_of_gradients = [tf.abs(g) for g in gradients_hypercolumn_2d_sheets]
        else:
            raise NotImplementedError(f'goodness_gradient_metric={self.goodness_gradient_metric} not implemented')

        self._update_spatial_loss(goodness_of_gradients)

    def _update_spatial_loss(self, goodness_of_gradients=None):
        spatial_loss_args = self.spatial_loss_args
        spatial_loss_args['goodness_of_gradients'] = goodness_of_gradients or [None] * 6
        self.spatial_loss_values = self.loss_obj(**spatial_loss_args)

    def _update_loss_and_metrics(self, y, y_pred, sample_weight):
        self.compiled_loss(
            y, y_pred, sample_weight,
            regularization_losses=[
                l if 'spatial_loss' in l.name else l
                for l in self.losses
            ]
        )
        self.compiled_metrics.update_state(y, y_pred, sample_weight)

    def _get_metrics_output(self):
        custom_metrics = []
        if 'spatial_loss_metric' in self.metrics_names:
            custom_metrics.append(self.metrics[self.metrics_names.index('spatial_loss_metric')])

        std_metrics = [m for m in self.metrics if m not in custom_metrics]
        metrics_output = {m.name: m.result() for m in std_metrics}
        for metric in custom_metrics:
            metrics_output = metric.fill_output(metrics_output)

        return metrics_output

    def gaussian_blur_kernel(self, size=3, sigma=1.0):
        ax = tf.range(-size // 2 + 1.0, size // 2 + 1.0)
        xx, yy = tf.meshgrid(ax, ax)
        kernel = tf.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        return kernel / tf.reduce_sum(kernel)

    def apply_gaussian_blur_to_weights(self, size=3, sigma=1.0):
        kernel = self.gaussian_blur_kernel(size=size, sigma=sigma)
        for layer in self.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                weights, biases = layer.get_weights()
                blurred_weights = tf.nn.depthwise_conv2d(
                    weights[:, :, :, tf.newaxis],
                    kernel[:, :, tf.newaxis, tf.newaxis],
                    strides=[1, 1, 1, 1],
                    padding="SAME"
                )
                layer.set_weights([blurred_weights[:, :, 0, :], biases])

##########################################################################################
# Metrics for spatial regularization loss
##########################################################################################
class SpatialLossMetric(tf.keras.metrics.Metric):
    """
    A custom Keras metric to log regularization losses
    """

    def __init__(self, spatial_loss_layers, alpha, hparams, **kwargs):
        super(SpatialLossMetric, self).__init__(name='spatial_loss_metric', **kwargs)
        self.spatial_loss_layers = spatial_loss_layers
        self.n_hypercolumns = len(spatial_loss_layers)
        self.alpha = alpha
        self.cross_entropy = tf.Variable(0, dtype=tf.float32)
        self.n_batches = tf.Variable(0, dtype=tf.float32)
        self.hparams = hparams

    def fill_output(self, output):
        for i in range(self.n_hypercolumns):
            output[f'spatial_loss_layer{i}'] = self.spatial_loss_layers[i]
        total_spatial_loss = tf.keras.backend.sum(self.spatial_loss_layers)
        output = {'spatial_loss': total_spatial_loss, **output}
        return output

    def reset_state(self):
        pass

    def update_state(self, y_true, y_pred, sample_weight=None):
        pass

    def result(self):
        return tf.keras.backend.sum(self.spatial_loss_layers) 
        

    def get_config(self):
        return {"spatial_loss_layers": self.spatial_loss_layers,
                "alpha": self.alpha,
                "hparams": self.hparams}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


##########################################################################################
# Helper functions for spatial model & spatial regularization loss
##########################################################################################

def compute_weights_cosine_distance(x, b, kernel_size, circular, return_maps=False, using_goodness_gradient_bias=False, layer_gs=None, eight_neighbourhood=False, save_path=None):
    """
    Returns the mean cosine distance between neighbouring neurons on a 2D sheet.

    Args:
        x: Weights of the locally connected layer.
        b: Biases of the locally connected layer.
        kernel_size: Kernel size of the locally connected layer.
        circular: If True, applies circular boundary conditions.
        return_maps: If True, returns the similarity maps.
        using_goodness_gradient_bias: If True, uses goodness of gradients for biasing.
        layer_gs: Goodness of gradients.
        eight_neighbourhood: If True, considers 8 neighbours instead of 2.
        save_path: Path to save the similarity maps.

    Returns:
        Mean cosine distance
    """

    # Extract weight shapes
    n_rows, n_cols, in_channels, c_dim = extract_weight_dims(x, kernel_size)

    # Reshape and permute dimensions
    x = k.reshape(x, shape=(n_rows, n_cols, kernel_size[0], kernel_size[1], in_channels, c_dim[0], c_dim[1]))
    x = k.permute_dimensions(x, [0, 5, 1, 6, 2, 3, 4])
    x = k.reshape(x, shape=(n_rows * c_dim[0], n_cols * c_dim[1], in_channels * kernel_size[0] * kernel_size[1]))

    # Reshape and add biases if available
    if b is not None:
        b = k.reshape(b, (n_rows, n_cols, c_dim[0], c_dim[1]))
        b = k.permute_dimensions(b, [0, 2, 1, 3])
        b = k.reshape(b, (n_rows * c_dim[0], n_cols * c_dim[1]))
        x = k.concatenate([x, tf.expand_dims(b, -1)], axis=2)

    # Normalize vectors
    x = tf.math.l2_normalize(x, axis=2)

    # Compute cosine similarities
    if not eight_neighbourhood:
        bottom_sim = x[:-1, :] * x[1:, :]
        right_sim = x[:, :-1] * x[:, 1:]

        if circular:
            bottom_sim = k.concatenate([bottom_sim, x[-1:, :] * x[:1, :]], axis=0)
            right_sim = k.concatenate([right_sim, x[:, -1:] * x[:, :1]], axis=1)

        bottom_sim = tf.reduce_sum(bottom_sim, axis=-1)
        right_sim = tf.reduce_sum(right_sim, axis=-1)
        mean_cos_sim = (k.mean(bottom_sim) + k.mean(right_sim)) / 2

        if save_path:
            np.save(os.path.join(save_path, 'bottom_sim.npy'), bottom_sim)
            np.save(os.path.join(save_path, 'right_sim.npy'), right_sim)
    else:
        mean_cos_sim = _compute_eight_neighbourhood_sim(x, circular, save_path)

    mean_cos_dist = (1 - mean_cos_sim) / 2

    if using_goodness_gradient_bias and layer_gs is not None:
        mean_cos_dist = _apply_goodness_gradient_bias(mean_cos_dist, layer_gs, bottom_sim, right_sim)

    if return_maps:
        return mean_cos_dist, bottom_sim, right_sim
    else:
        return mean_cos_dist

def _compute_eight_neighbourhood_sim(x, circular, save_path):
    """
    Computes cosine similarities considering 8 neighbours.
    """
    bottom_sim = x[:-1, :] * x[1:, :]
    right_sim = x[:, :-1] * x[:, 1:]
    bottom_right_sim = x[:-1, :-1] * x[1:, 1:]
    bottom_left_sim = x[:-1, 1:] * x[1:, :-1]
    upper_sim = x[1:, :] * x[:-1, :]
    left_sim = x[:, 1:] * x[:, :-1]
    upper_right_sim = x[1:, :-1] * x[:-1, 1:]
    upper_left_sim = x[1:, 1:] * x[:-1, :-1]

    if circular:
        bottom_sim = k.concatenate([bottom_sim, x[-1:, :] * x[:1, :]], axis=0)
        right_sim = k.concatenate([right_sim, x[:, -1:] * x[:, :1]], axis=1)
        upper_sim = k.concatenate([upper_sim, x[1:, :] * x[:-1, :]], axis=0)
        left_sim = k.concatenate([left_sim, x[:, 1:] * x[:, :-1]], axis=1)

    bottom_sim = tf.reduce_sum(bottom_sim, axis=-1)
    right_sim = tf.reduce_sum(right_sim, axis=-1)
    bottom_right_sim = tf.reduce_sum(bottom_right_sim, axis=-1)
    bottom_left_sim = tf.reduce_sum(bottom_left_sim, axis=-1)
    upper_sim = tf.reduce_sum(upper_sim, axis=-1)
    left_sim = tf.reduce_sum(left_sim, axis=-1)
    upper_right_sim = tf.reduce_sum(upper_right_sim, axis=-1)
    upper_left_sim = tf.reduce_sum(upper_left_sim, axis=-1)

    mean_cos_sim = (k.mean(bottom_sim) + k.mean(right_sim) + k.mean(bottom_right_sim) + k.mean(bottom_left_sim) + 
                    k.mean(upper_sim) + k.mean(left_sim) + k.mean(upper_right_sim) + k.mean(upper_left_sim)) / 8

    if save_path:
        np.save(os.path.join(save_path, 'bottom_sim.npy'), bottom_sim)
        np.save(os.path.join(save_path, 'right_sim.npy'), right_sim)
        np.save(os.path.join(save_path, 'bottom_right_sim.npy'), bottom_right_sim)
        np.save(os.path.join(save_path, 'bottom_left_sim.npy'), bottom_left_sim)
        np.save(os.path.join(save_path, 'upper_sim.npy'), upper_sim)
        np.save(os.path.join(save_path, 'left_sim.npy'), left_sim)
        np.save(os.path.join(save_path, 'upper_right_sim.npy'), upper_right_sim)
        np.save(os.path.join(save_path, 'upper_left_sim.npy'), upper_left_sim)

    return mean_cos_sim

def _apply_goodness_gradient_bias(mean_cos_dist, layer_gs, bottom_sim, right_sim):
    """
    Applies goodness gradient bias to the cosine distances.
    """
    highest_goodness = tf.reduce_max(layer_gs)

    bottom_coefficient = (1 - layer_gs / highest_goodness + 1 - k.concatenate([layer_gs[1:, :], layer_gs[:1, :]], 0) / highest_goodness) / 2
    right_coefficient = (1 - layer_gs / highest_goodness + 1 - k.concatenate([layer_gs[:, 1:], layer_gs[:, :1]], 1) / highest_goodness) / 2

    right_goodness_weighted_sim = right_sim * right_coefficient
    bottom_goodness_weighted_sim = bottom_sim * bottom_coefficient

    mean_goodness_cos_sim = (k.mean(right_goodness_weighted_sim) + k.mean(bottom_goodness_weighted_sim)) / 2
    mean_goodness_cos_dist = (1 - mean_goodness_cos_sim) / 2
    return mean_goodness_cos_dist
    
    
def extract_channel_dim(out_channels):
    if (np.sqrt(out_channels) % 1) == 0:
        channel_dim = [np.sqrt(out_channels).astype(int),
                       np.sqrt(out_channels).astype(int)]
    else:
        print(f'WARNING: out_channels is expected to be be square. found {out_channels}. Finding best rectangle instead')
        i = np.sqrt(out_channels).astype(int)
        while i > 0:
            if (out_channels % i) == 0:
                channel_dim = [i, int(out_channels / i)]
                break
            i -= 1
    return channel_dim


def channels_to_sheet(x_, c_dims=None, return_np=False):
    '''Converts from the 3d fully connected layer format to a 2d "hypercolumn" sheet
    x_: input rd fully conencted layer tensor. If return_np->np[1,h,w,c] or np[h,w,c]. Else: tf[batch_size,h,w,c]
    c_dims: x,y dimensions of each hypercolumn (normally: sqrt of channels). Inferred if return_np, else: required
    return_np: return a [h, w] np array instead of a [b,h,w,c] tf tensor
    '''

    if c_dims is None:
        c_dims = extract_channel_dim(x_.shape[-1])
    if isinstance(x_, np.ndarray):
        x_ = tf.convert_to_tensor(x_)
    if len(x_.shape) == 3:
        x_ = tf.expand_dims(x_, axis=0)

    x_ = k.reshape(x_, (-1, x_.shape[1], x_.shape[2], c_dims[0], c_dims[1]))
    x_ = k.permute_dimensions(x_, [0, 1, 3, 2, 4])
    x_ = k.reshape(x_, (-1, x_.shape[1] * c_dims[0], x_.shape[3] * c_dims[1], 1))

    if return_np:
        x_np = x_.numpy()
        if x_np.ndim == 4:
            if x_np.shape[0] == 1:
                return x_np[0,:,:,0]  # remove batch and channel dim if we have a single img batch
            else:
                return x_np[:,:,:,0]  # remove channel dim
        else:
            assert x_np.ndim == 3, f'problem with channels_to_sheet output. Should be ndim=3or4, got shape {x_np.shape}'
            return x_np[:,:,0]
    else:
        return x_


def sheet_to_channels(x_, c_dims, return_np=False):
    """Function that infers the n_row/n_col arguments based on c_dims"""

    if isinstance(x_, np.ndarray):
        x_ = tf.convert_to_tensor(x_)
    if tf.rank(x_) == 2:
        x_ = tf.expand_dims(x_, axis=0)  # expepcts dim0->batch_size, dim1->sheet_rows, dim2->sheet_cols
    batch_size, layer_rows, layer_cols = x_.shape[0], x_.shape[1]/c_dims[0], x_.shape[2]/c_dims[1]
    assert (layer_rows%1, layer_cols%1) == (0,0)
    layer_rows, layer_cols = int(layer_rows), int(layer_cols)
    x_ = k.reshape(x_, (batch_size, layer_rows, c_dims[0], layer_cols, c_dims[1]))
    x_ = k.permute_dimensions(x_, [0, 1, 3, 2, 4])
    x_ = k.reshape(x_, (batch_size, layer_rows, layer_cols, int(np.product(c_dims))))

    if return_np:
        x_np = x_.numpy()
        return x_np[0] if x_np.ndim == 4 else x_np  # remove batch_size if needed
    else:
        return x_


def extract_weight_dims(x, kernel_size):
    n_rows, n_cols = np.sqrt(x.shape[0]), np.sqrt(x.shape[0])
    assert (n_rows % 1, n_cols % 1) == (0, 0), 'Layers must be square, otherwise you need to code a way for compute_weights_cosine_distance to know n_rows and n_cols'
    n_rows, n_cols = int(n_rows), int(n_cols)
    weights_out_channels = x.shape[-1]
    in_channels = x.shape[1] // np.product(kernel_size)
    c_dim = extract_channel_dim(weights_out_channels)  # [row_channels_out, col_channels_out]
    return n_rows, n_cols, in_channels, c_dim


class SteepSigmoidMultiplierLayer(layers.Layer):
    def __init__(self, steepness=7.0, threshold=0.5, l1_regularization=1e-5, **kwargs):
        super(SteepSigmoidMultiplierLayer, self).__init__(**kwargs)
        self.steepness = steepness
        self.threshold = threshold
        self.l1_regularization = l1_regularization
        self.x = None

    def build(self, input_shape):
        # Initialize the learnable parameters x with ones
        self.x = self.add_weight(shape=input_shape[1:], initializer="ones", trainable=True)

    def call(self, inputs, training=False):
        steep_sigmoid_x = tf.math.sigmoid(self.x*self.steepness)
        thresholded_output = tf.cast(steep_sigmoid_x > self.threshold, tf.float32)

        if training:
            l1_penalty = tf.reduce_sum(tf.abs(steep_sigmoid_x)) * self.l1_regularization
            self.add_loss(l1_penalty)

        return inputs * steep_sigmoid_x


