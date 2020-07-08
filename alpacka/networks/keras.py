"""Network interface implementation using the Keras framework."""

import functools

import gin
import numpy as np
import tensorflow as tf
from tensorflow import keras

from alpacka import data
from alpacka.networks import core


def _make_inputs(input_signature):
    """Initializes keras.Input layers for a given signature.

    Args:
        input_signature (pytree of TensorSignatures): Input signature.

    Returns:
        Pytree of tf.keras.Input layers.
    """
    def init_layer(signature):
        return keras.Input(shape=signature.shape, dtype=signature.dtype)
    return data.nested_map(init_layer, input_signature)


def _make_output_heads(hidden, output_signature, output_activation, zero_init):
    """Initializes Dense layers for heads.

    Args:
        hidden (tf.Tensor): Output of the last hidden layer.
        output_signature (pytree of TensorSignatures): Output signature.
        output_activation (pytree of activations): Activation of every head. See
            tf.keras.layers.Activation docstring for possible values.
        zero_init (bool): Whether to initialize the heads with zeros. Useful for
            ensuring proper exploration in the initial stages of RL training.

    Returns:
        Pytree of head output tensors.
    """
    def init_head(signature, activation):
        assert signature.dtype == np.float32
        (depth,) = signature.shape
        kwargs = {'activation': activation}
        if zero_init:
            kwargs['kernel_initializer'] = 'zeros'
            kwargs['bias_initializer'] = 'zeros'
        return keras.layers.Dense(depth, **kwargs)(hidden)
    return data.nested_zip_with(
        init_head, (output_signature, output_activation)
    )


@gin.configurable
def mlp(
    network_signature,
    hidden_sizes=(32,),
    activation='relu',
    output_activation=None,
    output_zero_init=False,
):
    """Simple multilayer perceptron."""
    # TODO(xxx): Consider moving common boilerplate code to KerasNetwork.
    inputs = _make_inputs(network_signature.input)

    x = inputs
    for h in hidden_sizes:
        x = keras.layers.Dense(h, activation=activation)(x)

    outputs = _make_output_heads(
        x, network_signature.output, output_activation, output_zero_init
    )
    return keras.Model(inputs=inputs, outputs=outputs)


@gin.configurable
def convnet_mnist(
    network_signature,
    n_conv_layers=5,
    d_conv=64,
    d_ff=128,
    activation='relu',
    output_activation=None,
    output_zero_init=False,
    global_average_pooling=False,
    strides=(1, 1),
):
    """Simple convolutional network."""
    inputs = _make_inputs(network_signature.input)

    x = inputs
    for _ in range(n_conv_layers):
        x = keras.layers.Conv2D(
            d_conv,
            kernel_size=(3, 3),
            padding='same',
            activation=activation,
            strides=strides,
        )(x)
    if global_average_pooling:
        x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(d_ff, activation=activation)(x)

    outputs = _make_output_heads(
        x, network_signature.output, output_activation, output_zero_init
    )
    return keras.Model(inputs=inputs, outputs=outputs)


class KerasNetwork(core.TrainableNetwork):
    """TrainableNetwork implementation in Keras.

    Args:
        network_signature (NetworkSignature): Network signature.
        model_fn (callable): Function network_signature -> tf.keras.Model.
        optimizer: See tf.keras.Model.compile docstring for possible values.
        loss: See tf.keras.Model.compile docstring for possible values.
        loss_weights (list or None): Weights assigned to losses, or None if
            there's just one loss.
        weight_decay (float): Weight decay to apply to parameters.
        metrics: See tf.keras.Model.compile docstring for possible values
            (Default: None).
        train_callbacks: List of keras.callbacks.Callback instances. List of
            callbacks to apply during training (Default: None)
        **compile_kwargs: These arguments are passed to tf.keras.Model.compile.
    """

    def __init__(
        self,
        network_signature,
        model_fn=mlp,
        optimizer='adam',
        loss='mean_squared_error',
        loss_weights=None,
        weight_decay=0.0,
        metrics=None,
        train_callbacks=None,
        **compile_kwargs
    ):
        super().__init__(network_signature)
        self._model = model_fn(network_signature)
        self._add_weight_decay(self._model, weight_decay)
        self._model.compile(
            optimizer=optimizer,
            loss=loss,
            loss_weights=loss_weights,
            metrics=metrics or [],
            **compile_kwargs
        )

        self.train_callbacks = train_callbacks or []

    @staticmethod
    def _add_weight_decay(model, weight_decay):
        # Add weight decay in form of an auxiliary loss for every layer,
        # assuming that the weights to be regularized are in the "kernel" field
        # of every layer (true for dense and convolutional layers). This is
        # a bit hacky, but still better than having to add those losses manually
        # in every defined model_fn.
        for layer in model.layers:
            if hasattr(layer, 'kernel'):
                # Keras expects a parameterless function here. We use
                # functools.partial instead of a lambda to workaround Python's
                # late binding in closures.
                layer.add_loss(functools.partial(
                    keras.regularizers.l2(weight_decay), layer.kernel
                ))

    def train(self, data_stream, n_steps):
        """Performs one epoch of training on data prepared by the Trainer.

        Args:
            data_stream: (Trainer-dependent) Python generator of batches to run
                the updates on.
            n_steps: (int) Number of training steps in the epoch.

        Returns:
            dict: Collected metrics, indexed by name.
        """

        def dtypes(tensors):
            return data.nested_map(lambda x: x.dtype, tensors)

        def shapes(tensors):
            return data.nested_map(lambda x: x.shape, tensors)

        dataset = tf.data.Dataset.from_generator(
            generator=data_stream,
            output_types=dtypes((self._model.input, self._model.output)),
            output_shapes=shapes((self._model.input, self._model.output)),
        )

        # WA for bug: https://github.com/tensorflow/tensorflow/issues/32912
        history = self._model.fit(
            dataset, epochs=1, verbose=0, steps_per_epoch=n_steps,
            callbacks=self.train_callbacks
        )
        # history contains epoch-indexed sequences. We run only one epoch, so
        # we take the only element.
        return {name: values[0] for (name, values) in history.history.items()}

    def predict(self, inputs):
        """Returns the prediction for a given input.

        Args:
            inputs: (Agent-dependent) Batch of inputs to run prediction on.

        Returns:
            Agent-dependent: Network predictions.
        """

        return self._model.predict_on_batch(inputs)

    @property
    def params(self):
        """Returns network parameters."""

        return self._model.get_weights()

    @params.setter
    def params(self, new_params):
        """Sets network parameters."""

        self._model.set_weights(new_params)

    def save(self, checkpoint_path):
        """Saves network parameters to a file."""

        self._model.save_weights(checkpoint_path, save_format='h5')

    def restore(self, checkpoint_path):
        """Restores network parameters from a file."""

        self._model.load_weights(checkpoint_path)
