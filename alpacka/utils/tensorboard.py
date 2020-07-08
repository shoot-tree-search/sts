"""TensorBoard logger."""

import os
from datetime import datetime

import tensorflow as tf


class TensorBoardLogger:
    """Logging into TensorBoard without TensorFlow ops."""

    def __init__(self, log_dir):
        """Initializes a summary writer logging to log_dir."""
        path = os.path.join(
            log_dir, 'tb_' + datetime.now().strftime('%m-%dT%H%M%S'))
        os.makedirs(path)
        self._writer = tf.summary.create_file_writer(path)

    def log_scalar(self, name, step, value):
        """Logs a scalar to TensorBoard."""
        with self._writer.as_default():
            tf.summary.scalar(name, value, step=step)

    def log_property(self, name, value):
        """Logs a property to TensorBoard."""
        with self._writer.as_default():
            tf.summary.text(name, value, step=0)
