"""Supervised trainer."""

import lzma
import pickle
import time

import gin
import numpy as np

from alpacka import data
from alpacka.trainers import base
from alpacka.trainers import replay_buffers


@gin.configurable
def target_solved(episode):
    return np.full(
        shape=(episode.transition_batch.observation.shape[:1] + (1,)),
        fill_value=int(episode.solved),
    )


@gin.configurable
def target_return(episode):
    return np.cumsum(episode.transition_batch.reward[::-1],
                     dtype=np.float)[::-1, np.newaxis]


@gin.configurable
def target_discounted_return(episode):
    """Uses discounted_return calculated by agent."""
    return np.expand_dims(
        episode.transition_batch.agent_info['discounted_return'], axis=1
    )


@gin.configurable
def target_value(episode):
    return np.expand_dims(
        episode.transition_batch.agent_info['value'], axis=1
    )


@gin.configurable
def target_qualities(episode):
    return episode.transition_batch.agent_info['qualities']


@gin.configurable
def target_action_histogram(episode):
    return episode.transition_batch.agent_info['action_histogram']


@gin.configurable
def target_action_histogram_smooth(episode):
    return episode.transition_batch.agent_info['action_histogram_smooth']


class SupervisedTrainer(base.Trainer):
    """Supervised trainer.

    Trains the network based on (x, y) pairs generated out of transitions
    sampled from a replay buffer.
    """

    def __init__(
        self,
        network_signature,
        target=target_solved,
        batch_size=64,
        n_steps_per_epoch=1000,
        replay_buffer_capacity=1000000,
        replay_buffer_sampling_hierarchy=(),
    ):
        """Initializes SupervisedTrainer.

        Args:
            network_signature (pytree): Input signature for the network.
            target (pytree): Pytree of functions episode -> target for
                determining the targets for network training. The structure of
                the tree should reflect the structure of a target.
            batch_size (int): Batch size.
            n_steps_per_epoch (int): Number of optimizer steps to do per
                epoch.
            replay_buffer_capacity (int): Maximum size of the replay buffer.
            replay_buffer_sampling_hierarchy (tuple): Sequence of Episode
                attribute names, defining the sampling hierarchy.
        """
        super().__init__(network_signature)
        self._target_fn = lambda episode: data.nested_map(
            lambda f: f(episode), target
        )
        self._batch_size = batch_size
        self._n_steps_per_epoch = n_steps_per_epoch

        # (input, target)
        datapoint_sig = (network_signature.input, network_signature.output)
        self._replay_buffer = replay_buffers.HierarchicalReplayBuffer(
            datapoint_sig,
            capacity=replay_buffer_capacity,
            hierarchy_depth=len(replay_buffer_sampling_hierarchy),
        )
        self._sampling_hierarchy = replay_buffer_sampling_hierarchy

    def add_episode(self, episode):
        buckets = [
            getattr(episode, bucket_name)
            for bucket_name in self._sampling_hierarchy
        ]
        self._replay_buffer.add(
            (
                episode.transition_batch.observation,  # input
                self._target_fn(episode),  # target
            ),
            buckets,
        )

    def train_epoch(self, network):
        def data_stream():
            for _ in range(self._n_steps_per_epoch):
                yield self._replay_buffer.sample(self._batch_size)

        start_time = time.time()
        metrics = network.train(data_stream, self._n_steps_per_epoch)
        metrics['time'] = time.time() - start_time
        return metrics

    def save(self, path):
        # The only training state that we care about is the replay buffer.
        # It gets pretty large with visual observations, so we compress it using
        # lzma. The compression algo was chosen experimentally.
        # We set the protocol to 4, because older protocols don't support big
        # files.
        with lzma.open(path, 'wb') as f:
            pickle.dump(self._replay_buffer, f, protocol=4)

    def restore(self, path):
        with lzma.open(path, 'rb') as f:
            self._replay_buffer = pickle.load(f)
