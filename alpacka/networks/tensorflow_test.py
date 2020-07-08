"""Tests for alpacka.networks.tf_meta_graph."""

import os
import tempfile

import gym
import numpy as np
import pytest

from alpacka import agents
from alpacka.networks import tensorflow as tf_networks

# Fixed values for the Baseline ppo2 model in the GFootball Academy Corner with
# stacked "extracted" observations.
fixed_batch_size = 16
obs_space = gym.spaces.Box(low=0, high=255, shape=(72, 96, 16), dtype=np.uint8)
act_space = gym.spaces.Discrete(n=19)

# Fix for "Initializing libiomp5.dylib, but found libiomp5.dylib already
# initialized." on MacOS. See: https://github.com/dmlc/xgboost/issues/1715.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


@pytest.fixture(scope='module')
def network():
    return tf_networks.TFMetaGraphNetwork(
        network_signature=agents.SoftmaxAgent().network_signature(obs_space,
                                                                  act_space),
        model_path='fixtures/tf_metagraph_checkpoint/'
        'baseline_ppo2_in_gfootball_academy_corner'
    )


@pytest.mark.parametrize('batch_size', [fixed_batch_size//2, fixed_batch_size])
def test_model_valid(network, batch_size):
    # Set up
    zero_x = np.zeros((batch_size,) + obs_space.shape)

    # Run
    out = network.predict(zero_x)

    # Test
    assert out.shape == (batch_size, act_space.n)


def test_modify_weights(network):
    # Set up
    new_params = network.params
    for p in new_params:
        p *= 2

    # Run
    network.params = new_params

    # Test
    for new, current in zip(new_params, network.params):
        assert np.all(new == current)


def test_save_weights(network):
    # Set up, Run and Test
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = os.path.join(temp_dir, 'test')
        assert not os.listdir(temp_dir)
        network.save(checkpoint_path)
        assert os.listdir(temp_dir)


def test_restore_weights(network):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set up
        checkpoint_path = os.path.join(temp_dir, 'test')
        orig_params = network.params
        network.save(checkpoint_path)

        new_params = network.params
        for p in new_params:
            p *= 2
        network.params = new_params

        # Run
        network.restore(checkpoint_path)

        # Test
        for orig, current in zip(orig_params, network.params):
            assert np.all(orig == current)
