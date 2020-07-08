"""Tests for alpacka.envs."""

import itertools
import functools

import numpy as np
import pytest

from alpacka import envs
from alpacka.envs import gfootball


envs_to_test = [
    envs.BinPacking,
    envs.CartPole,
    envs.Octomaze,
    envs.Sokoban,
]
if gfootball.football_env is not None:
    envs_to_test.append(envs.GoogleFootball)

wrappers_to_test = [
    None,
    functools.partial(envs.TimeLimitWrapper, max_episode_steps=10),
]

@pytest.fixture(scope='module',
                params=itertools.product(envs_to_test, wrappers_to_test))
def env_fn(request):
    """Creates a factory that constructs and wraps an environment."""
    env_cls, wrapper_cls = request.param

    def _env_fn():
        env = env_cls()
        if wrapper_cls is not None:
            env = wrapper_cls(env)
        return env
    return _env_fn


def test_reset_and_step_observations_shape(env_fn):
    # Set up
    env = env_fn()

    # Run
    obs_reset = env.reset()
    (obs_step, _, _, _) = env.step(env.action_space.sample())

    # Test
    assert obs_reset.shape == env.observation_space.shape
    assert obs_step.shape == env.observation_space.shape


def test_restore_after_reset(env_fn):
    # Set up
    env = env_fn()
    obs = env.reset()
    state = env.clone_state()

    # Run
    env.reset()
    obs_ = env.restore_state(state)
    state_ = env.clone_state()

    # Test
    env.step(1)  # Test if can take step
    np.testing.assert_array_equal(obs, obs_)
    np.testing.assert_equal(state, state_)


def test_restore_after_step(env_fn):
    # Set up
    env = env_fn()
    obs = env.reset()
    state = env.clone_state()

    # Run
    # TODO(xxx): Make somehow sure the action changes env state/obs.
    env.step(1)
    obs_ = env.restore_state(state)
    state_ = env.clone_state()

    # Test
    env.step(1)  # Test if can take step
    np.testing.assert_array_equal(obs, obs_)
    np.testing.assert_equal(state, state_)


def test_restore_to_step_after_reset(env_fn):
    # Set up
    env = env_fn()
    env.reset()
    obs, _, _, _ = env.step(1)
    state = env.clone_state()

    # Run
    env.reset()
    obs_ = env.restore_state(state)
    state_ = env.clone_state()

    # Test
    env.step(1)  # Test if can take step
    np.testing.assert_array_equal(obs, obs_)
    np.testing.assert_equal(state, state_)


def test_restore_in_place_of_reset(env_fn):
    # Set up
    env_rolemodel, env_imitator = env_fn(), env_fn()
    obs = env_rolemodel.reset()
    state = env_rolemodel.clone_state()

    # Run
    obs_ = env_imitator.restore_state(state)
    state_ = env_imitator.clone_state()

    # Test
    env_imitator.step(1)  # Test if can take step
    np.testing.assert_array_equal(obs, obs_)
    np.testing.assert_equal(state, state_)
