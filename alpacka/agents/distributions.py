"""Distributions used by agents to sample actions."""

import numpy as np

from alpacka import data
from alpacka.utils import space as space_utils


class ProbabilityDistribution:
    """Base class for probability distributions."""

    def compute_statistics(self, logits):
        """Computes probabilities, log probabilities and distribution entropy.

        Args:
            logits (np.ndarray): Distribution parameters.

        Returns:
            dict: with keys 'prob', 'logp', 'entropy'.
        """
        raise NotImplementedError()

    def sample(self, logits):
        """Samples from the distribution.

        Args:
            logits (np.ndarray): Distribution parameters.

        Returns:
            Distribution-dependent: sample from the distribution.
        """
        raise NotImplementedError()

    def params_signature(self, action_space):  # pylint: disable=redundant-returns-doc,useless-return
        """Defines the signature of parameters this dist. is parameterized by.

        Overriding is optional.

        Args:
            action_space (gym.Space): Environment action space.

        Returns:
            TensorSignature or None: Either the parameters tensor signature or
            None if the distribution isn't parameterized.
        """
        del action_space
        return None


class CategoricalDistribution(ProbabilityDistribution):
    """Categorical probabilistic distribution.

    Softmax with temperature."""

    def __init__(self, temperature):
        """Initializes CategoricalDistribution.

        Args:
            temperature (float): Softmax temperature parameter.
        """
        super().__init__()
        self.temperature = temperature

    def compute_statistics(self, logits):
        """Computes softmax, log softmax and entropy with temperature."""
        w_logits = logits / self.temperature
        c_logits = w_logits - np.max(w_logits, axis=-1, keepdims=True)
        e_logits = np.exp(c_logits)
        sum_e_logits = np.sum(e_logits, axis=-1, keepdims=True)

        prob = e_logits / sum_e_logits
        logp = c_logits - np.log(sum_e_logits)
        entropy = -np.sum(prob * logp, axis=-1)

        return {'prob': prob, 'logp': logp, 'entropy': entropy}

    def sample(self, logits):
        """Sample from categorical distribution with temperature in log-space.

        See: https://stats.stackexchange.com/a/260248"""
        w_logits = logits / self.temperature
        u = np.random.uniform(size=w_logits.shape)
        return np.argmax(w_logits - np.log(-np.log(u)), axis=-1)

    @staticmethod
    def params_signature(action_space):
        return data.TensorSignature(
            shape=(space_utils.max_size(action_space),)
        )


class EpsilonGreedyDistribution(ProbabilityDistribution):
    """Epsilon-greedy probability distribution."""

    def __init__(self, epsilon):
        """Initializes EpsilonGreedyDistribution.

        Args:
            epsilon (float): Probability of taking random action.
        """
        super().__init__()
        self.epsilon = epsilon

    def compute_statistics(self, logits):
        prob = np.full(shape=logits.shape,
                       fill_value=self.epsilon/len(logits))
        prob[np.argmax(logits)] += 1 - self.epsilon

        logp = np.log(prob)

        entropy = -np.sum(prob * logp, axis=-1)

        return {'prob': prob, 'logp': logp, 'entropy': entropy}

    def sample(self, logits):
        if np.random.random_sample() < self.epsilon:
            return np.random.choice(len(logits))
        else:
            return np.argmax(logits)

    @staticmethod
    def params_signature(action_space):
        return data.TensorSignature(
            shape=(space_utils.max_size(action_space),)
        )
