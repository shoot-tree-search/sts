"""Data transformation utils."""

import scipy.signal


def discount_cumsum(x, discount):
    """Magic from rllab for computing discounted cumulative sums of vectors.

    Args:
        x (np.array): sequence of floats (eg. rewards from a single episode in
            RL settings)
        discount (float): discount factor (in RL known as gamma)

    Returns:
        Array of cumulative discounted sums. For example:

        If vector x has a form
            [x0,
             x1,
             x2]

        Then the output would be:
            [x0 + discount * x1 + discount^2 * x2,
             x1 + discount * x2,
             x2]
    """
    return scipy.signal.lfilter(
        [1], [1, float(-discount)], x[::-1], axis=0
    )[::-1]
