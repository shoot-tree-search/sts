"""Environments."""

import gin

from alpacka.envs import bin_packing
from alpacka.envs import cartpole
from alpacka.envs import gfootball
from alpacka.envs import octomaze
from alpacka.envs import sokoban
from alpacka.envs.base import *
from alpacka.envs.wrappers import *


# Configure envs in this module to ensure they're accessible via the
# alpacka.envs.* namespace.
def configure_env(env_class):
    return gin.external_configurable(
        env_class, module='alpacka.envs'
    )


ActionNoiseSokoban = configure_env(sokoban.ActionNoiseSokoban) # pylint: disable=invalid-name
BinPacking = configure_env(bin_packing.BinPacking)  # pylint: disable=invalid-name
CartPole = configure_env(cartpole.CartPole)  # pylint: disable=invalid-name
GoogleFootball = configure_env(gfootball.GoogleFootball)  # pylint: disable=invalid-name
Octomaze = configure_env(octomaze.Octomaze)  # pylint: disable=invalid-name
Sokoban = configure_env(sokoban.Sokoban)  # pylint: disable=invalid-name
