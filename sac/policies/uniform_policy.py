from rllab.core.serializable import Serializable

from rllab.misc.overrides import overrides
from sandbox.rocky.tf.policies.base import Policy

import numpy as np


class UniformPolicy(Policy, Serializable):
    """
    Fixed policy that randomly samples actions uniformly at random.

    Used for an initial exploration period instead of an undertrained policy.
    """
    def __init__(self, env_spec):
        Serializable.quick_init(self, locals())
        self._Da = env_spec.action_space.flat_dim

        super(UniformPolicy, self).__init__(env_spec)

    # Assumes action spaces are normalized to be the interval [-1, 1]
    @overrides
    def get_action(self, observation):
        return np.random.uniform(-1., 1., self._Da), None 

    @overrides
    def get_actions(self, observations):
        pass 

    @overrides
    def log_diagnostics(self, paths):
        pass

    @overrides
    def get_params_internal(self, **tags):
        pass 

