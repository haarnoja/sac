import numpy as np
import tensorflow as tf

from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides

from softqlearning.misc.nn import feedforward_net

from .nn_policy import NNPolicy


class StochasticNNPolicy(NNPolicy, Serializable):
    """Stochastic neural network policy."""

    def __init__(self,
                 env_spec,
                 hidden_layer_sizes,
                 squash=True,
                 smoothing_coeff=None,
                 name='policy'):
        Serializable.quick_init(self, locals())

        self._action_dim = env_spec.action_space.flat_dim
        self._observation_dim = env_spec.observation_space.flat_dim
        self._layer_sizes = list(hidden_layer_sizes) + [self._action_dim]
        self._squash = squash
        self._name = name

        assert smoothing_coeff is None or 0 <= smoothing_coeff <= 1
        self._alpha = smoothing_coeff
        if smoothing_coeff:
            self._beta = self._beta = np.sqrt(1 - self._alpha**2)
            self._x_prev = 0

        self._observations_ph = tf.placeholder(
            tf.float32,
            shape=[None, self._observation_dim],
            name='observations')

        self._actions, self._latents = self.actions_for(
            self._observations_ph, with_latents=True)

        super(StochasticNNPolicy, self).__init__(
            env_spec, self._observations_ph, self._actions, self._name)

    def actions_for(self,
                    observations,
                    n_action_samples=1,
                    reuse=False,
                    with_latents=False):

        n_state_samples = tf.shape(observations)[0]

        if n_action_samples > 1:
            observations = observations[:, None, :]
            latent_shape = (n_state_samples, n_action_samples,
                            self._action_dim)
        else:
            latent_shape = (n_state_samples, self._action_dim)

        latents = tf.random_normal(latent_shape)

        with tf.variable_scope(self._name, reuse=reuse):
            raw_actions = feedforward_net(
                (observations, latents),
                layer_sizes=self._layer_sizes,
                activation_fn=tf.nn.relu,
                output_nonlinearity=None)

        actions = tf.tanh(raw_actions) if self._squash else raw_actions

        return (actions, latents) if with_latents else actions

    @overrides
    def get_action(self, observation):
        if self._mode == 'execute' and self._alpha:
            x = self._alpha * self._x_prev + np.random.randn(self._action_dim)
            latents = self._beta * x
            self._x_prev = x

            feeds = {
                self._obs_pl: observation[None],
                self._latents: latents[None]
            }

            action = tf.get_default_session().run(self._action, feeds)

            return action.squeeze(), dict(latents=latents)

        else:
            return super(StochasticNNPolicy, self).get_action(observation)
