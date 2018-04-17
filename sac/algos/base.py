import abc
import gtimer as gt

import numpy as np

from rllab.misc import logger
from rllab.algos.base import Algorithm

from sac.core.serializable import deep_clone
from sac.misc import tf_utils
from sac.misc.sampler import rollouts


class RLAlgorithm(Algorithm):
    """Abstract RLAlgorithm.

    Implements the _train and _evaluate methods to be used
    by classes inheriting from RLAlgorithm.
    """

    def __init__(
            self,
            batch_size=64,
            n_epochs=1000,
            n_train_repeat=1,
            epoch_length=1000,
            min_pool_size=10000,
            max_path_length=1000,
            sampler=None,
            eval_n_episodes=10,
            eval_deterministic=True,
            eval_render=False,
    ):
        """
        Args:
            batch_size (`int`): Size of the sample batch to be used
                for training.
            n_epochs (`int`): Number of epochs to run the training for.
            n_train_repeat (`int`): Number of times to repeat the training
                for single time step.
            epoch_length (`int`): Epoch length.
            min_pool_size (`int`): Minimum size of the sample pool before
                running training.
            max_path_length (`int`): Number of timesteps before resetting
                environment and policy, and the number of paths used for
                evaluation rollout.
            scale_reward (`float`): Scaling factor for raw reward.
            eval_n_episodes (`int`): Number of rollouts to evaluate.
            eval_deterministic (`int`): Whether or not to run the policy in
                deterministic mode when evaluating policy.
            eval_render (`int`): Whether or not to render the evaluation
                environment.
        """
        self._batch_size = batch_size
        self._n_epochs = n_epochs
        self._n_train_repeat = n_train_repeat
        self._epoch_length = epoch_length
        self._min_pool_size = min_pool_size
        self._max_path_length = max_path_length

        self._eval_n_episodes = eval_n_episodes
        self._eval_deterministic = eval_deterministic
        self._eval_render = eval_render

        self._sess = tf_utils.get_default_session()
        self._sampler = sampler

        self._env = None
        self._policy = None
        self._pool = None

    def _train(self, env, policy, pool):
        self._init_training(env, policy, pool)
        self._sampler.initialize(env, policy, pool)

        with tf_utils.get_default_session().as_default():
            gt.rename_root('RLAlgorithm')
            gt.reset()
            gt.set_def_unique(False)

            for epoch in gt.timed_for(
                    range(self._n_epochs + 1), save_itrs=True):
                logger.push_prefix('Epoch #%d | ' % epoch)

                for t in range(self._epoch_length):
                    self._sampler.sample()
                    if not self._sampler.batch_ready():
                        continue
                    gt.stamp('sample')

                    for i in range(self._n_train_repeat):
                        self._do_training(
                            itr=t + epoch * self._epoch_length,
                            batch=self._sampler.random_batch())
                    gt.stamp('train')

                # TODO: no evaluations for now
                # self._evaluate(policy, evaluation_env)
                gt.stamp('eval')

                params = self.get_snapshot(epoch)
                logger.save_itr_params(epoch, params)

                time_itrs = gt.get_times().stamps.itrs
                time_eval = time_itrs['eval'][-1]
                time_total = gt.get_times().total
                time_train = time_itrs.get('train', [0])[-1]
                time_sample = time_itrs.get('sample', [0])[-1]

                logger.record_tabular('time-train', time_train)
                logger.record_tabular('time-eval', time_eval)
                logger.record_tabular('time-sample', time_sample)
                logger.record_tabular('time-total', time_total)
                logger.record_tabular('epoch', epoch)

                self._sampler.log_diagnostics()

                logger.dump_tabular(with_prefix=False)
                logger.pop_prefix()

            self._sampler.terminate()

    def _evaluate(self, epoch):
        """Perform evaluation for the current policy.

        :param epoch: The epoch number.
        :return: None
        """

        if self._eval_n_episodes < 1:
            return

        with self._policy.deterministic(self._eval_deterministic):
            paths = rollouts(self._eval_env, self._policy,
                             self._max_path_length, self._eval_n_episodes,
                             False)

        total_returns = [path['rewards'].sum() for path in paths]
        episode_lengths = [len(p['rewards']) for p in paths]

        logger.record_tabular('return-average', np.mean(total_returns))
        logger.record_tabular('return-min', np.min(total_returns))
        logger.record_tabular('return-max', np.max(total_returns))
        logger.record_tabular('return-std', np.std(total_returns))
        logger.record_tabular('episode-length-avg', np.mean(episode_lengths))
        logger.record_tabular('episode-length-min', np.min(episode_lengths))
        logger.record_tabular('episode-length-max', np.max(episode_lengths))
        logger.record_tabular('episode-length-std', np.std(episode_lengths))

        self._eval_env.log_diagnostics(paths)
        if self._eval_render:
            self._eval_env.render(paths)

        batch = self._pool.random_batch(self._batch_size)
        self.log_diagnostics(batch)

    @abc.abstractmethod
    def log_diagnostics(self, batch):
        raise NotImplementedError

    @abc.abstractmethod
    def get_snapshot(self, epoch):
        raise NotImplementedError

    @abc.abstractmethod
    def _do_training(self, itr, batch):
        raise NotImplementedError

    @abc.abstractmethod
    def _init_training(self, env, policy, pool):
        """Method to be called at the start of training.

        :param env: Environment instance.
        :param policy:  Policy instance.
        :return: None
        """

        self._env = env
        if self._eval_n_episodes > 0:
            self._eval_env = deep_clone(env)
        self._policy = policy
        self._pool = pool

    @property
    def policy(self):
        return self._policy

    @property
    def env(self):
        return self._env

    @property
    def pool(self):
        return self._pool
