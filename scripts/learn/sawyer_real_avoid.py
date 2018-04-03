"""
Learn real Sawyer reaching task
"""
import os
import numpy as np

from rllab.envs.normalized_env import normalize

from softqlearning.algorithms.sql import SQL
from softqlearning.misc.instrument import run_sql_experiment
from softqlearning.replay_buffers import SimpleReplayBuffer
from softqlearning.misc.utils import timestamp
from softqlearning.misc.remote_sampler import RemoteSampler
from softqlearning.policies.stochastic_policy import StochasticNNPolicy
from softqlearning.value_functions import NNQFunction

from softqlearning.environments.real.real_sawyer_avoid_and_reach import SawyerEnvAvoidingAndReaching

from softqlearning.misc.kernel import adaptive_isotropic_gaussian_kernel


def run(variant):  # parameter is unused
    joint_mask = [True, True, True, True, True, True, True]

    target = np.array([ 0.65813969,  0.05324293,  0.30079076]) # old target at block

    # target = np.array([ 0.64930338, -0.39607906,  1.0251332 ])

    # lorentz_bonus_param = {'delta': 0.001, 'c': 2., 'scale': 1., 'threshold': 0.03,
    #                        'bonus_reward': 3, 'downwards_force_reward': 5}

    target_angle = (-3.114475262108045, 0.007095421510716096, 1.5514081108775244)

    # avoid_box = np.array([[0.3, -1, .5], [1, 0.3, 0.7]]) # obstacle above target

    avoid_box = np.array([[0.0, 0.15, 0.0], [1, 0.25, 0.45]])
    avoid_env_kwargs = dict(
        target_pos=target,
        target_type='cartesian',
        angle_cost_coeff=1,
        target_angle=target_angle,
        avoid_box=avoid_box,
        avoid_cost_coeff=10,
        randomize_target=False,
        action_cost_coeff=0.001,
        joint_mask=joint_mask,
        include_xpos=True,
        include_pose=True,
        include_actual_torques=False,
        loss_type='l2',
        loss_param=None,
        reset_every_n=1,
    )

    env = normalize(
        SawyerEnvAvoidingAndReaching(**avoid_env_kwargs)
    )

    pool = SimpleReplayBuffer(
        env_spec=env.spec,
        max_replay_buffer_size=1E6
    )

    sampler = RemoteSampler(
        max_path_length=250,
        min_pool_size=250,
        batch_size=128)

    base_kwargs = dict(
        epoch_length=500,
        n_epochs=1000,
        n_train_repeat=1,
        eval_render=False,
        eval_n_episodes=0,  # do not evaluate
        sampler=sampler,
    )

    M = 256
    qf = NNQFunction(
        env_spec=env.spec,
        hidden_layer_sizes=(M, M),
    )

    policy = StochasticNNPolicy(env_spec=env.spec, hidden_layer_sizes=(M, M), smoothing_coeff=0.5, squash=True) # maybe dont squash?, more smoothing?

    algorithm = SQL(
        base_kwargs=base_kwargs,
        env=env,
        pool=pool,
        qf=qf,
        policy=policy,
        kernel_fn=adaptive_isotropic_gaussian_kernel,
        kernel_n_particles=32,
        kernel_update_ratio=0.5,
        value_n_particles=16,
        td_target_update_interval=1000,
        qf_lr=3e-4,
        policy_lr=3e-4,
        discount=0.99,
        reward_scale=1,
        save_full_state=True
    )
    algorithm.train()


def main():
    exp_prefix = 'sql-avoid-with-angle'
    exp_name = format(timestamp())
    run_sql_experiment(
        run,
        exp_prefix=exp_prefix,
        exp_name=exp_name,
        n_parallel=1,
        terminate_machine=True,
        snapshot_mode='gap',
        snapshot_gap=10,
        mode='local',
    )


if __name__ == "__main__":
    main()
