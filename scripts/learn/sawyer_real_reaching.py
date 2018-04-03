"""
Learn real Sawyer reaching task
"""
import os
import numpy as np

from rllab.envs.normalized_env import normalize

# from softqlearning.algorithms.sql import SQL
# from softqlearning.misc.instrument import run_sql_experiment
# from softqlearning.replay_buffers import SimpleReplayBuffer
# from softqlearning.misc.utils import timestamp
# from softqlearning.misc.remote_sampler import RemoteSampler
# from softqlearning.policies.stochastic_policy import StochasticNNPolicy
# from softqlearning.value_functions import NNQFunction


from softqlearning.environments.real.real_sawyer_reaching import SawyerEnvReaching

from softqlearning.misc.kernel import adaptive_isotropic_gaussian_kernel

from sac.replay_buffers.simple_replay_buffer import SimpleReplayBuffer
from sac.misc.remote_sampler import RemoteSampler
from sac.misc.utils import timestamp
from sac.policies.gmm import GMMPolicy
from sac.misc.instrument import run_sac_experiment


from sac.value_functions import NNQFunction, NNVFunction
from sac.algos.sac_algo import SAC

def run(variant): # parameter is unused
    joint_mask = [True, True, True, True, True, True, True]

    target = np.array([0.5, -0.4, 0.7])
    reaching_env_kwargs = dict(
        target_pos=target,
        target_type='cartesian',
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
        SawyerEnvReaching(**reaching_env_kwargs)
    )

    pool = SimpleReplayBuffer(
        env_spec=env.spec,
        max_replay_buffer_size=1E6
    )

    sampler = RemoteSampler(
        max_path_length=150,
        min_pool_size=150,
        batch_size=128)

    # incorporate remote sampler into base
    base_kwargs = dict(
        epoch_length=500,
        n_epochs=5000,
        # scale_reward=1,
        n_train_repeat=1,
        eval_render=False,
        eval_n_episodes=0, # do not evaluate
        sampler=sampler,
    )

    M = 128
    qf = NNQFunction(
        env_spec=env.spec,
        hidden_layer_sizes=[M, M],
    )

    vf = NNVFunction(
        env_spec=env.spec,
        hidden_layer_sizes=[M, M],
    )

    # policy = StochasticNNPolicy(env_spec=env.spec, hidden_layer_sizes=(M, M), smoothing_coeff=0.5)
    policy = GMMPolicy(
        env_spec=env.spec,
        K=1, # single Gaussian
        hidden_layer_sizes=[M, M],
        qf=qf,
        reg=0.001, # need smoothing
    )
    """
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
        reward_scale=100,
        save_full_state=False,
        save_pool=True
    )
    """
    algorithm = SAC(
        base_kwargs=base_kwargs,
        env=env,
        policy=policy,
        pool=pool,
        qf=qf,
        vf=vf,

        lr=3e-4,
        scale_reward=100,
        discount=0.99,
        tau=0.01,

        save_full_state=False,
    )

    algorithm.train()

def main():
    exp_prefix = 'sql-reaching'
    exp_name = format(timestamp())
    run_sac_experiment(
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
