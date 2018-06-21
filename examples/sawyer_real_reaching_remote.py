"""
Learn real Sawyer reaching task
"""
import numpy as np

from rllab.envs.normalized_env import normalize
from sac.envs.real.real_sawyer_reaching import SawyerEnvReaching

from sac.replay_buffers.simple_replay_buffer import SimpleReplayBuffer
from sac.misc.remote_sampler import RemoteSampler
from sac.misc.utils import timestamp
from sac.policies.gmm import GMMPolicy
from sac.misc.instrument import run_sac_experiment


from sac.value_functions import NNQFunction, NNVFunction
from sac.algos.sac_algo import SAC


def run(_):
    """
    # Example parameters for randomized targets
    random_target_set = [np.array([0.5, -0.4, 0.5]),
                            np.array([0.5, 0.0, 0.5]),
                            np.array([0.5, 0.4, 0.5])]
                            
    random_target_params = dict(method='set', 
                                targets=random_target_set)
    """
    
    """
    # Example paramaeters for randomized targets using a box
    random_target_params = dict(
                        method='box',
                        x=(0.3, 0.6),
                        y=(-0.4, 0.4),
                        z=(0.2, 0.7)
                        )
    """
    # Joint mask specifies which joints are being controlled. The state space is
    # selected accordingly.
    joint_mask = [True, True, True, True, True, True, True]

    # Reaching target in cartesian space.
    target = np.array([0.5, -0.4, 0.5])

    reaching_env_kwargs = dict(
        target_pos=target,
        target_type='cartesian',
        randomize_target=False,
        random_target_params=None,
        action_cost_coeff=0.001,
        joint_mask=joint_mask,
        include_xpos=True,
        include_pose=False,
        loss_type='l2_distance',
        loss_param=None,
        reset_every_n=1,
    )

    env = SawyerEnvReaching(**reaching_env_kwargs)
    env = normalize(env)

    pool = SimpleReplayBuffer(
        env_spec=env.spec,
        max_replay_buffer_size=1E6
    )
    sampler = RemoteSampler(
        max_path_length=150,
        min_pool_size=150,
        batch_size=256)

    base_kwargs = dict(
        epoch_length=150,
        n_epochs=5000,
        n_train_repeat=1,
        eval_render=False,
        eval_n_episodes=0,  # does not do evaluation episodes
        sampler=sampler,
    )

    M = 256
    qf = NNQFunction(
        env_spec=env.spec,
        hidden_layer_sizes=[M, M],
    )

    vf = NNVFunction(
        env_spec=env.spec,
        hidden_layer_sizes=[M, M],
    )

    policy = GMMPolicy(
        env_spec=env.spec,
        K=1,  # single Gaussian
        hidden_layer_sizes=[M, M],
        qf=qf,
        reg=0.001, 
        smoothing_coeff=0.75,
    )

    algorithm = SAC(
        base_kwargs=base_kwargs,
        env=env,
        policy=policy,
        pool=pool,
        qf=qf,
        vf=vf,

        lr=3e-4,
        scale_reward=30,
        discount=0.99,
        tau=0.001,

        save_full_state=False,
    )

    algorithm.train()


def main():
    exp_prefix = 'sac-reaching'
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
