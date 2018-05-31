import numpy as np
import time
import random
import os

from sac.envs.real.real_sawyer_base import SawyerEnv

from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides

import rospy
from intera_interface import CHECK_VERSION, limb, RobotEnable


class SawyerEnvReaching(SawyerEnv, Serializable):
    """
    Environment for reaching either in joint space or Cartesian space.

    If target_type is 'joints', then the goal is to reach a specified joint angle.

    If target_type is 'cartesian', then the goal is to have the end effector reach
        a certain position. 
    Cartesian reaching supports several loss types, including l2_distance, Huber 
    loss, and l2_distance_with_bonus. It also allows for a new target to be selected 
    randomly each episode if randomize_target is set to True. The target can be selected
    from a finite set of target locations or uniformly throughout a 3D box.
    """
    def __init__(self, target_pos=None, target_type='joints',
                 action_cost_coeff=1.0,
                 loss_type='l2',
                 loss_param=None,
                 joint_mask=None,
                 include_xpos=False,
                 include_pose=False,
                 randomize_target=False,
                 random_target_params=None,
                 reset_every_n=1,
                 *args, **kwargs):
        #
        # import ipdb; ipdb.set_trace()
        super(SawyerEnvReaching, self).__init__(action_cost_coeff, joint_mask, reset_every_n,
                 include_xpos, include_pose, *args, **kwargs)


        Serializable.quick_init(self, locals())

        self._target_pos = target_pos
        self._target_type = target_type

        """
        Includes the location of the new target as part of the obs.
        Currently only implemented for Cartesian reaching
        """
        if randomize_target:
            assert target_type == 'cartesian', target_type
            self._Do += 3

        self._loss_type = loss_type
        self._loss_param = loss_param

        self._randomize_target = randomize_target
        self._random_target_params = random_target_params
        if self._randomize_target:
            self.get_new_random_target()


    @overrides
    def reset(self):
        if self._randomize_target:
            self.get_new_random_target()
        return super(SawyerEnvReaching, self).reset()

    @overrides
    def _compute_reward(self, obs, action):
        end_episode_immediately = False

        if self._target_type == 'joints':
            actual = self.get_joint_angles()[-self.ADIM:]
            target = self._target_pos[-self.ADIM:]
            pos_cost = sum((actual - target) ** 2)

        elif self._target_type == 'cartesian':
            actual = self.get_end_effector_pos()

            if self._loss_type == 'l2_distance':
                pos_cost = np.sqrt(sum((self._target_pos - actual) ** 2))

            elif self._loss_type == 'huber':
                """Computes the Huber loss"""
                l2_dist = sum((self._target_pos - actual) ** 2)
                delta = self._loss_param['delta']
                if np.sqrt(l2_dist) < delta:
                    pos_cost = 0.5 * l2_dist
                else:
                    pos_cost = delta * (np.sqrt(l2_dist) - 0.5 * delta)

            elif self._loss_type == 'l2_distance_with_bonus':
                l2_dist = np.sqrt(sum((self._target_pos - actual) ** 2))

                """Adds a flat bonus if within a certain distance of the target and terminates the episode"""
                if l2_dist < self._loss_param['threshold']:
                    pos_cost = l2_dist - self._loss_param['bonus_reward']
                    end_episode_immediately = True
                else:
                    pos_cost = l2_dist
            else:
                raise ValueError
        else:
            raise ValueError

        action_cost = self._action_cost_coeff * sum(action ** 2)
        reward = -action_cost - pos_cost
        distance_from_goal = np.sqrt(sum((self.get_end_effector_pos() - self._target_pos) ** 2))
        end_effector_pos = self.get_end_effector_pos()
        target_pos = self._target_pos

        env_info = { 
            actual_torques=self.get_joint_torques(),
            action_cost=action_cost,
            pos_cost=pos_cost,
            distance_from_goal=distance_from_goal,
            end_effector_pos=end_effector_pos,
            target_pos=target_pos
        }

        return reward, end_episode_immediately, env_info

    """
    Overrides base method in SawyerEnv to include goal position.
    """
    @overrides
    def get_observation(self):
        obs = super(SawyerEnvReaching, self).get_observation()
        if self._randomize_target:
            obs = np.concatenate((obs, self._target_pos))
        return obs

    """
    Selects a new target for the next episode.
    """
    def get_new_random_target(self):
        if self._random_target_params["method"] == "box":
            low_x, up_x = self._random_target_params["x"]
            low_y, up_y = self._random_target_params["y"]
            low_z, up_z = self._random_target_params["z"]
            x = np.random.uniform(low_x, up_x)
            y = np.random.uniform(low_y, up_y)
            z = np.random.uniform(low_z, up_z)
            self._target_pos = np.array([x, y, z])
        elif self._random_target_params["method"] == "set":
            targets = self._random_target_params["targets"]
            self._target_pos = random.choice(targets)
        else:
            raise ValueError

if __name__ == '__main__':
    env = SawyerEnvReaching()
    env.initialize()
    import IPython
    IPython.embed()
