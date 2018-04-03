import numpy as np
import time
import random
import os
from sac.envs.real.real_sawyer_base import SawyerEnv

from rllab.core.serializable import Serializable
import rospy
from intera_interface import CHECK_VERSION, limb, RobotEnable
import tf


class SawyerEnvAvoidingAndReaching(SawyerEnv, Serializable):

    def __init__(self, target_pos=None, target_type='joints',
                 action_cost_coeff=1.0,
                 avoid_cost_coeff=5.,
                 avoid_box=np.zeros((2, 3)),
                 angle_cost_coeff=1,
                 target_angle=(0,0,0),
                 loss_type='l2',
                 loss_param=None,
                 joint_mask=None,
                 include_xpos=False,
                 include_pose=False,
                 include_actual_torques=False,
                 randomize_target=False,
                 random_target_params=None,
                 reset_every_n=1,
                 *args, **kwargs):
        #
        # import ipdb; ipdb.set_trace()
        super(SawyerEnvAvoidingAndReaching, self).__init__(action_cost_coeff, joint_mask, reset_every_n,
                 include_xpos, include_pose, include_actual_torques, *args, **kwargs)


        Serializable.quick_init(self, locals())

        self._target_pos = target_pos
        self._target_type = target_type
        if randomize_target:
            self._Do += 3
        self._avoid_box = avoid_box
        self._avoid_cos_coeff = avoid_cost_coeff

        self._angle_cost_coeff = angle_cost_coeff
        self._target_angle = target_angle
        self._target_rot_mat = tf.transformations.euler_matrix(*target_angle)

        self._loss_type = loss_type
        self._loss_param = loss_param

        self._randomize_target = randomize_target
        self._random_target_params = random_target_params
        if self._randomize_target:
            self.get_new_random_target()

    def _compute_reward(self, action):
        obs = self.get_observation()
        end_episode_immediately = False


        angle_diff = self.compute_angle_diff()
        angle_cost = self._angle_cost_coeff * angle_diff

        if self._target_type == 'joints':
            actual = self.get_joint_angles()[-self.ADIM:]
            target = self._target_pos[-self.ADIM:]
            pos_cost = sum((actual - target) ** 2)
        elif self._target_type == 'cartesian':
            actual = self.get_end_effector_pos()
            if self._loss_type == 'l2':
                pos_cost = np.sqrt(sum((self._target_pos - actual) ** 2))
            elif self._loss_type == 'huber':
                l2_dist = sum((self._target_pos - actual) ** 2)
                delta = self._loss_param['delta']
                if np.sqrt(l2_dist) < delta:
                    pos_cost = 0.5 * l2_dist
                else:
                    pos_cost = delta * (np.sqrt(l2_dist) - 0.5 * delta)
            elif self._loss_type == 'l2_with_bonus':
                l2_dist = np.sqrt(sum((self._target_pos - actual) ** 2))
                if l2_dist < self._loss_param['threshold']:
                    pos_cost = l2_dist - self._loss_param['bonus_reward']
                    end_episode_immediately = True
                else:
                    pos_cost = l2_dist
            else:
                raise ValueError
        else:
            raise ValueError
        pos_cost_coeff = 10
        y_pos = self.get_end_effector_pos()[1]

        block_target = np.array([0.65813969, 0.05324293, 0.30079076])
        # if y_pos < 0.2:
        #     cur_target = block_target
        # else:
        #     cur_target = self._target_pos

        action_cost = self._action_cost_coeff * sum(action ** 2)
        avoid_cost = self.avoidance_cost() * self._avoid_cos_coeff
        reward = -action_cost - pos_cost_coeff * pos_cost - avoid_cost - angle_cost
        distance_from_goal = np.sqrt(sum((self.get_end_effector_pos() - block_target) ** 2))
        end_effector_pos = self.get_end_effector_pos()
        target_pos = self._target_pos
        env_info = dict(
            actual_torques=self.get_joint_torques(),
            action_cost=action_cost,
            pos_cost=pos_cost,
            avoid_cost=avoid_cost,
            distance_from_goal=distance_from_goal,
            end_effector_pos=end_effector_pos,
            target_pos=target_pos
        )

        return obs, reward, end_episode_immediately, env_info


    def avoidance_cost(self):
        pos = self.get_end_effector_pos()
        box = self._avoid_box
        return np.all(pos > box[0]) and np.all(pos < box[1])

if __name__ == '__main__':
    env = SawyerEnvAvoidingAndReaching()
    import IPython
    import sys
    f = sys.stdout
    IPython.embed()
else:
    f = open("sawyer_env_log.txt", 'w')