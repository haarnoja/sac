import numpy as np
import time
import random
import os
from sac.envs.real.real_sawyer_base import SawyerEnv

from rllab.core.serializable import Serializable
import rospy
from intera_interface import CHECK_VERSION, limb, RobotEnable


class SawyerEnvReaching(SawyerEnv, Serializable):

    def __init__(self, target_pos=None, target_type='joints',
                 action_cost_coeff=1.0,
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
        super(SawyerEnvReaching, self).__init__(action_cost_coeff, joint_mask, reset_every_n,
                 include_xpos, include_pose, include_actual_torques, *args, **kwargs)


        Serializable.quick_init(self, locals())

        self._target_pos = target_pos
        self._target_type = target_type
        if randomize_target:
            self._Do += 3

        self._loss_type = loss_type
        self._loss_param = loss_param

        self._randomize_target = randomize_target
        self._random_target_params = random_target_params
        if self._randomize_target:
            self.get_new_random_target()


    # Overrides reset method in SawyerEnv to select a new target
    def reset(self):
        distance_from_goal = np.sqrt(sum((self.get_end_effector_pos() - self._target_pos) ** 2))
        f.write("Resetting\n")
        f.write("Target {}\n".format(self._target_pos))
        f.write("End Effector {}\n".format(self.get_end_effector_pos()))
        f.write("Distance {}\n".format(distance_from_goal))
        f.flush()
        # os.fsync(f)
        if self._num_iter % self._reset_every_n == 0:
            self.reset_arm_to_neutral()
        f.write("Finished resetting\n")
        f.flush()
        os.fsync(f)

        self._num_iter += 1
        time.sleep(1)
        if self._randomize_target:
            self.get_new_random_target()
        return self.get_observation()

    def _compute_reward(self, action):
        obs = self.get_observation()
        end_episode_immediately = False

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

        action_cost = self._action_cost_coeff * sum(action ** 2)
        reward = -action_cost - pos_cost
        distance_from_goal = np.sqrt(sum((self.get_end_effector_pos() - self._target_pos) ** 2))
        end_effector_pos = self.get_end_effector_pos()
        target_pos = self._target_pos
        env_info = dict(
            actual_torques=self.get_joint_torques(),
            action_cost=action_cost,
            pos_cost=pos_cost,
            distance_from_goal=distance_from_goal,
            end_effector_pos=end_effector_pos,
            target_pos=target_pos
        )

        return obs, reward, end_episode_immediately, env_info

    # Overrides base method in SawyerEnv to include goal position
    def get_observation(self):
        obs = super(SawyerEnvReaching, self).get_observation()
        if self._randomize_target:
            obs = np.concatenate((obs, self._target_pos))
        return obs

    def get_new_random_target(self):
        # Change the target after every reset, which happens each iteration if enabled
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
    import IPython
    import sys
    f = sys.stdout
    IPython.embed()
else:
    f = open("sawyer_env_log.txt", 'w')