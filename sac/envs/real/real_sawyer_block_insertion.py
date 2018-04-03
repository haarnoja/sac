import numpy as np
import scipy
import time
import random
import os
import tf
from sac.envs.real.real_sawyer_base import SawyerEnv


class SawyerEnvBlockInsertion(SawyerEnv):

    def __init__(self, target_pos=np.array([ 0.65813969,  0.05324293,  0.28679076]),
                 target_type='joints',
                 action_cost_coeff=1.0,
                 target_angle=(-3.114475262108045, 0.007095421510716096, 1.5514081108775244),
                 angle_cost_coeff=1.0,
                 loss_type='l2',
                 loss_param=None,
                 joint_mask=None,
                 include_xpos=False,
                 include_pose=False,
                 include_actual_torques=False,
                 reset_every_n=1,
                 *args, **kwargs):
        super(SawyerEnvBlockInsertion, self).__init__(action_cost_coeff, joint_mask, reset_every_n, include_xpos,
                                                      include_pose, include_actual_torques, *args, **kwargs)

        self._angle_cost_coeff = angle_cost_coeff
        self._target_pos = target_pos
        self._target_type = target_type
        self._target_angle = target_angle
        self._target_rot_mat = tf.transformations.euler_matrix(*target_angle)

        self._loss_type = loss_type
        self._loss_param = loss_param

    """
    # Overrides reset method in SawyerEnv
    def reset(self):
        distance_from_goal = np.sqrt(sum((self.get_end_effector_pos() - self._target_pos) ** 2))
        f.write("Resetting\n")
        f.write("Target {}\n".format(self._target_pos))
        f.write("End Effector {}\n".format(self.get_end_effector_pos()))
        f.write("Distance {}\n".format(distance_from_goal))
        f.flush()
        if self._num_iter % self._reset_every_n == 0:
            self.reset_arm_to_neutral()
        f.write("Finished resetting\n")
        f.flush()
        self._num_iter += 1
        time.sleep(1)
        return self.get_observation()
    """

    def _compute_reward(self, action):
        obs = self.get_observation()
        end_episode_immediately = False

        angle_diff = self.compute_angle_diff()
        angle_cost = self._angle_cost_coeff * angle_diff
        # angle_cost += 0.025 * np.log(angle_diff**2 + 0.1) # lorentz loss for angle

        if self._target_type == 'cartesian':
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
            elif self._loss_type == 'lorentz':
                l2_dist = sum(np.array([2.5, 2.5, 1]) * (self._target_pos - actual) ** 2)
                pos_cost = self._loss_param['scale'] * np.log(self._loss_param['c'] * l2_dist + self._loss_param['delta'])
            elif self._loss_type == 'l2_with_bonus':
                l2_dist = np.sqrt(sum((self._target_pos - actual) ** 2))
                if l2_dist < self._loss_param['threshold']:
                    pos_cost = l2_dist - self._loss_param['bonus_reward']
                    # end_episode_immediately = True
                else:
                    pos_cost = l2_dist
            elif self._loss_type == 'lorentz+l2':
                l2_dist = sum(np.array([1.25, 1.25, 1]) * (self._target_pos - actual) ** 2)
                pos_cost = self._loss_param['scale'] * np.log(self._loss_param['c'] * l2_dist + self._loss_param['delta'])
                pos_cost += np.sqrt(l2_dist)
            elif self._loss_type == 'lorentz+l2+bonus':
                l2_dist = sum(np.array([1, 1, 1.25]) * (self._target_pos - actual) ** 2)
                pos_cost = self._loss_param['scale'] * np.log(self._loss_param['c'] * l2_dist + self._loss_param['delta'])
                pos_cost += 10 * np.sqrt(l2_dist)
                # angle loss always less than 0.3, has never been chagned since adding
                if np.sqrt(l2_dist) < self._loss_param['threshold'] and angle_diff < 0.25:
                    # add reward for downwards force here?
                    pos_cost += self._loss_param['downwards_force_reward'] * self.compute_downwards_force(action)
                    pos_cost -= self._loss_param['bonus_reward'] / 2
                    if np.sqrt(l2_dist) < self._loss_param['threshold'] / 2 and angle_diff < 0.12:
                        pos_cost -= self._loss_param['bonus_reward'] / 2
            elif self._loss_type == 'lorentz+l2+bonus_multitarget':
                x, y, z = actual[0], actual[1], actual[2]
                range_mask = self._loss_param['range_mask']
                ranges = self._loss_param['ranges']
                dist = np.zeros(3)
                exact_dists = 0
                for i in range(3):
                    if range_mask[i]:
                        dist[i] = np.abs(actual[i] - self._target_pos[i])
                        exact_dists += dist[i]
                    else:
                        dist[i] = self.compute_l2_dist_outside_box(actual[i], ranges[i])
                pos_cost =  10 * np.sum(dist)
                if not self._loss_param['avoid'] or y > self._loss_param['avoid_y_thresh']:
                    pos_cost += self._loss_param['scale'] * np.log(
                        self._loss_param['c'] * exact_dists + self._loss_param['delta'])

                f.write("exact_dists: {}\n".format(exact_dists))
                if exact_dists < self._loss_param['threshold'] and angle_diff < 0.3:
                    if exact_dists < self._loss_param['threshold']/2:
                        pos_cost -= self._loss_param['bonus_reward']
                    # add reward for downwards force here?
                    if not self._loss_param['avoid']:
                        pos_cost += self._loss_param['downwards_force_reward'] * self.compute_downwards_force(action)
                if self._loss_param['avoid']:
                    # avoids going past a certain y
                    if y > self._loss_param['avoid_y_thresh'] and z < self._loss_param['avoid_z_thresh']:
                        pos_cost += self._loss_param['avoid_cost']
            elif self._loss_type == 'lorentz+l2+bonus_discrete_targets':
                targets = self._loss_param['targets']
                squared_dists = [(t - actual)**2 for t in targets]
                pos_cost = self._loss_param['l2_weight'] * np.sum(squared_dists)
                pos_cost += self._loss_param['scale'] * np.sum(
                    [np.log(self._loss_param['c'] * d + self._loss_param['delta']) for d in squared_dists])
                if np.min(np.sqrt(squared_dists)) < self._loss_param['threshold'] and angle_diff < 0.3:
                    pos_cost -= self._loss_param['bonus_reward']
                    pos_cost += self._loss_param['downwards_force_reward'] * self.compute_downwards_force(action)
            else:
                raise ValueError
        else:
            raise ValueError

        action_cost = self._action_cost_coeff * sum(action ** 2)
        reward = -action_cost - pos_cost - angle_cost
        distance_from_goal = np.sqrt(sum((self.get_end_effector_pos() - self._target_pos) ** 2))
        end_effector_pos = self.get_end_effector_pos()
        target_pos = self._target_pos
        env_info = dict(
            actual_torques=self.get_joint_torques(),
            action_cost=action_cost,
            pos_cost=pos_cost,
            angle_cost=angle_cost,
            distance_from_goal=distance_from_goal,
            end_effector_pos=end_effector_pos,
            target_pos=target_pos
        )

        return obs, reward, end_episode_immediately, env_info
    """
    def compute_reward_from_obs(self, obs, action):
        jointdim = sum(self._joint_mask)
        qpos = obs[:jointdim]
        qvel = obs[jointdim: jointdim * 2]
        xpos = obs[2*jointdim: 2*jointdim + 3]
        angle = obs[2*jointdim+3: 2*jointdim+6]
        actual_torques = obs[2*jointdim+6:]

        # rescaling actions to match env
        lb, ub = self.action_space.bounds
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        scaled_action = np.clip(scaled_action, lb, ub)

        angle_diff = self.compute_angle_loss_from_euler(angle)
        angle_cost = self._angle_cost_coeff * angle_diff

        if self._loss_type == 'lorentz+l2+bonus_multitarget':
            x, y, z = xpos[0], xpos[1], xpos[2]
            range_mask = self._loss_param['range_mask']
            ranges = self._loss_param['ranges']
            dist = np.zeros(3)
            exact_dists = 0
            for i in range(3):
                if range_mask[i]:
                    dist[i] = np.abs(xpos[i] - self._target_pos[i])
                    exact_dists += dist[i]
                else:
                    dist[i] = self.compute_l2_dist_outside_box(xpos[i], ranges[i])
            pos_cost = 10 * np.sum(dist)
            if not self._loss_param['avoid'] or y > self._loss_param['avoid_y_thresh']:
                pos_cost += self._loss_param['scale'] * np.log(
                    self._loss_param['c'] * exact_dists + self._loss_param['delta'])

            f.write("exact_dists: {}\n".format(exact_dists))
            if exact_dists < self._loss_param['threshold'] and angle_diff < 0.3:
                if exact_dists < self._loss_param['threshold'] / 2:
                    pos_cost -= self._loss_param['bonus_reward']
                # add reward for downwards force here?
                if not self._loss_param['avoid']:
                    pos_cost += self._loss_param['downwards_force_reward'] * self.compute_downwards_force(action)
            if self._loss_param['avoid']:
                # avoids going past a certain y
                if y > self._loss_param['avoid_y_thresh'] and z < self._loss_param['avoid_z_thresh']:
                    pos_cost += self._loss_param['avoid_cost']
        action_cost = self._action_cost_coeff * sum(scaled_action ** 2)
        reward = -action_cost - pos_cost - angle_cost
        return reward
    """

    # only use if allowing one axis of rotation
    def compute_euler_loss(self):
        euler_angle = self.compute_euler()
        euler_alpha = np.abs(euler_angle[0])
        euler_beta = np.abs(euler_angle[1])
        return ((np.pi - euler_alpha) ** 2 + euler_beta ** 2)

    # same as compute angle loss with an euler angle given as argument
    def compute_angle_loss_from_euler(self, angle):
        rot_mat = tf.transformations.euler_matrix(*angle)
        return np.linalg.norm(scipy.linalg.logm(rot_mat.dot(self._target_rot_mat.T)))

    def compute_downwards_force(self, action):
        jac = self._kinematics.jacobian_for('right_hand')
        jac_T_pinv = np.linalg.pinv(jac.T)

        return np.array(jac_T_pinv.dot(action)).squeeze()[2]

    def compute_l2_dist_outside_box(self, x, interval):
        x1, x2 = interval
        if x < x1:
            return x1 - x
        if x > x2:
            return x - x2
        return 0

if __name__ == '__main__':
    env = SawyerEnvBlockInsertion()
    import IPython
    import sys
    f = sys.stdout
    IPython.embed()
else:
    f = open("sawyer_env_log.txt", 'w')