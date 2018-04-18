from __future__ import absolute_import
import time
import numpy as np
import scipy
import tf # ros transformations library
import random
import os
import sys

import rospy

from intera_interface import CHECK_VERSION, limb, RobotEnable

# from softqlearning.misc.baxter_pykdl import baxter_kinematics
# from baxter_pykdl.baxter_pykdl import baxter_kinematics
# from softqlearning.misc.ros_pykdl import sawyer_kinematics
from sac.misc.ros_pykdl_baxter_urdf_parser import sawyer_kinematics

from rllab import spaces
from rllab.misc.overrides import overrides
from cached_property import cached_property

import config

from rllab.envs.base import Env

# test new neutral for ground lego field
# NEUTRAL = np.array([ 0.49363281, -0.29021289, -1.70333887,  1.78036523,  1.28074414,
#         1.61316406,  1.94795215])

# NEUTRAL = np.array([ 0.31316895, -0.44302539, -1.93846191,  1.28451953,  1.08844824,
#         2.01496582,  2.07264746])

#vertical neutral
# NEUTRAL = np.array([ 0.14158887, -1.22479492, -1.92339355,  0.63440039,  0.61677246,
#         2.44802441,  2.03218457])

# NEUTRAL = np.array([ 0.4340293 , -1.29651563, -2.01877051,  1.07372363,  1.58683691,
#         1.50760449,  1.81149805])

# close neutral
# NEUTRAL = np.array([ 0.29048438, -0.2022959 , -1.81892871,  1.37756934,  1.38645898,
#         1.81822852,  2.22537109])

# safe neutral?
NEUTRAL = np.array([ 0.28625879, -0.06652148, -1.45274023,  1.87639258,  1.47005664,
        1.47084668, -0.87656152])



BOX = np.array([
        [0.2, -0.7, 0.1],
        [0.9,  0.7,  0.9]
])

# define this
SAFE_LINKS = {
    # 'right_l0': BOX,
    # 'right_l1': BOX,
    # 'right_l2': BOX,
    'right_l3': BOX,  # BOX1,
    'right_l4': BOX,  # BOX1,
    'right_l5': BOX,
    'right_l6': BOX,
    'right_hand': BOX,
    # 'right_gripper_base': BOX,
    # 'right_gripper': BOX
}

ALL_JOINTS = (
    'right_l0',
    'right_l1',
    'right_l2',
    'right_l3',
    'right_l4',
    'right_l5',
    'right_l6',
    'right_hand'
)

ALL_LINKS = (
    'right_l0',
    'right_l1',
    'right_l2',
    'right_l3',
    'right_l4',
    'right_l5',
    'right_l6',
    'right_hand',
    'right_gripper_base',
    'right_gripper'
)


class SawyerEnv(Env):
    MAX_TORQUE = 3.
    MAX_TORQUES = 0.5 * np.array([8, 12, 6, 5, 4, 3, 6])

#     ADIM = 7
#     ODIM = 2 * ADIM + 3

    def __init__(self,
                 action_cost_coeff=1.0,
                 joint_mask=None,
                 reset_every_n=1,
                 include_xpos=False,
                 include_pose=False,
                 include_actual_torques=False):
        super(SawyerEnv, self).__init__()

        self._safety_box = BOX

        self.safety_box_magnitude = 3
        self.safety_end_effector_box = True

        if joint_mask is None:
            joint_mask = [True, True, True, True, True, True, True]

        self._joint_mask = joint_mask
        self._action_cost_coeff = action_cost_coeff

        self._Da = sum(self._joint_mask)
        self._Do = 2 * self._Da

        if include_xpos:
            self._Do += 3
        if include_pose:
            self._Do += 3
        if include_actual_torques:
            self._Do += sum(self._joint_mask) # all torques

        self._include_xpos = include_xpos
        self._include_pose = include_pose
        self._include_actual_torques = include_actual_torques

        self._num_iter = 0
        self._reset_every_n = reset_every_n

    def initialize(self, use_remote_name=False):
        """
        This function allows us to create instances of the env
        without interfacing with ROS. This is convenient when
        remote envs need to access the environment (and actual
        robot) to sample actions, while local processes also
        need the environement, but not to actually connect with
        the robot.
        :param use_remote_name: Should be true if being initialized in a remote environment
        :return: None
        """
        if use_remote_name:
            rospy.init_node("rllab_interface_remote")
        else:
            rospy.init_node("rllab_interface")
        rs = RobotEnable(CHECK_VERSION)
        init_state = rs.state().enabled
        if not init_state:
            rs.enable()
        rospy.on_shutdown(self.clean_shutdown)
        self._arm = limb.Limb('right')
        self._kinematics = sawyer_kinematics('right')
        self._rate = rospy.Rate(20)

    @cached_property
    @overrides
    def action_space(self):
        masked_limits = self.MAX_TORQUES[self._joint_mask]
        return spaces.Box(-masked_limits, masked_limits)

    @cached_property
    @overrides
    def observation_space(self):

        lim = np.zeros(self._Do) * 1E6
        return spaces.Box(-lim, lim)

    def step(self, action):
        angle = self.get_joint_angles()
        if not all(self._joint_mask):
            action_all = self._get_pd_torques(angle - NEUTRAL)
            action_all[self._joint_mask] = action
        else:
            action_all = action
        self.set_joint_torques(action_all)
        self._rate.sleep()
        return self._compute_reward(action)

    def _compute_reward(self, action):
        obs = self.get_observation()
        end_episode_immediately = False
        reward = 0
        env_info = dict(
            actual_torques=self.get_joint_torques()
        )
        return obs, reward, end_episode_immediately, env_info


    def _make_safe(self, torques):
        safe_coeff = self.safety_box_magnitude

        pos_dict = self._kinematics.forward_position_kinematics_all(
            link_names=SAFE_LINKS.keys(), components='vel'
        )
        jac_dict = self._kinematics.jacobian_all(link_names=SAFE_LINKS.keys(), components='vel')
        for link_name, pos in pos_dict.iteritems():
            if not self.joint_in_box(link_name, pos):
                print("joint violation", link_name, pos)
                box = SAFE_LINKS[link_name]
                jac = jac_dict[link_name][:3]
                diff = np.maximum(box[0] - pos, 0) + np.minimum(box[1] - pos, 0)

                safe_torques = np.zeros(self._Da)
                t = self.safety_box_magnitude*jac.T.dot(diff)
                n_jnts_involved = t.size
                safe_torques[:n_jnts_involved] = t.squeeze()
                safe_torques *= safe_coeff * self.MAX_TORQUES

                # torques = safe_torques
                torques += safe_torques
        return torques

    def _move_end_effector_upwards(self):
        r = rospy.Rate(10)
        for i in range(10):
            jac = self._kinematics.jacobian()[:3] # maybe the last 3 instead
            torques = jac.T.dot(np.array([0, 0, 3]))
            self.set_joint_torques(np.array(torques).squeeze())
            r.sleep()

    def downwards_effort(self, value):
        jac = self._kinematics.jacobian(components='vel')
        torques = jac.T.dot(np.array([0, 0, -5]))

        r = rospy.Rate(10)
        for i in range(5):
            self.set_joint_torques(np.array(torques).squeeze())
            r.sleep()

    def get_end_effector_pos(self):
        state_dict = self._arm.endpoint_pose()
        pos = state_dict['position']
        return np.array([
            pos.x,
            pos.y,
            pos.z
        ])

    def _get_pd_torques(self, error, p_coeff=15, d_coeff=5):
        joint_velocities = self.get_joint_velocities()

        torques = - p_coeff*error - d_coeff*joint_velocities
        # import ipdb; ipdb.set_trace()
        return torques

    def reset_arm_to_neutral(self):
        r = rospy.Rate(10)
        for i in range(100):
            joint_angles = self.get_joint_angles()
            error = joint_angles - NEUTRAL
            torques = self._get_pd_torques(error, 30, 5)
            self.set_joint_torques(torques)
            if sum(error ** 2) < 0.05:
                break
            r.sleep()

    def reset(self):
        if self._num_iter % self._reset_every_n == 0:
            self.reset_arm_to_neutral()
        self._num_iter += 1
        time.sleep(1)
        return self.get_observation()

    def float(self, time):
        r = rospy.Rate(10)
        for i in range(int(time * 10)):
           torques = np.zeros(self._Da)
           self.set_joint_torques(torques)
           r.sleep()

    def get_observation(self):
        pos = self.get_joint_angles()[self._joint_mask]
        vel = self.get_joint_velocities()[self._joint_mask]
        obs = np.concatenate((pos, vel))
        if self._include_xpos:
            obs = np.concatenate([obs, self.get_end_effector_pos()])
        if self._include_pose:
            obs = np.concatenate([obs, self.compute_euler()])
        if self._include_actual_torques:
            obs = np.concatenate([obs, self.get_joint_torques()])
        return obs

    def compute_euler(self):
        quat = self._arm.endpoint_pose()['orientation']
        return tf.transformations.euler_from_quaternion(quat)

    # computes phi_6 in http://ai2-s2-pdfs.s3.amazonaws.com/5617/8de1001efe54792ad93f6980de5d5e91906b.pdf
    # it would probably be hard to adapt this loss to ignore the rotation of the last joint
    def compute_angle_diff(self):
        quat = self._arm.endpoint_pose()['orientation']
        rot_mat = tf.transformations.quaternion_matrix(quat)
        return np.linalg.norm(scipy.linalg.logm(rot_mat.dot(self._target_rot_mat.T)))

    def ros_loop(self, fun):
        r = rospy.Rate(10)
        for i in range(100):
            print(fun())
            r.sleep()

    def box_finder(self):
        max_pos = -np.inf*np.ones(3)
        min_pos = np.inf*np.ones(3)
        r = rospy.Rate(10)

        try:
            while True:
                r.sleep()
                pose_dict = self._kinematics.forward_position_kinematics_all(
                    link_names=SAFE_LINKS.keys()
                )

                for name, pose in pose_dict.iteritems():
                    max_pos = np.maximum(max_pos, pose[0])
                    min_pos = np.minimum(min_pos, pose[0])

                limits = [(min_pos[i], max_pos[i]) for i in range(3)]
                print(limits)

        except:
            pass

    def start_observing_joints(self):
        r = rospy.Rate(10)
        while True:
            print(self.get_observation())
            r.sleep()

    def clean_shutdown(self):
        print("\nClosing rllab interface.")
        self._arm.exit_control_mode()

    def move_downwards_effort(self, val):
        delta_ee = np.zeros((6,))
        delta_ee[2] = val
        r = rospy.Rate(10)
        for t in range(3):

            delta_jts = np.asarray(self._kinematics.jacobian_transpose().dot(delta_ee))[0]
            self.set_joint_torques(delta_jts)
            r.sleep()

    """
    If components == 'vel' gets 3 x N jacobian only for cartesian coordinates
    If components == 'vel' gets 3 x N jacobian only for angular coordinates
    If components == 'both' gets both as a 6 x N array
    """
    def get_jacobian(self, components='both'):
        jac = self._kinematics.jacobian(components).getA()
        return jac

    def joint_in_box(self, joint_name, joint_pos, print_violation=True):
        BOX = SAFE_LINKS[joint_name]
        # if not self._use_bounding_box_z_axis:
        #     BOX[0][2] = -1.
        in_box = all(joint_pos > BOX[0]) and all(joint_pos < BOX[1])

        if print_violation and not in_box:
            print("Joint violation: ", joint_name, joint_pos)
            print("Box limits: ", BOX)
        return in_box

    def arm_in_box(self):
        pos_dict = self._kinematics.forward_position_kinematics_all(
            link_names=SAFE_LINKS.keys(), components='vel'
        )
        in_box = [self.joint_in_box(name, pos)
                  for name, pos in pos_dict.iteritems()]
        return all(in_box)

    def map_values_to_joints(self, values):
        """
        limb: String 'left' or 'right'
        values: List of numerical values to map to joints using above ordering
                Order '_s0','_s1','_e0','_e1','_w0','_w1','_w2'
        -----
        Returns a dictionary with joint_name:value
        """
        joint_names = self._arm.joint_names()
        return dict(zip(joint_names, values))

    def get_joint_angles(self):
        """
        Returns list of joint angles
        """
        ja_dict = self._arm.joint_angles()
        return np.array([ja_dict[k] for k in sorted(ja_dict)])

    def get_joint_velocities(self):
        """
        Returns list of joint velocities
        """
        jv_dict = self._arm.joint_velocities()
        return np.array([jv_dict[k] for k in sorted(jv_dict)])

    def make_safe_joint_angles(self, torques):
        joint_angles = self.get_joint_angles()
        for i in range(len(torques)):
            if np.abs(joint_angles[i]) > (160.* np.pi / 180) and joint_angles[i]*torques[i] > 0:
                print("Joint limit exceeded for joint {}, disabling torques".format(i))
                torques[i] = 0
        return torques

    def get_joint_torques(self):
        """
        Returns list of joint torques.
        """
        jt_dict = self._arm.joint_efforts()
        return np.array([jt_dict[k] for k in sorted(jt_dict)])

    def set_joint_torques(self, torques):
        """
        """
        if np.isnan(torques).any() or len(torques) != 7:
            return
        # self.get_logging_observations()
        torques = np.clip(np.asarray(torques),
                          -self.MAX_TORQUES, self.MAX_TORQUES)

        torques = self._make_safe(torques)

        # make sure torques do not push joints past the angle limits
        # torques = self.make_safe_joint_angles(torques)
        # print("Torques after enforcing joint limits: ", torques)
        # import pdb; pdb.set_trace()
        torques_dict = self.map_values_to_joints(torques)
        self._arm.set_joint_torques(torques_dict)

    def set_joint_angles(self, joint_angles):
        """
        limb: String 'left' or 'right'
        joint_angles: list of joint angles
        """
        angle_map = self.map_values_to_joints(joint_angles)
        self._arm.move_to_joint_positions(angle_map)

    def get_cartesian_positions(self):
        pose_dict = self._kinematics.forward_position_kinematics_all(
            link_names=ALL_JOINTS
        )

        return dict([(name, pose[0]) for name, pose in pose_dict.iteritems()])

    def get_logging_observations(self):
        print("Joint Angles", self.get_joint_angles())
        print("Joint Velocities", self.get_joint_velocities())
        print("Joint Cartesian Coordinates", self.get_cartesian_positions())


def get_env_dims(joint_mask, include_xpos, include_pose, include_actual_torques, **kwargs):
    Da = sum(joint_mask)
    Do = 2 * Da
    if include_xpos:
        Do += 3
    if include_pose:
        Do += 3
    if include_actual_torques:
        Do += 7  # all torques
    return dict(
        action_space_dim=Da,
        observation_space_dim=Do
    )
if __name__ == '__main__':
    env = SawyerEnv()
    env.initialize()
    import IPython
    import sys
    f = sys.stdout
    IPython.embed()
else:
    import sys
    f = open("sawyer_env_log.txt", 'w')
# sys.stdout = open("sawyer_env_log.txt", 'w')