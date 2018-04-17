#!/usr/bin/python

# Copyright (c) 2013-2014, Rethink Robotics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the Rethink Robotics nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import PyKDL

import rospy

# import baxter_interface

import intera_interface

# from baxter_kdl.kdl_parser import kdl_tree_from_urdf_model
from kdl_parser_py.urdf import treeFromUrdfModel
from urdf_parser_py.urdf import URDF


# converts cartesian vector of size 3 to numpy array
def vec_to_array(vector):
    return np.array([vector[i] for i in range(3)])

class sawyer_kinematics(object):
    """
    Baxter Kinematics with PyKDL
    """

    def __init__(self, limb):
        self._sawyer = URDF.from_parameter_server(key='robot_description')
        ok, self._kdl_tree = treeFromUrdfModel(self._sawyer) # kdl_tree_from_urdf_model(self._baxter)
        self._base_link = self._sawyer.get_root()
        self._tip_link = limb + '_hand' # limb + '_l6'
        self._tip_frame = PyKDL.Frame()
        self._arm_chain = self._kdl_tree.getChain(self._base_link,
                                                  self._tip_link)

        self._arm_chain_names_indices_map = dict([(self._arm_chain.getSegment(i).getName(), i)
                                 for i in range(self._arm_chain.getNrOfSegments())])

        # Sawyer Interface Limb Instances
        self._limb_interface = intera_interface.limb.Limb(limb)
        self._joint_names = self._limb_interface.joint_names()
        self._num_jnts = len(self._joint_names)

        # KDL Solvers
        self._fk_p_kdl = PyKDL.ChainFkSolverPos_recursive(self._arm_chain)
        self._fk_v_kdl = PyKDL.ChainFkSolverVel_recursive(self._arm_chain)
        self._ik_v_kdl = PyKDL.ChainIkSolverVel_pinv(self._arm_chain)
        self._ik_p_kdl = PyKDL.ChainIkSolverPos_NR(self._arm_chain,
                                                   self._fk_p_kdl,
                                                   self._ik_v_kdl)
        self._jac_kdl = PyKDL.ChainJntToJacSolver(self._arm_chain)
        self._dyn_kdl = PyKDL.ChainDynParam(self._arm_chain,
                                            PyKDL.Vector.Zero())


    def print_robot_description(self):
        nf_joints = 0
        for j in self._sawyer.joints:
            if j.type != 'fixed':
                print(j.name)
                nf_joints += 1
        print "URDF non-fixed joints: %d;" % nf_joints
        print "URDF total joints: %d" % len(self._sawyer.joints)
        print "URDF links: %d" % len(self._sawyer.links)
        print "KDL joints: %d" % self._kdl_tree.getNrOfJoints()
        print "KDL segments: %d" % self._kdl_tree.getNrOfSegments()

    def print_kdl_chain(self):
        for idx in xrange(self._arm_chain.getNrOfSegments()):
            print '* ' + self._arm_chain.getSegment(idx).getName()

    def name_to_chain_index(self, name):
        return self._arm_chain_names_indices_map[name]

    def joints_to_kdl(self, type, values=None):
        kdl_array = PyKDL.JntArray(self._num_jnts)

        if values is None:
            if type == 'positions':
                cur_type_values = self._limb_interface.joint_angles()
            elif type == 'velocities':
                cur_type_values = self._limb_interface.joint_velocities()
            elif type == 'torques':
                cur_type_values = self._limb_interface.joint_efforts()
        else:
            cur_type_values = values

        for idx, name in enumerate(cur_type_values):
            kdl_array[idx] = cur_type_values[name]
        if type == 'velocities':
            kdl_array = PyKDL.JntArrayVel(kdl_array)
        return kdl_array

    def kdl_to_mat(self, data):
        mat = np.mat(np.zeros((data.rows(), data.columns())))
        for i in range(data.rows()):
            for j in range(data.columns()):
                mat[i, j] = data[i, j]
        return mat

    def forward_position_kinematics(self, joint_values=None):
        end_frame = PyKDL.Frame()

        self._fk_p_kdl.JntToCart(self.joints_to_kdl('positions', joint_values),
                                 end_frame)

        pos = end_frame.p
        rot = PyKDL.Rotation(end_frame.M)
        import ipdb; ipdb.set_trace()
        rot = rot.GetQuaternion()
        return np.array([pos[0], pos[1], pos[2],
                         rot[0], rot[1], rot[2], rot[3]])

    """
    Returns positions and rotations of each link in joint_names
    If components == 'vel', returns 3xN representing the values for translational components
    If components == 'rot' returns 3xN matrix representing the values for rotational components
    If components == 'both' returns 6xN matrix with first 3 rows representing translational components,
                    and last 3 for rotational components.
    """
    def forward_position_kinematics_all(self, link_names, components='both',
                                        joint_values=None, as_vector=False):

        joint_indices = [self.name_to_chain_index(name) for name in link_names]
        frames = [PyKDL.Frame() for _ in link_names]

        for i in range(len(joint_indices)):
            self._fk_p_kdl.JntToCart(self.joints_to_kdl('positions', joint_values),
                                     frames[i], joint_indices[i] + 1)
        if components == 'vel':
            if as_vector:
                pose = [frame.p for frame in frames]
            else:
                pose = [vec_to_array(frame.p) for frame in frames]
        elif components == 'rot':
            pose = [PyKDL.Rotation(frame.M).GetQuaternion() for frame in frames]
        elif components == 'both':
            if as_vector:
                pose = [(frame.p, PyKDL.Rotation(frame.M).GetQuaternion()) for frame in frames]
            else:
                pose = [(vec_to_array(frame.p), PyKDL.Rotation(frame.M).GetQuaternion()) for frame in frames]
        else:
            raise Exception("Components must be either 'vel', 'rot', or 'both'")
        import ipdb; ipdb.set_trace()
        return dict(zip(link_names, pose))

    # todo components for velocity kinematics
    def forward_velocity_kinematics(self, joint_velocities=None):
        end_frame = PyKDL.FrameVel()
        import ipdb; ipdb.set_trace()
        self._fk_v_kdl.JntToCart(self.joints_to_kdl('velocities', joint_velocities),
                                 end_frame)
        return end_frame.GetTwist()

    def forward_velocity_kinematics_all(self, joint_values=None, joint_names=None):
        return 0

    def inverse_kinematics(self, position, orientation=None, seed=None):
        ik = PyKDL.ChainIkSolverVel_pinv(self._arm_chain)
        pos = PyKDL.Vector(position[0], position[1], position[2])
        if orientation != None:
            rot = PyKDL.Rotation()
            rot = rot.Quaternion(orientation[0], orientation[1],
                                 orientation[2], orientation[3])
        # Populate seed with current angles if not provided
        seed_array = PyKDL.JntArray(self._num_jnts)
        if seed != None:
            seed_array.resize(len(seed))
            for idx, jnt in enumerate(seed):
                seed_array[idx] = jnt
        else:
            seed_array = self.joints_to_kdl('positions')

        # Make IK Call
        if orientation:
            goal_pose = PyKDL.Frame(rot, pos)
        else:
            goal_pose = PyKDL.Frame(pos)
        result_angles = PyKDL.JntArray(self._num_jnts)

        if self._ik_p_kdl.CartToJnt(seed_array, goal_pose, result_angles) >= 0:
            result = np.array(list(result_angles))
            return result
        else:
            return None

    """
    Computes Jacobian matrix for the last link (right_hand)
    If components == 'vel', returns 3xN representing the Jacobian for translational components
    If components == 'rot' returns 3xN matrix representing the Jacobian for rotational components
    If components == 'both' returns 6xN matrix with first 3 rows representing translational components,
                    and last 3 for rotational components.
    """
    def jacobian(self, components='both', joint_values=None):
        jacobian = PyKDL.Jacobian(self._num_jnts)
        self._jac_kdl.JntToJac(self.joints_to_kdl('positions', joint_values), jacobian)
        if components == 'vel':
            return self.kdl_to_mat(jacobian)[:3]
        elif components == 'rot':
            return self.kdl_to_mat(jacobian)[3:]
        elif components == 'both':
            return self.kdl_to_mat(jacobian)
        else:
            raise Exception("Components must be either 'vel', 'rot', or 'both'")

    """
    Computes Jacobian matrices for each link in link_names, returning dict mapping names to Jacobians
    If components == 'vel', returns 3xN representing the Jacobian for translational components
    If components == 'rot' returns 3xN matrix representing the Jacobian for rotational components
    If components == 'both' returns 6xN matrix with first 3 rows representing translational components,
                    and last 3 for rotational components.
    """
    def jacobian_all(self, link_names, components='both', joint_values=None):
        jac = PyKDL.Jacobian(self._num_jnts)
        self._jac_kdl.JntToJac(self.joints_to_kdl('positions', joint_values), jac)

        pos_dict = self.forward_position_kinematics_all(link_names, joint_values=joint_values,
                                                         components='vel', as_vector=True)
        end_effector_pos = pos_dict[self._tip_link]

        jacs = [PyKDL.Jacobian(jac) for _ in link_names]

        for i, name in enumerate(link_names):
            jacs[i].changeRefPoint(pos_dict[name] - end_effector_pos)

        if components == 'vel':
            jac_mats = [self.kdl_to_mat(jac)[:3] for jac in jacs]
        elif components == 'rot':
            jac_mats = [self.kdl_to_mat(jac)[3:] for jac in jacs]
        elif components == 'both':
            jac_mats = [self.kdl_to_mat(jac) for jac in jacs]
        else:
            raise Exception("Components must be either 'vel', 'rot', or 'both'")
        return dict(zip(link_names, jac_mats))

    def jacobian_transpose(self, joint_values=None):
        return self.jacobian(joint_values).T

    def jacobian_pseudo_inverse(self, joint_values=None):
        return np.linalg.pinv(self.jacobian(joint_values))

    def inertia(self, joint_values=None):
        inertia = PyKDL.JntSpaceInertiaMatrix(self._num_jnts)
        self._dyn_kdl.JntToMass(self.joints_to_kdl('positions', joint_values), inertia)
        return self.kdl_to_mat(inertia)

    def cart_inertia(self, joint_values=None):
        js_inertia = self.inertia(joint_values)
        jacobian = self.jacobian(joint_values)
        return np.linalg.inv(jacobian * np.linalg.inv(js_inertia) * jacobian.T)
