import argparse
import numpy as np
import joblib
import tensorflow as tf
import os
from sac.misc import utils
from sac.policies.hierarchical_policy import FixedOptionPolicy
from sac.misc.sampler import rollouts
import scipy.stats
import re
import json
import copy
import matplotlib.pyplot as plt

def collect_expert_trajectories(expert_snapshot, max_path_length):
    tf.logging.info('Collecting expert trajectories')
    with tf.Session() as sess:
        data = joblib.load(expert_snapshot)
        policy = data['policy']
        env = data['env']
        num_skills = data['policy'].observation_space.flat_dim - data['env'].spec.observation_space.flat_dim
        traj_vec = []
        with policy.deterministic(True):
            for z in range(num_skills):
                fixed_z_policy = FixedOptionPolicy(policy, num_skills, z)
                new_paths = rollouts(env, fixed_z_policy,
                                     args.max_path_length, n_paths=1)
                path = new_paths[0]
                traj_vec.append(path)
    tf.reset_default_graph()
    return traj_vec



if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--expert_snapshot', type=str, help='Path to the snapshot to imitate.')
    parser.add_argument('--student_snapshot', type=str, help='Path to the snapshot of student.')
    parser.add_argument('--max-path-length', type=int, default=1000)

    args = parser.parse_args()

    expert_tag = os.path.basename(os.path.dirname(args.expert_snapshot))
    expert_exp = re.search('SAC_\d\d\d', args.expert_snapshot).group()
    student_tag = os.path.basename(os.path.dirname(args.student_snapshot))
    student_exp = re.search('SAC_\d\d\d', args.student_snapshot).group()

    # Store in the student folder
    base_folder = os.path.dirname(os.path.dirname(args.student_snapshot))

    student_itr = re.search('\d+', os.path.basename(args.student_snapshot)).group()
    expert_itr = re.search('\d+', os.path.basename(args.expert_snapshot)).group()
    tag = 'STUDENT_{}_{}__EXPERT_{}_{}'.format(student_exp, student_tag,
                                               expert_exp, expert_tag)
    itr = 'STUDENT_{}__EXPERT_{}'.format(student_itr, expert_itr)

    folder = os.path.join(base_folder, tag)
    try:
        os.makedirs(folder)
    except:
        pass
    assert os.path.exists(folder)
    matrix_filename = os.path.join(base_folder, tag,
                                   '{}_matrix.json'.format(itr))

    traj_vec = collect_expert_trajectories(args.expert_snapshot, args.max_path_length)
    num_skills_expert = len(traj_vec)

    tf.logging.info('Discriminating expert trajectories')
    with tf.Session() as sess:
        data = joblib.load(args.student_snapshot)
        policy = data['policy']
        env = data['env']
        num_skills_student = data['policy'].observation_space.flat_dim - data['env'].spec.observation_space.flat_dim

        M = np.zeros((num_skills_expert, num_skills_student))
        L = []
        # Matrix M stores the pairwise "distances"
        # Entry M[i, j] corresponds to expert i and student j.
        # Rows of M sum to 1

        discriminator = data['discriminator']
        for (expert_z, expert_path) in enumerate(traj_vec):
            log_p_z_vec = []
            obs_vec = expert_path['observations']
            action_vec = expert_path['actions']
            for (obs, action) in zip(obs_vec, action_vec):
                logits = discriminator.eval(obs[None], action[None])[0]
                log_p_z = np.log(utils._softmax(logits))
                log_p_z_vec.append(log_p_z)
            L.append(np.array(log_p_z_vec).tolist())
            log_p_z = np.sum(log_p_z_vec, axis=0)
            M[expert_z] = utils._softmax(log_p_z)

    tf.reset_default_graph()
    tf.logging.info('Collecting trajectories of students')
    student_traj_vec = collect_expert_trajectories(args.student_snapshot, args.max_path_length)
    dist_vec = []
    for (expert_z, expert_path) in enumerate(traj_vec):
        student_z = np.argmax(M[expert_z])
        student_path = student_traj_vec[student_z]
        student_x = student_path['observations'][:, 0]
        expert_x = expert_path['observations'][:, 0]
        if len(student_x) < len(expert_x):
            student_x = np.hstack([student_x, student_x[-1] * np.ones(len(expert_x) - len(student_x))])
        elif len(student_x) > len(expert_x):
            student_x = student_x[:len(expert_x)]

        tf.logging.warning('Using maximum x coordinate distance')
        # dist = np.linalg.norm(student_x - expert_x)
        dist = (np.max(student_x) - np.max(expert_x))**2
        dist_vec.append(dist)
        # plt.figure(figsize=(8, 2))
        # plt.plot(student_path['observations'][:, 0], label='student')
        # plt.plot(expert_path['observations'][:, 0], label='expert')
        # plt.legend()
        # l = L[expert_z][-1][student_z]
        # p = M[expert_z, student_z]
        # plt.title('e = %d, s = %d, d = %f, l = %f, p = %f, max_s = %f, max_e = %f' % (expert_z, student_z, dist, l, p, np.max(student_x), np.max(expert_x)))
        # plt.ylim([-1.5, 1.5])
        # plt.savefig(os.path.join(folder, '%02d_%02d.png' % (expert_z, student_z)))
        # plt.close()
    tf.logging.info('Average distance = %f' % np.mean(dist_vec))
    tf.logging.info('Std distance = %f' % np.std(dist_vec))

    tf.logging.info('Writing files')
    d = {
        'M': M.tolist(),
        'L': L,
        'dist_vec': dist_vec,
    }
    with open(matrix_filename, 'w') as f:
        json.dump(d, f)
