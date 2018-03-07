import datetime
import dateutil.tz
import os
import numpy as np


def timestamp():
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    return now.strftime('%Y-%m-%d-%H-%M-%S-%f-%Z')

def concat_obs_z(obs, z, num_skills):
    """Concatenates the observation to a one-hot encoding of Z."""
    assert np.isscalar(z)
    z_one_hot = np.zeros(num_skills)
    z_one_hot[z] = 1
    return np.hstack([obs, z_one_hot])

def split_aug_obs(aug_obs, num_skills):
    """Splits an augmented observation into the observation and Z."""
    (obs, z_one_hot) = (aug_obs[:-num_skills], aug_obs[-num_skills:])
    z = np.where(z_one_hot == 1)[0][0]
    return (obs, z)

def _make_dir(filename):
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)

def _save_video(paths, filename):
    import cv2
    assert all(['ims' in path for path in paths])
    ims = [im for path in paths for im in path['ims']]
    _make_dir(filename)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = 30.0
    (height, width, _) = ims[0].shape
    writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    for im in ims:
        writer.write(im)
    writer.release()

def _softmax(x):
    max_x = np.max(x)
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x)

def get_snapshots(snapshot_dir, num_seeds):
    """Returns a list of filenames of the most recent snapshot."""
    snapshots = []
    for subfolder in os.listdir(snapshot_dir):
        # Ignore the directories from the imitation experiments
        if subfolder.startswith('STUDENT'):
            continue
        max_itr = 0
        max_snapshot = None
        for snapshot in os.listdir(os.path.join(snapshot_dir, subfolder)):
            if snapshot.startswith('itr_') and snapshot.endswith('.pkl'):
                itr = int(os.path.splitext(snapshot)[0].split('_')[-1])
                if itr > max_itr:
                    max_snapshot = os.path.join(snapshot_dir, subfolder,
                                                snapshot)
                    max_itr = itr
        snapshots.append(max_snapshot)
    if num_seeds is None:
        return snapshots
    else:
        return snapshots[:num_seeds]


PROJECT_PATH = os.path.dirname(
    os.path.realpath(os.path.join(__file__, '..', '..')))
