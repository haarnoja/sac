# Soft Actor-Critic
Soft actor-critic is a deep reinforcement learning framework for training maximum entropy policies in continous domains. [TODO: paper reference]

## Getting Started

Soft Actor-Critic can be run either locally or through Docker.

### Prerequisites

You will need to have [Docker](https://docs.docker.com/engine/installation/) and [Docker Compose](https://docs.docker.com/compose/install/) installed unless you want to run the environment locally.

Most of the models require a [Mujoco](https://www.roboti.us/license.html) license.

### Installing

If you want to run the Mujoco environments, the docker environment needs to know where to find your Mujoco license key (`mjkey.txt`). You can either copy your key into `<PATH_TO_THIS_REPOSITY>/.mujoco/mjkey.txt`, or you can specify the path to the key in your environment variables:

```
export MUJOCO_LICENSE_PATH=<path_to_mujoco>/mjkey.txt
```

Once that's done, you can run the Docker container with

```
docker-compose up
```

Docker compose creates a Docker container named `soft-actor-critic` and automatically sets the needed environment variables and volumes.

You can access the container with the typical Docker [exec](https://docs.docker.com/engine/reference/commandline/exec/)-command, i.e.

```
docker exec -it soft-actor-critic bash
```

To clean up the setup:
```
docker-compose down
```

### Examples
#### Training and simulating an agent
1. To train the agent
```
python ./examples/mujoco_all_sac.py --env=swimmer --log_dir="/root/softqlearning-private/data/swimmer-experiment"
```

2. To simulate the agent
```
python ./scripts/sim_policy.py /root/softqlearning-private/data/swimmer-experiment/itr_<iteration>.pkl
```

This step currently fails with the Docker installation, due to missing display.

# Credits
The soft actor-critic algorithm was developed by Tuomas Haarnoja under the supervision of [Prof. Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/) and [Prof. Pieter Abbeel](https://people.eecs.berkeley.edu/~pabbeel/) at UC Berkeley. Special thanks to [Vitchyr Pong](https://github.com/vitchyr), who wrote some parts of the code, and [Kristian Hartikainen](https://github.com/hartikainen) who helped testing and documenting the code. The work was supported by [Berkeley Deep Drive](https://deepdrive.berkeley.edu/).

# Reference
TODO