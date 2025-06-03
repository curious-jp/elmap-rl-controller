# Robust Reinforcement Learning-Based Locomotion for Resource-Constrained Quadrupeds with Exteroceptive Sensing

Training code implementation of the paper: Robust Reinforcement Learning-Based Locomotion for Resource-Constrained Quadrupeds with Exteroceptive Sensing.


# Table of contents
1. [Overview](#overview)
2. [Demonstration Video](#video)
3. [License](#license)
2. [System Requirements](#requirements)
3. [Training a Model](#simulation)
    1. [Installation](#installation)
    2. [Environment and Model Configuration](#configuration)
    3. [Training and Logging](#training)
    4. [Analyzing the Policy](#analysis)
4. [Deploying a Model](#realworld)

## Overview <a name="overview"></a>

This repository provides an implementation of the paper [Robust Reinforcement Learning-Based Locomotion for Resource-Constrained Quadrupeds with Exteroceptive Sensing](https://arxiv.org/abs/2505.12537), accepted at the 2025 IEEE International Conference on Robotics & Automation (ICRA).



If you use this repository in your work, consider citing:

```
@article{plozza2025robust,
  title={Robust Reinforcement Learning-Based Locomotion for Resource-Constrained Quadrupeds with Exteroceptive Sensing},
  author={Plozza, Davide and Apostol, Patricia and Joseph, Paul and Schl{\"a}pfer, Simon and Magno, Michele},
  journal={arXiv preprint arXiv:2505.12537},
  year={2025}
}
```


This code builds on top of [Improbable AI's implementation](https://github.com/Improbable-AI/walk-these-ways) which is also MIT licensed.   of the paper  [Walk these Ways: Tuning Robot Control for Generalization with Multiplicity of Behavior](https://gmargo11.github.io/walk-these-ways/)
The environment builds on the [legged gym environment](https://leggedrobotics.github.io/legged_gym/) by Nikita
Rudin, Robotic Systems Lab, ETH Zurich (Paper: https://arxiv.org/abs/2109.11978) and the Isaac Gym simulator from 
NVIDIA (Paper: https://arxiv.org/abs/2108.10470). Training code builds on the 
[rsl_rl](https://github.com/leggedrobotics/rsl_rl) repository, also by Nikita
Rudin, Robotic Systems Lab, ETH Zurich. All redistributed code retains its
original [license](LICENSES/legged_gym/LICENSE).



## Demonstration Video <a name="video"></a>

Real-life performance of our controller is showcased in this [video](https://www.youtube.com/watch?v=tdEDdsTzjxE).

## License <a name="license"></a>

This repository is licensed under the MIT License (see [LICENSE](./LICENSE)).  
It is based on [Improbable AI's original repository](https://github.com/Improbable-AI/walk-these-ways) which is also MIT licensed.  
See `LICENSES/` for third-party license information.


## System Requirements <a name="requirements"></a>

**Simulated Training and Evaluation**: Isaac Gym requires an NVIDIA GPU. To train in the default configuration, we recommend a GPU with at least 10GB of VRAM. The code can run on a smaller GPU if you decrease the number of parallel environments (`Cfg.env.num_envs`). However, training will be slower with fewer environments.


## Training a Model <a name="simulation"></a>

### Installation <a name="installation"></a>

#### Install anaconda3 and create env

Install anaconda3 https://www.anaconda.com/download

Create python 3.8.16 environment
```bash
conda create --name robodog_gym python=3.8.16
```

#### Install pytorch

For RTX 3000 series: 1.10 with cuda-11.3

```bash
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

For RTX 4000 series: 2.2.0 with cuda-12.1 
```bash
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
```


#### Install Isaac Gym

1. Download and install Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym
2. unzip the file via:
    ```bash
    tar -xf IsaacGym_Preview_4_Package.tar.gz
    ```

3. now install the python package
    ```bash
    cd isaacgym/python && pip install -e .
    ```
4. Verify the installation by try running an example

    ```bash
    cd examples
    python 1080_balls_of_solitude.py
    ```
5. For troubleshooting check docs `isaacgym/docs/index.html`

If you get the error `ImportError: libpython3.7m.so.1.0: cannot open shared object file: No such file or directory`:
```bash
cd ~/anaconda3/envs/<env_name>
mkdir -p etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/<use-name>/anaconda3/envs/<env_name>/lib' >> etc/conda/activate.d/ld_library_path.sh
```

#### Install the `robodog_gym` package

In this repository, run `pip install -e .`

If get errors:
```
sudo apt-get install libcurl4-openssl-dev
sudo apt-get install libssl-dev
```

### Verifying the Installation

If everything is installed correctly, you should be able to run the test script with:

```bash
python scripts/test.py
```

The script should print `Simulating step {i}`.
The GUI is off by default. To turn it on, set `headless=False` in `test.py`'s main function call.

### Setup WandB logging

Create a WanB account/team (entity) and project.
You will need to change cfg.cfg_ppo.runner.wandb_entity and Cfg.cfg_ppo.runner.wandb_project in the training file accordingly.

Then login with your account.
```bash
wandb login
```


### Environment and Model Configuration <a name="configuration"></a>


**CODE STRUCTURE** The main environment for simulating a legged robot is
in [legged_robot.py](robodog_gym/envs/base/legged_robot.py). The default configuration parameters including reward
weightings are defined in [go1_backpack_config.py](robodog_gym/robodog/go1_backpack_config.py).

There are three scripts in the [scripts](scripts/) directory:

TODO: add additional types

```bash
scripts
├── __init__.py
├── play_teleop.py
├── test.py
└── train_<configuration>.py
```

You can run the `test.py` script to verify your environment setup.
If it runs then you have installed the gym environments correctly. To train an agent, run one of the `train_<configuration>.py` script. To evaluate a trained agent, run `play_teleop.py`.


We provie one of the trained agent checkpoints used in the paper in the [./runs/exteroceptive_robust_icra_proposed](runs/exteroceptive_robust_icra_proposed) directory.



### Training and Logging <a name="training"></a>

To train the Go1 controller from [Walk these Ways](https://sites.google.com/view/gait-conditioned-rl/), run: 

```bash
python scripts/train.py
```

After initializing the simulator, the script will print out a list of metrics every ten training iterations.

Training with the default configuration requires about 12GB of GPU memory. If you have less memory available, you can 
still train by reducing the number of parallel environments used in simulation (the default is `Cfg.env.num_envs = 4000`).

Runs are logged with both ML Dash and Weight and Biases

To visualize training progress in ML Dash, first start the ml_dash frontend app:
```bash
python -m ml_dash.app
```
then start the ml_dash backend server by running this command in the parent directory of the `runs` folder:
```bash
python -m ml_dash.server .
```

Finally, use a web browser to go to the app IP (defaults to `localhost:3001`) 
and create a new profile with the credentials:

Username: `runs` \
API: [server IP] (default is `http://localhost:8081`) \
Access Token: [blank] \

Now, clicking on the profile should yield a dashboard interface visualizing the training runs.


### WandB logging

We use WandB for logging. Model checkpoints and videos are also stored in the run.

To download checkpoint, configuration, video, and metrics data from an online wandb run, use the [scripts/utils/download_wandb_run.py](scripts/utils/download_wandb_run.py).

```bash
cd scripts
python utils/download_wandb_run.py <arguments>
```


### Analyzing the Policy <a name="analysis"></a>

To evaluate the most recently trained model, run:

```bash
python scripts/play_teleop.py
```

The robot can be controlled with the following key mapping.
- **W, A, S, D**: Control linear velocities in the X and Y directions.  
- **Q, E**: Adjust yaw velocity.  
- **Shift**: Increase speed.  
- **Arrow Keys**: Simulate external forces by pushing the robot.  


## Deploying a Model  <a name="realworld"></a>

Trained agents can be deployed with Improbable's AI [walk-these-ways](https://github.com/Improbable-AI/walk-these-ways/tree/master/go1_gym_deploy) deployment scripts, which needs to be adapted to inlclude elevation map sampling.

GPU-accelerated elevation map (running on a Jetson) can be obtained with RSL's open-source implementation [elevation_mapping_cupy](https://github.com/leggedrobotics/elevation_mapping_cupy).









