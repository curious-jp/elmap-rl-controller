import isaacgym

assert isaacgym
import torch
import numpy as np
import random


import glob
import os
import pickle as pkl

from robodog_gym.envs import *
from robodog_gym.envs.base.legged_robot_config import Cfg
from robodog_gym.envs.robodog.go1_config import config_go1
from robodog_gym.envs.robodog.velocity_tracking import VelocityTrackingEasyEnv

from params_proto.proto import Meta

from robodog_gym.utils.math_utils import get_scale_shift


import time


from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import *

assert gymtorch


from tqdm import tqdm

def convert_weights_to_jit(env,logdir, iteration,cfg):

    from robodog_gym_learn.ppo_cse.actor_critic import ActorCritic

    if iteration == -1:
        it_label = 'last'
    else:
        it_label = f'{iteration:06d}'


    # Load the model weights
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_weights = torch.load(f'{logdir}/checkpoints/ac_weights_{it_label}.pt', map_location=device)


    print(cfg.cfg_ppo.policy)

    if hasattr(cfg, 'cfg_ppo'):
        cfg_ppo = cfg.cfg_ppo
    else:
        cfg_ppo = None

    print(cfg.cfg_ppo.policy.actor_hidden_dims)

    # Create a new actor critic module and load the weights
    actor_critic = ActorCritic(env.num_policy_obs,
                               env.num_estimator_obs,
                               env.num_privileged_obs,
                               env.num_actions,
                               cfg_ppo,
                               ).to('cpu')
    actor_critic.load_state_dict(model_weights)


    adaptation_module_path = f'{logdir}/checkpoints/adaptation_module_{it_label}.jit'
    traced_script_adaptation_module = torch.jit.script(actor_critic.adaptation_module)
    traced_script_adaptation_module.save(adaptation_module_path)

    body_path = f'{logdir}/checkpoints/body_{it_label}.jit'
    traced_script_body_module = torch.jit.script(actor_critic.actor_body)
    traced_script_body_module.save(body_path)

    print(f"Converted weights for iteration {iteration} to JIT.")


def load_policy(logdir,iteration):
    if iteration == -1:
        label = 'last'
    else:
        label = f'{iteration:06d}'
    body = torch.jit.load(f'{logdir}/checkpoints/body_{label}.jit')
    adaptation_module = torch.jit.load(f'{logdir}/checkpoints/adaptation_module_{label}.jit')

    def policy(obs, info={}):
        i = 0
        latent = adaptation_module.forward(obs["estimator_obs"].to('cpu'))
        action = body.forward(torch.cat((obs["policy_obs"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action, latent

    return policy


def load_config(cfg_dict,Cfg_class):
    """ Load configuration parameteres from dict to class form
    Args:
        cfg_dict: dict containing the config params after unpickling
        Cfg_class: Configuration PrefixProto class
    """
    for key, value in cfg_dict.items():
        if hasattr(Cfg_class, key):
            subcfg = getattr(Cfg_class, key) # member config class
            for key2, value2 in cfg_dict[key].items():
                print(key2)
                if hasattr(subcfg, key2) and isinstance(getattr(subcfg, key2),Meta): # check if it is another nested class
                    for key3, value3 in cfg_dict[key][key2].items():
                        print(key2,key3)
                        setattr(getattr(subcfg, key2), key3, value3)
                else: # or if it is just an attribut
                    setattr(subcfg, key2, value2)

    return Cfg_class

def load_env(label,iteration=-1, headless=False):
    # if label does not specify the exact run time, take the most recent
    logdir = f"./runs/{label}"
    print(logdir)
    if not os.path.exists(os.path.join(f"./runs/{label}", "parameters.pkl")):
        dirs = glob.glob(f"./runs/{label}/*")
        logdir = sorted(dirs)[0]

    # load parameters from run, overwrite default parameters!
    # important to set all parameters correctly
    with open(logdir + "/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        cfg = pkl_cfg["Cfg"]
        # print(cfg.keys())

        # load all parameters into configuration class
        load_config(cfg,Cfg)


    np.random.seed(52)
    random.seed(42)
    torch.manual_seed(45)



    Cfg.asset.file = '{MINI_GYM_ROOT_DIR}/resources/robots/go1_backpack_v3/urdf/go1_backpack.urdf'



    # turn off DR for evaluation script
    Cfg.domain_rand.push_robots = False
    # Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_joint_friction = False
    Cfg.domain_rand.randomize_com_displacement = False

    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 1


    # Enforce specific friction
    Cfg.domain_rand.randomize_friction = True # if false uses default asset friction, no idea what it is
    Cfg.domain_rand.friction_range = [0.8, 0.8] # effective friction averaged with

    # Friction above will be averaged with this values
    # Cfg.terrain.static_friction = 1.0
    # Cfg.terrain.dynamic_friction = 1.0


    # Push robots
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.push_interval_s = 2.0
    Cfg.domain_rand.max_push_vel_xy = 1.0


    # asset setup
    Cfg.asset.self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter. This has also effect for terminal collisions!

    # -----------------
    # Env termination
    # -----------------

    # terminate on these conditions
    Cfg.rewards.use_terminal_foot_height = False
    Cfg.rewards.use_terminal_body_height = False
    Cfg.rewards.use_terminal_body_impact = True
    Cfg.rewards.terminal_body_height = 0.10
    Cfg.rewards.use_terminal_roll_pitch = True
    Cfg.rewards.terminal_body_ori = 1.5 #0.785 #in rads
    Cfg.reward_scales.termination = -0 # not


    # terrain_proportions defines the probabilities of each terrain type
    # terrain types:
    # 0: pyramid_sloped_terrain, slope
    # 1: random_uniform_terrain. Affected by terrain_smoothness
    # 3: pyramid_stairs_terrain. affected ny step_height
        # 2: defines if stairs inverted
    # 4: discrete_obstacles_terrain. discrete_obstacles_height, rectangle_min_size,rectangle_max_size,num_rectangles
    # 5: stepping_stones_terrain stepping_stones_size  stone_distance
    # 6: pass
    # 7: pass
    # 8: random_uniform_terrain, min_height=-cfg.terrain_noise_magnitude, max_height=cfg.terrain_noise_magnitude
    # 9: strange  ersion of random_uniform_terrain

    # terrain_noise_magnitude only affects terrain type 8
    # slope, step_height, discrete_obstacles_height, stepping_stones_size, stone_distance is defined by "difficulty",
    # which is defined by curriculum. If curriculum is disabled, difficulty is chosen at random


    # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete, stepping stones, none, smooth flat, rough flat]
    # Cfg.terrain.terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0, 1.0] #type 8, default
    #Cfg.terrain.terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]

    Cfg.terrain.curriculum = False #disable curriculum. If disabled, terrain "difficulty" is chosen at random
    Cfg.terrain.selected = False # else random parameters are used
    # if selected is true, need to pass terrain_kwargs   # Dict of arguments for selected terrain



    Cfg.terrain.terrain_proportions = [0, 0, 0, 0, 1, 0, 0, 0, 0]

    # Cfg.terrain.terrain_proportions = [0, 0, 0, 0, 1, 0, 0, 0, 0] # discrete steps
    # Cfg.terrain.max_platform_height = 0.1 # controls discrete steps
    Cfg.terrain.terrain_noise_magnitude = 0.1
    Cfg.terrain.slope_treshold = 0.0025 #0.75


    # Terrain params
    Cfg.terrain.mesh_type = "trimesh"  #"heightfield"
    Cfg.terrain.num_rows = 3
    Cfg.terrain.num_cols = 3
    Cfg.terrain.border_size = 10
    Cfg.terrain.terrain_length = 5.
    Cfg.terrain.terrain_width = 5.
    Cfg.terrain.x_init_range = 0.2
    Cfg.terrain.y_init_range = 0.2
    Cfg.terrain.teleport_robots = True
    Cfg.terrain.teleport_thresh = 2.0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.horizontal_scale = 0.10


    # disable max env length
    Cfg.env.episode_length_s = 1000000 # default 20 in seconds
    Cfg.commands.resampling_time = 1000000 # default 20 in seconds



    from robodog_gym.envs.wrappers.history_wrapper import HistoryWrapper

    debug_viz = True
    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=headless, cfg=Cfg, debug_viz=debug_viz)
    env = HistoryWrapper(env)

    # load policy
    from ml_logger import logger
    from robodog_gym_learn.ppo_cse.actor_critic import ActorCritic

    # convert iteration weights to jit
    convert_weights_to_jit(env,logdir,iteration,Cfg)

    policy = load_policy(logdir,iteration)

    return env, policy


class RobotController:
    def __init__(self):
        # Initialize the robot's control commands
        self.x_vel_cmd, self.y_vel_cmd, self.yaw_vel_cmd = 0.4, 0.0, 0.0
        self.body_height_cmd = 0.0
        self.step_frequency_cmd = 2.0
        # self.gait = torch.tensor(gaits["trotting"])
        self.footswing_height_cmd = 0.25
        self.pitch_cmd = 0.0
        self.roll_cmd = 0.0
        self.stance_width_cmd = 0.25

        self.push_robot = False
        self.x_vel_push = 0
        self.y_vel_push = 0

        self.keys = {
            'w': False,
            's': False,
            'a': False,
            'd': False,
            'q': False,
            'e': False,
            'up':False,
            'down':False,
            'right':False,
            'left':False,
            'Shift':False,
            'Space':False,
        }


        self.max_x_vel = 0.1
        self.max_y_vel = 0.2
        self.max_yaw_vel = 1.0

        self.max_x_vel_shift = 1.0
        self.max_y_vel_shift = 1.0
        self.max_yaw_vel_shift = 2.0

        self.max_push = 1.0

        try:
            from pynput import keyboard
            self.keyboard = keyboard
        except:
            self.keyboard = None


    def on_press(self,key):
        try:
            self.keys[key.char.lower()] = True
        except:
            if key==self.keyboard.Key.shift:
                self.keys['Shift'] = True
            if key==self.keyboard.Key.space:
                self.keys['Space'] = True
            if key==self.keyboard.Key.up:
                self.keys['up'] = True
            if key==self.keyboard.Key.down:
                self.keys['down'] = True
            if key==self.keyboard.Key.right:
                self.keys['right'] = True
            if key==self.keyboard.Key.left:
                self.keys['left'] = True
        # print(key)

    def on_release(self,key):
        try:
            self.keys[key.char.lower()] = False
        except:
            if key==self.keyboard.Key.shift:
                self.keys['Shift'] = False
            if key==self.keyboard.Key.space:
                self.keys['Space'] = False
            if key==self.keyboard.Key.up:
                self.keys['up'] = False
            if key==self.keyboard.Key.down:
                self.keys['down'] = False
            if key==self.keyboard.Key.right:
                self.keys['right'] = False
            if key==self.keyboard.Key.left:
                self.keys['left'] = False

    def update_control_commands(self):

        if self.keys['Shift']:
            max_x_vel = self.max_x_vel_shift
            max_y_vel = self.max_y_vel_shift
            max_yaw_vel = self.max_yaw_vel_shift
        else:
            max_x_vel = self.max_x_vel
            max_y_vel = self.max_y_vel
            max_yaw_vel = self.max_yaw_vel

        self.x_vel_cmd = 0.0
        self.y_vel_cmd = 0.0
        self.yaw_vel_cmd = 0.0

        if self.keys['w']:
            self.x_vel_cmd = max_x_vel
        elif self.keys['s']:
            self.x_vel_cmd = -max_x_vel

        if self.keys['a']:
            self.y_vel_cmd = max_y_vel
        elif self.keys['d']:
            self.y_vel_cmd = -max_y_vel

        if self.keys['q']:
            self.yaw_vel_cmd = max_yaw_vel
        elif self.keys['e']:
            self.yaw_vel_cmd = -max_yaw_vel


        # push robot
        self.x_vel_push = 0
        self.y_vel_push = 0
        if self.keys['up']:
            self.x_vel_push = self.max_push
        elif self.keys['down']:
            self.x_vel_push = -self.max_push
        if self.keys['left']:
            self.y_vel_push = self.max_push
        elif self.keys['right']:
            self.y_vel_push = -self.max_push


        self.push_robot = self.x_vel_push!=0 or self.y_vel_push!=0


    def get_velocity_commands(self):
        return self.x_vel_cmd, self.y_vel_cmd, self.yaw_vel_cmd



def play_go1(headless=True):
    from ml_logger import logger

    from pathlib import Path
    from robodog_gym import MINI_GYM_ROOT_DIR
    import glob
    import os



    # policy run
    label = "rsl_exteroceptive_simple/2025-10-28_04-48-55.533078"
    iteration =  10000

    # Create an instance of the RobotController
    robot_controller = RobotController()

    try:
        from pynput import keyboard
        # Set up the keyboard listener
        listener = keyboard.Listener(on_press=robot_controller.on_press,on_release=robot_controller.on_release)
        listener.start()
    except:
        print("Keyboard input not available")



    env, policy = load_env(label,iteration, headless=headless)

    num_eval_steps = 1000000

    # estimator target and output
    estimator_target = np.zeros((num_eval_steps,env.num_privileged_obs))
    estimator_pred = np.zeros((num_eval_steps,env.num_privileged_obs))

    print("Estimated parameters:",env.num_privileged_obs)
    obs = env.reset()

    # polcy updates: self.cfg.control.decimation * self.sim_params.dt (50 Hz)
    # sim updates: 4x50=200Hz
    # dt_policy = env.cfg.control.decimation * env.cfg.sim.dt #50hz
    dt = 1*env.dt # = 50Hz

    for i in tqdm(range(num_eval_steps)):
        start_time = time.time()  # Record the start time of the time step

        robot_controller.update_control_commands()

        with torch.no_grad():
            actions, latent = policy(obs)
        env.commands[:, 0] = robot_controller.x_vel_cmd
        env.commands[:, 1] = robot_controller.y_vel_cmd
        env.commands[:, 2] = robot_controller.yaw_vel_cmd

        obs, rew, done, info = env.step(actions)



        estimator_pred[i] = latent[0,:]
        estimator_target[i] = env.privileged_obs_buf[0,:].to('cpu')

        # Push robot
        if robot_controller.push_robot:
            #env.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (env.root_states.shape[0], 2),
            #                                                  device=env.device)  # lin vel x/y
            env.root_states[:, 7] = torch.ones_like(env.root_states[:,0])*robot_controller.x_vel_push
            env.root_states[:, 8] = torch.ones_like(env.root_states[:,0])*robot_controller.y_vel_push

            env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))

        elapsed_time = time.time() - start_time  # Calculate the time elapsed during the time step
        if elapsed_time < dt:
            time.sleep(dt - elapsed_time)  # Sleep to maintain the desired time step


if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    play_go1(headless=False)
