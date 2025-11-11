# Reproduction of RSL legged_gym flat training + extensive dyn rand + wtw estimator
# - Normal estimator: friction, restitution and line vel
# - Disable self collisions
# - Lin and ang vel not oberved
# - Full WTW dynamic randomization (friction, mass, pushes, gravity, restitution, motor strength, offset and lag)


def train_go1(headless=True):

    import isaacgym
    assert isaacgym
    import torch

    from robodog_gym.envs.base.legged_robot_config import Cfg
    from robodog_gym.envs.robodog.go1_config import config_go1
    from robodog_gym.envs.robodog.a1_config import config_a1
    from robodog_gym.envs.robodog.a1_backpack_config import config_a1_backpack
    from robodog_gym.envs.robodog.go1_backpack_config import config_go1_backpack

    from robodog_gym.envs.robodog.velocity_tracking import VelocityTrackingEasyEnv

    from ml_logger import logger

    from robodog_gym_learn.ppo_cse import Runner
    from robodog_gym.envs.wrappers.history_wrapper import HistoryWrapper


    config_go1_backpack(Cfg)

    # Cfg.cfg_ppo.runner.wandb_logging = False
    # Cfg.env.num_envs = 4
    # Cfg.terrain.num_cols = 3
    # Cfg.terrain.num_rows = 3
    # Cfg.terrain.center_span = 1
    debug_viz = False

    # curriculum configuration
    # Cfg.commands.num_lin_vel_bins = 1 no used
    # Cfg.commands.num_ang_vel_bins = 1
    # Cfg.curriculum_thresholds.tracking_ang_vel = 0.95
    # Cfg.curriculum_thresholds.tracking_lin_vel = 0.95
    # Cfg.curriculum_thresholds.tracking_contacts_shaped_vel = 0.90
    # Cfg.curriculum_thresholds.tracking_contacts_shaped_force = 0.90

    # asset setup
    Cfg.asset.self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter. This has also effect for terminal collisions!

    #-----------------------
    # control
    #-----------------------

    Cfg.control.control_type = "P" # actuator control type
    Cfg.control.stiffness = {'joint': 25.}  # [N*m/rad]
    Cfg.control.damping = {'joint': 0.5}  # [N*m*s/rad]
    Cfg.control.action_scale = 0.25 #0.25
    Cfg.control.hip_scale_reduction = 1.0 # consistent with rsl

    #-----------------------
    # domain randomization
    #-----------------------

    Cfg.domain_rand.rand_interval_s = 4

    # Nominal domain rand
    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True # wtw true

    Cfg.domain_rand.randomize_friction = True
    Cfg.domain_rand.friction_range = [0.0, 4.0]
    Cfg.domain_rand.randomize_restitution = True  # wtw true
    Cfg.domain_rand.restitution_range = [0.0, 0.4]
    Cfg.domain_rand.randomize_base_mass = True
    Cfg.domain_rand.added_mass_range = [-2.0, 2.0]
    Cfg.domain_rand.randomize_com_displacement = False
    Cfg.domain_rand.com_displacement_range = [-0.15, 0.15]
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.ground_friction_range = [0.0, 0.0]
    Cfg.domain_rand.randomize_motor_strength = True  # wtw true
    Cfg.domain_rand.motor_strength_range = [0.9, 1.1]
    Cfg.domain_rand.randomize_motor_offset = True # wtw true
    Cfg.domain_rand.motor_offset_range = [-0.02, 0.02]
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_rigids_after_start = True
    Cfg.domain_rand.randomize_friction_indep = False


    # gravity changes and pushes
    Cfg.domain_rand.randomize_gravity = True # wtw true
    Cfg.domain_rand.gravity_range = [-1.0, 1.0]
    Cfg.domain_rand.gravity_rand_interval_s = 8.0
    Cfg.domain_rand.gravity_impulse_duration = 0.99

    # pushes as in rsl
    Cfg.domain_rand.push_robots = True
    Cfg.domain_rand.max_push_vel_xy = 1.0
    Cfg.domain_rand.push_interval_s = 15.0



    #--------------------------
    # Priviledged observations
    #--------------------------

    # Normalization used for estimator
    Cfg.normalization.friction_range = [-0.5,6.0] ##from[0.1, 1.3]
    Cfg.normalization.restitution_range = [0, 0.4]
    Cfg.normalization.x_velocity_range = [-2, 2]
    Cfg.normalization.y_velocity_range = [-1, 1]
    Cfg.normalization.z_velocity_range = [-1, 1]
    Cfg.normalization.foot_height_range = [0.02, 0.08]
    Cfg.normalization.height_measurements_z_bias_range = [-0.05, 0.05]

    num_dof = Cfg.env.num_dof
    Cfg.env.privileged_observation_components = [
        ['friction',        1,      False],
        ['ground_friction', None,   False],
        ['restitution',     1,      False],
        ['base_mass',       1,      False],
        ['com_displacement ', 3,    False],
        ['motor_strength',  num_dof, False],
        ['motor_offset',    num_dof, False],
        ['body_height',     1,      False],
        ['body_velocity',   3,      False],
        ['body_angular_velocity', 3, False],
        ['gravity',         3,      False],
        ['clock_inputs',    4,      False],
        ['desired_contact_states', 4, False],
        ['contact_states',  4,      False],
        ['feet_height',     4,      False],
        ['height_measurements_bias', 3, False],
        ['height_measurements_z_bias', 1, False],
        ['zero',            1,      True]   # additional privileged observation that will always be zero, to use when disabling the estimator
    ]


    #--------------------------
    # Observations and commands
    #--------------------------

    num_dof = Cfg.env.num_dof
    Cfg.env.policy_observation_components = [
        ['commands',             None, True],  # size = cfg.commands.num_commands
        ['global_linear_vel',    3,     False], # linear velocity in global frame of reference
        ['linear_vel',           3,     False],
        ['angular_vel',          3,     False],
        ['projected_gravity',    3,     True],
        ['dof_positions',        num_dof, True], # joint positions
        ['dof_velocities',       num_dof, True],
        ['dof_position_history', num_dof * Cfg.env.dof_history_length, False], # Joint pos history n...n-m, size = 12 * dof_history_length. Assumes joint_positions are false
        ['dof_velocity_history', num_dof * Cfg.env.dof_history_length, False],
        ['last_actions',         num_dof, True], # n-1 actions
        ['action_history',       num_dof * Cfg.env.action_history_length, False], # n-1...n-m actions, size = 12 * action_history_length
        ['timing_parameter',     1,     False],
        ['clock_inputs',         4,     False],
        ['yaw',                  1,     False],
        ['contact_states',       4,     False],
        ['height_measurements',  None,  True], # size = len(cfg.terrain.measured_points_x)*len(cfg.terrain.measured_points_y)
    ]

    Cfg.env.estimator_observation_components = [
        ['commands',             None, False],  # size = cfg.commands.num_commands
        ['global_linear_vel',    3,     False], # linear velocity in global frame of reference
        ['linear_vel',           3,     False],
        ['angular_vel',          3,     False],
        ['projected_gravity',    3,     True],
        ['dof_positions',        num_dof, True], # joint positions
        ['dof_velocities',       num_dof, True],
        ['dof_position_history', num_dof * Cfg.env.dof_history_length, False], # Joint pos history n...n-m, size = 12 * dof_history_length. Assumes joint_positions are false
        ['dof_velocity_history', num_dof * Cfg.env.dof_history_length, False],
        ['last_actions',         num_dof, True], # n-1 actions
        ['action_history',       num_dof * Cfg.env.action_history_length, False], # n-1...n-m actions, size = 12 * action_history_length
        ['timing_parameter',     1,     False],
        ['clock_inputs',         4,     False],
        ['yaw',                  1,     False],
        ['contact_states',       4,     False],
        ['height_measurements',  None,  True], # size = len(cfg.terrain.measured_points_x)*len(cfg.terrain.measured_points_y)
    ]

    Cfg.terrain.measure_heights = True
    Cfg.terrain.measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
    Cfg.terrain.measured_points_y = [-0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3]

    Cfg.noise_scales.height_measurements = 0 #0.25
    Cfg.terrain.height_measurements_per_step_xy_noise_std = 0.05
    Cfg.terrain.height_measurements_per_step_z_noise_std = 0.025

    Cfg.terrain.height_measurements_per_env_xy_noise_std = 0.05
    Cfg.terrain.height_measurements_per_env_z_noise_std = 0.05
    Cfg.terrain.height_measurements_per_env_noise_prob = 0.20
    Cfg.terrain.height_measurements_per_env_resampling_s = 7



    # Cfg.env.estimator_observation_components = None # same as policy observations

    Cfg.env.num_observation_history = 10 #10 training way slower, 5 still trains ok #30 reduced because of height measurements
    # Cfg.env.sparse_obs_history = [0, 1, 3, 6, 10]
    # Cfg.env.sparse_obs_history = [0, 1, 2, 3, 4]

    # Cfg.env.num_estimator_obs_history = 10
    # Cfg.env.sparse_estimator_obs_history = [0,1,2]


    if Cfg.env.sparse_obs_history is not None:
      Cfg.cfg_ppo.runner.wandb_note = 'Extero, sparse hist ' + ','.join(str(e) for e in Cfg.env.sparse_obs_history)
    else:
      Cfg.cfg_ppo.runner.wandb_note = 'Extero, full hist ' + str(Cfg.env.num_observation_history)

    # commands
    Cfg.commands.num_commands = 3 # change!

    Cfg.normalization.clip_actions = 20.0 # was 10 way lower than RSL...
    Cfg.normalization.clip_observations = 100.0



    #--------------------------
    # Terrain configuration
    #--------------------------


    # terrain configuration
    # Cfg.domain_rand.tile_height_range = [-0.0, 0.0]
    # Cfg.domain_rand.tile_height_curriculum = False
    # Cfg.domain_rand.tile_height_update_interval = 1000000
    # Cfg.domain_rand.tile_height_curriculum_step = 0.01

    # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete, stepping stones, none, smooth flat, rough flat]
    Cfg.terrain.terrain_proportions = [0, 0, 0, 0, 1.0, 0, 0, 0.0, 0.0]
    Cfg.terrain.curriculum = False
    Cfg.terrain.max_platform_height = 0.15
    Cfg.terrain.slope_treshold = 0.25 ##added (maybe needs to be reduced)
    Cfg.terrain.terrain_noise_magnitude = 0.05
    Cfg.terrain.border_size = 10.0
    Cfg.terrain.mesh_type = "trimesh"
    Cfg.terrain.num_cols = 10
    Cfg.terrain.num_rows = 20
    Cfg.terrain.terrain_width = 5.0
    Cfg.terrain.terrain_length = 5.0
    Cfg.terrain.x_init_range = 0.2
    Cfg.terrain.y_init_range = 0.2
    Cfg.terrain.teleport_thresh = 0.3
    Cfg.terrain.teleport_robots = False
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 5
    Cfg.terrain.horizontal_scale = 0.1 #0.1 resolution of gridmap


    # original RSL plane
    # Cfg.terrain.mesh_type = 'plane'
    # Cfg.terrain.teleport_robots = False # else gives error for plane
    # Cfg.terrain.measure_heights = False

    # this will be averaged with domain_rand.friction_range if randomization is on
    Cfg.terrain.static_friction = 1.0
    Cfg.terrain.dynamic_friction = 1.0

    Cfg.terrain.yaw_init_range = 3.14


    # -----------------
    # Env termination
    # -----------------

    # terminate on these conditions
    Cfg.rewards.use_terminal_foot_height = False
    Cfg.rewards.use_terminal_body_height = False
    Cfg.rewards.use_terminal_body_impact = True
    Cfg.rewards.terminal_body_height = 0.10
    Cfg.rewards.use_terminal_roll_pitch = True
    Cfg.rewards.terminal_body_ori =  1.39626  # 80 degrees #1.22173 # 70 degrees
    Cfg.reward_scales.termination = -0 # not


    # ---------------------
    # Rewards
    # ---------------------

    Cfg.rewards.reward_container_name = "CoRLRewards"
    Cfg.rewards.only_positive_rewards = False
    Cfg.rewards.only_positive_rewards_ji22_style = False
    # Cfg.rewards.sigma_rew_neg = 0.2
    Cfg.rewards.total_reward_function = "leaky_relu_0.25"

    Cfg.rewards.reward_curriculum_factor_init = 0.001
    Cfg.rewards.reward_curriculum_factor_rate = 0.99


    #positive rewards
    Cfg.reward_scales.tracking_lin_vel = 1.0
    Cfg.rewards.tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
    Cfg.rewards.end_sigma_curriculum_iter = 0 # disable curriculum
    Cfg.reward_scales.tracking_ang_vel = 0.5 ## was 3
    Cfg.rewards.tracking_sigma_yaw = 0.25
    Cfg.rewards.end_sigma_yaw_curriculum_iter = 0


    # air time

    Cfg.reward_scales.feet_air_time_rsl = 3.0 # was 2 # 0.75 #0.75 #2.0
    Cfg.rewards.feet_air_time_rsl_period = 0.25 #0.5 # original 0.5
    Cfg.rewards.feet_air_time_rsl_curriculum = True

    Cfg.reward_scales.feet_air_time = 0.0 # 0.5
    Cfg.rewards.use_adaptive_period = False
    Cfg.rewards.contact_condition_scale = -1


    #negative rewards
    Cfg.reward_scales.base_height = -5 #-10
    Cfg.rewards.base_height_target = 0.30
    Cfg.reward_scales.orientation = -2.0 #-4.0

    # go1 urdf weight no backpack: 11.308932. Backpack weight: 3.211. Increase by 28%
    Cfg.reward_scales.torques = -0.00020 # best for A1: -0.00025
    Cfg.rewards.torque_hip_weight = 1.0
    Cfg.rewards.torque_pos_thigh_weight = 1.0
    Cfg.rewards.torque_pos_calf_weight = 1.0

    Cfg.reward_scales.dof_pos = -0.1
    Cfg.rewards.dof_pos_hip_weight = 3.0
    Cfg.rewards.dof_pos_thigh_weight = 1.0
    Cfg.rewards.dof_pos_calf_weight = 0.0

    Cfg.reward_scales.dof_vel = 0
    Cfg.reward_scales.dof_acc = -2.5e-7
    Cfg.reward_scales.action_rate = -0.01
    Cfg.reward_scales.feet_slip = -0.05
    Cfg.reward_scales.collision = -1.
    Cfg.reward_scales.lin_vel_z = -2.0
    Cfg.reward_scales.ang_vel_xy = -0.05

    Cfg.reward_scales.dof_pos_limits = -10.0
    Cfg.rewards.soft_dof_pos_limit = 0.9

    # add torque limits, and torque reduction!
    # Cfg.asset.torque_limits_factor = 0.8
    Cfg.rewards.soft_torque_limit = 0.7
    Cfg.reward_scales.torque_limits = -10.0



    # disabled
    # Cfg.reward_scales.feet_air_time = 0.0
    Cfg.reward_scales.feet_contact_forces = 0.0
    Cfg.reward_scales.tracking_contacts_shaped_vel = 0
    Cfg.reward_scales.raibert_heuristic = -0.0
    Cfg.reward_scales.feet_clearance_cmd_linear = -0.0
    Cfg.reward_scales.action_smoothness_1 = 0
    Cfg.reward_scales.action_smoothness_2 = 0
    Cfg.reward_scales.feet_impact_vel = -0.0

    # Cfg.reward_scales.dof_pos_stancemode = -0.075#from -0.075
    # Cfg.rewards.use_adaptive_stancemode = True
    # Cfg.rewards.stancemode_multiplier = 15
    # Cfg.rewards.hip_weight = 30
    # Cfg.rewards.thigh_weight = 1
    # Cfg.rewards.calf_weight = 10
    # unused
    Cfg.rewards.kappa_gait_probs = 0.07
    Cfg.rewards.gait_force_sigma = 100.
    Cfg.rewards.gait_vel_sigma = 10.


    #-------------
    # Learing config
    #-------------

    Cfg.cfg_ppo.algorithm.schedule = 'adaptive' # 'adaptive' # if not adaptive let Adam handle it. Adaptive is KL thing from RSL paper
    Cfg.cfg_ppo.algorithm.learning_rate = 1.e-3  # Initial learning rate
    Cfg.cfg_ppo.algorithm.desired_kl = 0.01 # default 0.01 # Used by adaptive learning rate
    Cfg.cfg_ppo.algorithm.lr_adaptive_schedule_decay = 1.25 # 1.1 1.001 # defaul 1.5 decay factor for adaptive KL-based learning rate schedule
    # try to increase entropy!

    #-------------
    # Commands
    #-------------


    Cfg.commands.resampling_time = 10 # twice per episode


    # heading command
    Cfg.commands.heading_command = False
    Cfg.commands.heading = [-3.14, 3.14]

    # Cfg.commands.lin_vel_x = [-1.0, 1.0]
    # Cfg.commands.limit_vel_x = [-1.0, 1.0]
    # Cfg.commands.lin_vel_y = [-1.0, 1.0]
    # Cfg.commands.limit_vel_y = [-1.0, 1.0]
    # Cfg.commands.ang_vel_yaw = [-1.5, 1.5]
    # Cfg.commands.limit_vel_yaw = [-1.5, 1.5]


    Cfg.commands.lin_vel_x = [-0.5, 0.5]
    Cfg.commands.limit_vel_x = [-1.0, 1.5]
    Cfg.commands.lin_vel_y = [-0.5, 0.5]
    Cfg.commands.limit_vel_y = [-1.0, 1.0]
    Cfg.commands.ang_vel_yaw = [-1.0, 1.0]
    Cfg.commands.limit_vel_yaw = [-1.5, 1.5]

    # Cfg.commands.lin_vel_x = [-0.5, 0.5]
    # Cfg.commands.limit_vel_x = [-1.5, 1.5]
    # Cfg.commands.lin_vel_y = [-0.5, 0.5]
    # Cfg.commands.limit_vel_y = [-1.0, 1.0]
    # Cfg.commands.ang_vel_yaw = [-1.0, 1.0]
    # Cfg.commands.limit_vel_yaw = [-1.5, 1.5]

    Cfg.commands.body_height_cmd = [0.0, 0.0]
    Cfg.commands.gait_frequency_cmd_range = [0.0, 0.0]
    Cfg.commands.gait_phase_cmd_range = [0.0, 0.0]
    Cfg.commands.gait_offset_cmd_range = [0.0, 0.0]
    Cfg.commands.gait_bound_cmd_range = [0.0, 0.0]
    Cfg.commands.gait_duration_cmd_range = [0.0, 0.0]
    Cfg.commands.footswing_height_range = [0.0, 0.0]
    Cfg.commands.body_pitch_range = [0.0, 0.0]
    Cfg.commands.body_roll_range = [0.0, 0.0]
    Cfg.commands.stance_width_range = [0.0, 0.0]
    Cfg.commands.stance_length_range = [0.0, 0.0]
    Cfg.commands.aux_reward_coef_range = [0.0, 0.0]


    Cfg.commands.limit_body_height = [0.0, 0.0]
    Cfg.commands.limit_gait_frequency = [0.0, 0.0]
    Cfg.commands.limit_gait_phase = [0.0, 0.0]
    Cfg.commands.limit_gait_offset = [0.0, 0.0]
    Cfg.commands.limit_gait_bound = [0.0, 0.0]
    Cfg.commands.limit_gait_duration = [0.0, 0.0]
    Cfg.commands.limit_footswing_height = [0.0, 0.0]
    Cfg.commands.limit_body_pitch = [0.0, 0.0]
    Cfg.commands.limit_body_roll = [0.0, 0.0]
    Cfg.commands.limit_stance_width = [0.0, 0.0]
    Cfg.commands.limit_stance_length = [0.0, 0.0]
    Cfg.commands.limit_aux_reward_coef = [0.0, 0.0]

    Cfg.commands.num_bins_vel_x = 10 ##from 21
    Cfg.commands.num_bins_vel_y = 10 ## from 1
    Cfg.commands.num_bins_vel_yaw = 10 ## from 21
    Cfg.commands.num_bins_body_height = 1
    Cfg.commands.num_bins_gait_frequency = 1
    Cfg.commands.num_bins_gait_phase = 1
    Cfg.commands.num_bins_gait_offset = 1
    Cfg.commands.num_bins_gait_bound = 1
    Cfg.commands.num_bins_footswing_height = 1
    Cfg.commands.num_bins_body_roll = 1
    Cfg.commands.num_bins_body_pitch = 1
    Cfg.commands.num_bins_stance_width = 1

    #Cfg.commands.gaitwise_curricula = False

    # cmd vx,vy,vtheta exactly 0 with standing_still_prob
    Cfg.commands.train_standing_still = True
    Cfg.commands.standing_still_prob = 0.05

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=headless, cfg=Cfg, debug_viz=debug_viz)

    # log the experiment parameters
    logger.log_params(Cfg=vars(Cfg))

    env = HistoryWrapper(env)
    gpu_id = 0
    runner = Runner(env, cfg = Cfg, device=f"cuda:{gpu_id}")
    runner.learn(num_learning_iterations=100000, init_at_random_ep_len=True, eval_freq=50)


if __name__ == '__main__':
    import os
    from pathlib import Path
    from ml_logger import logger
    from robodog_gym import MINI_GYM_ROOT_DIR

    logger.configure(logger.utcnow(f'rsl_exteroceptive_simple/%Y-%m-%d_%H-%M-%S.%f'),
                     root=Path(f"{MINI_GYM_ROOT_DIR}/runs").resolve(), )
    print("Loggind directory:", os.path.join(logger.root,logger.prefix))
    logger.log_text("""
                charts:
                - yKey: train/episode/rew_total/mean
                  xKey: iterations
                - yKey: train/episode/rew_tracking_lin_vel/mean
                  xKey: iterations
                - yKey: train/episode/rew_tracking_ang_vel/mean
                  xKey: iterations
                - yKey: train/episode/rew_feet_air_time_rsl/mean
                  xKey: iterations
                - yKey: train/episode/rew_feet_air_time/mean
                  xKey: iterations
                - yKey: train/episode/rew_orientation/mean
                  xKey: iterations
                - yKey: train/episode/rew_base_height/mean
                  xKey: iterations
                - yKey: train/episode/rew_torques/mean
                  xKey: iterations
                - yKey: train/episode/rew_collision/mean
                  xKey: iterations


                - yKey: train/episode/rew_dof_pos/mean
                  xKey: iterations
                - yKey: train/episode/rew_dof_vel/mean
                  xKey: iterations
                - yKey: train/episode/rew_dof_acc/mean
                  xKey: iterations
                - yKey: train/episode/rew_action_rate/mean
                  xKey: iterations
                - yKey: train/episode/rew_ang_vel_xy/mean
                  xKey: iterations
                - yKey: train/episode/rew_lin_vel_z/mean
                  xKey: iterations
                - yKey: train/episode/rew_feet_slip/mean
                  xKey: iterations


                - yKey: train/episode/command_area_nominal/mean
                  xKey: iterations
                - yKey: train/episode/max_terrain_height/mean
                  xKey: iterations
                - type: video
                  glob: "videos/*.mp4"

                - yKey: mean_surrogate_loss/mean
                  xKey: iterations
                - yKey: mean_value_loss/mean
                  xKey: iterations
                - yKey: adaptation_loss/mean
                  xKey: iterations
                - yKey: mean_adaptation_module_test_loss/mean
                  xKey: iterations
                - yKey: learning_rate/mean
                  xKey: iterations
                """, filename=".charts.yml", dedent=True)

    # to see the environment rendering, set headless=False
    train_go1(headless=True)
