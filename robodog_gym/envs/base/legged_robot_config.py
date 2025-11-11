# License: see [LICENSE, LICENSES/legged_gym/LICENSE]

from params_proto import PrefixProto, ParamsProto


class Cfg(PrefixProto, cli=False):
    class env(PrefixProto, cli=False):
        num_envs = 4096

        num_dof = 12
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 20  # episode length in seconds

        # observation history. Either use dense history or sparse history
        num_observation_history = 1 # Dense history: number of previous observations to stack (full frame stacking)
        sparse_obs_history = None

        num_estimator_obs_history = None # if not none, overrides num_observation_history for the estimator
        sparse_estimator_obs_history = None # if not none, overrides sparse_obs_history for the estimator

        # sparse_obs_history = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # timestep indices of observations to keep in history [n_1,n_2,n_3,...]. Current timestep is t, so t-n_1, t-n_2, t-n_3, ... are kept
        # if list not None, will be used instead of dense history, num_observation_history will be ignored
        # sparse_obs_history = [0, 1, 3, 6, 10]
        # sparse_obs_history = [0, 1, 2, 4, 8]
        # sparse_obs_history = [0, 2, 4, 8, 10]
        # sparse_obs_history = [0, 2, 4, 8, 10, 12, 14, 16, 18, 20]

        dof_history_length = 2  # number of previous dof states to stack >=1
        dof_history_step_skip = 0  # skipped steps between dof history states. So dof_history_steps = 2 and dof_history_step_skip = 1 will stack (n default), n-2 and n-4
        action_history_length = 2  # number of previous actions to stack >=1
        action_history_step_skip = 0  # skipped steps between action history states

        # main policy observation components in order, each component is a tuple (name, size, enabled). If size is None, the size will be determined by the environment later
        policy_observation_components = [
            ['commands',             None, True],  # size = cfg.commands.num_commands
            ['global_linear_vel',    3,     False], # linear velocity in global frame of reference
            ['linear_vel',           3,     False],
            ['angular_vel',          3,     False],
            ['projected_gravity',    3,     True],
            # ['commands', None, True],  # size = cfg.commands.num_commands
            ['dof_positions',        num_dof, True], # joint positions
            ['dof_velocities',       num_dof, True],
            ['dof_position_history', num_dof * dof_history_length, False], # Joint pos history n...n-m, size = 12 * dof_history_length. Assumes joint_positions are false
            ['dof_velocity_history', num_dof * dof_history_length, False],
            ['last_actions',         num_dof, True], # n-1 actions
            ['action_history',       num_dof * action_history_length, False], # n-1...n-m actions, size = 12 * action_history_length
            ['timing_parameter',     1,     False],
            ['clock_inputs',         4,     False],
            ['yaw',                  1,     False],
            ['contact_states',       4,     False],
            ['height_measurements',  None,  False], # size = len(cfg.terrain.measured_points_x)*len(cfg.terrain.measured_points_y)
        ]

        # observation vector for estimator. If None, the main observation vector is used
        estimator_observation_components = None

        privileged_observation_components = [
            ['friction',        1,      True],
            ['ground_friction', None,   False],
            ['restitution',     1,      False],
            ['base_mass',       1,      True],
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
            ['height_measurements_bias', 3, False], # xyz bias for height measurements (xy are in world frame!)
            ['height_measurements_z_bias', 1, False], # z only bias for height measurements (same in world and local frame)
        ]


        compute_true_feet_height = False # compute the feet height (self.feet_height) relative to terrain, else assume terrain is at y=0 (disable on flat terrain for improved performance)


        record_video = True
        recording_width_px = 368
        recording_height_px = 240
        recording_mode = "COLOR"
        num_recording_envs = 1
        debug_viz = False
        all_agents_share = False


    class terrain(PrefixProto, cli=False):
        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25 # [m]
        curriculum = True
        legacy_curriculum = True # if true use RSL terrain curriculum, else use curriculum based on cmd vel residual
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        terrain_noise_magnitude = 0.1
        # rough terrain only:
        terrain_smoothness = 0.005

        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        min_init_terrain_level = 0
        max_init_terrain_level = 5  # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces

        # wtw additions
        difficulty_scale = 1.
        x_init_range = 1. # randomized initial position around env tile center
        y_init_range = 1.
        yaw_init_range = 0.
        x_init_offset = 0.
        y_init_offset = 0.
        teleport_robots = True
        teleport_thresh = 2.0
        max_platform_height = 0.2
        max_step_height = 0.23
        center_robots = False # center robots in the middle of the terrain grid, only makes sense without curriculum (wtw addition)
        center_span = 5

        # height_measurements params. Should be put in a separate class
        measure_heights = True
        # 1mx1.6m rectangle (without center line)
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        # noise
        # per timestep noise: sampled at every policy timestep, for every point
        # per episode noise: sampled once for each episode and environment, same for every point
        # all are normal distributions with zero mean
        height_measurements_per_step_xy_noise_std = 0 #0.025 #0.05
        height_measurements_per_step_z_noise_std = 0# 0.05 #0.075

        height_measurements_per_env_xy_noise_std = 0.0
        height_measurements_per_env_z_noise_std = 0.0

        height_measurements_per_env_noise_prob = 0.1 # probability that per env noise is defined by above paramaters, else it is zero
        height_measurements_per_env_resampling_s = 3 # resampling time for per env nois


    class commands(PrefixProto, cli=False):
        command_curriculum = False
        max_reverse_curriculum = 1.
        max_forward_curriculum = 1.
        yaw_command_curriculum = False
        max_yaw_curriculum = 1.
        exclusive_command_sampling = False
        num_commands = 3   # 3 for only x y yaw control, 4 if also heading_command=True, more only used for WTW
        resampling_time = 10.  # time before command are changed[s]
        subsample_gait = False
        gait_interval_s = 10.  # time between resampling gait params
        vel_interval_s = 10.
        jump_interval_s = 20.  # time between jumps
        jump_duration_s = 0.1  # duration of jump
        jump_height = 0.3
        heading_command = True  # if true: compute ang vel command from heading error
        global_reference = False
        observe_accel = False
        distributional_commands = False
        curriculum_type = "RewardThresholdCurriculum" # tere are no other options (it should be GridAdaptive curriculum)
        lipschitz_threshold = 0.9

        num_lin_vel_bins = 20
        lin_vel_step = 0.3
        num_ang_vel_bins = 20
        ang_vel_step = 0.3
        distribution_update_extension_distance = 1
        curriculum_seed = 100

        lin_vel_x = [-1.0, 1.0]  # min max [m/s] # initial command sampling range
        lin_vel_y = [-1.0, 1.0]  # min max [m/s]
        ang_vel_yaw = [-1, 1]  # min max [rad/s]
        body_height_cmd = [-0.05, 0.05]
        impulse_height_commands = False

        limit_vel_x = [-10.0, 10.0] # maximum command sampling range
        limit_vel_y = [-0.6, 0.6]
        limit_vel_yaw = [-10.0, 10.0]
        limit_body_height = [-0.05, 0.05]
        limit_gait_phase = [0, 0.01]
        limit_gait_offset = [0, 0.01]
        limit_gait_bound = [0, 0.01]
        limit_gait_frequency = [2.0, 2.01]
        limit_gait_duration = [0.49, 0.5]
        limit_footswing_height = [0.06, 0.061]
        limit_body_pitch = [0.0, 0.01]
        limit_body_roll = [0.0, 0.01]
        limit_aux_reward_coef = [0.0, 0.01]
        limit_compliance = [0.0, 0.01]
        limit_stance_width = [0.0, 0.01]
        limit_stance_length = [0.0, 0.01]

        num_bins_vel_x = 25
        num_bins_vel_y = 3
        num_bins_vel_yaw = 25
        num_bins_body_height = 1
        num_bins_gait_frequency = 11
        num_bins_gait_phase = 11
        num_bins_gait_offset = 2
        num_bins_gait_bound = 2
        num_bins_gait_duration = 3
        num_bins_footswing_height = 1
        num_bins_body_pitch = 1
        num_bins_body_roll = 1
        num_bins_aux_reward_coef = 1
        num_bins_compliance = 1
        num_bins_compliance = 1
        num_bins_stance_width = 1
        num_bins_stance_length = 1

        heading_range = [-3.14, 3.14] # just used in heading_command

        gait_phase_cmd_range = [0.0, 0.01]
        gait_offset_cmd_range = [0.0, 0.01]
        gait_bound_cmd_range = [0.0, 0.01]
        gait_frequency_cmd_range = [2.0, 2.01]
        gait_duration_cmd_range = [0.49, 0.5]
        footswing_height_range = [0.06, 0.061]
        body_pitch_range = [0.0, 0.01]
        body_roll_range = [0.0, 0.01]
        aux_reward_coef_range = [0.0, 0.01]
        # compliance_range = [0.0, 0.01] not used...
        stance_width_range = [0.0, 0.01]
        stance_length_range = [0.0, 0.01]

        exclusive_phase_offset = True
        binary_phases = False
        pacing_offset = False # used in the foot timing variables computation
        balance_gait_distribution = True
        gaitwise_curricula = True
        train_standing_still = False
        standing_still_prob = 0.1
        standing_still_explicit_flag = False # if true, input 15 (aux_reward_coef) is set to 1 if command velocities are all 0

    class curriculum_thresholds(PrefixProto, cli=False):
        tracking_lin_vel = 0.8  # closer to 1 is tighter
        tracking_ang_vel = 0.5
        tracking_contacts_shaped_force = 0.8  # closer to 1 is tighter
        tracking_contacts_shaped_vel = 0.8

    class init_state(PrefixProto, cli=False):
        pos = [0.0, 0.0, 1.]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        # target angles when action = 0.0
        default_joint_angles = {"joint_a": 0., "joint_b": 0.}

    class control(PrefixProto, cli=False):
        control_type = 'actuator_net' #'P'  # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        hip_scale_reduction = 1.0
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(PrefixProto, cli=False):
        file = ""
        foot_name = "None"  # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        collapse_fixed_joints = True
        fix_base_link = False  # fixe the base of the robot
        default_dof_drive_mode = 3  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        # replace collision cylinders with capsules, leads to faster/more stable simulation
        replace_cylinder_with_capsule = True
        flip_visual_attachments = True  # Some .obj meshes must be flipped from y-up to z-up

        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand(PrefixProto, cli=False):
        rand_interval_s = 10 # has only effect on randomized parameters that change over time (no friction etc)
        randomize_rigids_after_start = True
        randomize_friction = True # per environment, fixed over time
        friction_range = [0.5, 1.25]
        randomize_restitution = False # per environment, fixed over time
        restitution_range = [0, 1.0]
        randomize_base_mass = False
        # add link masses, increase range, randomize inertia, randomize joint properties
        added_mass_range = [-1., 1.]
        randomize_com_displacement = False
        # add link masses, increase range, randomize inertia, randomize joint properties
        com_displacement_range = [-0.15, 0.15]
        randomize_motor_strength = False
        motor_strength_range = [0.9, 1.1]
        randomize_Kp_factor = False
        Kp_factor_range = [0.8, 1.3]
        randomize_Kd_factor = False
        Kd_factor_range = [0.5, 1.5]
        gravity_rand_interval_s = 7
        gravity_impulse_duration = 1.0
        randomize_gravity = False
        gravity_range = [-1.0, 1.0]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.
        randomize_lag_timesteps = True
        lag_timesteps = 6

    class rewards(PrefixProto, cli=False):
        reward_curriculum_factor_init = 0.001 # initial value for reward curriculum factor, that tends towards 1.0
        reward_curriculum_factor_rate = 0.98 # convergence rate c_k+1 = c_k**rate, k=iteration

        # only one of the following should be true
        # they affect the total reward for each timestep, before adding to the episode sum
        only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards_ji22_style = False
        total_reward_function = "" #"leaky_relu" "elu" etc # if both are false, the the reward for the timestep is modulated with a function. empty "" is bypass

        sigma_rew_neg = 5
        reward_container_name = "CoRLRewards"
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        tracking_sigma_lat = 0.25  # tracking reward = exp(-error^2/sigma)
        tracking_sigma_long = 0.25  # tracking reward = exp(-error^2/sigma)
        tracking_sigma_yaw = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1.  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1. # used for _reward_torque_limits
        base_height_target = 1.
        feet_air_time_rsl_period = 0.5
        feet_air_time_rsl_curriculum = False # if true, multiply reward by reward_curriculum_factor
        feet_air_time_period = 0.25 # air and stance time target for each feet, used for feet_air_time reward
        feet_air_time_max = 0.25 # air time reward for each leg is clamped to this value
        max_contact_force = 100.  # forces above this value are penalized
        use_terminal_body_impact = True
        use_terminal_body_height = False
        terminal_body_height = 0.20
        use_terminal_foot_height = False
        terminal_foot_height = -0.005
        use_terminal_roll_pitch = False
        terminal_body_ori = 0.5
        kappa_gait_probs = 0.07
        gait_force_sigma = 50.
        gait_vel_sigma = 0.5
        footswing_height = 0.09
        hip_weight = 3
        thigh_weight = 1
        calf_weight = 1
        contact_condition_scale = -1
        use_adaptive_period = 0
        use_adaptive_stancemode = 0
        start_tracking_sigma = 1
        end_sigma_curriculum_iter = 10000
        start_tracking_sigma_yaw = 0.66
        end_sigma_yaw_curriculum_iter = 10000
        stancemode_multiplier = 10

        dof_pos_hip_weight = 3.0
        dof_pos_thigh_weight = 1.0
        dof_pos_calf_weight = 1.0

        torque_hip_weight = 1.0
        torque_thigh_weight = 1.0
        torque_calf_weight = 1.0

    class reward_scales(ParamsProto, cli=False):
        termination = -0.0
        tracking_lin_vel = 1.0
        tracking_ang_vel = 0.5
        lin_vel_z = -2.0
        ang_vel_xy = -0.05
        orientation = -0.
        torques = -0.00001
        dof_vel = -0.
        dof_acc = -2.5e-7
        base_height = -0.
        feet_air_time = 1.0
        feet_air_time_rsl = 0.0
        collision = -1.
        feet_stumble = -0.0
        action_rate = -0.
        stand_still = -0.
        tracking_lin_vel_lat = 0.
        tracking_lin_vel_long = 0.
        tracking_contacts = 0.
        tracking_contacts_shaped = 0.
        tracking_contacts_shaped_force = 0.
        tracking_contacts_shaped_vel = 0.
        jump = 0.0
        energy = 0.0
        energy_expenditure = 0.0
        survival = 0.0
        dof_pos_limits = 0.0
        feet_contact_forces = 0.
        feet_slip = 0.
        feet_clearance_cmd_linear = 0.
        thigh_angle = 0.
        dof_pos = 0.
        action_smoothness_1 = 0.
        action_smoothness_2 = 0.
        base_motion = 0.
        feet_impact_vel = 0.0
        raibert_heuristic = 0.0
        dof_pos_stancemode = 0.0

        torque_limits = 0.0

    class normalization(PrefixProto, cli=False):
        clip_observations = 100.
        clip_actions = 100.

        friction_range = [0.05, 4.5]
        ground_friction_range = [0.05, 4.5]
        restitution_range = [0, 1.0]
        added_mass_range = [-1., 3.]
        com_displacement_range = [-0.1, 0.1]
        motor_strength_range = [0.9, 1.1]
        motor_offset_range = [-0.05, 0.05]
        Kp_factor_range = [0.8, 1.3]
        Kd_factor_range = [0.5, 1.5]
        joint_friction_range = [0.0, 0.7]
        contact_force_range = [0.0, 50.0]
        contact_state_range = [0.0, 1.0]
        body_velocity_range = [-6.0, 6.0]
        x_velocity_range = [-2.0, 2.0]
        y_velocity_range = [-1.0, 1.0]
        z_velocity_range = [-1.0, 1.0]
        pitch_velocity_range = [-2.0, 2.0]
        roll_velocity_range = [-2.0, 2.0]
        yaw_velocity_range = [-2.0, 2.0]
        foot_height_range = [0.0, 0.15]
        body_height_range = [0.0, 0.60]
        gravity_range = [-1.0, 1.0]
        motion = [-0.01, 0.01]
        height_measurements_bias_range = [-0.1, 0.1]
        height_measurements_z_bias_range = [-0.1, 0.1]

    class obs_scales(PrefixProto, cli=False):
        # if not specified, the scale is 1.0

        # lin_vel = 2.0
        # ang_vel = 0.25
        # dof_pos = 1.0
        # dof_vel = 0.05

        # global_linear_vel = 2.0
        linear_vel = 2.0
        angular_vel = 0.25
        projected_gravity = 1.0
        dof_positions = 1.0
        dof_velocities = 0.05
        # dof_position_history
        # dof_velocity_history
        height_measurements = 5.0

        body_height_cmd = 2.0
        gait_phase_cmd = 1.0
        gait_freq_cmd = 1.0
        footswing_height_cmd = 0.15
        body_pitch_cmd = 0.3
        body_roll_cmd = 0.3
        aux_reward_cmd = 1.0
        compliance_cmd = 1.0
        stance_width_cmd = 1.0
        stance_length_cmd = 1.0
    class obs_bias(PrefixProto, cli=False):
        # scalar bias added to the measurements
        height_measurements = 0.3

    class noise(PrefixProto, cli=False):
        add_noise = True
        noise_level = 1.0  # scales other values

    class noise_scales(PrefixProto, cli=False):
        # for each observation component, the noise level is multiplied by the corresponding value
        # if not specified for an observation component, the noise level is multiplied by 1.0
        # noise scales ending with "history" are handled separately, by only applying noise to the most recent observation

        # dof_pos = 0.01
        # dof_vel = 1.5
        # lin_vel = 0.1
        # ang_vel = 0.2
        # gravity = 0.05
        contact_states = 0.05
        height_measurements = 0.1

        commands = 0
        global_linear_vel = 0.1
        linear_vel = 0.1
        angular_vel = 0.2
        projected_gravity = 0.05
        dof_positions = 0.01
        dof_velocities = 1.5
        dof_position_history = 0.01
        dof_velocity_history = 1.5
        last_actions = 0
        action_history = 0
        timing_parameter = 0
        clock_inputs = 0
        yaw = 0


    # viewer camera:
    class viewer(PrefixProto, cli=False):
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim(PrefixProto, cli=False):
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        use_gpu_pipeline = True

        class physx(PrefixProto, cli=False):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)



    class cfg_ppo(PrefixProto, cli=False):
        seed = 1
        runner_class_name = 'OnPolicyRunner'

        class policy(PrefixProto, cli=False):
            init_noise_std = 1.0
            actor_hidden_dims = [512, 256, 128]
            critic_hidden_dims = [512, 256, 128]
            activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

            adaptation_module_branch_hidden_dims = [256, 128]

            use_decoder = False

            # only for 'ActorCriticRecurrent':
            # rnn_type = 'lstm'
            # rnn_hidden_size = 512
            # rnn_num_layers = 1

        class algorithm(PrefixProto, cli=False):
            # training params
            value_loss_coef = 1.0
            use_clipped_value_loss = True
            clip_param = 0.2
            entropy_coef = 0.01
            num_learning_epochs = 5
            num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
            learning_rate = 1.e-3  # 5.e-4
            adaptation_module_learning_rate = 1.e-3
            num_adaptation_module_substeps = 1
            schedule = 'adaptive'  # could be adaptive, fixed
            lr_adaptive_schedule_decay = 1.5 # decay factor for adaptive KL-based learning rate schedule
            gamma = 0.99
            lam = 0.95
            desired_kl = 0.01
            max_grad_norm = 1.

            selective_adaptation_module_loss = False

        class runner(PrefixProto, cli=False):
            # policy_class_name = 'ActorCritic'
            # algorithm_class_name = 'PPO'
            algorithm_class_name = 'RMA' #somebody was trying to reimplement RMA paper eh?
            num_steps_per_env = 24  # per iteration
            max_iterations = 1500  # number of policy updates

            # logging
            save_interval = 500 # 400  # check for potential saves every this many iterations
            save_video_interval = 250
            save_curriculum_plot_interval = 10
            log_freq = 10

            # load and resume
            resume = False
            load_run = -1  # -1 = last run
            checkpoint = -1  # -1 = last saved model
            resume_path = None  # updated from load_run and chkpt
            resume_curriculum = True

            wandb_logging = True
            wandb_project = 'robodog_test'
            wandb_entity = 'curious-ks-jp-keio-jp'
            wandb_note = 'default'
