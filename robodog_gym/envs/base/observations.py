# License: see [LICENSE]

"""  Observations class for the environment

This class is used to compute the observations for the environment.
It handles observation buffer creation, observation computation and observation scaling.
"""

import torch
import numpy as np
from robodog_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from isaacgym.torch_utils import *
from isaacgym import gymapi


class Observations:
    """
    This class handles the computation of all observations buffers for the environment.
    Also implements observation history for some observation components

    Attributes:
        env (object): The environment object.
        cfg (object): The configuration object.

    Methods:
        __init__(self, env): Initializes the Observations object.
        _compute_obs_buf(self, active_policy_obs_components, add_noise=False): Computes a single observation vector for the specified observation components.
        compute_obs(self): Computes all observations.
    
    """
    def __init__(self, env):
        self.env = env
        self.cfg = env.cfg

        # process observation configuration

        # use policy observation components for estimator if not specified
        if  self.cfg.env.estimator_observation_components is  None:
            self.cfg.env.estimator_observation_components = self.cfg.env.policy_observation_components


        # fill missing sizes
        for component in  self.cfg.env.policy_observation_components:
            if component[2] and component[0] == 'commands':
                component[1] = self.cfg.commands.num_commands
            if component[2] and component[0] == 'height_measurements':
                component[1] = self.cfg.env.num_height_points

        for component in  self.cfg.env.estimator_observation_components:
            if component[2] and component[0] == 'commands':
                component[1] = self.cfg.commands.num_commands
            if component[2] and component[0] == 'height_measurements':
                component[1] = self.cfg.env.num_height_points
        
            
        # Get only observation components that are active, based on boolean flag
        self.active_policy_obs_components = [comp[:2] for comp in self.cfg.env.policy_observation_components if comp[-1]]
        self.active_estimator_obs_components = [comp[:2] for comp in self.cfg.env.estimator_observation_components if comp[-1]]
        self.active_privileged_obs_components = [comp[:2] for comp in self.cfg.env.privileged_observation_components if comp[-1]]

        # unition of active policy and estimator observation components
        self.active_policy_estimator_components = self.active_policy_obs_components + [component for component in self.active_estimator_obs_components if component not in self.active_policy_obs_components]
        
        # dict if obs component is active either in policy or estimator obs
        is_estimator_obs_component_active = {comp[0] : comp[2] for comp in self.cfg.env.estimator_observation_components}
        self.is_obs_component_active = {comp[0] : comp[2] or is_estimator_obs_component_active[comp[0]] for comp in self.cfg.env.policy_observation_components}

        # self.is_privileged_obs_component_active = {comp[0] : comp[2] for comp in self.cfg.env.privileged_observation_components}

        num_policy_observations = sum([comp[1] for comp in self.active_policy_obs_components])
        num_estimator_observations = sum([comp[1] for comp in self.active_estimator_obs_components])
        num_privileged_observations = sum([comp[1] for comp in self.active_privileged_obs_components])

        self.env.num_policy_obs = num_policy_observations
        self.env.num_estimator_obs = num_estimator_observations
        self.env.num_privileged_obs = num_privileged_observations

        print(f"Number of policy observations: {num_policy_observations}")
        print(f"Number of estimator observations: {num_estimator_observations}")
        print(f"Number of privileged observations: {num_privileged_observations}")

        # Create observation noise buffer dict
        # It is used to ensure that at each iteration the noise is the same for same observation components in policy and estimator
        # dict of noise_name : noise tensor of noise_name size
        self.obs_noise_components = {obs_name:torch.zeros(self.env.num_envs, size, dtype=torch.float, device=self.env.device, requires_grad=False)
                                      for (obs_name,size) in self.active_policy_estimator_components}


        # Initialize observation history buffers
        dof_history_total_length = ((self.cfg.env.dof_history_length-1)*(self.cfg.env.dof_history_step_skip+1)+1)
        self.dof_position_history = torch.zeros(self.env.num_envs, dof_history_total_length,self.env.num_actuated_dof, dtype=torch.float,
                                       device=self.env.device, requires_grad=False)
        self.dof_velocity_history = torch.zeros(self.env.num_envs, dof_history_total_length,self.env.num_actuated_dof, dtype=torch.float,
                                       device=self.env.device, requires_grad=False)

        action_history_total_length = ((self.cfg.env.action_history_length-1)*(self.cfg.env.action_history_step_skip+1)+1)
        self.action_history = torch.zeros(self.env.num_envs, action_history_total_length, self.env.num_actuated_dof, dtype=torch.float,
                                       device=self.env.device, requires_grad=False)
        
        # initialize additional buffers for observation computation
        self.last_contact_states = torch.zeros(self.env.num_envs, len(self.env.feet_indices), dtype=torch.bool, device=self.env.device,
                                         requires_grad=False)

       

    def reset_idx(self, env_ids):
        """ Reset history buffers some environments.

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """

        if len(env_ids) == 0:
            return

        # reset observation history for terminated envs
        self.dof_position_history[env_ids, :, :] = 0
        self.dof_velocity_history[env_ids, :, :] = 0
        self.action_history[env_ids, :, :] = 0

        # reset additional buffers
        self.last_contact_states[env_ids, :] = False

    
    def compute_obs(self):
        """ Compute all observations
        Returns:
            obs_buf (torch.Tensor): clipped observation buffer for main policy
            estimator_obs_buf (torch.Tensor): clipped observation buffer for the estimator
            privileged_obs_buf (torch.Tensor): clipped privileged observation buffer
        """
        # debug
        self.debug_flag = False

        self._update_observation_histories()


        # TODO improve: update noise for each active observation component
        for obs_name in self.obs_noise_components.keys():
            if 'history' not in obs_name:
                self.obs_noise_components[obs_name] = (2 * torch.rand_like(self.obs_noise_components[obs_name]) - 1) 

        # -----------------------
        # build policy obs
        # -----------------------

        policy_obs_buf  = self._compute_obs_buf(self.active_policy_obs_components, prefix ="", add_noise =self.env.cfg.noise.add_noise)
            

        # -----------------------
        # build estimator obs
        # -----------------------

        estimator_obs_buf = self._compute_obs_buf(self.active_estimator_obs_components, prefix ="", add_noise =self.env.cfg.noise.add_noise)

    
        # -----------------------
        # build privileged obs
        # -----------------------

        privileged_obs_buf = self._compute_obs_buf(self.active_privileged_obs_components, prefix ="priv_", add_noise = False)


        # Clip observations
        clip_obs = self.env.cfg.normalization.clip_observations
        policy_obs_buf = torch.clip(policy_obs_buf, -clip_obs, clip_obs)
        estimator_obs_buf = torch.clip(estimator_obs_buf, -clip_obs, clip_obs)
        if privileged_obs_buf is not None:
            privileged_obs_buf = torch.clip(privileged_obs_buf, -clip_obs, clip_obs)


    
         
        return policy_obs_buf, estimator_obs_buf, privileged_obs_buf


    def _compute_obs_buf(self,active_policy_obs_components,prefix,add_noise=False):
        """ Compute a single observation vector for the specified observation components
        Add scaled noise if needed

        Args:
            active_policy_obs_components (List): List of tuples defining the active observation components (name, size)
            add_noise (bool): Whether to add noise to the observations
            prefix (str): Prefix for the observation components ("", or "priv_")
        Returns:
            obs_buf (torch.Tensor): concatenated observation buffer
        """

        noise_scales = self.env.cfg.noise_scales
        noise_level = self.env.cfg.noise.noise_level

        obs_list = []
        for obs_name, size_info in active_policy_obs_components:
            # Determine the size, which might be a callable or a static value
            size = size_info(cfg) if callable(size_info) else size_info
            tensor_method_name = f'_calculate_{prefix}{obs_name}'
            if hasattr(self, tensor_method_name):
                tensor = getattr(self, tensor_method_name)()
                if add_noise and 'history' not in obs_name:
                    # find corresponding noise scale, for non-history observations
                    scale = 1.0
                    if hasattr(noise_scales, obs_name):
                        scale = getattr(noise_scales, obs_name)
                    
                    obs_list.append(tensor + self.obs_noise_components[obs_name]*noise_level*scale)
                else:
                    obs_list.append(tensor)
            else:
                raise NotImplementedError(f"Calculation method for {obs_name} not implemented.")
        
        # Perform a single concatenation operation
        obs_buf = torch.cat(obs_list, dim=-1).to(self.env.device)

        if self.debug_flag:
            # self.debug_flag = False
            
            # print if there is termination
            if self.env.reset_buf[0] > 0:
                print("Reset")
            
            if self.env.common_step_counter%10==0 and  prefix == "priv_":
                obs_np = [obs.cpu().numpy() for obs in obs_list]
                
                print(f"it {self.env.common_step_counter}",np.round(obs_np[1],2))

        return obs_buf


    def _update_observation_histories(self):
        """ Update observation history buffers
        Also adds noise to the newly added observations
        """

        noise_scales = self.env.cfg.noise_scales
        noise_level = self.env.cfg.noise.noise_level
        add_noise = self.env.cfg.noise.add_noise
        
        # DOF position and velocity history

        if self.is_obs_component_active['dof_position_history']:
            # Calculate the new DOF positions and add noise
            new_dof_positions = self._calculate_dof_positions()
            if add_noise:
                new_dof_positions = new_dof_positions + (2 * torch.rand_like(new_dof_positions) - 1) * noise_level * noise_scales.dof_position_history

            self.dof_position_history = torch.cat(
                (self.dof_position_history[:, 1:, :],  # Remove the oldest history step
                new_dof_positions.unsqueeze(1)),  # Add the new positions at the last position
                dim=1  # Concatenate along the history axis
            )

        # update history buffer
        if self.is_obs_component_active['dof_velocity_history']:
            # Calculate the new DOF positions and add noise
            new_dof_velocities = self._calculate_dof_velocities()
            if add_noise:
                new_dof_velocities = new_dof_velocities + (2 * torch.rand_like(new_dof_velocities) - 1) * noise_level * noise_scales.dof_velocity_history

            self.dof_velocity_history = torch.cat(
                (self.dof_velocity_history[:, 1:, :],  # Remove the oldest history step
                new_dof_velocities.unsqueeze(1)),  # Add the new positions at the last position
                dim=1  # Concatenate along the history axis
            )

        # Action history
        if self.is_obs_component_active['action_history']:
            # Calculate the new DOF positions and add noise
            new_actions = self._calculate_last_actions()

            self.action_history = torch.cat(
                (self.action_history[:, 1:, :],  # Remove the oldest history step
                new_actions.unsqueeze(1)),  # Add the new positions at the last position
                dim=1  # Concatenate along the history axis
            )



    # -----------------------
    # Normal observations
    # -----------------------

    def _calculate_commands(self):
        return self.env.commands * self.env.commands_scale

    def _calculate_global_linear_vel(self):
        return self.env.root_states[:self.env.num_envs, 7:10] * self.env.obs_scales.linear_vel

    def _calculate_linear_vel(self):
        return self.env.base_lin_vel * self.env.obs_scales.linear_vel

    def _calculate_angular_vel(self):
        return self.env.base_ang_vel * self.env.obs_scales.angular_vel

    def _calculate_projected_gravity(self):
        return self.env.projected_gravity*self.env.obs_scales.projected_gravity

    def _calculate_dof_positions(self):
        return (self.env.dof_pos[:, :self.env.num_actuated_dof] - self.env.default_dof_pos[:, :self.env.num_actuated_dof]) * self.env.obs_scales.dof_positions

    def _calculate_dof_velocities(self):
        return  self.env.dof_vel[:, :self.env.num_actuated_dof] * self.env.obs_scales.dof_velocities

    def _calculate_dof_position_history(self):
        
        # Select only non-skipped steps from the history based on the configuration
        selected_history = self.dof_position_history[:, ::self.cfg.env.dof_history_step_skip + 1, :]

        # Reshape the selected history to flatten the history and DOF dimensions into a single dimension
        return selected_history.reshape(selected_history.shape[0], -1)
        
    def _calculate_dof_velocity_history(self):

        # Select only non-skipped steps from the history based on the configuration
        selected_history = self.dof_velocity_history[:, ::self.cfg.env.dof_history_step_skip + 1, :]

        # Reshape the selected history to flatten the history and DOF dimensions into a single dimension
        return selected_history.reshape(selected_history.shape[0], -1)

            
    def _calculate_last_actions(self):
        return self.env.actions

    def _calculate_action_history(self):
        
        # Select only non-skipped steps from the history based on the configuration
        selected_history = self.action_history[:, ::self.cfg.env.action_history_step_skip + 1, :]

        # Reshape the selected history to flatten the history and DOF dimensions into a single dimension
        return selected_history.reshape(selected_history.shape[0], -1)
        

    def _calculate_timing_inputs(self):
        return self.env.gait_indices.unsqueeze(1)

    def _calculate_clock_inputs(self):
        return self.env.clock_inputs

    def _calculate_yaw(self):
        forward = quat_apply(self.env.base_quat, self.env.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0]).unsqueeze(1)
        return heading
    
    def _calculate_contact_states(self):
        return (self.env.contact_forces[:, self.env.feet_indices, 2] > 1.).view(self.env.num_envs, -1) * 1.0
        
    def _calculate_height_measurements(self):
        # height_measurements noise is handled separately
        return torch.clip(self.env.noisy_measured_heights - self.env.root_states[:, 2].unsqueeze(1) + self.env.obs_bias.height_measurements, -1, 1.) * self.env.obs_scales.height_measurements


    # -----------------------
    # Privileged observations
    # -----------------------

    def _calculate_priv_friction(self):
        friction_coeffs_scale, friction_coeffs_shift = get_scale_shift(self.env.cfg.normalization.friction_range)
        return (self.env.friction_coeffs[:, 0].unsqueeze(1) - friction_coeffs_shift) * friction_coeffs_scale

    def _calculate_priv_ground_friction(self):
        self.env.ground_friction_coeffs = self.env._get_ground_frictions(range(self.env.num_envs))
        ground_friction_coeffs_scale, ground_friction_coeffs_shift = get_scale_shift(
            self.env.cfg.normalization.ground_friction_range)
        return (self.env.ground_friction_coeffs.unsqueeze(1) - ground_friction_coeffs_shift) * ground_friction_coeffs_scale

    def _calculate_priv_restitution(self):
        restitutions_scale, restitutions_shift = get_scale_shift(self.env.cfg.normalization.restitution_range)
        return (self.env.restitutions[:, 0].unsqueeze(1) - restitutions_shift) * restitutions_scale

    def _calculate_priv_base_mass(self):
        payloads_scale, payloads_shift = get_scale_shift(self.env.cfg.normalization.added_mass_range)
        return (self.env.payloads.unsqueeze(1) - payloads_shift) * payloads_scale

    def _calculate_priv_com_displacement(self):
        com_displacements_scale, com_displacements_shift = get_scale_shift(
            self.env.cfg.normalization.com_displacement_range)
        return (self.env.com_displacements - com_displacements_shift) * com_displacements_scale

    def _calculate_priv_motor_strength(self):
        motor_strengths_scale, motor_strengths_shift = get_scale_shift(self.env.cfg.normalization.motor_strength_range)
        return (self.env.motor_strengths - motor_strengths_shift) * motor_strengths_scale

    def _calculate_priv_motor_offset(self):
        motor_offset_scale, motor_offset_shift = get_scale_shift(self.env.cfg.normalization.motor_offset_range)
        return (self.env.motor_offsets - motor_offset_shift) * motor_offset_scale

    def _calculate_priv_body_height(self):
        body_height_scale, body_height_shift = get_scale_shift(self.env.cfg.normalization.body_height_range)
        return ((self.env.root_states[:self.env.num_envs, 2]).view(self.env.num_envs, -1) - body_height_shift) * body_height_scale

    def _calculate_priv_body_velocity(self):
        x_velocity_scale, x_velocity_shift = get_scale_shift(self.env.cfg.normalization.x_velocity_range)
        y_velocity_scale, y_velocity_shift = get_scale_shift(self.env.cfg.normalization.y_velocity_range)
        z_velocity_scale, z_velocity_shift = get_scale_shift(self.env.cfg.normalization.z_velocity_range)

        # print("Unscaled env 0 body velocity: ", self.env.base_lin_vel[0])
        x_vel = self.env.base_lin_vel[:, 0].view(self.env.num_envs, -1)
        y_vel = self.env.base_lin_vel[:, 1].view(self.env.num_envs, -1)
        z_vel = self.env.base_lin_vel[:, 2].view(self.env.num_envs, -1)

        # Normalize each component separately
        normalized_x_vel = (x_vel - x_velocity_shift) * x_velocity_scale
        normalized_y_vel = (y_vel - y_velocity_shift) * y_velocity_scale
        normalized_z_vel = (z_vel - z_velocity_shift) * z_velocity_scale

        # Concatenate the normalized components
        normalized_velocity = torch.cat((normalized_x_vel, normalized_y_vel, normalized_z_vel), dim=1)
        # print("Env 0 normalized velocity: ", normalized_velocity[0])

        return normalized_velocity

    
    def _calculate_priv_body_angular_velocity(self):
        # print("Privileged observation: body angular velocity")
        pitch_velocity_scale, pitch_velocity_shift = get_scale_shift(self.env.cfg.normalization.pitch_velocity_range)
        roll_velocity_scale, roll_velocity_shift = get_scale_shift(self.env.cfg.normalization.roll_velocity_range)
        yaw_velocity_scale, yaw_velocity_shift = get_scale_shift(self.env.cfg.normalization.yaw_velocity_range)

        # print("Unscaled env 0 body angular velocity: ", self.env.base_ang_vel[0])
        pitch_vel = self.env.base_ang_vel[:, 0].view(self.env.num_envs, -1) # roll pitch yaw OR pitch roll yaw ???
        roll_vel = self.env.base_ang_vel[:, 1].view(self.env.num_envs, -1)
        yaw_vel = self.env.base_ang_vel[:, 2].view(self.env.num_envs, -1)

        # Normalize each component separately
        normalized_pitch_vel = (pitch_vel - pitch_velocity_shift) * pitch_velocity_scale
        normalized_roll_vel = (roll_vel - roll_velocity_shift) * roll_velocity_scale
        normalized_yaw_vel = (yaw_vel - yaw_velocity_shift) * yaw_velocity_scale

        # Concatenate the normalized components
        normalized_angular_velocity = torch.cat((normalized_pitch_vel, normalized_roll_vel, normalized_yaw_vel), dim=1)
        # print("Env 0 normalized angular velocity: ", normalized_angular_velocity[0])

        return normalized_angular_velocity

    def _calculate_priv_gravity(self):
        gravity_scale, gravity_shift = get_scale_shift(self.env.cfg.normalization.gravity_range)
        return (self.env.gravities - gravity_shift) * gravity_scale

    def _calculate_priv_clock_inputs(self):
        return self.env.clock_inputs

    def _calculate_priv_desired_contact_states(self):
        return self.env.desired_contact_states
    
    def _calculate_priv_contact_states(self):
        # use filtering methods from wtw rewards
        contact = self.env.contact_forces[:, self.env.feet_indices, 2] > 1.
        contact_states_filt = torch.logical_or(contact, self.last_contact_states)
        self.last_contact_states = contact

        return (contact_states_filt.float() - 0.5)*2.0

    def _calculate_priv_feet_height(self):
        feet_height_scale, feet_height_shift = get_scale_shift(self.env.cfg.normalization.foot_height_range)
        return (self.env.feet_height - feet_height_shift) * feet_height_scale
    
    def _calculate_priv_height_measurements_bias(self):
        height_measurements_bias_scale, height_measurements_bias_shift = get_scale_shift(self.env.cfg.normalization.height_measurements_bias_range)
        return (self.env.height_measurements_per_env_xyz_noise - height_measurements_bias_shift) * height_measurements_bias_scale

    def _calculate_priv_height_measurements_z_bias(self):
        height_measurements_z_bias_scale, height_measurements_z_bias_shift = get_scale_shift(self.env.cfg.normalization.height_measurements_z_bias_range)
        return (self.env.height_measurements_per_env_xyz_noise[:,-1:] - height_measurements_z_bias_shift) * height_measurements_z_bias_scale
   
    def _calculate_priv_zero(self):
        # boilerplate zero value to use when disabling the estimator, as it's output size can't be zero
        return torch.zeros(self.env.num_envs, 1, dtype=torch.float, device=self.env.device, requires_grad=False)


