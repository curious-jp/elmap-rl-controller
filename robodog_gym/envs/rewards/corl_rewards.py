import torch
import numpy as np
from robodog_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from isaacgym.torch_utils import *
from isaacgym import gymapi

class CoRLRewards:
    def __init__(self, env):
        self.env = env

    def load_env(self, env):
        self.env = env

    # ------------ reward functions----------------


    def _reward_tracking_lin_vel(self):
        # sigma curriculum
        # ppo iteration should be simulator step / 24 (num_steps_per_env hardcoded in RunnerArgs)
        # (in output, "iterations" is simulat steps * num envs 4000?)
        iteration = self.env.common_step_counter / self.env.cfg.cfg_ppo.runner.num_steps_per_env
        tracking_sigma = self.env.cfg.rewards.tracking_sigma # TODO  implement curriculum
        target_iteration = self.env.cfg.rewards.end_sigma_curriculum_iter
        start_tracking_sigma = self.env.cfg.rewards.start_tracking_sigma
        if iteration >= target_iteration:
            tracking_sigma = self.env.cfg.rewards.tracking_sigma
        else:
            tracking_sigma = start_tracking_sigma - (start_tracking_sigma - self.env.cfg.rewards.tracking_sigma) * (iteration / target_iteration)
        # if iteration%10==0:
        #     print("Tracking sigma: ", tracking_sigma, " Iteration: ", iteration)
        # Tracking of linear velocity commands (xy axes)
        # lin_vel_error = torch.sum(torch.pow(self.env.commands[:, :2] - self.env.base_lin_vel[:, :2],4), dim=1)
        lin_vel_error = torch.sum(torch.square(self.env.commands[:, :2] - self.env.base_lin_vel[:, :2]), dim=1)
        #return torch.exp(-lin_vel_error / self.env.cfg.rewards.tracking_sigma)
        return torch.exp(-lin_vel_error / tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # sigma curriculum
        # ppo iteration should be simulator step / 24 (num_steps_per_env hardcoded in RunnerArgs)
        # (in output, "iterations" is simulat steps * num envs 4000?)
        iteration = self.env.common_step_counter / 24
        tracking_sigma_yaw = self.env.cfg.rewards.tracking_sigma_yaw # TODO  implement curriculum
        target_iteration = self.env.cfg.rewards.end_sigma_yaw_curriculum_iter 
        start_tracking_sigma_yaw = self.env.cfg.rewards.start_tracking_sigma_yaw
        if iteration >= target_iteration:
            tracking_sigma_yaw = self.env.cfg.rewards.tracking_sigma_yaw
        else:
            tracking_sigma_yaw = start_tracking_sigma_yaw - (start_tracking_sigma_yaw - self.env.cfg.rewards.tracking_sigma_yaw) * (iteration / target_iteration)
        # if iteration%10==0:
        #     print("Tracking sigma: ", tracking_sigma_yaw, " Iteration: ", iteration)
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.env.commands[:, 2] - self.env.base_ang_vel[:, 2])
        #return torch.exp(-ang_vel_error / self.env.cfg.rewards.tracking_sigma_yaw)
        return torch.exp(-ang_vel_error / tracking_sigma_yaw)

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.env.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.env.base_ang_vel[:, :2]), dim=1)

    def _reward_ang_vel_xy_linear(self):
        # Penalize xy axes base angular velocity, linearly
        return torch.sum(torch.abs(self.env.base_ang_vel[:, :2]), dim=1)
    
    def _reward_ang_vel_xy_sqrt(self):
        # Penalize xy axes base angular velocity, with a square root
        return torch.sum(torch.sqrt(torch.abs(self.env.base_ang_vel[:, :2])), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        #TODO: add parameters for pitch and roll scale
        return torch.sum(torch.square(self.env.projected_gravity[:, :2]), dim=1)

    def _reward_torques(self):
        # Penalize torques
        weights = torch.ones(self.env.num_envs ,12, device=self.env.device) # (12) tensor
        hip_joint_indices =   [0, 3, 6, 9]
        thigh_joint_indices = [1, 4, 7, 10]
        calf_joint_indices =  [2, 5, 8, 11]
        weights[:,hip_joint_indices] = self.env.cfg.rewards.torque_hip_weight
        weights[:,thigh_joint_indices] = self.env.cfg.rewards.torque_thigh_weight
        weights[:,calf_joint_indices] = self.env.cfg.rewards.torque_calf_weight

        return torch.sum(torch.square(self.env.torques) * weights, dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.env.last_actions - self.env.actions), dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1. * (torch.norm(self.env.contact_forces[:, self.env.penalised_contact_indices, :], dim=-1) > 0.1),
                         dim=1)

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.env.dof_pos - self.env.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.env.dof_pos - self.env.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.env.dof_vel) - self.env.dof_vel_limits*self.env.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.env.torques) - self.env.torque_limits*self.env.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_jump(self):
        # body height tracking, no idea why it is called jump
        reference_heights = torch.mean((self.env.foot_positions[:, :, 2]).view(self.env.num_envs, -1) - self.env.feet_height, dim=1) #avg terrain height under the feet
        body_height = self.env.base_pos[:, 2]  - reference_heights
        if self.env.cfg.commands.num_commands > 3:
            jump_height_target = self.env.commands[:, 3] + self.env.cfg.rewards.base_height_target
        else:
            jump_height_target = self.env.cfg.rewards.base_height_target
        reward = - torch.square(body_height - jump_height_target)
        return reward


    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.env.root_states[:, 2].unsqueeze(1) - self.env.measured_heights, dim=1)
        return torch.square(base_height - self.env.cfg.rewards.base_height_target)

    def _reward_tracking_contacts_shaped_force(self):
        # penalize nonzero contact forces during swing phase
        foot_forces = torch.norm(self.env.contact_forces[:, self.env.feet_indices, :], dim=-1)
        desired_contact = self.env.desired_contact_states

        reward = 0
        for i in range(4):
            reward += - (1 - desired_contact[:, i]) * (
                        1 - torch.exp(-1 * foot_forces[:, i] ** 2 / self.env.cfg.rewards.gait_force_sigma))
            
        return reward / 4

    def _reward_tracking_contacts_shaped_vel(self):
        # penalize nonzero xy foot velocities during stance phase 
        foot_velocities = torch.norm(self.env.foot_velocities, dim=2).view(self.env.num_envs, -1)
        desired_contact = self.env.desired_contact_states
        reward = 0
        for i in range(4):
            reward += - (desired_contact[:, i] * (
                        1 - torch.exp(-1 * foot_velocities[:, i] ** 2 / self.env.cfg.rewards.gait_vel_sigma)))
            
        return reward / 4

    def _reward_dof_pos(self):
        # Penalize dof positions
        # return torch.sum(torch.square(self.env.dof_pos - self.env.default_dof_pos), dim=1)
        
        # Penalize dof position different from nominal    
        reward = torch.square(self.env.dof_pos - self.env.default_dof_pos) # (env_num x 12) tensor
        
        # Penalize hip joint positions more
        weights = torch.ones(self.env.num_envs ,12, device=self.env.device) # (12) tensor
        hip_joint_indices =   [0, 3, 6, 9]
        thigh_joint_indices = [1, 4, 7, 10]
        calf_joint_indices =  [2, 5, 8, 11]
        weights[:,hip_joint_indices] = self.env.cfg.rewards.dof_pos_hip_weight
        weights[:,thigh_joint_indices] = self.env.cfg.rewards.dof_pos_thigh_weight
        weights[:,calf_joint_indices] = self.env.cfg.rewards.dof_pos_calf_weight

        return torch.sum(reward * weights, dim=1) # (env_num) tensor


    def _reward_dof_pos_stancemode(self):
        # Penalize dof positions
        reward = torch.square(self.env.dof_pos - self.env.default_dof_pos) # (env_num x 12) tensor
        
        # Penalize hip joint positions more
        velocity_commands = self.env.commands[:, :3]
        weights = torch.ones(self.env.num_envs ,12, device=self.env.device) # (12) tensor
        hip_joint_indices =   [0, 3, 6, 9]
        thigh_joint_indices = [1, 4, 7, 10]
        calf_joint_indices =  [2, 5, 8, 11]
        weights[:,hip_joint_indices] = self.env.cfg.rewards.hip_weight
        weights[:,thigh_joint_indices] = self.env.cfg.rewards.thigh_weight
        weights[:,calf_joint_indices] = self.env.cfg.rewards.calf_weight

        if(self.env.cfg.rewards.use_adaptive_stancemode):
            y_cmd_intesity = torch.abs(velocity_commands[:,1]/self.env.cfg.commands.limit_vel_y[1])
            yaw_cmd_intensity = torch.abs(velocity_commands[:,2]/self.env.cfg.commands.limit_vel_yaw[1])
            cmd_intesity = torch.max(y_cmd_intesity,yaw_cmd_intensity)
            cmd_intesity = torch.min(torch.ones_like(cmd_intesity)*0.5, cmd_intesity)
            weights[:,hip_joint_indices] = weights[:,hip_joint_indices]*(torch.ones_like(cmd_intesity)-cmd_intesity).view(-1,1)
        # todo print weights/reward
        reward = torch.sum(reward * weights, dim=1) # (env_num) tensor

        # use stancemode alternative reward if command vector is below a certain threshold
        if self.env.cfg.commands.train_standing_still:
            velocity_commands = self.env.commands[:, :3] # (env_num x 3) tensor, x/y/yaw velocity commands
            cmd_norm = torch.norm(velocity_commands, dim=1)
            env_stancemode = cmd_norm < 0.01 #0.1
            
            # penalize dof positions more if in stancemode
            reward = torch.where(env_stancemode, reward*self.env.cfg.rewards.stancemode_multiplier, reward)
            
        return reward


    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.env.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.env.last_dof_vel - self.env.dof_vel) / self.env.dt), dim=1)

    def _reward_action_smoothness_1(self):
        # Penalize changes in actions
        diff = torch.square(self.env.joint_pos_target[:, :self.env.num_actuated_dof] - self.env.last_joint_pos_target[:, :self.env.num_actuated_dof])
        diff = diff * (self.env.last_actions[:, :self.env.num_dof] != 0)  # ignore first step
        return torch.sum(diff, dim=1)

    def _reward_action_smoothness_2(self):
        # Penalize changes in actions
        diff = torch.square(self.env.joint_pos_target[:, :self.env.num_actuated_dof] - 2 * self.env.last_joint_pos_target[:, :self.env.num_actuated_dof] + self.env.last_last_joint_pos_target[:, :self.env.num_actuated_dof])
        diff = diff * (self.env.last_actions[:, :self.env.num_dof] != 0)  # ignore first step
        diff = diff * (self.env.last_last_actions[:, :self.env.num_dof] != 0)  # ignore second step
        return torch.sum(diff, dim=1)

    def _reward_feet_slip(self):
        contact = self.env.contact_forces[:, self.env.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.env.last_contacts)
        self.env.last_contacts = contact
        foot_velocities = torch.square(torch.norm(self.env.foot_velocities[:, :, 0:2], dim=2).view(self.env.num_envs, -1)) # squared norm of xy velocities, results in 2D tensor
        rew_slip = torch.sum(contact_filt * foot_velocities, dim=1) # sum over the four feet, results in 1D tensor
        return rew_slip
    
    def _reward_feet_air_time(self):
        # print("CAREFUL: this reward may be bugged!")
        contact = self.env.contact_forces[:, self.env.feet_indices, 2] > 1. # 1N contact force threshold
        contact_filt = torch.logical_or(contact, self.env.last_contacts_air_time) # shows whether contact was present in the last two steps
       
        transition_reset_condition = contact_filt ^ self.env.last_contacts_filt_air_time # shows whether a transition from contact to no contact or vice versa happened
        # print("The following env 0 feet just transitioned: ", transition_reset_condition[0, :])

        # reset the touchdown and takeoff timers if a transition happens
        self.env.feet_touchdown_air_time = torch.where(transition_reset_condition, torch.zeros_like(self.env.feet_touchdown_air_time), self.env.feet_touchdown_air_time)
        self.env.feet_takeoff_air_time = torch.where(transition_reset_condition, torch.zeros_like(self.env.feet_takeoff_air_time), self.env.feet_takeoff_air_time)

        # increment the timers
        self.env.feet_touchdown_air_time = torch.where(contact_filt, self.env.feet_touchdown_air_time + self.env.dt, self.env.feet_touchdown_air_time)
        self.env.feet_takeoff_air_time = torch.where(~contact_filt, self.env.feet_takeoff_air_time + self.env.dt, self.env.feet_takeoff_air_time)
        
        # save contact states for next iteration
        self.env.last_contacts_air_time = contact
        self.env.last_contacts_filt_air_time = contact_filt

        velocity_commands = self.env.commands[:, :3] # (env_num x 3) tensor, x/y/yaw velocity commands
        cmd_norm = torch.norm(velocity_commands, dim=1)

        # use stancemode alternative reward if command vector is below a certain threshold
        if self.env.cfg.commands.train_standing_still:
            env_stancemode = cmd_norm < 0.001
        else:
            env_stancemode = cmd_norm < 0 # disabled

        # calculate the desired stance time
        air_time_period = torch.ones_like(contact_filt)*self.env.cfg.rewards.feet_air_time_period      

        if(self.env.cfg.rewards.use_adaptive_period):
            cmd_int_x = torch.abs(velocity_commands[:,0])/self.env.cfg.commands.limit_vel_x[0]
            cmd_int_y = torch.abs(velocity_commands[:,1])/self.env.cfg.commands.limit_vel_x[1]
            cmd_intesity = torch.max(cmd_int_x,cmd_int_y)
            cmd_intesity = torch.min(torch.ones_like(cmd_intesity)*0.5, cmd_intesity)
            air_time_period = air_time_period*((torch.ones_like(cmd_intesity)-cmd_intesity).view(-1,1))

        # set reward to 0 if feet remain in one state for too long


        reward_condition_contact = self.env.feet_touchdown_air_time < air_time_period #(desired_stance_time.unsqueeze(1) + 0.05)
        reward_condition_nocontact = self.env.feet_takeoff_air_time <  air_time_period
       

        rew_air_time_contact = torch.where(reward_condition_contact, torch.min(self.env.feet_touchdown_air_time, air_time_period), torch.tensor(0.0, device=self.env.device))
        rew_air_time_nocontact = torch.where(reward_condition_nocontact, torch.min(self.env.feet_takeoff_air_time, air_time_period), torch.tensor(0.0, device=self.env.device))
        # print("The contact reward of env 0 feet are: ", rew_air_time_contact[0, :])
        # print("The nocontact reward of env 0 feet are: ", rew_air_time_nocontact[0, :])

        # calculate the final reward
        rew_air_time_foot_normal = torch.where(contact_filt, rew_air_time_contact, rew_air_time_nocontact)
        rew_air_time_foot_stancemode = torch.clip(self.env.feet_touchdown_air_time - self.env.feet_takeoff_air_time, min=-air_time_period, max=air_time_period)
        # print("The normal reward of env 0 feet are: ", rew_air_time_foot_normal[0, :])
        # print("The stancemode reward of env 0 feet are: ", rew_air_time_foot_stancemode[0, :])
        
        # enforce trotting gait
        # Joint order: FL FR RL RR
        # rew_air_time_foot_normal[:,0] = torch.where(torch.logical_xor(contact_filt[:,0],contact_filt[:,1]),rew_air_time_foot_normal[:,0],rew_air_time_foot_normal[:,0]*self.env.cfg.rewards.contact_condition_scale)
        # rew_air_time_foot_normal[:,1] = torch.where(torch.logical_xor(contact_filt[:,0],contact_filt[:,1]),rew_air_time_foot_normal[:,1],rew_air_time_foot_normal[:,1]*self.env.cfg.rewards.contact_condition_scale)
        # rew_air_time_foot_normal[:,2] = torch.where(torch.logical_xor(contact_filt[:,2],contact_filt[:,3]),rew_air_time_foot_normal[:,2],rew_air_time_foot_normal[:,2]*self.env.cfg.rewards.contact_condition_scale)
        # rew_air_time_foot_normal[:,3] = torch.where(torch.logical_xor(contact_filt[:,2],contact_filt[:,3]),rew_air_time_foot_normal[:,3],rew_air_time_foot_normal[:,3]*self.env.cfg.rewards.contact_condition_scale)
        
        if(self.env.cfg.rewards.use_adaptive_period): #normalize reward => small period would give small reward else
            rew_air_time_foot_normal *= torch.square((torch.divide(torch.ones_like(cmd_intesity),torch.ones_like(cmd_intesity)-cmd_intesity)).view(-1,1))


        rew_air_time_foot = torch.where(env_stancemode.unsqueeze(1), rew_air_time_foot_stancemode, rew_air_time_foot_normal)
        # print("The final reward of env 0 feet are: ", rew_air_time_foot[0, :])
        
        rew_air_time = torch.sum(rew_air_time_foot, dim=1)
        # print("The final reward of env 0 is: ", rew_air_time[0])

        
        return rew_air_time
        
    def _reward_feet_air_time_rsl(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.env.contact_forces[:, self.env.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.env.last_contacts) 
        self.env.last_contacts = contact
        first_contact = (self.env.feet_air_time > 0.) * contact_filt
        self.env.feet_air_time += self.env.dt
        rew_airTime = torch.sum((self.env.feet_air_time - self.env.cfg.rewards.feet_air_time_rsl_period) * first_contact, dim=1) # reward only on first contact with the ground
        # rew_airTime = torch.sum((self.env.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.env.commands[:, :2], dim=1) > 0.01 #no reward for zero command
        self.env.feet_air_time *= ~contact_filt

        if self.env.cfg.rewards.feet_air_time_rsl_curriculum:
            rew_airTime *= self.env.reward_curriculum_factor

        return rew_airTime
        


    def _reward_feet_contact_vel(self):
        reference_heights = 0
        near_ground = self.env.foot_positions[:, :, 2] - reference_heights < 0.03
        foot_velocities = torch.square(torch.norm(self.env.foot_velocities[:, :, 0:3], dim=2).view(self.env.num_envs, -1))
        rew_contact_vel = torch.sum(near_ground * foot_velocities, dim=1)
        return rew_contact_vel

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.env.contact_forces[:, self.env.feet_indices, :],
                                     dim=-1) - self.env.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_feet_clearance_cmd_linear(self):
        phases = 1 - torch.abs(1.0 - torch.clip((self.env.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)
        foot_height = (self.env.foot_positions[:, :, 2]).view(self.env.num_envs, -1) # reference_heights
        target_height = self.env.commands[:, 9].unsqueeze(1) * phases + 0.02 # offset for foot radius 2cm
        rew_foot_clearance = torch.square(target_height - foot_height) * (1 - self.env.desired_contact_states) # penalize height only when contact is not desired (swing phase)
        rew_foot_clearance = torch.sum(rew_foot_clearance, dim=1)
        
        return rew_foot_clearance
    
    def _reward_feet_clearance_ji22(self):
        # get filtered contact states
        contact = self.env.contact_forces[:, self.env.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.env.last_contacts_feet_clearance_ji22) ##why or? shouldn't it be an and ?
        self.env.last_contacts_feet_clearance_ji22 = contact
        
        target_height = self.env.cfg.rewards.feet_clearance_ji22_target + 0.02 # target + offset for foot radius 2cm
        
        rew_foot_clearance = torch.square(torch.where(self.env.feet_height < target_height, self.env.feet_height, torch.full_like(self.env.feet_height, fill_value=target_height)))/target_height**2 
        rew_foot_clearance = torch.where(contact_filt, torch.zeros_like(rew_foot_clearance), rew_foot_clearance) # set reward to 0 if foot is in contact
        
        
        
        return torch.sum(rew_foot_clearance, dim=1)
    
    def _reward_thigh_angle(self):
        """ Penalize knee height (or thigh joint angle) to avoid collisions with backpack
        """

        max_thigh_angle = self.env.cfg.rewards.max_thigh_angle

        weights = torch.zeros(12, device=self.env.device) # (12) tensor
        thigh_joint_indices = [1, 4, 7, 10]
        weights[thigh_joint_indices] = 1.0

        #thigh_joint_indices =  torch.tensor([1, 4, 7, 10],device=self.env.device)
        
        rew_thigh_angle = torch.clamp(self.env.dof_pos - max_thigh_angle, min=0, max=None)*weights
    
        #print(self.env.dof_pos[:,1])

        return torch.sum(rew_thigh_angle, dim=1)
        
        

    def _reward_feet_impact_vel(self):
        prev_foot_velocities = self.env.prev_foot_velocities[:, :, 2].view(self.env.num_envs, -1)
        contact_states = torch.norm(self.env.contact_forces[:, self.env.feet_indices, :], dim=-1) > 1.0 ##why: non need to filter? 

        rew_foot_impact_vel = contact_states * torch.square(torch.clip(prev_foot_velocities, -100, 0))

        return torch.sum(rew_foot_impact_vel, dim=1)


    def _reward_orientation_control(self):
        # Penalize non flat base orientation
        # Actually nope, also do pitch control!
        roll_pitch_commands = self.env.commands[:, 10:12]
        quat_roll = quat_from_angle_axis(-roll_pitch_commands[:, 1],
                                         torch.tensor([1, 0, 0], device=self.env.device, dtype=torch.float))
        quat_pitch = quat_from_angle_axis(-roll_pitch_commands[:, 0],
                                          torch.tensor([0, 1, 0], device=self.env.device, dtype=torch.float))

        desired_base_quat = quat_mul(quat_roll, quat_pitch)
        desired_projected_gravity = quat_rotate_inverse(desired_base_quat, self.env.gravity_vec)

        return torch.sum(torch.square(self.env.projected_gravity[:, :2] - desired_projected_gravity[:, :2]), dim=1)

    def _reward_raibert_heuristic(self):
        cur_footsteps_translated = self.env.foot_positions - self.env.base_pos.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.env.num_envs, 4, 3, device=self.env.device)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = quat_apply_yaw(quat_conjugate(self.env.base_quat),
                                                              cur_footsteps_translated[:, i, :])

        # nominal positions: [FR, FL, RR, RL]
        if self.env.cfg.commands.num_commands >= 13:
            desired_stance_width = self.env.commands[:, 12:13]
            desired_ys_nom = torch.cat([desired_stance_width / 2, -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], dim=1)
        else:
            desired_stance_width = 0.3
            desired_ys_nom = torch.tensor([desired_stance_width / 2,  -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], device=self.env.device).unsqueeze(0)

        if self.env.cfg.commands.num_commands >= 14:
            desired_stance_length = self.env.commands[:, 13:14]
            desired_xs_nom = torch.cat([desired_stance_length / 2, desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], dim=1)
        else:
            desired_stance_length = 0.45
            desired_xs_nom = torch.tensor([desired_stance_length / 2,  desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], device=self.env.device).unsqueeze(0)

        # raibert offsets
        phases = torch.abs(1.0 - (self.env.foot_indices * 2.0)) * 1.0 - 0.5
        frequencies = self.env.commands[:, 4]
        x_vel_des = self.env.commands[:, 0:1]
        yaw_vel_des = self.env.commands[:, 2:3]
        y_vel_des = yaw_vel_des * desired_stance_length / 2
        desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
        desired_ys_offset[:, 2:4] *= -1
        desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1))

        desired_ys_nom = desired_ys_nom + desired_ys_offset
        desired_xs_nom = desired_xs_nom + desired_xs_offset

        desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)

        err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])

        reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))

        return reward