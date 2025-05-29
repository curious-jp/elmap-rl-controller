"""
This module defines a gym environment for tracking the ground truth velocity of the base of a legged robot.

Classes:
    VelocityTrackingEasyEnv(LeggedRobot): a gym environment for velocity tracking.

"""

from isaacgym import gymutil, gymapi
import torch
from params_proto import Meta
from typing import Union

from robodog_gym.envs.base.legged_robot import LeggedRobot
from robodog_gym.envs.base.legged_robot_config import Cfg


class VelocityTrackingEasyEnv(LeggedRobot):
    def __init__(self, sim_device, headless, num_envs=None, prone=False, deploy=False,
                 cfg: Cfg = None, eval_cfg: Cfg = None, initial_dynamics_dict=None, physics_engine="SIM_PHYSX", debug_viz=False):
        """
        Initializes the VelocityTrackingEasyEnv gym environment.

        Note that foot contacts are binarized.
        Some indices are hardcoded for the go1 and A1 models, so this environment may not work with other robots.

        Args:
            sim_device: the device on which to simulate the environment.
            headless:   a boolean indicating whether to display the simulation window.
            num_envs:   the number of environments to create (optional).
            prone:      a boolean indicating whether the robot starts in a prone position (optional).
            deploy:     a boolean indicating whether to deploy the environment (optional).
            cfg:        a configuration object for the environment (optional).
            eval_cfg:   a configuration object for evaluation of the environment (optional).
            initial_dynamics_dict:  a dictionary of initial dynamics settings for the environment (optional).
            physics_engine:         the physics engine to use for simulation (default is "SIM_PHYSX").

        Returns:
            None
        """

        if num_envs is not None:
            cfg.env.num_envs = num_envs

        sim_params = gymapi.SimParams()
        gymutil.parse_sim_config(vars(cfg.sim), sim_params)
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless, eval_cfg, initial_dynamics_dict,debug_viz=debug_viz)


    def step(self, actions):
        """
        Perform a step in the environment.

        Important: the tensors may be not cloned from internal tensors, so they should be cloned before calling step() again

        Args:
            actions (array-like): The actions to take in the environment.

        Returns:
            tuple: A tuple containing the following elements:
                - dict: A dictionary containing the observation buffers for the policy, estimator, and privileged observations.
                - array-like: The reward buffer.
                - array-like: The reset buffer.
                - dict: A dictionary containing additional information.

        """

        policy_obs_buf, estimator_obs_buf, privileged_obs_buf, rew_buf, reset_buf, extras = super().step(actions)

        # extract foot positions, with hardcoded indices from go1 or A1 models
        # The  robot has 1 body and the number of joints is 12 leg joints + 1 base joint
        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices,
                               0:3]

        extras.update({
            "joint_pos": self.dof_pos.cpu().numpy(),
            "joint_vel": self.dof_vel.cpu().numpy(),
            "joint_pos_target": self.joint_pos_target.cpu().detach().numpy(),
            "joint_vel_target": torch.zeros(12),
            "body_linear_vel": self.base_lin_vel.cpu().detach().numpy(),
            "body_angular_vel": self.base_ang_vel.cpu().detach().numpy(),
            "body_linear_vel_cmd": self.commands.cpu().numpy()[:, 0:2],
            "body_angular_vel_cmd": self.commands.cpu().numpy()[:, 2:],
            "contact_states": (self.contact_forces[:, self.feet_indices, 2] > 1.).detach().cpu().numpy().copy(),
            "foot_positions": (self.foot_positions).detach().cpu().numpy().copy(),
            "body_pos": self.root_states[:, 0:3].detach().cpu().numpy(),
            "torques": self.torques.detach().cpu().numpy()
        })

        return {'policy_obs': policy_obs_buf, 'estimator_obs': estimator_obs_buf, 'privileged_obs': privileged_obs_buf}, rew_buf, reset_buf, extras
    
    def get_observations(self):
        """
        Get the observations from the environment.

        Important: the tensors may be not cloned from internal tensors, so they should be cloned before calling step()

        Returns:
            dict: A dictionary containing the observation buffers for the policy, estimator, and privileged observations.

        """
        policy_obs_buf, estimator_obs_buf, privileged_obs_buf = super().get_observations()
        return {'policy_obs': policy_obs_buf, 'estimator_obs': estimator_obs_buf, 'privileged_obs': privileged_obs_buf}

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs_dict, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs_dict

