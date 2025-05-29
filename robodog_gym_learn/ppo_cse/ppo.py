import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from params_proto import PrefixProto

from robodog_gym_learn.ppo_cse import ActorCritic
from robodog_gym_learn.ppo_cse import RolloutStorage
from robodog_gym_learn.ppo_cse import caches



class PPO:
    actor_critic: ActorCritic

    def __init__(self, actor_critic, cfg_ppo, device='cpu'):

        self.device = device

        self.cfg_ppo = cfg_ppo

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.cfg_ppo.algorithm.learning_rate)
        self.adaptation_module_optimizer = optim.Adam(self.actor_critic.parameters(),
                                                      lr=self.cfg_ppo.algorithm.adaptation_module_learning_rate)
        if self.actor_critic.decoder:
            self.decoder_optimizer = optim.Adam(self.actor_critic.parameters(),
                                                          lr=self.cfg_ppo.algorithm.adaptation_module_learning_rate)
        self.transition = RolloutStorage.Transition()

        self.learning_rate = self.cfg_ppo.algorithm.learning_rate

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, estimator_obs_shape, privileged_obs_shape,
                     action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, estimator_obs_shape, privileged_obs_shape,
                                      action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, estimator_obs, privileged_obs):
        # Important to detech and clone tensors obtained from environment step function
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs,estimator_obs).detach()
        self.transition.values = self.actor_critic.evaluate(obs, privileged_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs.detach().clone()
        self.transition.critic_observations = self.transition.observations
        self.transition.estimator_observations = estimator_obs.detach().clone()
        self.transition.privileged_observations = privileged_obs.detach().clone()
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        # Important to detech and clone tensors obtained from environment step function 
        # (if not modified by non-in-place operations)
        self.transition.rewards = rewards.detach().clone()
        self.transition.dones = dones.detach().clone()
        self.transition.env_bins = infos["env_bins"]
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.cfg_ppo.algorithm.gamma * torch.squeeze(
                self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs, last_critic_privileged_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs, last_critic_privileged_obs).detach()
        self.storage.compute_returns(last_values, self.cfg_ppo.algorithm.gamma, self.cfg_ppo.algorithm.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_adaptation_module_loss = 0
        mean_decoder_loss = 0
        mean_decoder_loss_student = 0
        mean_adaptation_module_test_loss = 0
        mean_decoder_test_loss = 0
        mean_decoder_test_loss_student = 0
        generator = self.storage.mini_batch_generator(self.cfg_ppo.algorithm.num_mini_batches, self.cfg_ppo.algorithm.num_learning_epochs)
        for obs_batch, critic_obs_batch, estimator_obs_batch, privileged_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, masks_batch, env_bins_batch in generator:

            self.actor_critic.act(obs_batch, estimator_obs_batch, masks=masks_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(obs_batch, privileged_obs_batch, masks=masks_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL
            if self.cfg_ppo.algorithm.desired_kl != None and self.cfg_ppo.algorithm.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (
                                torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (
                                2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.cfg_ppo.algorithm.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / self.cfg_ppo.algorithm.lr_adaptive_schedule_decay)
                    elif kl_mean < self.cfg_ppo.algorithm.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.cfg_ppo.algorithm.clip_param,
                                                                               1.0 + self.cfg_ppo.algorithm.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.cfg_ppo.algorithm.use_clipped_value_loss:
                value_clipped = target_values_batch + \
                                (value_batch - target_values_batch).clamp(-self.cfg_ppo.algorithm.clip_param,
                                                                          self.cfg_ppo.algorithm.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.cfg_ppo.algorithm.value_loss_coef * value_loss - self.cfg_ppo.algorithm.entropy_coef * entropy_batch.mean()

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.cfg_ppo.algorithm.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

            data_size = privileged_obs_batch.shape[0]
            num_train = int(data_size // 5 * 4)

            # Adaptation module gradient step

            for epoch in range(self.cfg_ppo.algorithm.num_adaptation_module_substeps):

                adaptation_pred = self.actor_critic.adaptation_module(estimator_obs_batch)
                with torch.no_grad():
                    adaptation_target = privileged_obs_batch
                    # residual = (adaptation_target - adaptation_pred).norm(dim=1)
                    # caches.slot_cache.log(env_bins_batch[:, 0].cpu().numpy().astype(np.uint8),
                    #                       sysid_residual=residual.cpu().numpy())

                # print("The adaptation module has target: ", adaptation_target)
                selection_indices = torch.linspace(0, adaptation_pred.shape[1]-1, steps=adaptation_pred.shape[1], dtype=torch.long)
                if self.cfg_ppo.algorithm.selective_adaptation_module_loss:
                    # mask out indices corresponding to swing feet
                    selection_indices = 0

                adaptation_loss = F.mse_loss(adaptation_pred[:num_train, selection_indices], adaptation_target[:num_train, selection_indices])
                adaptation_test_loss = F.mse_loss(adaptation_pred[num_train:, selection_indices], adaptation_target[num_train:, selection_indices])



                self.adaptation_module_optimizer.zero_grad()
                adaptation_loss.backward()
                self.adaptation_module_optimizer.step()

                mean_adaptation_module_loss += adaptation_loss.item()
                mean_adaptation_module_test_loss += adaptation_test_loss.item()

        num_updates = self.cfg_ppo.algorithm.num_learning_epochs * self.cfg_ppo.algorithm.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_adaptation_module_loss /= (num_updates * self.cfg_ppo.algorithm.num_adaptation_module_substeps)
        mean_decoder_loss /= (num_updates * self.cfg_ppo.algorithm.num_adaptation_module_substeps)
        mean_decoder_loss_student /= (num_updates * self.cfg_ppo.algorithm.num_adaptation_module_substeps)
        mean_adaptation_module_test_loss /= (num_updates * self.cfg_ppo.algorithm.num_adaptation_module_substeps)
        mean_decoder_test_loss /= (num_updates * self.cfg_ppo.algorithm.num_adaptation_module_substeps)
        mean_decoder_test_loss_student /= (num_updates * self.cfg_ppo.algorithm.num_adaptation_module_substeps)
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_adaptation_module_loss, mean_decoder_loss, mean_decoder_loss_student, mean_adaptation_module_test_loss, mean_decoder_test_loss, mean_decoder_test_loss_student
