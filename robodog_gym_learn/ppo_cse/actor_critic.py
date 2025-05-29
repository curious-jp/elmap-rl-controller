import torch
import torch.nn as nn
from params_proto import PrefixProto
from torch.distributions import Normal


class Cfg_ppo(PrefixProto, cli=False):
    class policy(PrefixProto, cli=False):
        # policy
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

        adaptation_module_branch_hidden_dims = [256, 128]

        use_decoder = False


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self, num_obs,
                 num_estimator_obs,
                 num_privileged_obs,
                 num_actions,
                 cfg_ppo = None,
                 **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        self.cfg_ppo = cfg_ppo
        if self.cfg_ppo is None:
            self.cfg_ppo = Cfg_ppo

        self.decoder = self.cfg_ppo.policy.use_decoder
        super().__init__()

        self.num_obs = num_obs
        self.num_estimator_obs = num_estimator_obs
        self.num_privileged_obs = num_privileged_obs

        activation = get_activation(self.cfg_ppo.policy.activation)

        # Estimator module (it is called adaptation module for legay reasons, seemingly adopted from RMA architecture)
        adaptation_module_layers = []
        adaptation_module_layers.append(nn.Linear(self.num_estimator_obs, self.cfg_ppo.policy.adaptation_module_branch_hidden_dims[0]))
        adaptation_module_layers.append(activation)
        for l in range(len(self.cfg_ppo.policy.adaptation_module_branch_hidden_dims)):
            if l == len(self.cfg_ppo.policy.adaptation_module_branch_hidden_dims) - 1:
                adaptation_module_layers.append(
                    nn.Linear(self.cfg_ppo.policy.adaptation_module_branch_hidden_dims[l], self.num_privileged_obs))
            else:
                adaptation_module_layers.append(
                    nn.Linear(self.cfg_ppo.policy.adaptation_module_branch_hidden_dims[l],
                              self.cfg_ppo.policy.adaptation_module_branch_hidden_dims[l + 1]))
                adaptation_module_layers.append(activation)
        self.adaptation_module = nn.Sequential(*adaptation_module_layers)



        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(self.num_privileged_obs + self.num_obs, self.cfg_ppo.policy.actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(self.cfg_ppo.policy.actor_hidden_dims)):
            if l == len(self.cfg_ppo.policy.actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(self.cfg_ppo.policy.actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(self.cfg_ppo.policy.actor_hidden_dims[l], self.cfg_ppo.policy.actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor_body = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(self.num_privileged_obs + self.num_obs, self.cfg_ppo.policy.critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(self.cfg_ppo.policy.critic_hidden_dims)):
            if l == len(self.cfg_ppo.policy.critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(self.cfg_ppo.policy.critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(self.cfg_ppo.policy.critic_hidden_dims[l], self.cfg_ppo.policy.critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic_body = nn.Sequential(*critic_layers)

        print(f"Adaptation Module: {self.adaptation_module}")
        print(f"Actor MLP: {self.actor_body}")
        print(f"Critic MLP: {self.critic_body}")

        # Action noise
        self.std = nn.Parameter(self.cfg_ppo.policy.init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations, estimator_observations):
        latent = self.adaptation_module(estimator_observations)
        # Possible issue: latent gradients should be detached, as without the PPO gradients are flowing trough the estimator
        # However, detaching the gradients causes policy training to behave differently, requiring retuning of rsl-rl hyperparameters,
        # Especially the target entropy should be lowered.
        mean = self.actor_body(torch.cat((observations, latent), dim=-1))
        self.distribution = Normal(mean, mean * 0. + self.std)

    def act(self, observations, estimator_observations, **kwargs):
        self.update_distribution(observations, estimator_observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_expert(self, ob, policy_info={}):
        return self.act_teacher(ob["obs"], ob["privileged_obs"])

    def act_inference(self, ob, policy_info={}):
        return self.act_student(ob["obs"], policy_info=policy_info)

    def act_student(self, observations, estimator_observations, policy_info={}):
        latent = self.adaptation_module(estimator_observations)
        # Latent gradients should be detached, check comment in update_distribution
        actions_mean = self.actor_body(torch.cat((observations, latent), dim=-1))
        policy_info["latents"] = latent.detach().cpu().numpy()
        return actions_mean

    def act_teacher(self, observations, privileged_info, policy_info={}):
        actions_mean = self.actor_body(torch.cat((observations, privileged_info), dim=-1))
        policy_info["latents"] = privileged_info
        return actions_mean

    def evaluate(self, observations, privileged_observations, **kwargs):
        value = self.critic_body(torch.cat((observations, privileged_observations), dim=-1))
        return value

    def get_student_latent(self, estimator_observations):
        return self.adaptation_module(estimator_observations)

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
