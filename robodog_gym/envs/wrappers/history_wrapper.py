import isaacgym
assert isaacgym
import torch
import gym
import numpy as np

class HistoryWrapper(gym.Wrapper):
    """ Perform frame stacking for policy and estimator observations if obs_history_length > 1
    Additionally it changes return values of step(), get_observations() to be compatible with learning framework
    """
    def __init__(self, env):
        super().__init__(env)
        self.env = env


        # Set up policy history
        if self.env.cfg.env.sparse_obs_history is None:
            # sparse_policy_obs_history: list of timestep indices of observations timesteps in the history
            # [n_1,n_2,n_3,...]. Current timestep is t, so t-n_1, t-n_2, t-n_3, ... are in history
            # The history buffer is then reversed, so the oldest observations come first
            self.sparse_policy_obs_history = list(range(self.env.cfg.env.num_observation_history))
        else:
            self.sparse_policy_obs_history = self.env.cfg.env.sparse_obs_history

        # Set up estimator history
        if self.env.cfg.env.num_estimator_obs_history is None and self.env.cfg.env.sparse_estimator_obs_history is None:
            # use the same history as policy
            self.sparse_estimator_obs_history = self.sparse_policy_obs_history
        elif self.env.cfg.env.sparse_estimator_obs_history is None:
            self.sparse_estimator_obs_history = list(range(self.env.cfg.env.num_estimator_obs_history))
        else:
            self.sparse_estimator_obs_history = self.env.cfg.env.sparse_estimator_obs_history

        # compute observation history indices
        self.policy_obs_history_ids = -1-torch.tensor(list(reversed(self.sparse_policy_obs_history)))
        self.estimator_obs_history_ids = -1-torch.tensor(list(reversed(self.sparse_estimator_obs_history)))

        self.policy_obs_history_length = len(self.sparse_policy_obs_history)
        self.policy_obs_history_length_full = np.max(self.sparse_policy_obs_history) + 1

        self.estimator_obs_history_length = len(self.sparse_estimator_obs_history)
        self.estimator_obs_history_length_full = np.max(self.sparse_estimator_obs_history) + 1

       
        self.policy_obs_history = torch.zeros(self.env.num_envs, self.policy_obs_history_length_full,self.num_policy_obs,
                                       dtype=torch.float, device=self.env.device, requires_grad=False)
        self.estimator_obs_history = torch.zeros(self.env.num_envs, self.estimator_obs_history_length_full,self.num_estimator_obs,
                                       dtype=torch.float, device=self.env.device, requires_grad=False)


        # initialize selected history by running a dry step
        self._step_history(torch.zeros(self.env.num_envs, self.num_policy_obs, dtype=torch.float, device=self.env.device, requires_grad=False),
                            torch.zeros(self.env.num_envs, self.num_estimator_obs, dtype=torch.float, device=self.env.device, requires_grad=False))


        # overwrite the number of observations
        self.num_policy_obs = self.policy_obs_history_length*self.num_policy_obs
        self.num_estimator_obs = self.estimator_obs_history_length*self.num_estimator_obs

        print("num_policy_obs: ", self.num_policy_obs)
        print("num_estimator_obs: ", self.num_estimator_obs)
        # self.env.reset_idx = self.reset_idx
        # self.debug_step = 0


    def step(self, action):
        """ Perform a step in the environment
        Important: the tensors may be not cloned from internal tensors, so they should be cloned before calling step() again
        """

        # privileged information and observation history are stored in info
        obs_dict, rew, done, info = self.env.step(action)
        policy_obs, privileged_obs, estimator_obs = obs_dict["policy_obs"], obs_dict["privileged_obs"], obs_dict["estimator_obs"]

        # reset history for terminated envs
        env_ids = done.nonzero(as_tuple=False).flatten()
        self.reset_idx_history(env_ids)
                

        self._step_history(policy_obs, estimator_obs)


        return {'policy_obs': self.selected_policy_obs_history, 'estimator_obs': self.selected_estimator_obs_history, 'privileged_obs': privileged_obs}, rew, done, info


    def get_observations(self):
        """ Return observations for policy, privileged and estimator networks
        Important: the tensors may be not cloned from internal tensors, so they should be cloned before calling step()
        Called only once at the beginning of learning loop, aftre calling reset()
        Shoud only be called after calling reset() at least once before calling step()

        Returns:
            dict: dictionary containing policy_obs, privileged_obs, estimator_obs
        """
        obs_dict = self.env.get_observations()
        policy_obs, privileged_obs, estimator_obs = obs_dict["policy_obs"], obs_dict["privileged_obs"], obs_dict["estimator_obs"]

        return {'policy_obs': self.selected_policy_obs_history, 'estimator_obs': self.selected_estimator_obs_history, 'privileged_obs': privileged_obs}


    def reset_idx_history(self, env_ids):

        if len(env_ids) == 0:
            return

        # reset observation history for terminated envs
        self.policy_obs_history[env_ids, :] = 0
        self.estimator_obs_history[env_ids, :] = 0
        

    def reset(self):
        # return value is not used 
        obs_dict = super().reset() # calls step() internally
        policy_obs, privileged_obs, estimator_obs = obs_dict["policy_obs"], obs_dict["privileged_obs"], obs_dict["estimator_obs"]

        self.policy_obs_history[:, :] = 0
        self.estimator_obs_history[:, :] = 0
        
        self._step_history(policy_obs, estimator_obs)

        return {'policy_obs': self.selected_policy_obs_history, 'estimator_obs': self.selected_estimator_obs_history, 'privileged_obs': privileged_obs}
    

    def _step_history(self, policy_obs, estimator_obs):
        """ perform a step of the history buffer, computes sparse history
        
        Args:
            policy_obs (torch.Tensor): (num_envs,num_policy_obs) policy observations
            estimator_obs (torch.Tensor): (num_envs,num_estimator_obs)  estimator observations
        """


        # Shift the history buffer
        # .clone() is needed to avoid in-place overwriting
        self.policy_obs_history[:, :-1, :] = self.policy_obs_history[:, 1:, :].clone()
        self.estimator_obs_history[:, :-1, :] = self.estimator_obs_history[:, 1:, :].clone()


        # Insert the latest observations at the last index
        self.policy_obs_history[:, -1, :] = policy_obs
        self.estimator_obs_history[:, -1, :] = estimator_obs

        # Calculate selected history for output
        self.selected_policy_obs_history = self.policy_obs_history[:, self.policy_obs_history_ids, :].reshape(-1, self.num_policy_obs)
        self.selected_estimator_obs_history = self.estimator_obs_history[:, self.estimator_obs_history_ids, :].reshape(-1, self.num_estimator_obs)


if __name__ == "__main__":
    from tqdm import trange
    import matplotlib.pyplot as plt

    import ml_logger as logger

    from robodog_gym_learn.ppo import Runner
    from robodog_gym.envs.wrappers.history_wrapper import HistoryWrapper
    from robodog_gym_learn.ppo.actor_critic import AC_Args

    from robodog_gym.envs.base.legged_robot_config import Cfg
    from robodog_gym.envs.mini_cheetah.mini_cheetah_config import config_mini_cheetah
    config_mini_cheetah(Cfg)

    test_env = gym.make("VelocityTrackingEasyEnv-v0", cfg=Cfg)
    env = HistoryWrapper(test_env)

    env.reset()
    action = torch.zeros(test_env.num_envs, 12)
    for i in trange(3):
        obs, rew, done, info = env.step(action)
        print(obs.keys())
        print(f"obs: {obs['obs']}")
        print(f"privileged obs: {obs['privileged_obs']}")
        print(f"obs_history: {obs['obs_history']}")

        img = env.render('rgb_array')
        plt.imshow(img)
        plt.show()
