import time
from collections import deque
import copy
import os

import torch
import numpy as np
from ml_logger import logger
# from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter
import wandb
import yaml
import matplotlib.pyplot as plt

from params_proto import PrefixProto

from .actor_critic import ActorCritic
from .rollout_storage import RolloutStorage



def cfg_to_dict(obj) -> dict:
    """ Recursively convert class and nested classes to dictionary, excluding 'copy' attribute. """
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key, val in obj.__dict__.items():
        # Skip the 'copy' attribute at any level of nesting
        if key == "copy":
            continue
        if isinstance(val, list):
            # Process list items, potentially nested classes or primitives
            element = [cfg_to_dict(item) if hasattr(item, "__dict__") else item for item in val]
            result[key] = element
        elif isinstance(val, dict):
            # Process dictionary items, applying the same exclusion to nested items
            sub_dict = {sub_key: cfg_to_dict(sub_val) for sub_key, sub_val in val.items() if sub_key != "copy"}
            result[key] = sub_dict
        else:
            # Recurse into nested objects
            result[key] = cfg_to_dict(val)
    return result

def filter_arrays_from_dict(d):
    """ Recursively convert or remove numpy arrays and PyTorch tensors from the dictionary. """
    if isinstance(d, dict):
        return {k: filter_arrays_from_dict(v) for k, v in d.items() if not isinstance(v, (np.ndarray, torch.Tensor))}
    elif isinstance(d, list):
        return [filter_arrays_from_dict(v) for v in d if not isinstance(v, (np.ndarray, torch.Tensor))]
    elif isinstance(d, (np.ndarray, np.generic, torch.Tensor)):
        return "defined_at_runtime"  # return a placeholder string
    else:
        return d

class CustomNoAliasDumper(yaml.SafeDumper):
    """ Custom yaml dumper that prevents PyYAML from using aliases and enforces proper list formatting.
    """
    def ignore_aliases(self, data):
        # Prevents PyYAML from using aliases
        return True

    def increase_indent(self, flow=False, indentless=False):
        # Ensures that dictionaries are indented properly without affecting list formatting
        return super(CustomNoAliasDumper, self).increase_indent(flow, False)

    def represent_sequence(self, tag, sequence, flow_style=None):
        # Determine the style based on the nesting level of the sequence
        style = flow_style
        if isinstance(sequence, list) and sequence and isinstance(sequence[0], list):
            # If it's a nested list, we expand it
            style = False
        else:
            # For top-level lists, use in-line style
            style = True
        return super(CustomNoAliasDumper, self).represent_sequence(tag, sequence, flow_style=style)


class DataCaches:
    def __init__(self, curriculum_bins):
        from robodog_gym_learn.ppo.metrics_caches import SlotCache, DistCache

        self.slot_cache = SlotCache(curriculum_bins)
        self.dist_cache = DistCache()


caches = DataCaches(1)



class Runner:
    """ (autogen) A class that handles training and evaluation of a PPO algorithm on a given environment.
    Adapted from rsl_rl OnPolicyRunner class

    Args:
        env (gym.Env): The environment to train the algorithm on.
        cfg: Cfg from robodog_gym.base.legged_robot_config
        # cfg_ppo: Cfg.cfg_ppo from robodog_gym.base.legged_robot_config
        device (str, optional): The device to use for computation (default: 'cpu').

    """

    def __init__(self, env, cfg, device='cpu'):
        from .ppo import PPO

        self.device = device
        self.env = env
        self.cfg = env.cfg
        self.cfg_ppo = cfg.cfg_ppo

        actor_critic = ActorCritic(self.env.num_policy_obs,
                                    self.env.num_estimator_obs,
                                    self.env.num_privileged_obs,
                                    self.env.num_actions,
                                    cfg_ppo=cfg.cfg_ppo,
                                    ).to(self.device)

        if self.cfg_ppo.runner.resume:
            # load pretrained weights from resume_path
            from ml_logger import ML_Logger
            loader = ML_Logger(root="http://escher.csail.mit.edu:8080", #TODO hardcoded!
                               prefix=self.cfg_ppo.runner.resume_path)
            weights = loader.load_torch("checkpoints/ac_weights_last.pt")
            actor_critic.load_state_dict(state_dict=weights)

            if hasattr(self.env, "curricula") and self.cfg_ppo.runner.resume_curriculum:
                # load curriculum state
                distributions = loader.load_pkl("curriculum/distribution.pkl")
                distribution_last = distributions[-1]["distribution"]
                gait_names = [key[8:] if key.startswith("weights_") else None for key in distribution_last.keys()]
                for gait_id, gait_name in enumerate(self.env.category_names):
                    self.env.curricula[gait_id].weights = distribution_last[f"weights_{gait_name}"]
                    print(gait_name)

        self.alg = PPO(actor_critic, cfg_ppo=self.cfg_ppo, device=self.device)
        self.num_steps_per_env = self.cfg_ppo.runner.num_steps_per_env

        # init storage and model
        self.alg.init_storage(self.env.num_train_envs, self.num_steps_per_env, [self.env.num_policy_obs],
                              [self.env.num_estimator_obs], [self.env.num_privileged_obs], [self.env.num_actions])

        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.last_recording_it = 0

        # Log
        self.run_path = os.path.join(logger.root,logger.prefix)
        self.writer = None

        self.wandb_run = None

        self.env.reset() # also perform first env step

    def learn(self, num_learning_iterations, init_at_random_ep_len=False, eval_freq=100, curriculum_dump_freq=500, eval_expert=False):
        """ (autogen) Train the algorithm for a given number of iterations.

        Args:
            num_learning_iterations (int): The number of learning iterations to run.
            ? init_at_random_ep_len (bool, optional): Whether to initialize episode lengths at random (default: False).
            ? eval_freq (int, optional): The frequency (in iterations) to evaluate the model (default: 100).
            curriculum_dump_freq (int, optional): The frequency (in iterations) to save curriculum information (default: 500).
            eval_expert (bool, optional): Whether to use the expert policy during evaluation (default: False) makes sense only during
                                            teacher student training, else leave at false.

        Returns:
            None#TODO hardcoded!
        """

        from ml_logger import logger
        # initialize writer
        assert logger.prefix, "you will overwrite the entire instrument server"

        logger.start('start', 'epoch', 'episode', 'run', 'step')

        # Wandb init stuff
        run_dir = os.path.basename(os.path.dirname(self.run_path))  # Gets the last directory name
        run_name = os.path.basename(self.run_path)  # Gets the filename or the last segment

        run_full_name = run_dir + "/" + run_name

        # Store the config file
        cfg_dict = cfg_to_dict(self.cfg)
        cfg_dict = filter_arrays_from_dict(cfg_dict) # sanitize the config by removing numpy and  torch arrays

        # store config file as yaml
        config_file_path = os.path.join(self.run_path, "parameters.yaml")
        with open(config_file_path, 'w') as file:
            yaml.dump(cfg_dict, file, sort_keys=False, default_flow_style=False, Dumper=CustomNoAliasDumper)

        if self.cfg.cfg_ppo.runner.wandb_logging:
            self.wandb_run = wandb.init(project = self.cfg.cfg_ppo.runner.wandb_project,
                                        entity = self.cfg.cfg_ppo.runner.wandb_entity,
                                        config = cfg_dict,
                                        dir = os.path.dirname(os.path.dirname(self.run_path)), # change?
                                        name = run_name,
                                        group = run_dir,
                                        notes =self.cfg.cfg_ppo.runner.wandb_note)

            # store config file
            self.wandb_run.save(config_file_path, base_path=self.run_path, policy="now")
            # also store pickle version
            config_file_pickle_path = os.path.join(self.run_path, "parameters.pkl")
            self.wandb_run.save(config_file_pickle_path, base_path=self.run_path, policy="now")



        if init_at_random_ep_len:
            # Initialize the episode length buffer with random values (between 0 and max_episode_length)
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        # split train and test envs
        num_train_envs = self.env.num_train_envs

        # get first observations from the environment
        obs_dict = self.env.get_observations()  # TODO: check, is this correct on the first step?
        policy_obs, privileged_obs, estimator_obs = obs_dict["policy_obs"], obs_dict["privileged_obs"], obs_dict["estimator_obs"]
        policy_obs, privileged_obs, estimator_obs = policy_obs.to(self.device), privileged_obs.to(self.device), estimator_obs.to(
            self.device)
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)

        # define dequeues to track rewards and episode lengths
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        rewbuffer_eval = deque(maxlen=100)
        lenbuffer_eval = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                # Collect actions and perform a step in the environment
                for i in range(self.num_steps_per_env):
                    actions_train = self.alg.act(policy_obs[:num_train_envs], estimator_obs[:num_train_envs], privileged_obs[:num_train_envs])
                    if eval_expert:
                        actions_eval = self.alg.actor_critic.act_teacher(policy_obs[num_train_envs:],
                                                                         privileged_obs[num_train_envs:])
                    else:
                        actions_eval = self.alg.actor_critic.act_student(policy_obs[num_train_envs:], estimator_obs[num_train_envs:])

                    # step gym environment and collect observations
                    obs_dict, rewards, dones, infos = self.env.step(torch.cat((actions_train, actions_eval), dim=0))
                    policy_obs, privileged_obs, estimator_obs = obs_dict["policy_obs"], obs_dict["privileged_obs"], obs_dict["estimator_obs"]


                    # move obsvervations to device. If we train fully on GPU, no need to move
                    policy_obs, privileged_obs, estimator_obs, rewards, dones = policy_obs.to(self.device), privileged_obs.to(
                        self.device), estimator_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards[:num_train_envs], dones[:num_train_envs], infos)

                    # log metrics
                    if 'train/episode' in infos:
                        with logger.Prefix(metrics="train/episode"):
                            logger.store_metrics(**infos['train/episode'])
                            # ep_infos.append(infos['train/episode'])

                    if 'eval/episode' in infos:
                        with logger.Prefix(metrics="eval/episode"):
                            logger.store_metrics(**infos['eval/episode'])
                            # ep_infos.append(infos['eval/episode'])

                    if 'curriculum' in infos:

                        cur_reward_sum += rewards
                        cur_episode_length += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)

                        new_ids_train = new_ids[new_ids < num_train_envs]
                        rewbuffer.extend(cur_reward_sum[new_ids_train].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids_train].cpu().numpy().tolist())
                        cur_reward_sum[new_ids_train] = 0
                        cur_episode_length[new_ids_train] = 0

                        new_ids_eval = new_ids[new_ids >= num_train_envs]
                        rewbuffer_eval.extend(cur_reward_sum[new_ids_eval].cpu().numpy().tolist())
                        lenbuffer_eval.extend(cur_episode_length[new_ids_eval].cpu().numpy().tolist())
                        cur_reward_sum[new_ids_eval] = 0
                        cur_episode_length[new_ids_eval] = 0

                    if 'curriculum/distribution' in infos:
                        distribution = infos['curriculum/distribution']

                stop = time.time()
                collection_time = stop - start
                start = stop

                # Learning step
                self.alg.compute_returns(policy_obs[:num_train_envs], privileged_obs[:num_train_envs])

                if it % curriculum_dump_freq == 0:
                    logger.save_pkl({"iteration": it,
                                     **caches.slot_cache.get_summary(),
                                     **caches.dist_cache.get_summary()},
                                    path=f"curriculum/info.pkl", append=True)

                    if 'curriculum/distribution' in infos:
                        logger.save_pkl({"iteration": it,
                                         "distribution": distribution},
                                         path=f"curriculum/distribution.pkl", append=True)

            mean_value_loss, mean_surrogate_loss, mean_adaptation_module_loss, mean_decoder_loss, mean_decoder_loss_student, mean_adaptation_module_test_loss, mean_decoder_test_loss, mean_decoder_test_loss_student = self.alg.update()
            stop = time.time()
            learn_time = stop - start

            logger.store_metrics(
                # total_time=learn_time - collection_time,
                time_elapsed=logger.since('start'),
                time_iter=logger.split('epoch'),
                adaptation_loss=mean_adaptation_module_loss,
                mean_value_loss=mean_value_loss,
                mean_surrogate_loss=mean_surrogate_loss,
                mean_decoder_loss=mean_decoder_loss,
                mean_decoder_loss_student=mean_decoder_loss_student,
                mean_decoder_test_loss=mean_decoder_test_loss,
                mean_decoder_test_loss_student=mean_decoder_test_loss_student,
                mean_adaptation_module_test_loss=mean_adaptation_module_test_loss,
                learning_rate = self.alg.learning_rate
            )

            if it % self.cfg_ppo.runner.save_curriculum_plot_interval == 0 and self.env.cfg.terrain.curriculum:
                self.log_terrain_curriculum_plot(it)

            if self.cfg_ppo.runner.save_video_interval:
                self.log_video(it)

            self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
            if logger.every(self.cfg_ppo.runner.log_freq, "iteration", start_on=1):
                # if it % Config.log_freq == 0:
                logger.log_metrics_summary(key_values={"timesteps": self.tot_timesteps, "iterations": it})
                logger.job_running()

            if it % self.cfg_ppo.runner.save_interval == 0:
                with logger.Sync():
                    logger.torch_save(self.alg.actor_critic.state_dict(), f"checkpoints/ac_weights_{it:06d}.pt")
                    logger.duplicate(f"checkpoints/ac_weights_{it:06d}.pt", f"checkpoints/ac_weights_last.pt")

                    path = './tmp/legged_data'

                    os.makedirs(path, exist_ok=True)

                    adaptation_module_path = f'{path}/adaptation_module_latest.jit'
                    adaptation_module = copy.deepcopy(self.alg.actor_critic.adaptation_module).to('cpu')
                    traced_script_adaptation_module = torch.jit.script(adaptation_module)
                    traced_script_adaptation_module.save(adaptation_module_path)

                    body_path = f'{path}/body_latest.jit'
                    body_model = copy.deepcopy(self.alg.actor_critic.actor_body).to('cpu')
                    traced_script_body_module = torch.jit.script(body_model)
                    traced_script_body_module.save(body_path)

                    logger.upload_file(file_path=adaptation_module_path, target_path=f"checkpoints/", once=False)
                    logger.upload_file(file_path=body_path, target_path=f"checkpoints/", once=False)


                # upload files to wandb
                if self.wandb_run:
                    self.wandb_run.save(os.path.join(self.run_path, f"checkpoints/ac_weights_{it:06d}.pt"),
                                        base_path=self.run_path, policy="now")

                    self.wandb_run.save(os.path.join(self.run_path, f"checkpoints/ac_weights_last.pt"),
                                        base_path=self.run_path, policy="now")
                    self.wandb_run.save(os.path.join(self.run_path, f"checkpoints/adaptation_module_latest.jit"),
                                        base_path=self.run_path, policy="now")
                    self.wandb_run.save(os.path.join(self.run_path, f"checkpoints/body_latest.jit"),
                                        base_path=self.run_path, policy="now")


            # Also log with Tensorbaord
            if self.wandb_run:
                self.log_wandb(locals())

            self.current_learning_iteration += num_learning_iterations

        with logger.Sync():
            logger.torch_save(self.alg.actor_critic.state_dict(), f"checkpoints/ac_weights_{it:06d}.pt")
            logger.duplicate(f"checkpoints/ac_weights_{it:06d}.pt", f"checkpoints/ac_weights_last.pt")

            path = './tmp/legged_data'

            os.makedirs(path, exist_ok=True)

            adaptation_module_path = f'{path}/adaptation_module_latest.jit'
            adaptation_module = copy.deepcopy(self.alg.actor_critic.adaptation_module).to('cpu')
            traced_script_adaptation_module = torch.jit.script(adaptation_module)
            traced_script_adaptation_module.save(adaptation_module_path)

            body_path = f'{path}/body_latest.jit'
            body_model = copy.deepcopy(self.alg.actor_critic.actor_body).to('cpu')
            traced_script_body_module = torch.jit.script(body_model)
            traced_script_body_module.save(body_path)

            logger.upload_file(file_path=adaptation_module_path, target_path=f"checkpoints/", once=False)
            logger.upload_file(file_path=body_path, target_path=f"checkpoints/", once=False)


        # upload final model to wandb
        if self.wandb_run:
            self.wandb_run.save(os.path.join(self.run_path, f"checkpoints/ac_weights_last.pt"),
                                base_path=self.run_path, policy="now")
            self.wandb_run.save(os.path.join(self.run_path, f"checkpoints/adaptation_module_latest.jit"),
                                base_path=self.run_path, policy="now")
            self.wandb_run.save(os.path.join(self.run_path, f"checkpoints/body_latest.jit"),
                                base_path=self.run_path, policy="now")

            # Cleanup
            wandb.finish()


    def log_wandb(self, locs):

        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']


        self.wandb_run.log({
            "Loss/mean_value_loss": locs['mean_value_loss'],
            "Loss/mean_surrogate_loss": locs['mean_surrogate_loss'],
            "Loss/mean_adaptation_module_loss": locs['mean_adaptation_module_loss'],
            "Loss/mean_adaptation_module_test_loss": locs['mean_adaptation_module_test_loss'],
            "Loss/learning_rate": self.alg.learning_rate,
            "Policy/mean_noise_std": self.alg.actor_critic.std.mean().item(),
            "Perf/total_fps": int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time'])),
            "Perf/collection_time": locs['collection_time'],
            "Perf/learning_time": locs['learn_time'],
            "Perf/iteration_time": iteration_time,
        }, locs['it'])

        # Train statistics (over the whole iteration)
        if len(locs['rewbuffer']) > 0:
            self.wandb_run.log({
                "Train/mean_reward": statistics.mean(locs['rewbuffer']),
                "Train/mean_episode_length": statistics.mean(locs['lenbuffer']),
            }, locs['it'])
            self.wandb_run.log({
                "Train/mean_reward/time": statistics.mean(locs['rewbuffer']),
                "Train/mean_episode_length/time": statistics.mean(locs['lenbuffer']),
            }, self.tot_time)

        # Rewards (mean of reward sums over whole episode, logged at end of each episode)
        if 'train/episode' in locs["infos"]:
            for key in locs["infos"]['train/episode']:
                if "rew" in key:
                    self.wandb_run.log({"EpisodeRew/" + key: locs["infos"]['train/episode'][key]}, locs['it'])
                else:
                    self.wandb_run.log({"EpisodeMisc/" + key: locs["infos"]['train/episode'][key]}, locs['it'])



    def log_tensorboard(self, locs):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        # ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                # ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

    def log_video(self, it):

        if it - self.last_recording_it >= self.cfg_ppo.runner.save_video_interval:
            self.env.start_recording()
            if self.env.num_eval_envs > 0:
                self.env.start_recording_eval()
            print("START RECORDING")
            self.last_recording_it = it

        frames = self.env.get_complete_frames()
        if len(frames) > 0:
            self.env.pause_recording()
            print("LOGGING VIDEO")
            # record at half the framerate
            logger.save_video(frames[::2], f"videos/{it:05d}.mp4", fps=1/self.env.dt/2)

            # Log the video with a specific iteration tag
            if self.wandb_run:
                self.wandb_run.log({"training_videos": wandb.Video(os.path.join(self.run_path, f"videos/{it:05d}.mp4"),
                                                                   fps=1/self.env.dt/2, format="mp4")}, step=it)

        if self.env.num_eval_envs > 0:
            frames = self.env.get_complete_frames_eval()
            if len(frames) > 0:
                self.env.pause_recording_eval()
                print("LOGGING EVAL VIDEO")
                logger.save_video(frames, f"videos/{it:05d}_eval.mp4", fps=1 / self.env.dt)

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_expert_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_expert


    def log_terrain_curriculum_plot(self, it):


        terrain_types_np = self.env.terrain_types.cpu().numpy()
        levels_np = self.env.terrain_levels.cpu().numpy()

        # Create a 2D histogram (density plot)
        plt.figure(figsize=(4, 4))
        plt.hist2d(levels_np, terrain_types_np, bins=[self.env.cfg.terrain.max_terrain_level,self.env.cfg.terrain.num_cols],
                   range=[[0,self.env.cfg.terrain.max_terrain_level],[0,self.env.cfg.terrain.num_cols]], cmap='Blues')

        # Add colorbar to show the density scale
        plt.colorbar(label='Density')

        # Labeling the axes
        plt.xlabel('Levels')
        plt.ylabel('Terrain Types')


        image_path = os.path.join(self.run_path, f"figures/terrain_curriculum_{it:05d}.jpg")
        if not os.path.exists(f"{self.run_path}/figures/"):
            os.makedirs(f"{self.run_path}/figures/")
        plt.savefig(image_path)

        if self.wandb_run:
            self.wandb_run.log({"terrain_curriculum": wandb.Image(image_path)}, step=it)