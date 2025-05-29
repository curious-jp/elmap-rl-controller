"""
This script converts the weights of a run to a JIT format.

"""

import os
import argparse

import copy
import torch

from params_proto.proto import Meta

from robodog_gym.envs.base.legged_robot_config import Cfg
from robodog_gym_learn.ppo_cse.actor_critic import ActorCritic


LEGACY_PARAMS = False


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
                if hasattr(subcfg, key2) and isinstance(getattr(subcfg, key2),Meta): # check if it is another nested class
                    for key3, value3 in cfg_dict[key][key2].items():
                        setattr(getattr(subcfg, key2), key3, value3)
                else: # or if it is just an attribut
                    setattr(subcfg, key2, value2)

    return Cfg_class


def convert_weights_to_jit_cpu(runs_dir, run_group, run_name, iteration):
    """
    Convert the weights of a run to a JIT format.

    The run files should be found at <runs_dir>/<group>/<name>

    Args:
        runs_dir (str): Directory where runs are stored.
        run_group (str): Group folder of the experiment.
        run_name (str): Run name of the experiment
        iteration (str): Iteration of the model to convert to JIT. Either number padded with zeros to 7 polaces, like "035200", "last" for only the last model 
    """

    model_folder = "checkpoints"  # folder where models are saved
    params_filename = "parameters.yaml"
    legacy_params_filename = "parameters.pkl"


    # check that run exists at specified location <runs_dir>/<run_group>/<run_name>
    run_path = os.path.join(runs_dir, run_group, run_name)
    if not os.path.exists(run_path):
        raise Exception(f"Run {run_path} does not exist.")

    
    # ---------------------------------
    # Load the configuration parameters
    # ---------------------------------

    if LEGACY_PARAMS:
        # Lecacy code using pickle parameter file
        # if want to unpickle the parameters.pkl file on a GPU-less machine, you need to use a customm upickler
        # https://github.com/pytorch/pytorch/issues/16797
        import pickle
        import io
        class CPU_Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'torch.storage' and name == '_load_from_bytes':
                    return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
                else: return super().find_class(module, name)
        with open(os.path.join(run_path,legacy_params_filename), 'rb') as file:
            pkl_cfg = CPU_Unpickler(file).load()
        cfg_dict = pkl_cfg["Cfg"]

    else:
        import yaml
        with open(os.path.join(run_path,params_filename), 'r') as file:
            cfg_dict = yaml.safe_load(file)

    cfg = load_config(cfg_dict,Cfg)  # Load config parameters into Cfg class
    

    # ----------------------------------------
    # Extract relevant parameters from config
    # ----------------------------------------

    # compute policy and estimator observation components
    num_policy_observations = sum([comp[1] for comp in cfg.env.policy_observation_components if comp[2]])
    num_estimator_observations = sum([comp[1] for comp in cfg.env.estimator_observation_components if comp[2]])
    num_privileged_observations = sum([comp[1] for comp in cfg.env.privileged_observation_components if comp[2]])
    num_actions = cfg.env.num_actions

    # compute effective policy history length
    if cfg.env.sparse_obs_history is not None:
        num_policy_obs_history = len(cfg.env.sparse_obs_history)
    else:
        num_policy_obs_history = cfg.env.num_observation_history

    # compute effective history length, if different from policy
    if cfg.env.sparse_estimator_obs_history is None and cfg.env.num_estimator_obs_history is None:
        # just use the same history length as the policy
        num_estimator_obs_history = num_policy_obs_history
    else:
        if cfg.env.sparse_estimator_obs_history is not None:
            num_estimator_obs_history = len(cfg.env.sparse_estimator_obs_history)
        else:
            num_estimator_obs_history = cfg.env.num_estimator_obs_history
    

    # compute effective observation sizes (accounting for history)
    num_policy_observations_eff = num_policy_observations * num_policy_obs_history
    num_estimator_observations_eff = num_estimator_observations * num_estimator_obs_history

    
    print(f"num_policy_obs: {num_policy_observations}")
    print(f"num_estimator_observations: {num_estimator_observations}")
    print(f"num_privileged_obs: {num_privileged_observations}")
    print(f"num_policy_obs_history: {num_policy_obs_history}")
    print(f"num_estimator_obs_history: {num_estimator_obs_history}")
    print(f"num_policy_obs_eff: {num_policy_observations_eff}")
    print(f"num_estimator_obs_eff: {num_estimator_observations_eff}")
    

    if hasattr(cfg, 'cfg_ppo'):
        cfg_ppo = cfg.cfg_ppo
    else:
        cfg_ppo = None

    # ---------------------------------
    # Load the model and convert to JIT
    # ---------------------------------

    actor_critic = ActorCritic(num_policy_observations_eff,
                               num_estimator_observations_eff,
                               num_privileged_observations,
                               num_actions,
                               cfg_ppo=cfg_ppo,
                               ).to(device='cpu')
    

    weights = torch.load(f"{run_path}/{model_folder}/ac_weights_{iteration}.pt", map_location='cpu')
    actor_critic.load_state_dict(state_dict=weights)

    adaptation_module = copy.deepcopy(actor_critic.adaptation_module).to('cpu')
    traced_script_adaptation_module = torch.jit.script(adaptation_module)
    traced_script_adaptation_module.save(f"{run_path}/{model_folder}/adaptation_module_{iteration}.jit")

    body_model = copy.deepcopy(actor_critic.actor_body).to('cpu')
    traced_script_body_module = torch.jit.script(body_model)
    traced_script_body_module.save(f"{run_path}/{model_folder}/body_{iteration}.jit")

    print("Conversion finished.")


if __name__ == "__main__":
        
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Convert model to JIT format.')
    parser.add_argument('--runs_dir', default = "../runs/" ,type=str, help='Directory where runs are stored')
    parser.add_argument('--run_label', default = "test/2024-07-05_14-01-16.171329", type=str, help='Run group and name of the experiment, informat <group>/<name>.')
    parser.add_argument('--iter', default= "last", type=str, help='Iteration of the model to convert to jit. Either number padded with zeros to 7 polaces, like "035200", "last" for only the last model ')

    args = parser.parse_args()

    # split run group and name
    try:   
        run_group, run_name = args.run_label.split("/")
    except:
        raise Exception("Run name must be in the format <group>/<name>")
    
    
    convert_weights_to_jit_cpu(args.runs_dir, run_group, run_name, args.iter)