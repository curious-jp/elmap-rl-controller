"""
This script download a run from wandb and saves it in the runs folder.
"""

import os
import argparse
import yaml
import wandb


def find_runs_by_name(wandb_entity, wandb_project, run_name):
    """
    Retrieve all runs with the specified name from a wandb project.

    Args:
        wandb_entity (str): The entity under which the project is hosted.
        wandb_project (str): The project name.
        run_name (str): The name of the run to search for.

    Returns:
        list: A list of runs that match the given name.
    """
    # Initialize the wandb API
    wandb_api = wandb.Api()

    # Retrieve all runs from the specified project and entity
    project_path = f"{wandb_entity}/{wandb_project}"
    runs = wandb_api.runs(project_path)

    # Filter runs by name
    matching_runs = [run for run in runs if run.name == run_name]

    if not matching_runs:
        print(f"No runs found with the name '{run_name}'.")
    else:
        print(f"Found {len(matching_runs)} run(s) with the name '{run_name}':")
        for run in matching_runs:
            print(f"- Run ID: {run.id}, Name: {run.name}")

    return matching_runs


def download_run(runs_dir, run_group, run_name, iteration, videos, metrics, wandb_project, wandb_entity):
    """
    Download a run from wandb and save it in the runs folder.

    The run is saved in the runs folder with the following structure:
    runs_dir/
        run_group/
            run_name/
                checkpoints/
                    ac_weights_<iteration>.pt
                media/videos/
                    <iteration>.mp4 #filenmae may change
                parameters.yaml

    Args:
        runs_dir (str): Directory where the runs are saved (relative or absolute)
        run_group (str): Run group, used to organize runs. Does not have to match the group on wandb 
        run_name (str): Wandb run name     
        iteration (str): Either number padded with zeros to 7 polaces, like "035200", "last" for only the last model or "all" to download all checkpoints
        videos (bool): Download mp4 videos
        metrics (bool): Download metrics as CSV file
        wandb_project (str): Wandb project name
        wandb_entity (str): Wandb entity name
    """


    model_folder = "checkpoints"  # folder where models are saved
    video_folder = "videos"
    params_filename = "parameters.yaml"
    legacy_params_filename = "parameters.pkl"

    # Find runs with the specified name
    runs = find_runs_by_name(wandb_entity, wandb_project, run_name)
    if not runs:
        raise Exception(f"No runs found with the name '{run_name}'.")
    if len(runs) > 1:
        raise Exception(f"Multiple runs found with the name '{run_name}'. Please specify a unique run name.")

    # Get the ID of the first run
    run_id = runs[0].id

    # Initialize wandb API
    wandb_api = wandb.Api()

    # Build the wandb run ID
    run_full_id = f"{wandb_entity}/{wandb_project}/{run_id}"

    # Attempt to access the run
    run = wandb_api.run(run_full_id)

    # Construct the full run path
    run_path = os.path.join(runs_dir, run_group, run_name)
    os.makedirs(run_path, exist_ok=True)

    # Download parameters.yaml and legacy parameters.pkl
    run.file(params_filename).download(root=run_path, replace=True)
    run.file(legacy_params_filename).download(root=run_path, replace=True)

    if iteration == "all":
        # Download all model checkpoints
        for file in run.files(per_page=100):
            if model_folder in file.name and file.name.endswith('.pt'):
                file.download(root=run_path, replace=True)
    else:
        # Download specific or latest checkpoint
        checkpoint_file = f"ac_weights_{iteration}.pt" if iteration != "last" else "ac_weights_last.pt"
        run.file(f"{model_folder}/{checkpoint_file}").download(root=run_path, replace=True)

    if videos:
        # video_path = os.path.join(run_path, video_folder)
        # os.makedirs(video_path, exist_ok=True)
        for file in run.files(per_page=100):
            if video_folder in file.name and file.name.endswith('.mp4'):
                file.download(root=run_path, replace=True)

    if metrics:
        # download metrics as csv
        metrics_dataframe = run.history()
        metrics_dataframe.to_csv(os.path.join(run_path,"metrics.csv"))


    

if __name__ == "__main__":

    #*******************************************************************************
    # Set up argument parsing
    #*******************************************************************************

    parser = argparse.ArgumentParser(description='Download WandB run.')
    parser.add_argument('--runs_dir', default = "../runs/" ,type=str, help='Directory where runs are stored')
    # parser.add_argument('--run_group', default = "test", type=str, help='Group of the experiment')
    parser.add_argument('--run_label', default = "test/2024-07-05_14-01-16.171329", type=str, help='Run group and name of the experiment, informat <group>/<name>. Specified group can be different from wandb group')
    parser.add_argument('--iter', default= "last", type=str, help='Either number padded with zeros to 7 polaces, like "035200", "last" for only the last model or "all" to download all checkpoints')
    parser.add_argument('--videos', action="store_true", help='Download mp4 videos')
    parser.add_argument('--metrics', action="store_true", help='Download metrics as CSV file')

    parser.add_argument('--wandb_project', default= 'my-wandb-project', type=str, help='Wandb project name')
    parser.add_argument('--wandb_entity', default= 'my-wandb-entity', type=str, help='Wandb entity name')

    args = parser.parse_args()

    # split run group and name
    try:   
        run_group, run_name = args.run_label.split("/")
    except:
        raise Exception("Run name must be in the format <group>/<name>")

    download_run(args.runs_dir, run_group, run_name, args.iter, args.videos, args.metrics, args.wandb_project, args.wandb_entity)
