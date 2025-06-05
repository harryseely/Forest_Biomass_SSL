# NOTE: need to run in terminal from root dir as:
# uv run -m B_analysis.s6_effect_of_train_dataset_size

from ray import tune
import torch
import yaml
from pprint import pprint
from datetime import datetime as dt

from utils.train_parallel import ray_parallel
from utils.training_utils import check_matching_dict_keys


#Define function to set batch size based on number of training samples

def set_batch_size(n_train_val_samples, default_batch_size):
    """
    Set the batch size based on the number of training samples.
    If sufficient samples, use the default batch size.

    :param n_train_val_samples: Number of training samples
    :param batch_size: Default batch size
    """

    if n_train_val_samples <= 10:
        batch_size = 2
    elif n_train_val_samples <= 30:
        batch_size = 4
    else:
        batch_size = default_batch_size
    
    print(f"\nGiven {n_train_val_samples} training samples, using batch size of {batch_size}.\n")

    return batch_size


if __name__ == "__main__":

    TARGET_MODEL = "ocnn_lenet"

    DT = str(dt.now().strftime('%Y_%m_%d_%H_%M_%S'))

    #Load all config
    with open("config.yaml", "r") as yamlfile:
        cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
    
    #Load optimal model HPs
    with open(f"optimal_hyperparameters/{TARGET_MODEL}_hps.yaml", "r") as yamlfile:
        optimal_hps = yaml.load(yamlfile, Loader=yaml.FullLoader)
    
    #Update cfg with optimal HPs
    cfg = check_matching_dict_keys(og_dict=cfg, update_dict=optimal_hps)

    #Set the model checkpoint path to use for finetuning
    model_ckpt_fpath = cfg['model_ckpt_fpath']

    # Run logistics
    cfg['model_nm'] = TARGET_MODEL
    cfg['project_nm'] = f"rq3_effect_of_train_dataset_size_{TARGET_MODEL}"
    cfg['early_stopping'] = True

    # Use a higher dropout rate 
    cfg['dropout'] = 0.5
    print(f"\nUsing dropout rate of {cfg['dropout']} for {TARGET_MODEL} to limit overfitting for small datasets\n")

    search_dict = {

        "task": tune.grid_search(["finetune", "scratch"]),

        "cv_fold_num": tune.grid_search([1, 2, 3, 4, 5]),

        "n_train_val_samples": tune.grid_search(range(10, 180, 10)),

        "batch_size": tune.sample_from(lambda x: set_batch_size(x.config.n_train_val_samples, cfg['batch_size'])),

        "model_ckpt_fpath": tune.sample_from(lambda x: None if x.config.task == "scratch" else model_ckpt_fpath),

    }

    print(f"\nConfig:")
    pprint(cfg)
    print("\n")

    # Implement the parallel model training using Ray Tune
    ray_parallel(search_space=search_dict,
                 static_cfg=cfg,
                 resources_per_trial={"cpu": 1, "gpu": 1},
                 n_concurrent_trials=torch.cuda.device_count(),
                 time_budget_s=60 * 60 * 24 * 7,
                 )