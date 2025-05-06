# NOTE: need to run in terminal from root dir as:
# uv run -m B_analysis.s3_hp_tuning

from ray import tune
import torch
import yaml
from datetime import datetime as dt
from pprint import pprint

from utils.train_parallel import ray_parallel


if __name__ == "__main__":

    TARGET_MODEL = 'ocnn_lenet'

    DT = str(dt.now().strftime('%Y_%m_%d_%H_%M_%S'))

    # Load all config
    with open("config.yaml", "r") as yamlfile:
        cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)

    #Training logistics
    cfg['model_nm'] = TARGET_MODEL
    cfg['project_nm'] = f"rq3_hp_tuning_{TARGET_MODEL}"
    cfg['task'] = "scratch"

    # Run logistics
    cfg['ddp'] = False
    cfg['cv_fold_num'] = 1

    # Hyperparameters
    cfg['early_stopping'] = True
    cfg['weight_decay'] = 0.01
    cfg['dropout'] = 0.0

    # Set up search grid
    search_dict = {
        'lr': tune.choice([0.1, 0.01, 0.001, 0.0001, 0.00001]),
        'batch_size': tune.choice([8, 16, 32, 64, 128]),
        'n_epochs': tune.choice([100, 200, 300]),
        'dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
    }

    print(f"\nConfig:")
    pprint(cfg)
    print("\n")

    # Implement the parallel model training using Ray Tune
    ray_parallel(search_space=search_dict,
                 static_cfg=cfg,
                 apply_to_test_set=False,
                 resources_per_trial={"cpu": 1, "gpu": 1},
                 n_concurrent_trials=torch.cuda.device_count(),
                 time_budget_s=60 * 60 * 24 * 7,
                 num_samples=-1
                 )