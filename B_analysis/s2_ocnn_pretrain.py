# NOTE: need to run in terminal from root dir as:
# uv run -m B_analysis.s2_ocnn_pretrain

import os
import yaml
from pprint import pprint

from utils.train_model import train_fn


if __name__ == "__main__":

    TARGET_MODEL = 'ocnn_lenet'

    #Read base config
    with open("config.yaml", "r") as yamlfile:
        cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)

    #Training logistics
    cfg['model_nm'] = TARGET_MODEL
    cfg['project_nm'] = f"rq3_pretrain_{TARGET_MODEL}"
    cfg['task'] = 'pretrain'
    cfg['early_stopping'] = False
    cfg['ddp'] = False

    #Hyperparameters
    cfg['n_epochs'] = 100
    cfg['batch_size'] = 128
    cfg['lr'] = 0.0001
    cfg['dropout'] = 0.5
    cfg['weight_decay'] = 0.05

    #Checkpointing
    cfg['ckpt_freq_epochs'] = 5
    cfg['n_checkpoints'] = -1  # -1 means keep all checkpoints (if the model has improved)

    print(f"\nConfig:")
    pprint(cfg)
    print("\n")

    train_fn(cfg)
