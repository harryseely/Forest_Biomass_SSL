import os
from datetime import datetime as dt
import random
import string
from pathlib import Path
import yaml
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def is_slurm_cluster():
    slurm_vars = ['SLURM_JOB_ID', 'SLURM_NODELIST', 'SLURM_CPUS_ON_NODE']
    return any(var in os.environ for var in slurm_vars)

def name_run(cfg):

    #Name run locally, or using global random ID and datetime if on SLURM
    if is_slurm_cluster():
        t_now = str(os.environ['DATETIME'])
        run_id = str(os.environ['SLURM_JOB_ID'])
    else:
        t_now = dt.now().strftime('%Y_%m_%d_%H_%M_%S')
        run_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))

    run_name = f"{cfg['task']}_{cfg['model_nm']}_{t_now}_id{run_id}"

    return run_name

def create_run_dir(cfg, ckpt_root_dir):

    # Set directory to save checkpoints and config file
    save_dir = Path(os.path.join(ckpt_root_dir, cfg['project_nm'], cfg['run_name']))
    cfg_fpath = os.path.join(save_dir, f"config.yaml")

    # Create the run directory for checkpointing and logging
    try: 
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
    except FileExistsError:
        print(f"Directory {save_dir} already exists.")
    
    # Create the config yaml file for the run
    try: 
        if not os.path.exists(cfg_fpath):
            with open(cfg_fpath, 'x') as file:
                yaml.dump(cfg, file)

    except FileExistsError:
        print(f"Config file {cfg_fpath} already exists.")
    
    return save_dir



def set_up_logging_and_checkpoints(cfg, ckpt_root_dir):

    #Only create the dir on one node and one process
    save_dir = create_run_dir(cfg, ckpt_root_dir)

    if is_slurm_cluster():
        logger = CSVLogger(save_dir=save_dir,
                           name=cfg['run_name'],
                            )

    else:
        logger = WandbLogger(
            save_dir=str(save_dir),
            config=cfg,
            id=cfg['run_name'],
            project=cfg['project_nm'],
            allow_val_change=True,
            job_type='training',
            resume="allow",
            mode="online",
            log_model=False,
        )

    return logger, save_dir


def get_pl_callbacks(cfg, model_ckpt, save_dir):
    callback_list = list()

    if model_ckpt:
        checkpoint_callback = ModelCheckpoint(dirpath=save_dir,
                                              filename='{epoch}-{val_loss:.2f}-{val_r2:.2f}',
                                              monitor="val_loss",
                                              mode="min",
                                              save_top_k=cfg['n_checkpoints'],
                                              every_n_epochs=cfg['ckpt_freq_epochs'],
                                              verbose=cfg['verbose']
                                              )

        # Add to callback list
        callback_list.append(checkpoint_callback)

    if cfg['monitor_lr'] & (save_dir is not None):
        # Set up lr logging
        callback_list.append(LearningRateMonitor(logging_interval='step'))

    if cfg['early_stopping']:
        # Add early stopping to callback list
        callback_list.append(EarlyStopping(monitor="val_loss",
                                           mode="min",
                                           min_delta=cfg['early_stop_min_delta'],
                                           patience=cfg['patience'],
                                           check_finite=True,
                                           check_on_train_epoch_end=False))


    # Set callback list to None if it contains zero elements
    if len(callback_list) == 0:
        callback_list = None

    return callback_list

def check_matching_dict_keys(og_dict, update_dict):
        
        # Ensure all keys specified in input config are in the base config YAML file
        og_keys = og_dict.keys()
        update_keys = update_dict.keys()

        # Check to see which keys are missing from the base config
        missing_keys = set(update_keys) - set(og_keys)

        assert set(update_keys).issubset(
            og_keys), f"The following key(s) not in applicable to cfg: {missing_keys}"

        # Update dict with new values
        og_dict.update(update_dict)

        return og_dict
