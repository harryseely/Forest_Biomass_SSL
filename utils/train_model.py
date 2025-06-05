import os
from pprint import pprint
import pytorch_lightning as pl
import torch
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.plugins.environments import SLURMEnvironment
from torch.cuda import device_count
import yaml
import time
import wandb

from utils.dataset import PointCloudDataModule
from utils.test_model import test_model
from utils.training_utils import get_pl_callbacks, set_up_logging_and_checkpoints, name_run, is_slurm_cluster, check_matching_dict_keys
from utils.get_model import get_model
import warnings

warnings.simplefilter("ignore", category=FutureWarning)


def train_fn(tune_cfg, static_cfg=None, apply_to_test_set=True):
    """
    Executes DNN training on 1 GPU or using DDp on multiple GPUs.
    Function is set up to be run standalone or within Ray Tune as a trainable.

    :param tune_cfg: config dict used for training. Updated for each trial when using Ray Tune.
    :param static_cfg: config dict used for training. Only used with Ray Tune to hold config that is not tuned.
    :param apply_to_test_set: bool to determine if model is applied to test dataset after training.
    :return:
    """

    #Determine if using SLURM cluster
    ON_SLURM = is_slurm_cluster()

    #Set the root GPU index to use for logging and single GPU training/testing
    ROOT_GPU_IDX = 0

    #Set matrix multiplication precision
    #https://pytorch.org/docs/2.5/generated/torch.set_float32_matmul_precision.html#torch-set-float32-matmul-precision
    torch.set_float32_matmul_precision('high')

    # Update config with tuning config (needed if using function with ray tune)
    if static_cfg is not None:

        # Ensure all keys specified in input config are in the base config YAML file
        cfg = check_matching_dict_keys(og_dict=static_cfg, update_dict=tune_cfg)

    else:

        cfg = tune_cfg

    # Only print on main device
    if cfg['verbose']:
        print("\n\033[92mTraining with the following config:\033[0m\n")
        pprint(cfg)
    
    # Whether to load lightning checkpoint state for resuming training
    if cfg['resume_training']:
        resume_ckpt_fpath = cfg['model_ckpt_fpath']
        print(f"\033[92m\nResuming training with training state and checkpoint saved at:\n'{resume_ckpt_fpath}'.\033[0m")
        #Load the run name from the checkpoint dir
        cfg['run_name'] = os.path.basename(os.path.dirname(resume_ckpt_fpath))
    else:
        resume_ckpt_fpath = None
        cfg['run_name'] = name_run(cfg)

    # Add job ID to config if on SLURM
    cfg['job_id'] = os.environ['SLURM_JOB_ID'] if ON_SLURM else None

    if cfg['logging']:
        logger, save_dir = set_up_logging_and_checkpoints(cfg, ckpt_root_dir="checkpoints_and_config")
        model_checkpointing = True
    else:
        logger = False
        save_dir = None
        model_checkpointing = False

    #Add save dir to config as string (for yaml export)
    cfg['save_dir'] = str(save_dir)

    #Set some SLURM-related config
    n_nodes = int(os.environ['N_NODES']) if ON_SLURM else 1
    n_gpus  = int(os.environ['N_GPUS']) if ON_SLURM else device_count()
    bit_precision = "32-true" if ON_SLURM else "16-mixed"
    use_pbar = False if n_nodes > 1 else True

    if cfg['ddp']:
        
        backend = "nccl" if ON_SLURM else "gloo"

        #Using gradient_as_bucket_view=True following PL docs
        #https://lightning.ai/docs/pytorch/stable/advanced/ddp_optimizations.html#gradient-as-bucket-view
        gpu_strategy = DDPStrategy(process_group_backend=backend, 
                                   find_unused_parameters=False,
                                   gradient_as_bucket_view=True)
    else:
        n_gpus = 1
        gpu_strategy = "auto"

    # Get list of lightning callbacks
    callbacks_ls = get_pl_callbacks(cfg, model_ckpt=model_checkpointing, save_dir=save_dir)

    # Get lightning plugin list
    plugins_ls = [SLURMEnvironment(auto_requeue=False)] if ON_SLURM else None

    #Instatiate trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=n_nodes,
        strategy=gpu_strategy,
        logger=logger,
        default_root_dir=save_dir,
        max_epochs=cfg['n_epochs'],
        enable_checkpointing=model_checkpointing,
        callbacks=callbacks_ls,
        enable_progress_bar=use_pbar,
        precision=bit_precision,
        plugins=plugins_ls,
    )

    model = get_model(cfg)

    # Set up data module
    data_module = PointCloudDataModule(cfg)

    #Start recording time
    t0 = time.time()

    # Train the model
    trainer.fit(model, datamodule=data_module, ckpt_path=resume_ckpt_fpath)

    #Log train and val sizes on the root GPU
    if logger and not ON_SLURM:
        try: 
            #todo: deal with wandb bug where this code works with 1 GPU but fails when using DDP
            logger.experiment.config.update({
                "save_dir": save_dir,
                "n_train_samples": len(data_module.train_dataset),
                "n_val_samples": len(data_module.val_dataset)
            }, allow_val_change=True)
        
        except Exception as e:
            print(e, "\n")
            print("Unable to log save dir, n_train_samples, n_val_samples")
                    
    # Test model on root GPU (may need to modify)
    if apply_to_test_set and (torch.cuda.current_device() == ROOT_GPU_IDX) and (cfg['task'] != "pretrain") and logger:
        print(f"\n\033[95mTesting {os.path.basename(save_dir)} model on GPU Rank: {ROOT_GPU_IDX}\033[0m\n")

        test_out_dict = test_model(
                                cfg=cfg,
                                device_id=ROOT_GPU_IDX,
                                logger=logger,
                                save_dir=save_dir)
    else:
        test_out_dict = {}
    

    if logger and torch.cuda.current_device() == ROOT_GPU_IDX:

        #Update run config with test results
        cfg.update(test_out_dict)

        #Export final version of config
        with open(os.path.join(save_dir, f"config.yaml"), 'w') as file:
            yaml.dump(cfg, file, sort_keys=False, default_flow_style=False)

        # Stop logging
        wandb.finish()

        print(f"\033[92mTraining logs and config saved to: {save_dir}\033[0m")


    # Ensure DDP is stopped
    if cfg['ddp']:
        torch.distributed.destroy_process_group()
    
    #End recording time and calcute runtime in hours
    t1 = time.time()
    cfg['runtime_hours'] = round((t1 - t0) / 3600, 3)

    print(f"\nTraining completed with a runtime of {cfg['runtime_hours']} hours.\n")

if __name__ == "__main__":
    with open("config.yaml", "r") as yamlfile:
        cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)

    #Run logistics
    cfg['model_nm'] = "ocnn_lenet"
    cfg['task'] = "scratch"
    cfg['project_nm'] = "misc"
    cfg['logging'] = False
    cfg['ddp'] = False
    cfg['n_train_val_samples'] = "all"
    cfg['model_ckpt_fpath'] = None
    cfg['resume_training'] = False


    #Hyperparameters
    cfg['n_epochs'] = 2
    cfg['batch_size'] = 16

    train_fn(cfg)

