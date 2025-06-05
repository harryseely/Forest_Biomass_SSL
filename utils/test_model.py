import pytorch_lightning as pl
import pandas as pd
from pathlib import Path
import numpy as np
import sklearn.metrics as metrics
from math import sqrt
from statistics import mean

from utils.dataset import PointCloudDataModule
from models.ocnn_lightning import OCNNLightning


def test_model(cfg, device_id, logger, save_dir):
    """
    Test the model on the test set and log the results
    :param cfg: dictionary containing model configuration
    :param device_id: integer indicating GPU device to use (e.g., 0)
    :param logger: pytorch lightning logger object
    :param save_dir: directory where config and model checkpoint are stored
    :return:
    """

    # If you do not have a trainer and model available, provide the directory where config and model checkpoint are stored
    ckpt_files = list(Path(save_dir).glob("*" + 'ckpt'))
    if len(ckpt_files) > 1:
        raise ValueError("Multiple checkpoint files found in the directory. Please provide a single checkpoint file.")
    else:
        ckpt_file = ckpt_files[0]

    # Rebuild model for testing uprojesing correct config
    model = OCNNLightning(cfg)

    # Set up data module with config that model was trained with
    data_module = PointCloudDataModule(cfg=cfg)
    
    pred_df = model_predict(model, data_module, ckpt_file, device_id, cfg['biomass_comp_nms'])

    # Calculate model performance metrics and log
    comp_nms_w_agb = cfg['biomass_comp_nms'] + ['total_agb']
    metrics_df = calc_metrics(pred_df, comp_nms_w_agb)

    # Log metrics to wandb
    if logger:

        # Record the number of parameters and the test dataset size
        n_parameters = sum(p.numel() for p in model.parameters())
        logger.experiment.config.update(d={"n_parameters": n_parameters,
                               "n_test_samples": len(data_module.test_dataset.pc_fnames)
                               }, allow_val_change=True)

        # Convert the df into a dict for logging
        metrics_dict = metrics_df.to_dict()

        metrics_dict_full = {}

        # Rename keys
        for metric in metrics_dict.keys():
            for comp in metrics_dict[metric].keys():
                metrics_dict_full[f"{comp}_{metric}"] = metrics_dict[metric][comp]

        # Save performance metrics to wandb
        logger.experiment.config.update(d=metrics_dict_full, allow_val_change=True)

    return cfg

def model_predict(model, data_module, ckpt_file, device_id, biomass_comp_nms):

    # Instantiate pl trainer object
    trainer = pl.Trainer(accelerator="gpu",
                         devices=[device_id],
                         logger=False,
                         enable_checkpointing=False,
                         enable_progress_bar=False)

    # Generate predictions
    out = trainer.predict(model, datamodule=data_module, ckpt_path=str(ckpt_file), return_predictions=True)

    # Separate predictions and targets from output
    pred = np.concatenate([batch[0] for batch in out])
    target = np.concatenate([batch[1] for batch in out])

    # Combine preds and targets into single array
    pred_target_arr = np.concatenate([pred, target], axis=1)

    # Generate df from obs and pred with foliage, bark, branch, and wood biomass
    col_names = [f'{comp}_pred' for comp in biomass_comp_nms] + [f'{comp}_obs' for comp in biomass_comp_nms]
    df = pd.DataFrame(pred_target_arr, columns=col_names)

    #Insert plot ID from test dataset
    df['PlotID'] = data_module.test_dataset.PlotID_ls

    # Generate total AGB columns
    df['total_agb_obs'] = df['foliage_obs'] + df['bark_obs'] + df['branch_obs'] + df['wood_obs']
    df['total_agb_pred'] = df['foliage_pred'] + df['bark_pred'] + df['branch_pred'] + df['wood_pred']

    return df


def calc_bias(y_true, y_pred) -> float:
    """
    Calculate the bias between observed and predicted values

    :param y_true:
    :param y_pred:
    :return: numeric value indicating the bias
    """

    bias = mean(y_true - y_pred)

    return bias


def calc_metrics(df, biomass_comp_nms):
    """
    Calculate model performance metrics for each biomass component
    :param df:
    :param biomass_comp_nms:
    :return: df with model performance metrics
    """

    # Create a data frame to store component metrics
    metrics_df = pd.DataFrame(columns=["r2", "rmse", "bias"],
                              index=biomass_comp_nms)

    comp_list = metrics_df.index.tolist()

    # Loop through each biomass component get model performance metrics
    for comp in comp_list:
        # R2
        metrics_df.loc[comp, "r2"] = round(metrics.r2_score(y_true=df[f"{comp}_obs"], y_pred=df[f"{comp}_pred"]), 4)

        # RMSE
        metrics_df.loc[comp, "rmse"] = round(
            sqrt(metrics.mean_squared_error(y_true=df[f"{comp}_obs"], y_pred=df[f"{comp}_pred"])), 4)

        # Bias
        metrics_df.loc[comp, "bias"] = calc_bias(y_true=df[f"{comp}_obs"], y_pred=df[f"{comp}_pred"])

    print(metrics_df)

    return metrics_df

