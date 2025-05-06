import pandas as pd
import yaml
from tqdm.auto import tqdm
import os

def drop_runs(df, n_runs_per):
    """
    Drop runs from a dataframe so that each model-task-cv_fold_num combination has the same number of runs.
    
    Args:
    df: pd.DataFrame
    n_runs_per: int, number of runs to keep per model-task-cv_fold_num combination

    Returns:
    df: pd.DataFrame, dataframe with n_runs_per runs per model-task-cv_fold_num combination
    """

    df_sub_ls = []

    for model_nm in df['model_nm'].unique():

        for task in df['task'].unique():

            for cv_fold_num in df['cv_fold_num'].unique():

                for n_train_val in df['n_train_val_samples'].unique():

                    df_sub = df[(df['model_nm'] == model_nm) & (df['task'] == task) & (df['cv_fold_num'] == cv_fold_num) & (df['n_train_val_samples'] == n_train_val)]

                    n_runs = df_sub.shape[0]

                    if n_runs > n_runs_per:

                        df_sub = df_sub.sample(n=n_runs_per)

                    assert df_sub.shape[0] == n_runs_per, f"N runs per model-task-cv_fold_num not equal to {n_runs_per}"

                    df_sub_ls.append(df_sub)

    # Stitch back into one df
    df = pd.concat(df_sub_ls).reset_index(drop=True)

    #Summarize to ensure all have the same number of runs
    print(df.groupby(["model_nm", "task", "cv_fold_num", "n_train_val_samples"]).size())
    print(f"Number of runs total: {df.shape[0]}")

    return df

def pivot_comp_metric_long(df, 
                           metric_nms = ["r2"], 
                           id_cols = ['run_name', 'model_nm', 'task', 'n_train_val_samples', 'cv_fold_num']):
    """
    Pivot the dataframe to long format for the comp and metric columns

    Args:
    df: pd.DataFrame
    metric_nms: list of strings
    id_cols: list of strings

    Returns:
    metrics_df: pd.DataFrame
    """

    #Target cols
    metrics_cols = df.columns[df.columns.str.contains("|".join(metric_nms))].tolist()

    #Subset df to target columns
    metrics_df = df[metrics_cols + id_cols]

    #Create a new data frame for test RMSE
    metrics_df = pd.melt(metrics_df, id_vars=id_cols, value_vars=metrics_cols, var_name='comp_metric', value_name='value')

    # #Rename total_agb comp to avoid delim issues
    metrics_df['comp_metric'] = metrics_df['comp_metric'].str.replace('total_agb', 'Total AGB')

    # #Split cols
    metrics_df[['comp', 'metric']] = metrics_df['comp_metric'].str.split('_', expand=True)

    # #Drop comp metric col
    metrics_df.drop(columns=['comp_metric'], inplace=True)

    #Fix comp and task strings
    metrics_df['comp'] = [nm.title() if nm != "Total AGB" else "Total AGB" for nm in metrics_df['comp']]
    metrics_df['task'] = [nm.title() for nm in metrics_df['task']]

    #Convert comp and metric to ordered factor
    metrics_df['comp'] = pd.Categorical(metrics_df['comp'],
                                        categories=["Total AGB", "Wood", "Branch", "Bark", "Foliage"],
                                        ordered=True)

    return metrics_df

#Define functions
def read_run_cfg_to_df(run_dir):
    
    #Read the config file
    run_cfg = yaml.load(open(os.path.join(run_dir, 'config.yaml'), 'r'), Loader=yaml.Loader)

    #Convert to a df
    run_df = pd.DataFrame.from_dict(run_cfg, orient='index').T

    #Get the pretrain epoch name
    run_df['pretrain_epoch_nm'] = os.path.basename(run_df['model_ckpt_fpath'].values[0]).split("-")[0]

    #Get the pretrained epoch as integer
    run_df['pretrain_epoch'] = int(run_df['pretrain_epoch_nm'].str.split("=")[0][1])
    
    return run_df