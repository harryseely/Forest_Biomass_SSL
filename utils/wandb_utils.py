import os
import pandas as pd
import os
import wandb
from tqdm.auto import tqdm
import pytz

def download_wandb_project_data(entity, project_nm, csv_out_dir, startdt=None, enddt=None):
    """
    Function to download per epoch metrics for a wandb project
    :param entity: wandb entity name
    :param project_nm: wandb project name
    :param csv_out_dir: directory to save the downloaded data
    :param startdt: start date for filtering runs (must be datetime object)
    :param enddt: end date for filtering runs (must be datetime object)
    :return: full_cfg_df, full_epoch_df
    """

    #Ensure out dir exists
    if not os.path.exists(csv_out_dir):
        print(f"Creating directory {csv_out_dir} to save data.")
        os.makedirs(csv_out_dir)

    # Set date range filter based on: https://github.com/wandb/wandb/issues/3355#issuecomment-1067742338
    if startdt is not None and enddt is not None:

        pacific = pytz.timezone("US/Pacific")
        startdt = pacific.localize(startdt).isoformat()
        enddt = pacific.localize(enddt).isoformat()

        print(f"Filtering runs from {startdt} to {enddt}")

        date_filter = {
            "$and": [{
                'created_at': {
                    "$lt": enddt,
                    "$gt": startdt
                }
            }]
        }
    else:
        date_filter = {}

    api = wandb.Api()
    source = f"{entity}/{project_nm}"
    runs = api.runs(source, filters=date_filter)
    

    print(f"Downloading data from {source} for {len(runs)} runs.\n")

    epoch_df_ls = []
    cfg_df_ls = []

    for run in tqdm(runs, desc="Downloading run data..."):
        try:
            # Get epoch level metrics
            epoch_df = run.history(pandas=True)
            epoch_df = epoch_df.groupby('epoch').first().reset_index()
            epoch_df['run_name'] = run.name
            epoch_df_ls.append(epoch_df)

            # Get config parameters from run
            cfg_df = pd.DataFrame.from_dict(run.config, orient='index').T

            # Add runtime to config df
            cfg_df['runtime_seconds'] = run.summary['_runtime']

            # Add to list
            cfg_df_ls.append(cfg_df)

        except Exception as e:
            print(f"Failed to download run data from {source} for {run.name} due to following error\n{e}")

    # Combine epoch and config df lists into single dataframes
    full_epoch_df = pd.concat(epoch_df_ls)
    full_cfg_df = pd.concat(cfg_df_ls)

    # Export dfs
    full_epoch_df.to_csv(f"{csv_out_dir}/{project_nm}_epoch_metrics.csv", index=False)
    full_cfg_df.to_csv(f"{csv_out_dir}/{project_nm}_config_summary.csv", index=False)

    print(f"Saved data to {csv_out_dir}.")

    return full_cfg_df, full_epoch_df


def load_wandb_project(project_name, csv_dir, download_data=True, entity="irss", startdt=None, enddt=None):
    """
    Download or load wandb project data.
    :param project_name: name of the wandb project
    :param csv_dir: directory to save the downloaded data
    :param download_data: whether to download the data
    :param entity: wandb entity name
    :param startdt: start date for filtering runs (must be datetime object)
    :param enddt: end date for filtering runs (must be datetime object)
    :return: full_cfg_df, full_epoch_df
    """

    if download_data:

        df, epoch_df = download_wandb_project_data(entity, project_name, csv_dir, startdt, enddt)

    else:
        df = pd.read_csv(os.path.join(csv_dir, f'{project_name}_config_summary.csv'))

        epoch_df = pd.read_csv(os.path.join(csv_dir,  f'{project_name}_epoch_metrics.csv'))
    
    return(df, epoch_df)



if __name__ == '__main__':

    entity = "irss"
    project_name = "misc"
    csv_out_dir = "D:/Sync/RQ3/analysis/outputs"

    # Get run metrics for main comparison ALS (base) vs. Fusion Modules
    full_cfg_df, full_epoch_df = download_wandb_project_data(entity, project_name, csv_out_dir)