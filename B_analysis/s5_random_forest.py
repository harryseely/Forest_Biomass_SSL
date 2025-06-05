# NOTE: need to run in terminal from root dir as:
# uv run -m B_analysis.s5_random_forest

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
import yaml
from tqdm.auto import tqdm
import seaborn as sns
from datetime import datetime as dt
import string
import random

def train_rf(df, fold_col_nm, target_ls, metrics_ls, n_samp_train="all"):
    """
    Train a random forest model on the training data and apply to the test data.
    Calculate metrics for the test data.

    Args:
    df: pd.DataFrame containing the response and predictor variables
    fold_col_nm: str column name in df containing the split (fold)
    target_ls: list of str column names in df containing the target variables
    metrics_ls: list of str column names in df containing the predictor variables
    n_samp_train: int number of samples to use in the training data

    Returns:
    metrics_df: pd.DataFrame containing the metrics for the test data
    """
    
    #Set a run name
    t_now = dt.now().strftime('%Y_%m_%d_%H_%M_%S')
    rand_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
    run_name = f"rf_{t_now}-id{rand_id}"

    #Split data into train and test
    if n_samp_train == "all":
        train_df = df[df[fold_col_nm] == "train"]
        n_samp_train = train_df.shape[0]
    else:
        train_df = df[df[fold_col_nm] == "train"].sample(n=n_samp_train, random_state=15)

    test_df = df[df[fold_col_nm] == "test"]

    #Create and fit the model
    model = RandomForestRegressor(n_jobs=1)
    model.fit(X=train_df[metrics_ls], y=train_df[target_ls])

    #Apply the model to the test data
    test_pred_df = pd.DataFrame(model.predict(test_df[metrics_ls]), 
                                columns=[c + "_pred" for c in target_ls])

    #Calculate total AGB from comp predictions
    test_pred_df['total_Mg_ha_pred'] = test_pred_df[[c + "_pred" for c in target_ls]].sum(axis=1)

    #Join with the test data
    test_df = pd.concat([test_df.reset_index(drop=True), test_pred_df], axis=1)


    #Pivot to long format to calculate metrics
    obs_pred_cols = [c for c in test_df.columns if "Mg_ha" in c]


    obs_pred_df = test_df.melt(id_vars=['PlotID'],
                                value_vars=obs_pred_cols,
                                var_name='comp_var',
                                value_name='val')

    #Add obs to non-pred columns
    obs_pred_df['comp_var'] = obs_pred_df['comp_var'].apply(lambda c: c + "_obs" if "_pred" not in c else c)

    #Split the columns
    obs_pred_df['val_type'] = obs_pred_df['comp_var'].apply(lambda c: c.split("_")[-1])

    #Create comp only col and remove comp_var
    obs_pred_df['comp'] = obs_pred_df['comp_var'].apply(lambda x: x.replace("_obs", "").replace("_pred", ""))
    obs_pred_df = obs_pred_df.drop(columns=['comp_var'])

    #Pivot to wide format
    obs_pred_df = obs_pred_df.pivot_table(index=['PlotID', 'comp'], columns=['val_type'], values='val').reset_index()

    #Calculate metrics
    metrics_df = obs_pred_df.groupby('comp').apply(lambda x: pd.Series({
        'r2': r2_score(x['obs'], x['pred']),
        'rmse': root_mean_squared_error(x['obs'], x['pred']),
        'mean_bias': np.mean(x['obs'] - x['pred']),
    }), include_groups=False).reset_index()

    #Update train and test dataset sizes
    metrics_df['n_train'] = n_samp_train
    metrics_df['n_test'] = test_df.shape[0]

    #Add run name
    metrics_df['run_name'] = run_name
    obs_pred_df['run_name'] = run_name

    return metrics_df, obs_pred_df

#Load config
with open("config.yaml", "r") as yamlfile:
    cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)

#Set filepaths
biomass_fpath = 'data/biomass_labels.csv'
plot_metrics_fpath = 'data/labeled_plot_metrics.csv'

#Load data
df = pd.read_csv(biomass_fpath)
metrics_df = pd.read_csv(plot_metrics_fpath).rename(columns={'plot_id':'PlotID'})

#Extract lidar metric names
metrics_ls = [c for c in metrics_df.columns if c != "PlotID"]

#Add metrics to main df
df = df.merge(metrics_df, on='PlotID')

#Set target variables (z scored biomass)
target_ls = [nm + '_Mg_ha' for nm in cfg['biomass_comp_nms']]

print(df.shape)
print(metrics_ls)
print("Number of metrics: ", len(metrics_ls))
print(target_ls)

#Get the largest number of training samples using 70% of all plots
max_n_train = df[df['strat_fold_1'] == "train"].shape[0]

#Set number of train samples
n_train_ls = list(range(20, 180, 10))
n_train_ls.append(max_n_train)

print(n_train_ls)

#Iterate through each dataset size and stratified sampling CV folds
metrics_df_ls = []
obs_pred_df_ls = []

#Train each model variant 5 times
for n_train in tqdm(n_train_ls, 
                    desc="Iterating through different dataset sizes.", 
                    leave=True):

    for cv_fold_num in tqdm(range(1, 6), 
                    desc="Iterating through different CV folds.",
                    leave=False):
            
            for _ in tqdm(range(5), 
                          desc="Iterating 5 times for CV fold.",
                          leave=False):


                metrics_df_i, obs_pred_df_i = train_rf(df, "strat_fold_" + str(cv_fold_num), target_ls, metrics_ls, n_train)
                
                metrics_df_i['cv_fold_num'] = cv_fold_num
                obs_pred_df_i['cv_fold_num'] = cv_fold_num

                #Record whether using reduced training set
                if n_train != max_n_train:
                    metrics_df_i['reduced_train_set'] = True
                    obs_pred_df_i['reduced_train_set'] = True
                else:
                    metrics_df_i['reduced_train_set'] = False
                    obs_pred_df_i['reduced_train_set'] = False

                metrics_df_ls.append(metrics_df_i)
                obs_pred_df_ls.append(obs_pred_df_i)

#Combine all dataset size and fold metrics
metrics_df = pd.concat(metrics_df_ls)
obs_pred_df = pd.concat(obs_pred_df_ls)

#Get the mean r2 and rmse across all components
mean_metrics_df = metrics_df.groupby('n_train').\
    agg(mean_r2=('r2', 'mean'),
        mean_rmse=('rmse', 'mean')).reset_index()

#Plot the R2 vs. dataset size
sns.lineplot(data=mean_metrics_df, x='n_train', y='mean_r2')

#Subset stratified df to runs that only use all the training data
metrics_df_all_train = metrics_df[metrics_df['reduced_train_set'] == False]

#Summarize mean performance across all stratified folds
strat_fold_df = metrics_df_all_train[pd.notna(metrics_df_all_train['cv_fold_num'])].groupby('comp').agg(
                                                                                   strat_fold_r2=("r2", "mean"),
                                                                                   strat_fold_rmse=("rmse", "mean"),
                                                                                   n_runs_strat_fold=("cv_fold_num", "count")
                                                                                   ).reset_index()

#Export the RF metrics
root_dir = os.getcwd()

metrics_out_fpath = os.path.join(root_dir, 'results', 'rf_metrics.csv')
metrics_df.to_csv(metrics_out_fpath, index=False)
print("Exported RF metrics to: ", metrics_out_fpath)

obs_pred_out_fpath = os.path.join(root_dir, 'results', 'rf_obs_pred.csv')
obs_pred_df.to_csv(obs_pred_out_fpath, index=False)
print("Exported RF obs and pred to: ", obs_pred_out_fpath)

