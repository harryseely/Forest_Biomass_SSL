import os
from typing import Optional
from pathlib import Path
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from random import sample, seed

from utils.data_utils import read_las_to_np, convert_vals_to_z_score
from utils.ocnn_utils import load_octree_sample, CustomCollateBatch


class PointCloudDataset(Dataset):
    """
    Loads a dataset of point clouds from a directory of las files

    *** IMPORTANT NOTE: The target biomass tensor order is [foliage, bark, branch, wood]

    """

    def __init__(self, cfg, split, pretraining_pc_fpaths=None):
        """
        :param cfg: config dictionary
        :param split: split name, either "train", "val", or "test"
=        :param pretraining_pc_fpaths: list of point cloud file paths (only needed for pretraining)
        """

        assert split == "train" or split == "val" or split == "test", "Invalid split name!"

        self.cfg = cfg
        self.split = split
        self.labeled_pc_dir = r"data/plot_point_clouds"

        # List to store plot IDs for each sample
        self.PlotID_ls = []

        #Determine whether to apply data augmentation to samples for Octree-CNN
        if split == "train":
            self.augment = True
        else:
            self.augment = False

        #Set up dataset depending on task
        if cfg['task'] == "pretrain":
            self._setup_pretraining(pretraining_pc_fpaths)

        elif cfg['task'] in ["finetune", "scratch"]:
            self._setup_finetuning_scratch()

        else:
            raise ValueError(f"Invalid task {self.cfg['task']}) specified in config dictionary.")

        if cfg['verbose']:
            print(f"Number of {split} files: {len(self.pc_fpaths)}")

        super().__init__()

    def _setup_pretraining(self, pretraining_pc_fpaths):
        """
        Basic setup for pretraining. No labels are used and files are not loaded into memory.
        :param pretraining_pc_fpaths:
        :return:
        """

        self.pc_fpaths = pretraining_pc_fpaths

        #Load lidar metrics
        self.metrics_df = pd.read_csv("data/unlabeled_plot_metrics.csv")

        #Subset metrics df to target metrics only
        self.metrics_df = self.metrics_df[self.cfg['target_lidar_metrics'] + ['fname']]

        #Extract fnames of unlabeled point clouds
        self.fpath_df = pd.DataFrame({"pc_fpath": self.pc_fpaths})
        self.fpath_df['fname'] = [fpath.stem for fpath in self.fpath_df['pc_fpath']]

        #Ensure fname is same data type in both DFs
        self.metrics_df['fname'] = self.metrics_df['fname'].astype(float)
        self.fpath_df['fname'] = self.fpath_df['fname'].astype(float)

        #Join fpaths with metrics df for loading files later
        self.metrics_df = pd.merge(self.metrics_df, self.fpath_df, left_on='fname', right_on='fname', how='inner')

        #z-score normalize metrics
        for metric in self.cfg['target_lidar_metrics']:
            self.metrics_df[metric + "_z"] = convert_vals_to_z_score(self.metrics_df[metric])

        #Check for NAs in df
        na_count = self.metrics_df.isna().any(axis=1).sum()
        if na_count > 0:
            print(f"\033[91m\nRemoving {na_count} rows with NA values from metrics df.\n\033[0m")
            self.metrics_df = self.metrics_df.dropna()

        #Reduce pc_fpaths to only those with associated metrics
        self.pc_fpaths = self.metrics_df['pc_fpath'].tolist()


    def _setup_finetuning_scratch(self):
        """
        Load las files and corresponding biomass labels for finetuning or training from scratch.
        Setup method ensures the dataset is filtered to the target split and that las files are loaded into memory.
        :return:
        """

        self.pc_fpaths = list(Path(self.labeled_pc_dir).glob(f"*las"))
        self.biomass_df = pd.read_csv(self.cfg['labels_rel_fpath'])

        assert len(self.pc_fpaths) > 0, f"No las files found in {self.labeled_pc_dir}!"

        # Create a df with las files and join with plot DF
        las_files_df = pd.DataFrame(self.pc_fpaths, columns=["pc_fpath"])
        las_files_df['PlotID'] = las_files_df['pc_fpath'].apply(
            lambda x: int(os.path.basename(x).split(".")[0].replace("plot_", "")))
        self.biomass_df = pd.merge(self.biomass_df, las_files_df, on='PlotID', how='inner')

        #Set cross validation method (stratified/spatial) and associated fold number
        if self.cfg['spatial_cv']:
            raise ValueError("Spatial CV not currently supported.")
        else:
            fold_col_nm = f"strat_fold_{self.cfg['cv_fold_num']}"
        
        #Select specific stratified/spatial fold column
        self.biomass_df = self.biomass_df[self.biomass_df[fold_col_nm] == self.split]

        # Reduce train/val dataset size if specified
        if (self.split != "test") and (self.cfg['n_train_val_samples'] != "all"):

            #Set prop of samples to use based on split (80% train, 20% val)
            n_samp_prop = 0.8 if self.split == "train" else 0.2

            #Get specific number of samples for split
            n_samp = int(self.cfg['n_train_val_samples'] * n_samp_prop)

            #Reduce dataset to target n samples
            self.biomass_df = self.biomass_df.sample(n=n_samp, random_state=15)

            print(f"Reduced {self.split} dataset to {n_samp} samples.\nAssuming 80/20 train-val split.")

        # Get target las files for split and corresponding plot ids
        self.pc_fpaths = self.biomass_df['pc_fpath'].tolist()

        # Record pc filenames for indexing corresponding biomass labels
        self.pc_fnames = [os.path.basename(f) for f in self.pc_fpaths]

        # Load las files into memory
        self.point_clouds = []

        for f in tqdm(self.pc_fpaths, leave=False, position=0,
                      desc=f"Reading las files from {self.labeled_pc_dir} for {self.split} set..."):
            pc = read_las_to_np(str(f), centralize_coords=True, unit_sphere=True)

            self.point_clouds.append(pc)

    def __len__(self):
        return len(self.pc_fpaths)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.cfg['task'] in ["finetune", "scratch"]:

            # Use point cloud already loaded into memory
            pc = self.point_clouds[idx]

            # Get plot ID from filename
            PlotID = int(os.path.basename(self.pc_fnames[idx]).split(".")[0].replace("plot_", ""))
            self.PlotID_ls.append(PlotID)

            # Extract matching row from self.df
            biomass_row = self.biomass_df.loc[self.biomass_df["PlotID"] == PlotID]

            # Extract biomass component z-scores and convert to tensor
            target = np.array([biomass_row[comp + "_z"].values[0] for comp in self.cfg['biomass_comp_nms']])
            target = torch.from_numpy(target).float()

        elif self.cfg['task'] == "pretrain":

            # Select row idx from metrics df
            row_idx = self.metrics_df.iloc[idx]

            # Read associated point cloud
            pc = np.load(row_idx['pc_fpath'])

            # Select z-scored target metrics specified in config
            target_cols = [metric + "_z" for metric in self.cfg['target_lidar_metrics']]
            target = row_idx[target_cols].to_numpy(dtype=np.float32)

            # Convert target to tensor
            target = torch.tensor(target, dtype=torch.float32)

        else:
            raise ValueError("Invalid task specified in config dictionary.")

        #Convert point cloud to octree
        sample = load_octree_sample(pc, idx, depth=self.cfg['octree_depth'], 
                                    full_depth=self.cfg['ocnn_full_depth'], 
                                    augment=self.augment)

        sample['target'] = target


        return sample


class PointCloudDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        """

        :param cfg: config dictionary
        """
        super().__init__()
        self.cfg = cfg

        self.unlabeled_pc_dir = r"data/unlabeled_point_clouds"

        # For unlabelled pretraining, use a random train/val split
        if cfg['task'] == "pretrain":

            self.pc_fpaths = list(Path(self.unlabeled_pc_dir).glob(f"*npy"))

            assert len(self.pc_fpaths) > 0, f"No npy files found in {self.unlabeled_pc_dir}!"

            #Reduce dataset to target n samples if specified
            if self.cfg['n_train_val_samples'] != "all":
                seed(25)
                self.pc_fpaths = sample(self.pc_fpaths, cfg['n_train_val_samples'])

            # Split data into train and val sets
            self.train_fpaths, self.val_fpaths = train_test_split(self.pc_fpaths,
                                                                  test_size=1 - cfg['unlabeled_train_val_split'],
                                                                  random_state=42)
            
        else:
            self.train_fpaths = None
            self.val_fpaths = None

        # Use custom collate function for OCNN model
        self.collate_fn = CustomCollateBatch(batch_size=cfg['batch_size'])


    def setup(self, stage: Optional[str] = None):

        if stage == "fit":
            self.train_dataset = PointCloudDataset(self.cfg, split="train", pretraining_pc_fpaths=self.train_fpaths)

            self.val_dataset = PointCloudDataset(self.cfg, split="val", pretraining_pc_fpaths=self.val_fpaths)

        # Using predict stage in PL for testing instead of built-in test stage
        elif stage == "predict":
            self.test_dataset = PointCloudDataset(self.cfg, split="test", pretraining_pc_fpaths=self.train_fpaths)

        else:
            raise ValueError("Invalid stage specified in setup method.")


    # NOTE: pin_memory=True is not currently feasible since OCNN uses custom type batches
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.cfg['batch_size'],
                          shuffle=True,
                          collate_fn=self.collate_fn,
                          drop_last=True,
                          num_workers=self.cfg['n_data_workers']
                          )

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.cfg['batch_size'],
                          shuffle=False,
                          collate_fn=self.collate_fn,
                          drop_last=True,
                          num_workers=self.cfg['n_data_workers']
                          )

    def predict_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.cfg['batch_size'],
                          shuffle=False,
                          collate_fn=self.collate_fn,
                          drop_last=True
                          )
