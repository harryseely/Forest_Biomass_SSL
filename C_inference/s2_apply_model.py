# NOTE: need to run in terminal from root dir as:
# uv run -m C_inference.s2_apply_model

import os
import pandas as pd
import torch
import numpy as np
import yaml
import multiprocessing as mp
from pathlib import Path
import pytorch_lightning as pl
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import geopandas as gpd
from lightning.pytorch.utilities import disable_possible_user_warnings
import gc

from utils.training_utils import check_matching_dict_keys
from models.ocnn_lightning import OCNNLightning
from utils.ocnn_utils import CustomCollateBatch
from utils.data_utils import update_z_score_conversion_info
from C_inference.inference_dataset import InferenceDataset, rasterize_gridded_points


def apply_model_to_tile(tile_df_row, gpu_id, cfg, model, model_ckpt_fpath, pred_out_dir, 
                        centroids, out_res):
    """
    Applies the trained model to a single lidar tile and saves the predictions to a GeoTIFF.
    Function designed to run on a single GPU so mulitple instances can be run in parallel.
    
    :param tile_df_row: DataFrame row containing the tile information
    :param gpu_id: ID of the GPU to use for inference
    :param cfg: Configuration dictionary containing model parameters, batch size, etc.
    :param model: Instatiated pytorch model architecture to use for inference
    :param model_ckpt_fpath: Path to the model checkpoint file
    :param pred_out_dir: Output directory for the predicted rasters
    :param centroids: GeoDataFrame of the centroids representing the center of each raster cell
    :param out_res: Output resolution of the raster in meters
    """

    #Disable warnings from PyTorch Lightning
    disable_possible_user_warnings()

    if not os.path.exists(pred_out_dir):
        os.makedirs(pred_out_dir)
    
    assert tile_df_row.shape[0] == 1, "tile_df_row should only contain one row!"

    collate_fn = CustomCollateBatch(batch_size=cfg['batch_size'])

    cell_id_ls = tile_df_row['cell_id_ls'].values[0].split(",")
    cell_id_ls = [int(x) for x in cell_id_ls]

    tile_centroids = centroids[centroids['cell_id'].isin(cell_id_ls)]
    tile_centroids = tile_centroids.reset_index(drop=True)

    tile_fpath = tile_df_row['filename'].values[0]

    torch.set_float32_matmul_precision('high')

    trainer = pl.Trainer(accelerator="gpu",
                            devices=1,
                            logger=False,
                            enable_checkpointing=False,
                            enable_model_summary=False,
                            inference_mode=True,
                            precision="16-mixed")

    inference_dataset = InferenceDataset(pc_fpath=tile_fpath, 
                                        grid_centroids=centroids, 
                                        cell_id_ls=cell_id_ls, 
                                        res=out_res, 
                                        gpu_id=gpu_id,
                                        octree_depth=cfg['octree_depth'], 
                                        full_depth=cfg['ocnn_full_depth'])

    inference_dataloader = DataLoader(inference_dataset,
                                    batch_size=cfg['batch_size'],
                                    shuffle=False,
                                    collate_fn=collate_fn,
                                    drop_last=False
                                    )

    try: 

        out = trainer.predict(model, 
                            dataloaders=inference_dataloader, 
                            ckpt_path=model_ckpt_fpath, 
                            return_predictions=True)

        pred = np.concatenate([batch[0] for batch in out])
        cell_ids = np.concatenate([batch[1] for batch in out])

        pred_df = pd.DataFrame(pred, columns=cfg['biomass_comp_nms'])

        pred_df["cell_id"] = cell_ids

        pred_df = centroids.merge(pred_df, on="cell_id", how="right")
        
        # Remove original file extension and replace with .tif
        out_fname = f"{os.path.basename(tile_fpath).split('.')[0]}.tif"

        out_ras_fpath = os.path.join(pred_out_dir, out_fname)

        rasterize_gridded_points(centroids=pred_df, 
                                out_ras_fpath=out_ras_fpath, 
                                col_names=cfg['biomass_comp_nms'], 
                                out_resolution=out_res)
    
    except Exception as e:
        print(f"\nError predicting to raster for tile {tile_fpath}:\n{str(e)}\n")
    
    # Clean up
    del tile_df_row, cell_id_ls, tile_centroids
    del inference_dataset, inference_dataloader
    del out, pred, cell_ids, pred_df

    gc.collect()

    torch.cuda.empty_cache()

    print(f"\nFinished predicting tile {os.path.basename(tile_fpath)}.\n")
    

def apply_model_to_tile_list(tile_names_list, gpu_id, cfg, tiles_df, model, model_ckpt_fpath, pred_out_dir, centroids, out_res):

        for tile_nm in tqdm(tile_names_list, desc="Predicting biomass for tiles: "):

            #Subset df to target tile
            tile_df_row = tiles_df[tiles_df['tile_name'] == tile_nm].reset_index(drop=True)

            print("\nApplying model to tile: ", tile_df_row['tile_name'].values[0])


            apply_model_to_tile(tile_df_row=tile_df_row,
                                gpu_id=gpu_id,
                                cfg=cfg, 
                                model=model,
                                model_ckpt_fpath=model_ckpt_fpath,
                                pred_out_dir=pred_out_dir,
                                centroids=centroids, 
                                out_res=out_res)


    


if __name__ == "__main__":

    RUN_IN_PARALLEL = True

    BATCH_SIZE = 128
    
    OUT_CELL_RES = 11.28 * 2

    OUT_DIR = r"E:/rq3_rmf_inference"
    MODEL_CKPT_FPATH = r"D:/Sync/RQ3/analysis/checkpoints_and_config/rq3_finetune_vs_scratch_ocnn_lenet/finetune_ocnn_lenet_2025_03_28_13_57_17_idVPZN/epoch=32-val_loss=0.09-val_r2=0.75.ckpt"
    BIOMASS_LABELS_FPATH = r"D:/Sync/RQ3/analysis/data/biomass_labels.csv"
    CENTROIDS_FPATH = r"E:/rq3_rmf_inference/rmf_centroids.gpkg"
    CFG_FPATH = r"config.yaml"
    HPS_FPATH = r"optimal_hyperparameters/ocnn_lenet_hps.yaml"

    TILE_PREDICTION_OUT_DIR = os.path.join(OUT_DIR, "tile_predictions")
    TILES_W_CELL_IDS_FPATH=os.path.join(OUT_DIR, "rmf_tiles_with_cells.gpkg")

    print(f"\nReading centroids from {CENTROIDS_FPATH}\n")
    CENTROIDS = gpd.read_file(CENTROIDS_FPATH)[["cell_id", "geometry"]]

    tiles_w_cells_df = gpd.read_file(TILES_W_CELL_IDS_FPATH)

    #Get the current tiles with predicted rasters
    existing_pred_fpaths = list(Path(TILE_PREDICTION_OUT_DIR).glob("*.tif"))
    existing_pred_tile_name = [str(os.path.basename(fpath)) for fpath in existing_pred_fpaths]
    existing_pred_tile_name = [fpath for fpath in existing_pred_tile_name if not fpath.endswith(".aux.xml")]
    existing_pred_tile_name = [fpath.split(".")[0] for fpath in existing_pred_tile_name]

    print(f"\nFinished {len(existing_pred_tile_name)}/{len(tiles_w_cells_df)} tiles.\n")

    #Filter df to only include tiles that have not been predicted yet
    tiles_w_cells_df = tiles_w_cells_df[~tiles_w_cells_df["tile_name"].isin(pd.Series(existing_pred_tile_name))]

    # Filter centroids to remaining target tiles
    cell_id_ls_ls = [x for x in tiles_w_cells_df['cell_id_ls'].values]
    cell_id_ls_ls = [x.split(",") for x in cell_id_ls_ls]
    cell_id_ls = [int(x) for sublist in cell_id_ls_ls for x in sublist]  
    CENTROIDS = CENTROIDS[CENTROIDS['cell_id'].isin(cell_id_ls)].reset_index(drop=True)

    #Extract tile names list
    tile_names_list = tiles_w_cells_df["tile_name"].tolist()

    print(f"\nTiles remaining to predict: {len(tiles_w_cells_df)} ({CENTROIDS.shape[0]} cells)\n")

    # Set up config

    with open(CFG_FPATH, "r") as yamlfile:
        cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)

    with open(HPS_FPATH, "r") as yamlfile:
        optimal_hps = yaml.load(yamlfile, Loader=yaml.FullLoader)
    
    cfg = check_matching_dict_keys(og_dict=cfg, update_dict=optimal_hps)

    cfg['batch_size'] = BATCH_SIZE

    # Set up model

    z_info_dict = update_z_score_conversion_info(biomass_labels_fpath=BIOMASS_LABELS_FPATH,
                                                biomass_comp_nms=cfg['biomass_comp_nms'])

    model = OCNNLightning(cfg, z_info=z_info_dict, predict_output="pred_and_cellid")

    if RUN_IN_PARALLEL:

        cuda_ids = [i for i in range(torch.cuda.device_count())]

        # Split las fpaths into N GPU lists for parallel processing
        n_gpus = len(cuda_ids)
        tile_names_list_chunks = [tile_names_list[i::n_gpus] for i in range(n_gpus)]

        args = [(tile_names_list,
                gpu_id,
                cfg,
                tiles_w_cells_df,
                model,
                MODEL_CKPT_FPATH,
                TILE_PREDICTION_OUT_DIR,
                CENTROIDS,
                OUT_CELL_RES) for
                tile_names_list, gpu_id in zip(tile_names_list_chunks, cuda_ids)]

        print(f"\nApplying model to tiles in parallel using {len(cuda_ids)} GPUs...\n")

        with mp.Pool(n_gpus) as p:
            p.starmap(apply_model_to_tile_list, args)


    else:

        print("\nApplying model to tiles sequentially...\n")

        apply_model_to_tile_list(tile_names_list=tile_names_list,
                                gpu_id=0,
                                cfg=cfg,
                                tiles_df=tiles_w_cells_df,
                                model=model,
                                model_ckpt_fpath=MODEL_CKPT_FPATH,
                                pred_out_dir=TILE_PREDICTION_OUT_DIR,
                                centroids=CENTROIDS,
                                out_res=OUT_CELL_RES)
