# NOTE: need to run in terminal from root dir as:
# uv run -m C_inference.s3_trim_pred_rasters

import rasterio
import geopandas as gpd
from pathlib import Path
import os
from rasterio.windows import from_bounds
from rasterio.plot import show
import matplotlib.pyplot as plt
from tqdm.auto import tqdm



def trim_ras(ras_fpath, tile_bbox, trim_dir, plot_comparison=False):
    """
    Trim a raster to the bounds of a tile, with a trim distance
    to avoid edge effects.
    """

    with rasterio.open(ras_fpath) as src:

        trim_dist = src.res[0]

        tile_window = from_bounds(left=tile_bbox[0] + trim_dist, 
                                bottom=tile_bbox[1] + trim_dist,
                                right=tile_bbox[2] - trim_dist, 
                                top=tile_bbox[3] - trim_dist, 
                                transform=src.transform)

        ras = src.read(window=tile_window)

        clipped_transform = src.window_transform(tile_window)
        
        out_fpath = os.path.join(trim_dir, os.path.basename(ras_fpath))

        # Write the clipped raster to disk
        with rasterio.open(
                out_fpath,
                'w',
                driver=src.driver,
                height=ras.shape[1],
                width=ras.shape[2],
                count=src.count,
                dtype=ras.dtype,
                crs=src.crs,
                transform=clipped_transform
            ) as dst:
                dst.write(ras)
    
    if plot_comparison:
         
        with rasterio.open(ras_fpath) as src:
            og_ras = src.read()

        with rasterio.open(out_fpath) as src:
            trim_ras = src.read()

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        show(og_ras[0], ax=ax[0], title="Original Raster")
        show(trim_ras[0], ax=ax[1], title="Trimmed Raster")

        plt.show()


if __name__ == "__main__":

    RMF_INDEX_FPATH = r"E:/RMF/RMF_SPL100/LAS_Classified_Point_Clouds_Normalized_index.gpkg"

    PRED_RAS_DIR = r"E:/rq3_rmf_inference/tile_predictions"

    TILE_TRIM_DIR = r"E:/rq3_rmf_inference/tile_predictions_trimmed"

    rmf_index = gpd.read_file(RMF_INDEX_FPATH)

    rmf_index['tile_name'] = [os.path.basename(f).replace(".laz", "") for f in rmf_index['filename']]

    rmf_index['pred_fpath'] = rmf_index['tile_name'].apply(lambda x: os.path.join(PRED_RAS_DIR, x + ".tif"))

    pred_ras_fpath_ls = list(Path(PRED_RAS_DIR).rglob("*.tif"))

    tile_name_ls = [os.path.basename(f).replace(".tif", "") for f in pred_ras_fpath_ls]

    rmf_index['tile_name'].isin(tile_name_ls).all()

    i = 11

    ras_fpath = rmf_index['pred_fpath'][i]

    tile_bbox = rmf_index['geometry'][i].bounds

    trim_ras(ras_fpath, tile_bbox, TILE_TRIM_DIR, plot_comparison=True)

    for i in tqdm(range(len(rmf_index))):

        ras_fpath = rmf_index['pred_fpath'][i]

        tile_bbox = rmf_index['geometry'][i].bounds

        trim_ras(ras_fpath, tile_bbox, TILE_TRIM_DIR)

