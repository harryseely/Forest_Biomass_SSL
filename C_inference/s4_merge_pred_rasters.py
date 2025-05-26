# NOTE: need to run in terminal from root dir as:
# uv run -m C_inference.s4_merge_pred_rasters

import rasterio
import geopandas as gpd
from pathlib import Path
import os
from rasterio.merge import merge
import time

if __name__ == "__main__":

    OUT_CELL_RES = 11.28 * 2

    RMF_INDEX_FPATH = r"E:/RMF/RMF_SPL100/LAS_Classified_Point_Clouds_Normalized_index.gpkg"

    TILE_TRIM_DIR = r"E:/rq3_rmf_inference/tile_predictions_trimmed"

    MERGED_RAS_OUT_FPATH = r"E:/rq3_rmf_inference/tile_predictions_merged.tif"

    rmf_index = gpd.read_file(RMF_INDEX_FPATH)

    rmf_index['tile_name'] = [os.path.basename(f).replace(".laz", "") for f in rmf_index['filename']]

    rmf_index['trim_fpath'] = rmf_index['tile_name'].apply(lambda x: os.path.join(TILE_TRIM_DIR, x + ".tif"))

    pred_ras_fpath_ls = list(Path(TILE_TRIM_DIR).rglob("*.tif"))

    tile_name_ls = [os.path.basename(f).replace(".tif", "") for f in pred_ras_fpath_ls]

    assert rmf_index['tile_name'].isin(tile_name_ls).all(), "Not all tiles have been trimmed!"

    start_time = time.time()

    dataset_ls = []

    ras_fpath_ls = rmf_index['trim_fpath']

    print(f"\nOpening {len(ras_fpath_ls)} datasets...\n")
    [dataset_ls.append(rasterio.open(fpath, mode="r")) for fpath in ras_fpath_ls]

    merge(dataset_ls, 
            res=OUT_CELL_RES, 
            dst_path=MERGED_RAS_OUT_FPATH,
            nodata=-99999)

    print(f"\nClosing {len(dataset_ls)} datasets...\n")
    closed = [ras.close() for ras in dataset_ls]

    end_time = time.time()
    print(f"\nTime taken: {(end_time - start_time)/60} minutes\n")
    print(f"Rate: {len(ras_fpath_ls)/(end_time - start_time)} rasters/second\n")