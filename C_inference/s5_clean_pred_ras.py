# NOTE: need to run in terminal from root dir as:
# uv run -m C_inference.s5_clean_pred_ras

import numpy as np
import rasterio
from rasterio.fill import fillnodata
from rasterio.enums import Resampling
from rasterio.warp import reproject
import time


if __name__ == "__main__":

    MERGED_RAS_OUT_FPATH = r"E:/rq3_rmf_inference/tile_predictions_merged.tif"

    CHM_FPATH = r"E:/RMF/LiDAR Summary Metrics/CHM_22.56m_resampled.tif"

    CLEANED_RAS_OUT_FPATH = r"E:/rq3_rmf_inference/tile_predictions_clean.tif"

    start_time = time.time()

    #Read the predicted biomass raster
    with rasterio.open(MERGED_RAS_OUT_FPATH) as src:
        pred_ras = src.read()

        pred_ras_profile = src.profile

        pred_ras_nodata_val = src.nodata

    # Fill NA values in predicted raster
    na_mask = np.invert(pred_ras == pred_ras_nodata_val).astype(int)

    bands_filled_ls = [fillnodata(pred_ras[i], max_search_distance=2, mask=na_mask) for i in range(pred_ras.shape[0])]

    pred_ras_filled = np.stack(bands_filled_ls, axis=0)

    pred_ras_filled[pred_ras_filled == pred_ras_nodata_val] = np.nan

    pred_ras_filled[pred_ras_filled <= 0] = 0

    total_agb = np.expand_dims(np.nansum(pred_ras_filled, axis=0), axis=0)

    pred_ras_filled = np.concatenate([pred_ras_filled, total_agb], axis=0)

    # Create an NA mask
    na_mask = np.where(np.isnan(pred_ras_filled[0]), np.nan, 1)

    # Read the CHM
    with rasterio.open(CHM_FPATH) as src:

        chm = src.read() 

        chm[chm == src.nodata] = np.nan

        aligned_data = np.empty((pred_ras_profile['height'], pred_ras_profile['width']), dtype=chm.dtype)

        chm, chm_transform = reproject(
            source=chm,
            destination=aligned_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=pred_ras_profile['transform'],
            dst_crs=pred_ras_profile['crs'],
            dst_width=pred_ras_profile['width'],
            dst_height=pred_ras_profile['height'],
            resampling=Resampling.nearest  
        )

    assert pred_ras.shape[1:] == chm.shape, "Pred ras and CHM ras do not have the same shape!"

    # Mask areas with canopy height below 1.5m
    forested_area = np.where(chm > 1.5, 1, 0)
    pred_ras_masked = np.where(forested_area == 1, pred_ras_filled, 0)

    # Ensure areas outside RMF are set as NA
    pred_ras_masked = pred_ras_masked * na_mask

    with rasterio.open(
        CLEANED_RAS_OUT_FPATH,
        "w",
        driver="GTiff",
        height=pred_ras_masked.shape[1],
        width=pred_ras_masked.shape[2],
        count=pred_ras_masked.shape[0],
        dtype=pred_ras_masked.dtype,
        crs=pred_ras_profile['crs'],
        transform=pred_ras_profile['transform'],
    ) as dst:
        dst.write(pred_ras_masked)

    print(f"\nCleaned raster saved to {CLEANED_RAS_OUT_FPATH}\n")

    # Record runtime
    end_time = time.time()
    runtime_minutes = round((end_time - start_time) / 60, 2)

    print(f"Finished cleaning predicted raster. Time required:\n{runtime_minutes} minute(s).")
