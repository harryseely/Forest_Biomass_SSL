import numpy as np
import glob
from pathlib import Path
import os
import multiprocessing

# Run in terminal using code below:
# uv run -m preprocessing.s6_points_to_np

from utils.data_utils import read_las_to_np


def las_to_np(las_fpath, np_save_dir):
    try:
        pc = read_las_to_np(las_fpath, unit_sphere=False, centralize_coords=False)

        np_out_fpath = os.path.join(np_save_dir, f"{las_fpath.stem}.npy")

        np.save(np_out_fpath, pc)

    except Exception as e:
        print(f"Error with converting {las_fpath} to np file: ", e)


def connvert_las_to_np_parallel(las_dir, np_dir, n_cores=4):
    # Get all las files in the unlabelled point cloud folder
    las_fpaths = list(Path(las_dir).glob(f"*las"))

    # Clear out dir before saving new numpy files
    clear_dir = input("Clear out numpy dir? (y/n): ")
    if clear_dir == "y":
        for f in glob.glob(f"{np_dir}/*.npy"):
            os.remove(f)

    #List all the numpy files in the directory
    ids_npy = [f.stem for f in (list(Path(np_dir).glob(f"*npy")))]
    ids_npy_set = set(ids_npy)

    las_fpaths = [f for f in las_fpaths if f.stem not in ids_npy_set]

    print(f"Converting {len(las_fpaths)} las files to np files")

    # Save files using parallel processing
    with multiprocessing.Pool(n_cores) as pool:
        pool.starmap(las_to_np, [(fpath, np_dir) for fpath in las_fpaths])

    print("Done converting las files to np files")


if __name__ == "__main__":
    las_dir = r"E:/RMF/RMF_SPL100/clipped_unlabeled_samples"

    np_dir = r"D:/Sync/RQ3/analysis/data/unlabeled_point_clouds"

    #Use half the cores
    n_cores = multiprocessing.cpu_count() // 2
    connvert_las_to_np_parallel(las_dir, np_dir, n_cores=n_cores)
