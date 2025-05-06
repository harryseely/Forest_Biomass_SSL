
import os
import cupy as cp
from shapely.geometry import Point
import numpy as np
from tqdm.auto import tqdm
import traceback
from geocube.api.core import make_geocube
from geocube.rasterize import rasterize_points_griddata
from time import time
import os
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import Dataset

from utils.data_utils import read_las_to_np, pc_to_unit_sphere
from utils.ocnn_utils import load_octree_sample


def create_square_bbox(point, buffer_distance):
    x, y = float(point.x), float(point.y)
    min_x = x - buffer_distance
    max_x = x + buffer_distance
    min_y = y - buffer_distance
    max_y = y + buffer_distance
    bbox_array = np.array([min_x, max_x, min_y, max_y])
    return bbox_array

def clip_tile_to_cells(in_pc_fpath, grid_centroids, cell_id_ls, res, gpu_id):
    """
    :param in_pc_fpath: input las filepath
    :param grid_centroids: geopandas dataframe containing grid centroids and intersecting las tile filenames
    :param cell_id_ls: list of cell ids that intersect the tile (cell id is integer)
    :param res: resolution of grid cells (in metres)
    :param gpu_id: id of GPU to be used by CuPy
    :return:
    """

    print(f"Clipping {os.path.basename(in_pc_fpath)} on GPU {gpu_id}")

    # Set CuPy to only use target GPU
    with cp.cuda.Device(gpu_id):
            
        tile_clipped_pc_ls = []

        #Subset centroids to only include cells intersecting tile
        tile_centroids = grid_centroids[grid_centroids['cell_id'].isin(cell_id_ls)]
        tile_centroids = tile_centroids.reset_index(drop=True)
        
        # Create bboxes from grid_centroids in subsetted dataframe and send to gpu
        bboxes = cp.asarray(
            [create_square_bbox(point=Point(x, y), buffer_distance=res/2) for x, y in
            zip(tile_centroids.geometry.x, tile_centroids.geometry.y)])

        try:

            pc = read_las_to_np(in_pc_fpath, centralize_coords=False, unit_sphere=False)

            # Send to GPU
            pc = cp.asarray(pc)

            # Get extent of pc tile
            min_x, min_y = cp.nanmin(pc[:, 0]), cp.nanmin(pc[:, 1])
            max_x, max_y = cp.nanmax(pc[:, 0]), cp.nanmax(pc[:, 1])

            # Iterate through grid_centroids
            for bbox, cell_id in tqdm(zip(bboxes, cell_id_ls), 
                                desc=f"Clipping cells for tile: {in_pc_fpath}", 
                                leave=False,
                                total=len(cell_id_ls)):

                # Clip the pc to the cell bbox
                try:

                    # Check if bbox is within the bounds of the pc extent
                    if not (bbox[0] < min_x and bbox[1] > max_x and bbox[2] < min_y and bbox[3] > max_y):

                        # Create mask based on bbox
                        mask = ((pc[:, 0] >= bbox[0]) & (pc[:, 0] <= bbox[1]) & (pc[:, 1] >= bbox[2]) & (
                                pc[:, 1] <= bbox[3]))
                        
                        #Clip pc to mask
                        pc_clipped = pc[mask]

                        #Send points to CPU
                        pc_clipped_cpu = cp.asnumpy(pc_clipped)  # send to cpu  
                        
                        if pc_clipped_cpu.shape[0] == 0:
                            na_pc = np.empty((10, 3), dtype=pc_clipped_cpu.dtype)
                            id_pc_tuple = (cell_id, na_pc)
                        else:
                            id_pc_tuple = (cell_id, pc_clipped_cpu)

                        # Append the clipped pc to the list
                        tile_clipped_pc_ls.append(id_pc_tuple)

                except Exception as e:
                    print(f"Error in: {in_pc_fpath} for cell_id: {cell_id}")
                    print(str(e) + "\n")

            # Clear GPU of all cp objects and synchronize
            del pc, pc_clipped, mask, min_x, min_y, max_x, max_y, bbox
            cp.cuda.Stream.null.synchronize()

        except Exception as e:
            print(f"Error in: {in_pc_fpath}")
            print(str(e) + "\n")
            traceback.print_exc()
        
    return tile_clipped_pc_ls


def rasterize_gridded_points(centroids, out_ras_fpath, col_names, out_resolution, record_time=False):
    """

    IMPORTANT: this function assumes input centroids are on an even grid!

    Uses the geocube package to convert a gridded point dataset in the form of a geaodataframe to a rioxarray and then
    exports this to geotiff.
    :param centroids: geopandas dataframe of evenly spaced grid centroids
    :param out_ras_fpath: output raster file path for exported geotiff
    :param col_names: list of column names in centroids gdf to rasterize
    :param out_resolution: output resolution of raster in units of the CRS
    :param record_time: whether to report the runtime of the function
    :return:
    """

    if record_time:
        start_time = time()

    geo_grid = make_geocube(
        vector_data=centroids,
        measurements=col_names,
        resolution=out_resolution,
        output_crs=centroids.crs,
        rasterize_function=rasterize_points_griddata,
        fill=-9999,  # NA value
    )

    geo_grid.rio.to_raster(out_ras_fpath)

    print(f"Exported raster to {out_ras_fpath}")

    if record_time:
        end_time = time()
        runtime_seconds = round((end_time - start_time), 2)

        print(f"Finished rasterizing. Time required:\n{runtime_seconds} seconds.")


class InferenceDataset(Dataset):
    def __init__(self, pc_fpath, grid_centroids, cell_id_ls, res, gpu_id, octree_depth, full_depth):
        """

        :param pc_fpath: Path to lidar tile file (.las/.laz)
        :param octree_depth: Depth of octree to use for inference
        :param full_depth: Full depth of octree to use for inference
        """

        self.octree_depth = octree_depth
        self.full_depth = full_depth

        id_pc_ls = clip_tile_to_cells(in_pc_fpath=pc_fpath, 
                                      grid_centroids=grid_centroids, 
                                      cell_id_ls=cell_id_ls, 
                                      res=res, 
                                      gpu_id=gpu_id)
        
        self.cell_id_ls = [tup[0] for tup in id_pc_ls]
        self.pc_ls = [tup[1] for tup in id_pc_ls]

        print(f"\nGenerating predictions for {len(self.cell_id_ls)} cells.\n")

        print(f"\nIncluding cell IDs:\n{self.cell_id_ls[1:10]}...\n")

        super().__init__()

    def __len__(self):
        return len(self.cell_id_ls)

    def __getitem__(self, idx):

        target_cell_id = self.cell_id_ls[idx]
        pc = self.pc_ls[idx]

        pc = pc - np.mean(pc, axis=0)
        pc = pc_to_unit_sphere(pc, verbose=False)

        sample = load_octree_sample(pc, 
                                    idx, 
                                    depth=self.octree_depth,
                                    full_depth=self.full_depth,
                                    augment=False)
        
        sample['cell_id'] = target_cell_id


        return sample