# NOTE: need to run in terminal from root dir as:
# uv run -m C_inference.s1_create_grid

from shapely.geometry import Polygon
import os
import geopandas as gpd
import numpy as np
from tqdm.auto import tqdm
import time


def grid_and_centroid(tiles_gdf, res, tile_fpath_field, 
                          centroids_out_path, updated_tiles_out_path):
    """
    Creates a shapefile grid with specified resolution from an input polygon (lidar index).
    The grid is then intersected with the lidar index to get overlapping lidar tiles for each grid cell.
    Centroids are also created for each grid cell to be used for clipping the lidar tiles.

    :param tiles_gdf: Geopandas dataframe containing lidar index for which to create a grid
    :param res: resolution of each grid cell (in metres)
    :param tile_fpath_field: field name for column that contains the lidar tile filepath
    :param centroids_out_path: path for output centroids shapefile
    :param updated_tiles_out_path: path for output lidar index file with grid cell ids
    :return: grid and centroids geopandas dataframes
    """

    # Record start time and date
    start_time = time.time()

    # Extract name for each tile
    tiles_gdf['tile_name'] = tiles_gdf[tile_fpath_field].apply(lambda x: os.path.basename(x).split(".")[0])

    # Get bounding box coordinates
    xmin, ymin, xmax, ymax = tiles_gdf.total_bounds

    cols = list(np.arange(xmin, xmax + res, res))
    rows = list(np.arange(ymin, ymax + res, res))

    # Build polygon grid
    polygons = []
    for x in tqdm(cols[:-1], desc="Building grid..."):
        for y in rows[:-1]:
            polygons.append(Polygon([(x, y), (x + res, y), (x + res, y + res), (x, y + res)]))

    grid = gpd.GeoDataFrame({'geometry': polygons}, crs=tiles_gdf.crs)

    # Assign unique IDs to each grid cell
    grid['cell_id'] = np.arange(1, len(grid) + 1)

    # Perform intersection of grid and original gdf to retain information
    print("Performing intersection between grid and lidar tiles...")
    grid_intsct = grid.overlay(tiles_gdf, how='intersection')

    # Extract the cell ids that overlap with multiple lidar tiles
    tiles_w_cells_df = (grid_intsct.groupby('tile_name').\
                                        agg(list)[["cell_id"]].\
                                        reset_index().\
                                        rename(columns={"cell_id": "cell_id_ls"}))

    #Count the number of cells in each tile
    tiles_w_cells_df['cell_count'] = tiles_w_cells_df['cell_id_ls'].apply(lambda x: len(x))

    # Convert list of cell ids to comma separated string
    tiles_w_cells_df['cell_id_ls'] = [",".join(map(str, id_ls)) for id_ls in tiles_w_cells_df['cell_id_ls']]

    #Re-add the geometry for the tiles index
    tiles_gdf = tiles_gdf.merge(tiles_w_cells_df, on='tile_name', how='left')

    # Get centroid of each grid cell
    centroids = gpd.GeoDataFrame({'geometry': grid.centroid.geometry,'cell_id': grid['cell_id']}, crs=tiles_gdf.crs)
    centroids.reset_index(drop=True, inplace=True)

    # Export results

    print("Exporting grid and centroids...")

    centroids.to_file(centroids_out_path)

    tiles_gdf.to_file(updated_tiles_out_path)

    # Record runtime
    end_time = time.time()
    runtime_minutes = round((end_time - start_time) / 60, 2)

    print(f"Finished building grid. Time required:\n{runtime_minutes} minutes.")


if __name__ == "__main__":

    RMF_INDEX = gpd.read_file("E:/RMF/RMF_SPL100/LAS_Classified_Point_Clouds_Normalized_index.gpkg")

    OUT_DIR = r"E:/rq3_rmf_inference"

    OUT_CELL_RES = 11.28 * 2 

    # Create grid and centroids
    grid_and_centroid(tiles_gdf=RMF_INDEX,
                        res=OUT_CELL_RES,
                        tile_fpath_field="filename",
                        centroids_out_path=os.path.join(OUT_DIR, "rmf_centroids.gpkg"),
                        updated_tiles_out_path=os.path.join(OUT_DIR, "rmf_tiles_with_cells.gpkg"))

