import numpy as np
from laspy import read
import pandas as pd
import torch


def pc_to_unit_sphere(pc, verbose=True):
    """
    :param pc: point cloud with shape (n, 3)
    :return: point cloud with shape (n, 3) that is normalized to a unit sphere
    """
    try: 
        max_norm = np.max(
            np.linalg.norm(pc[:, :3], axis=-1, keepdims=True),
            axis=-2,
            keepdims=True,
        )
        pc[:, :3] = pc[:, :3] / max_norm
    except Exception as e:
        if verbose:
            print(f"Error normalizing point cloud to unit sphere: {e}")
            print(f"Point cloud shape: {pc.shape}")

    return pc


def read_las_to_np(las_fpath, centralize_coords, unit_sphere):
    """
    :param las_fpath: filepath to las file
    :param centralize_coords: whether to make all coords relative to center of point cloud (center is 0,0,0)
    :return: point cloud numpy array with shape (n, 3) where n is the number of points, and 3 is the x, y, z coords
    :param unit_sphere: whether to convert point cloud to unit sphere
    """

    # Read LAS for given plot ID
    inFile = read(las_fpath)

    # Get coords coordinates
    points = np.vstack([inFile.x, inFile.y, inFile.z]).transpose()

    # Centralize coordinates to a unit sphere following the method in PointNet++
    if centralize_coords:
        points = points - np.mean(points, axis=0)

    # Convert to unit sphere
    if unit_sphere:
        points = pc_to_unit_sphere(points)

    # Check for NANs and report
    if np.isnan(points).any():
        raise ValueError('NaN values in input point cloud!')

    return points




def update_z_score_conversion_info(biomass_labels_fpath, biomass_comp_nms):
    """
    Loads a csv of the reference data and gets the mean and sd for use in converting back from z score.
    Global config dictionary (cfg) is updated with these values for each biomass component for conversion during training.

    :param biomass_comp_nms: list of biomass component names in the order they are predicted by the model
    :param biomass_labels_fpath: path to biomass labels csv
    :return: updated global config dictionary.
    """

    ref_data = pd.read_csv(biomass_labels_fpath)

    z_info = dict()

    for comp in biomass_comp_nms:
        z_info[f'{comp}_Mg_ha_mn'] = np.mean(ref_data[f'{comp}_Mg_ha'])
        z_info[f'{comp}_Mg_ha_sd'] = np.std(ref_data[f'{comp}_Mg_ha'])

    return z_info


def convert_vals_to_z_score(values):
    """
    Converts a list of values to z-score format

    Z = (X - mean) / standard_deviation

    :param values: values to be converted to z-score
    :return: input values converted to z-scores
    """

    mean = np.mean(values)
    sd = np.std(values)

    z_scored_values = [(x - mean) / sd for x in values]

    return z_scored_values

def convert_from_z_score(z_vals, sd, mean):
    """
    Converts z-score back to original value using mean and sd
    :param z_vals: z-score values to be converted
    :param sd: standard deviation of original data
    :param mean: mean of original data
    :return: input values converted to back to original units
    """

    # X = Z * standard_deviation + mean
    z_score_val = z_vals * sd + mean

    return z_score_val


def convert_z_to_mg_ha(z_info: dict, z_components_arr: np.array, biomass_comp_nms: tuple):
    """
    Converts array of component z score value back to biomass value in Mg/ha
    ***IMPORTANT: array needs to enter function with columns as follows: bark, branch, foliage, wood

    :param z_info: dict that contains the mean and sd values for each component needed for conversion
    :param z_components_arr: input np array of 'branch', 'bark', 'foliage', 'wood' values (in z score format)
    :param biomass_comp_nms: list of biomass component names in the order they are predicted by the model
    :return: tensor -> input values converted to Mg/ha units, note that this tensor no longer has gradients and is only for calculating performance metrics
    """

    # Convert tensor to np array on CPU
    if torch.is_tensor(z_components_arr):
        converted_arr = z_components_arr.detach().cpu().numpy()
    else:
        converted_arr = z_components_arr

    # Re-convert z-score to original value for each component
    for col_number, comp in enumerate(biomass_comp_nms):

        #Ensure array is correct shape
        if converted_arr.ndim < 2:
            print(f"\n\033[93mArray shape is incorrect: {converted_arr.shape} reshaping\033[0m\n")
            converted_arr = np.expand_dims(converted_arr, axis=0)

        comp_z_vals = converted_arr[:, col_number]
        converted_arr[:, col_number] = convert_from_z_score(z_vals=comp_z_vals,
                                                            sd=z_info[f'{comp}_Mg_ha_sd'],
                                                            mean=z_info[f'{comp}_Mg_ha_mn'])

    return converted_arr
