
import matplotlib.pyplot as plt
import numpy as np

def plot_pc(pc, fig=None, ax=None, label=None,  pt_size=60, plt_size=5, cmap='jet',
            margin_lim=0.05, zmin=None, zmax=None, bg_col="black", text_col="white", text_size=15, 
            out_fpath=None, camera_elev=25, camera_azim=45):
    """
    Function for plotting point cloud data.
    :param pc: input numpy array with shape (n, 3) where n is number of points and 3 is x, y, z coordinates
    :param fig: figure for the plot (if None, a new figure is created)
    :param ax: axis for the plot (if None, a new figure is created)
    :param label: label for the plot
    :param fpath: filename for saving the plot
    :param pt_size: size of points in the plot
    :param plt_size: size of the plot (plt_size x plt_size)
    :param cmap: colormap for the plot
    :param patch_radius: radius for the rounded rectangle patch
    :param patch_pad: padding for the rounded rectangle patch
    :param margin_lim: margin for the plot
    :param zmin: minimum z value for the plot
    :param zmax: maximum z value for the plot
    :param bg_col: background color for the plot
    :param text_col: text color for the plot
    :param text_size: text size for the plot
    :param out_fpath: output path for saving the plot
    :param camera_elev: elevation angle for the camera
    :param camera_azim: azimuth angle for the camera
    :return:
    """

    if fig is None:
        fig = plt.figure(figsize=(plt_size, plt_size))
        plt.subplots_adjust(top=1 - margin_lim,
                        bottom=0 + margin_lim,
                        right=1 - margin_lim,
                        left=0 + margin_lim,
                        hspace=0,
                        wspace=0)
    
    if ax is None:
        ax = fig.add_subplot(projection='3d')

    ax.scatter(pc[:, 0],
               pc[:, 1],
               pc[:, 2],
               c=pc[:, 2],
               cmap=cmap,
               linewidth=1,
               alpha=1,
               s=pt_size)

    # Get point cloud bounds
    x_min, x_max = pc[:, 0].min(), pc[:, 0].max()
    ymin, y_max = pc[:, 1].min(), pc[:, 1].max()

    if zmin is None:
        zmin = pc[:, 2].min()
    
    if zmax is None:
        zmax = pc[:, 2].max()

    # Remove axis
    ax.set_axis_off()

    # Set the background color
    ax.set_facecolor(bg_col)

    # Set camera position
    ax.view_init(elev=camera_elev, azim=camera_azim) 

    if label is not None:
        ax.title.set_text(label)
        ax.title.set_color(text_col)
        ax.title.set_fontsize(text_size)
        ax.title.set_fontweight(800)
    
    if out_fpath is not None:
        plt.savefig(out_fpath, dpi=400, bbox_inches='tight', pad_inches=0)
        print(f"Saved plot to {out_fpath}")


def plot_voxels(xyz, resolution=1, bg_col="white", out_fpath=None, camera_elev=25, camera_azim=45,
                linewidth=0.25, edgecolor='k', plt_size=5, margin_lim=0.1):
    """
    Converts XYZ coordinates to a voxel grid, assigns colors based on height (Z-axis),
    and plots the voxels in 3D space.
    :param xyz: input numpy array with shape (n, 3) where n is number of points and 3 is x, y, z coordinates
    :param resolution: size of each voxel
    :param bg_col: background color for the plot
    :param out_fpath: output path for saving the plot
    :param camera_elev: elevation angle for the camera
    :param camera_azim: azimuth angle for the camera
    :param linewidth: line width for voxel edges
    :param edgecolor: color for voxel edges
    :param plt_size: size of the plot
    """

    # Normalize the coordinates to fit into a grid
    xyz_min = np.min(xyz, axis=0)
    xyz_max = np.max(xyz, axis=0)
    grid_size = np.ceil((xyz_max - xyz_min) / resolution).astype(int)

    # Create a voxel grid
    voxel_grid = np.zeros(grid_size, dtype=bool)
    indices = ((xyz - xyz_min) / resolution).astype(int)
    voxel_grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True

    # Assign colors based on height (Z-axis)
    z_values = indices[:, 2]
    z_min, z_max = np.min(z_values), np.max(z_values)
    z_norm = (z_values - z_min) / (z_max - z_min)  # Normalize Z values to [0, 1]
    colors = plt.cm.jet(z_norm)  # Use the "jet" colormap

    # Create a color array for the voxel grid
    color_grid = np.zeros(voxel_grid.shape + (4,), dtype=float)  # RGBA
    for i, (x, y, z) in enumerate(indices):
        color_grid[x, y, z] = colors[i]

    # Plot the voxel grid
    fig = plt.figure(figsize=(plt_size, plt_size))
    plt.subplots_adjust(top=1 - margin_lim,
                bottom=0 + margin_lim,
                right=1 - margin_lim,
                left=0 + margin_lim,
                hspace=0,
                wspace=0)
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(voxel_grid, facecolors=color_grid, edgecolor=edgecolor, linewidth=linewidth)

    # Set axis limits to zoom in on the voxel grid
    ax.set_xlim(0, grid_size[0])
    ax.set_ylim(0, grid_size[1])
    ax.set_zlim(0, grid_size[2])

    # Remove axis
    ax.set_axis_off()

    # Set the background color
    ax.set_facecolor(bg_col)

    # Set camera position
    ax.view_init(elev=camera_elev, azim=camera_azim) 

    # Ensure axes are equal
    ax.set_aspect('equal')

    if out_fpath is not None:
        plt.savefig(out_fpath, dpi=400, bbox_inches='tight', pad_inches=0)
        print(f"Saved plot to {out_fpath}")

    plt.show()