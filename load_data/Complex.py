"""
This script introduces the 'Complex' class that is fulling the Database object
A Complex takes as input a pdb, a mrc and some extra selection tools and outputs a grid aligned with the mrc.
"""

import os
import sys

import numpy as np
from sklearn.gaussian_process.kernels import RBF

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from utils import mrc_utils, pymol_utils

"""
Make the conversion from (n,3+features) matrices to the grid format.
"""


def just_one(coord, xi, yi, zi, sigma, feature, total_grid, use_multiprocessing=False):
    """

    :param coord: x,y,z
    :param xi: a range of x coordinates
    :param yi: a range of x coordinates
    :param zi: a range of x coordinates
    :param feature: The feature to put around this given point
    :param sigma: The scale of RBF
    :param total_grid: The grid to add to
    :param use_multiprocessing: If this is to be used in a multiprocessing setting
    :return:
    """
    #  Find subgrid
    nx, ny, nz = xi.size, yi.size, zi.size

    bound = int(4 * sigma)
    x, y, z = coord
    binx = np.digitize(x, xi)
    biny = np.digitize(y, yi)
    binz = np.digitize(z, zi)
    min_bounds_x, max_bounds_x = max(0, binx - bound), min(nx, binx + bound)
    min_bounds_y, max_bounds_y = max(0, biny - bound), min(ny, biny + bound)
    min_bounds_z, max_bounds_z = max(0, binz - bound), min(nz, binz + bound)

    X, Y, Z = np.meshgrid(xi[min_bounds_x: max_bounds_x],
                          yi[min_bounds_y: max_bounds_y],
                          zi[min_bounds_z:max_bounds_z],
                          indexing='ij')
    X, Y, Z = X.flatten(), Y.flatten(), Z.flatten()

    #  Compute RBF
    rbf = RBF(sigma)
    subgrid = rbf(coord, np.c_[X, Y, Z])
    subgrid = subgrid.reshape((max_bounds_x - min_bounds_x,
                               max_bounds_y - min_bounds_y,
                               max_bounds_z - min_bounds_z))

    # Broadcast the feature throughout the local grid.
    subgrid = subgrid[None, ...]
    feature = feature[:, None, None, None]
    subgrid_feature = subgrid * feature

    #  Add on the first grid
    if not use_multiprocessing:
        total_grid[:,
        min_bounds_x: max_bounds_x,
        min_bounds_y: max_bounds_y,
        min_bounds_z:max_bounds_z] += subgrid_feature
    else:
        return min_bounds_x, max_bounds_x, min_bounds_y, max_bounds_y, min_bounds_z, max_bounds_z, subgrid_feature


def fill_grid_from_coords(coords, bins, features=None, sigma=1.):
    """
    Generate a grid from the coordinates
    :param coords: (n,3) array
    :param bins: 3 arrays of bins. They can originate from raw coords or from another mrc.
    :param features: (n,k) array or None
    :param sigma:
    :return:
    """

    xi, yi, zi = bins
    nx, ny, nz = xi.size, yi.size, zi.size
    features = np.ones((len(coords), 1)) if features is None else features
    feature_len = features.shape[1]
    total_grid = np.zeros(shape=(feature_len, nx, ny, nz))

    for i, coord in enumerate(coords):
        just_one(coord, feature=features[i], xi=xi, yi=yi, zi=zi, sigma=sigma, total_grid=total_grid)

    return total_grid.astype(np.float32)


def get_bins(coords, spacing, padding, xyz_min=None, xyz_max=None):
    """
    Compute the 3D bins from the coordinates
    """
    if xyz_min is None:
        xm, ym, zm = coords.min(axis=0) - padding
    else:
        xm, ym, zm = xyz_min - padding
    if xyz_max is None:
        xM, yM, zM = coords.max(axis=0) + padding
    else:
        xM, yM, zM = xyz_max + padding

    xi = np.arange(xm, xM, spacing)
    yi = np.arange(ym, yM, spacing)
    zi = np.arange(zm, zM, spacing)
    return xi, yi, zi


def build_grid_from_coords(coords, features=None, spacing=2., padding=3, xyz_min=None, xyz_max=None, sigma=1.):
    """
    Generate a grid from the coordinates
    :param coords: (n,3) array
    :param features: (n,k) array or None
    :param spacing:
    :param padding:
    :param xyz_min:
    :param xyz_max:
    :param sigma:
    :return:
    """

    return fill_grid_from_coords(coords=coords,
                                 features=features,
                                 bins=get_bins(coords, spacing, padding, xyz_min, xyz_max),
                                 sigma=sigma)


class Complex:
    """
    Object containing a protein and a density
    The main difficulty arises from the creation of the grid for the output,
      because we need those to align with the input mrc
    """

    def __init__(self, mrc_path, pdb_name, pdb_path, antibody_selection=None):
        # First get the MRC data
        mrc = mrc_utils.MRCGrid(mrc_path)

        # Then get the corresponding empty grid, this follows 'resample' with origin offset
        bins = [np.arange(start=mrc.origin[i],
                          stop=(mrc.origin + mrc.data.shape * mrc.voxel_size)[i],
                          step=mrc.voxel_size[i])
                for i in range(3)]

        # Now let's get the relevant coordinates to embed in this grid
        antibody_coords = pymol_utils.get_protein_coords(pdb_name=pdb_name, pdb_path=pdb_path,
                                                         pymol_selection=antibody_selection)
        antigen_coords = pymol_utils.get_protein_coords(pdb_name=pdb_name, pdb_path=pdb_path,
                                                        pymol_selection=f"not ({antibody_selection})")

        # Get the corresponding grid
        antibody_grid = fill_grid_from_coords(coords=antibody_coords, bins=bins)
        antigen_grid = fill_grid_from_coords(coords=antigen_coords, bins=bins)
        void_grid = np.maximum(0, 1 - antibody_grid - antigen_grid)
        target_tensor = np.concatenate((antibody_grid, antigen_grid, void_grid))

        self.mrc = mrc
        self.target_tensor = target_tensor
        # self.save_mrc_lig()
        pass


if __name__ == '__main__':
    # dataset = ABDataset()
    # point = dataset[17]
    # print(point)

    # systems = process_csv('../data/reduced_clean.csv')
    # print(systems)

    comp = Complex(mrc='../data/pdb_em/3IXX_5103/5103_carved.mrc',
                   pdb_path='../data/pdb_em/3IXX_5103/3IXX.mmtf.gz',
                   pdb_name='3IXX',
                   antibody_selection='chain G or chain H or chain I or chain J')