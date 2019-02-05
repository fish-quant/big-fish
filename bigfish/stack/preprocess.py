# -*- coding: utf-8 -*-

"""
Functions used to format any input tensor loaded in bigfish.
"""

import os

import numpy as np
import pandas as pd

from scipy.sparse import coo_matrix
from scipy import ndimage as ndi










def check_volume(cyto_coord, nuc_coord):
    """
    cyto_coord: list
    nuc_coord: list
    """
    # get coordinates
    cyto = np.array(cyto_coord)
    nuc = np.array(nuc_coord)

    max_x = max(cyto[:, 0].max() + 5, nuc[:, 0].max() + 5)
    max_y = max(cyto[:, 1].max() + 5, nuc[:, 1].max() + 5)

    # build the dense representation for the cytoplasm
    values = [1] * cyto.shape[0]
    cyto = coo_matrix((values, (cyto[:, 0], cyto[:, 1])),
                      shape=(max_x, max_y)).todense()

    # build the dense representation for the nucleus
    values = [1] * nuc.shape[0]
    nuc = coo_matrix((values, (nuc[:, 0], nuc[:, 1])),
                     shape=(max_x, max_y)).todense()

    # check if the volume is valid
    mask_cyto = ndi.binary_fill_holes(cyto)
    mask_nuc = ndi.binary_fill_holes(nuc)
    frame = np.zeros((max_x, max_y))
    diff = frame - mask_cyto + mask_nuc
    diff = (diff > 0).sum()

    if diff > 0:
        return False
    else:
        return True

data_cell["valid"] = data_cell.apply(lambda row: check_volume(row["pos_cell"], row["pos_nuc"]), axis=1)


background_to_remove = []
for i in data_cell.index:
    if np.logical_not(data_cell.loc[i, "valid"]):
        plot_volume(data_cell, i)
        background_to_remove.append(data_cell.loc[i, "name_img_BGD"])

data_clean = data[~data["name_img_BGD"].isin(background_to_remove)]
print(data.shape)
print(data_clean.shape)


def check_rna(rna_coord, nb_rna):
    """
    rna_coord: list
    nb_rna: int
    """
    return nb_rna - len(rna_coord)


data_clean.apply(lambda row: check_rna(row["RNA_pos"], row["n_RNA"]), axis=1).value_counts()


def check_rna(cyto_coord, rna_coord):
    """
    cyto_coord: list
    rna_coord: list
    """
    # get coordinates
    cyto = np.array(cyto_coord)
    if not isinstance(rna_coord[0], list):
        # it means we have only one spot
        return False
    rna = np.array(rna_coord)

    # check if the coordinates are positive
    if rna.min() < 0:
        return False

    max_x = int(max(cyto[:, 0].max() + 5, rna[:, 0].max() + 5))
    max_y = int(max(cyto[:, 1].max() + 5, rna[:, 1].max() + 5))

    # build the dense representation for the cytoplasm
    values = [1] * cyto.shape[0]
    cyto = coo_matrix((values, (cyto[:, 0], cyto[:, 1])),
                      shape=(max_x, max_y)).todense()

    # build the dense representation for the rna
    values = [1] * rna.shape[0]
    rna = coo_matrix((values, (rna[:, 0], rna[:, 1])),
                     shape=(max_x, max_y)).todense()
    rna = (rna > 0)

    # check if the coordinates are valid
    mask_cyto = ndi.binary_fill_holes(cyto)
    frame = np.zeros((max_x, max_y))
    diff = frame - mask_cyto + rna
    diff = (diff > 0).sum()

    if diff > 0:
        return False
    else:
        return True

data_clean["valid"] = data_clean.apply(lambda row: check_rna(row["pos_cell"], row["RNA_pos"]), axis=1)

data_clean = data_clean[data_clean["valid"]]
print(data_clean.shape)
data_clean.head()

def count_rna(rna_coord):
    """
    rna_coord: list, rna spots coordinates
    """
    return len(rna_coord)

data_clean["nb_rna"] = data_clean.apply(lambda row: count_rna(row["RNA_pos"]), axis=1)


data_final = data_clean[['RNA_pos', 'cell_ID', 'pattern_level', 'pattern_name', 'pos_cell', 'pos_nuc', "nb_rna"]]
print(data_final.shape)
data_final.head()


path_output = os.path.join(main_directory, "data_cleaned")
data_final.to_pickle(path_output)