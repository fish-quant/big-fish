# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Utility functions for bigfish.data subpackage.
"""

import os
import bigfish.stack as stack


# ### Global variables ###

filename_input_dapi = "experience_1_dapi_fov_1.tif"
url_input_dapi = "https://github.com/fish-quant/big-fish/releases/download/data/experience_1_dapi_fov_1.tif"
hash_input_dapi = "3ce6dcfbece75da41326943432ada4cc9bacd06750e59dc2818bb253b6e7fdcd"
filename_input_smfish = "experience_1_smfish_fov_1.tif"
url_input_smfish = "https://github.com/fish-quant/big-fish/releases/download/data/experience_1_smfish_fov_1.tif"
hash_input_smfish = "bc6aec1f3da4c25f3c6b579c274584ce1e88112c7f980e5437b5ad5223bc8ff6"


# ### Function ###

def check_input_data(input_directory):
    """Check input images exists and download them if necessary.

    Returns
    -------
    input_directory : str
        Path of the input data directory.

    """
    # check if input dapi image exists
    path = os.path.join(input_directory, filename_input_dapi)
    if os.path.isfile(path):

        # check that image is not corrupted
        try:
            stack.check_hash(path, hash_input_dapi)
            print("{0} is already in the directory"
                  .format(filename_input_dapi))

        # otherwise download it
        except IOError:
            print("{0} seems corrupted".format(filename_input_dapi))
            print("downloading {0}...".format(filename_input_dapi))
            stack.load_and_save_url(url_input_dapi,
                                    input_directory,
                                    filename_input_dapi)
            stack.check_hash(path, hash_input_dapi)

    # if file does not exist we directly download it
    else:
        print("downloading {0}...".format(filename_input_dapi))
        stack.load_and_save_url(url_input_dapi,
                                input_directory,
                                filename_input_dapi)
        stack.check_hash(path, hash_input_dapi)

    # check if input smfish image exists
    path = os.path.join(input_directory, filename_input_smfish)
    if os.path.isfile(path):

        # check that image is not corrupted
        try:
            stack.check_hash(path, hash_input_smfish)
            print("{0} is already in the directory"
                  .format(filename_input_smfish))

        # otherwise download it
        except IOError:
            print("{0} seems corrupted".format(filename_input_smfish))
            print("downloading {0}...".format(filename_input_smfish))
            stack.load_and_save_url(url_input_smfish,
                                    input_directory,
                                    filename_input_smfish)
            stack.check_hash(path, hash_input_smfish)

    # if file does not exist we directly download it
    else:
        print("downloading {0}...".format(filename_input_smfish))
        stack.load_and_save_url(url_input_smfish,
                                input_directory,
                                filename_input_smfish)
        stack.check_hash(path, hash_input_smfish)

    return
