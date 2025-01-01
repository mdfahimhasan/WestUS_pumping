# Author : Md Fahim Hasan
# PhD Candidate
# Colorado State university
# Fahim.Hasan@colostate.edu

import os
import sys
from glob import glob
from os.path import dirname, abspath

sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.download_preprocess.tiles_utils import train_val_test_split

# ----------------------------------------------------------------------------------------------------------------------
# 1. Train-validation-test split
# ----------------------------------------------------------------------------------------------------------------------

# directories
multiband_tile_dir = '../../Data_main/rasters/multibands/training/tiles'
target_csv = '../../Data_main/rasters/multibands/training/tiles/target.csv'
train_dir = '../../Data_main/rasters/multibands/train_val_test/train'
val_dir = '../../Data_main/rasters/multibands/train_val_test/val'
test_dir = '../../Data_main/rasters/multibands/train_val_test/test'

# flags
skip_split_train_val_test = False  ######################################################################################

train_val_test_split(target_data_csv=target_csv, input_tile_dir=multiband_tile_dir,
                     train_dir=train_dir, val_dir=val_dir, test_dir=test_dir,
                     train_size=0.7, val_size=0.2, test_size=0.1,
                     random_state=42, skip_processing=skip_split_train_val_test)


