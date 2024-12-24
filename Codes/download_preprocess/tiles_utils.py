# # Steps of implementation for readying datasets for the DL model

# create tiles (we will do odd size images, preferred size is 7*7 or 9*9 for now)
# randomly split train-test-validation data
# standardize or normalize input variables of training data. Might have to consider mean and std of all input variables (of training data)
# generally standardizing/normalizing both input features and target variables are good practice (normalizing target variable might hurt interpretability)
# use DataLoader to create batches with standardized data (DataLoader class in with the CNN script as that uses pyTorch)

import os
import shutil
import pickle
import numpy as np
import pandas as pd
from glob import glob
import rasterio as rio
from rasterio.windows import Window, transform
from sklearn.model_selection import train_test_split

from Codes.utils.system_ops import makedirs
from Codes.utils.raster_ops import read_raster_arr_object, write_array_to_raster

no_data_value = -9999


def org_vars(list_of_temporal_var_dirs, years_list, list_of_static_var_dirs=None, month_range=None):
    """
    Arranges the variables (their paths) in an order that all input variables/features of a year/month or static
    will be in a nested list inside the main output list.

    :param month_range: Month range, as (4, 10), for which variables/datasets are available.
                        Default set to None as our primary target is annual datasets.
    :param list_of_temporal_var_dirs: List of main directory files of the temporal variables.
    :param years_list: A list of years for which variables/datasets are available.
    :param list_of_static_var_dirs: List of main directory files of the static variables. Default set to None.

    :return: A list that consists of multiple nested lists. Each nested list contains the file paths of all variables
             of a particular year and month.
    """
    # # processing temporal variables
    if month_range is None:  # if the datasets are annual
        all_data_paths = []  # final list to append the data paths

        # 1st loop for each year and 2nd loop for each dataset
        for year in years_list:
            data_paths = []

            for var in list_of_temporal_var_dirs:
                data = glob(os.path.join(var, f'*{year}.*tif'))

                data_paths.extend(data)

            all_data_paths.append(data_paths)

    else:  # if the datasets are monthly
        all_data_paths = []  # final list to append the data paths

        # 1st loop for each year, 2nd loop for each month, and 2nd loop for each dataset
        for year in years_list:
            months = list(range(month_range[0], month_range[1] + 1))

            for mon in months:
                data_paths = []
                for var in list_of_temporal_var_dirs:
                    data = glob(os.path.join(var, f'*{year}_{mon}.*tif'))

                    data_paths.extend(data)

                all_data_paths.append(data_paths)

    # # processing static variables
    if list_of_static_var_dirs is not None:
        for var in list_of_static_var_dirs:
            data = glob(os.path.join(var, '*.tif'))[0]

            for nes_list in all_data_paths:
                nes_list.append(data)

    return all_data_paths


def create_multiband_raster(input_files_list, band_key_list, output_file, nodata=no_data_value):
    """
    Create a multi-band image from a list of images.

    *** The output files can be arranged with temporal bands (each representing a particular time's dataset)
        or feature (each band representing a variable)

    :param input_files_list: List of image file paths to be included in the multi-band image.
    :param band_key_list: List of strings that will be added before the band names while saving.
                          Should be in the same serial as input_files_list

    :param output_file: Filepath of output raster.
    :param nodata: Default set to -9999.

    :return: None
    """
    # reading first dataset to extract essential metadata
    raster_arr, raster_file = read_raster_arr_object(input_files_list[0])
    height, width = raster_arr.shape[0], raster_arr.shape[1]

    # opening the output file in write mode and saving each band
    with rio.open(
            output_file,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            dtype=raster_arr.dtype,
            count=len(input_files_list),
            crs=raster_file.crs,
            transform=raster_file.transform,
            nodata=nodata
    ) as dst:
        for id, (layer, layer_name) in enumerate(zip(input_files_list, band_key_list), start=1):
            with rio.open(layer) as src:
                dst.write_band(id, src.read(1))
                dst.set_band_description(id, layer_name)


def make_multiband_datasets(list_of_temporal_var_dirs, list_of_static_var_dirs,
                            band_key_list, output_dir,
                            years_list, skip_processing=False):
    """
    Make multi-band raster dataset from individual temporal/static attributes.

    **Caution**:
        1. The first band of the multiband raster must be target band (train data) because the make_training_tiles()
        class wil consider the first band as trainingd data.


    :param list_of_temporal_var_dirs: List of directories of temporal attributes/datasets.
    :param list_of_static_var_dirs: List of directories of static attributes/datasets.
    :param band_key_list: keyword to add as band names in the multi-band dataset.
    :param output_dir: Path of output dir where the multi-band datasets will be stored.
    :param years_list: List of years to process the datasets.
    :param skip_processing: Set to True to skip this process.

    :return: None.
    """
    if not skip_processing:
        print('creating multi-band rasters...')

        global output_file_path

        makedirs([output_dir])

        # arranging the variables (their paths) in an order so that all input variables of a year/month will be in a
        # nested list inside the main output list
        data_paths_lists = org_vars(list_of_temporal_var_dirs=list_of_temporal_var_dirs,
                                    list_of_static_var_dirs=list_of_static_var_dirs,
                                    years_list=years_list)

        # creating a nan mask (1 - valid, 0 -  nan) raster and adding it's path to each nested list in data_paths_lists
        data_paths_lists_with_nan = []

        for nan_no, paths in enumerate(data_paths_lists):
            path_arrs = np.array([read_raster_arr_object(each, get_file=False) for each in paths])
            nan_mask = np.any(np.isnan(path_arrs), axis=0)

            # inverting the mask: True becomes 0, False becomes 1
            # Then, converting boolean to integer: True as 0, False as 1
            inverted_mask = ~nan_mask
            nan_arr = inverted_mask.astype(int)

            # saving as a raster and adding to each nested list in data_paths_lists
            nan_mask_dir = '../../Data_main/rasters/nan_mask'
            makedirs([nan_mask_dir])

            nan_raster_path = os.path.join(nan_mask_dir, f'nanmask{nan_no + 1}.tif')
            _, file = read_raster_arr_object(paths[0])
            write_array_to_raster(nan_arr, file, file.transform, nan_raster_path)

            data_paths_lists_with_nan.append(paths + [nan_raster_path])

        # # multi-band raster creation
        for paths in data_paths_lists_with_nan:
            # checking to see if the data is annual or monthly data, and then setting output name accordingly.
            # basically checking if the last block of the name is 4 digit (for year), 2 digit (month) or str (static data).

            last_name_block = os.path.basename(paths[0]).split('.')[0].split('_')[-1]

            if (len(last_name_block) == 4) & (last_name_block.isdigit()):  # check for annual data
                year = os.path.basename(paths[0]).split('.')[0].split('_')[-1]
                output_file_path = os.path.join(output_dir, f'{year}.tif')

            elif (len(last_name_block) <= 2) & (last_name_block.isdigit()):  # check for monthly data
                year = os.path.basename(paths[0]).split('.')[0].split('_')[-2]
                month = os.path.basename(paths[0]).split('.')[0].split('_')[-1]
                output_file_path = os.path.join(output_dir, f'{year}_{month}.tif')

            # creating multi-band rasters
            create_multiband_raster(input_files_list=paths, band_key_list=band_key_list, output_file=output_file_path)


class make_training_tiles:
    """
    Processes a multi-band raster image into tiles of training data and associated feature attributes.

     **Caution**:
        1. The first band of the multiband raster must be target band (train data). This class extracts the center pixel
        value of the first band as the target variable for each loop and saves it separately, while the remaining
        bands are used as features (saved a tiled multi-band GeoTIFF) for the deep learning model.

        2. The `band_key_list` should only contain the names of the feature bands and exclude the target variable's name.
    """

    def __init__(self, tiff_path, band_key_list, tile_output_dir, target_data_output_csv,
                 tile_size=7, nodata_value=-9999, nodata_threshold=50,
                 skip_processing=False):
        """
        :param tiff_path (str): Path to the multi-band raster TIFF file to process.
        :param band_key_list (list): List of keys or descriptions for each band.
        :param tile_output_dir (str): Directory where the processed tiles will be saved.
        :param target_data_output_csv (str): Filepath to save the target training data as CSV.
        :param tile_size (int): Size of the square tile (e.g., 7x7). Must be odd to ensure a center pixel.
        :param nodata_value (int/float): NoData value in the raster (default: -9999).
        :param nodata_threshold (int): Maximum percentage of NoData values allowed in a tile (default: 50).
        :param skip_processing (bool): If True, skips processing and initializes the class without performing operations.
        """
        self.tile_output_dir = tile_output_dir
        self.target_data_output_csv = target_data_output_csv
        self.tile_size = tile_size
        self.nodata_value = nodata_value
        self.nodata_threshold = nodata_threshold

        if not skip_processing:
            self.process_tiles(tiff_path, band_key_list)

    def process_tiles(self, tiff_path, band_key_list):
        """
        Processes a multi-band raster image into tiles of training data and associated feature attributes.

        :param tiff_path: Path to the multi-band raster TIFF file to process.
        :param band_key_list: List of keys or descriptions for each band.

        :return: None.
        """
        makedirs([self.tile_output_dir])

        # initiating training data storage dictionary
        target_data = {'tile_no': [], 'target_value': []}

        # opening a multi-band image file. The tile-ing operation will be done inside the opened image file.
        with rio.open(tiff_path) as tiff:
            tiff_height, tiff_width = tiff.height, tiff.width
            tile_radius = self.tile_size // 2

            # reading the training data band
            training_band_index = 0  # The training data is set as band 0 in the multiband dataset in make_multiband_datasets() function
            training_band = tiff.read(training_band_index + 1)  # added +1 as rasterio read uses 1-based index

            # initiating tile number
            tile_no = 1

            # The first loop iterates across the height (rows) and the second loop across the width (columns) of the raster.
            # Both loops start at `tile_radius` to ensure the tile remains fully within the raster boundaries, avoiding edge cases.
            # Both loops end at `tiff_height - tile_radius` (for rows) and `tiff_width - tile_radius` (for columns) to handle edge cases.
            # The loops move by 1 cell at a time, check if there is a valid training data value at the center pixel,
            # and create a tile around it if valid.
            for row in range(tile_radius, tiff_height - tile_radius):
                for col in range(tile_radius, tiff_width - tile_radius):
                    center_value = training_band[row, col]

                    # skipping NoData center pixels
                    if center_value == self.nodata_value:
                        continue

                    # if the center value has valid value, create a window around it and read the data
                    window = Window(col_off=col - tile_radius, row_off=row - tile_radius,
                                    width=self.tile_size, height=self.tile_size)
                    tile_arr = tiff.read(window=window)

                    # keeping only the arrays except the first band (indexed 0, as that is training data)
                    tile_arr = tile_arr[1:]

                    # checking if any array in the windowed tiff in entirely null (only no data values)
                    if self.is_image_null(tile_arr):
                        continue

                    # checking if the tile has too many NoData values
                    nodata_percentage = self.calculate_nodata_percentage(tile_arr)
                    if any(perc > self.nodata_threshold for perc in nodata_percentage):
                        continue

                    # replacing NoData values with np.nan
                    tile_arr[tile_arr == self.nodata_value] = np.nan

                    # saving the tile
                    crs = tiff.crs
                    window_transform = transform(window, tiff.transform)  # tiled window's affine transformation

                    tile_name = f'tile_{tile_no}.tif'
                    output_file = os.path.join(self.tile_output_dir, tile_name)
                    self.save_tile(output_file, tile_arr, crs, window_transform, band_key_list)

                    # saving the target value
                    target_data['tile_no'].append(tile_no)
                    target_data['target_value'].append(center_value)

                    tile_no += 1

        # Save the training data to a CSV file
        target_df = pd.DataFrame(target_data)
        target_df.to_csv(self.target_data_output_csv, index=False)

    def calculate_nodata_percentage(self, tile_arr):
        """
        Calculates the percentage of NoData pixels in each band of the tile and returns a list.

        :param tile_arr: Multi-band tile array.

        :return: List of NoData percentages for each band.
        """
        perc_counts_all_bands = []

        for num_band in range(0, tile_arr.shape[0]):
            single_arr = tile_arr[num_band]

            valid_pixels = np.count_nonzero(np.where(single_arr != self.nodata_value, 1, 0), keepdims=False)

            total_pixels = single_arr.shape[0] * single_arr.shape[1]

            perc_no_data = (total_pixels - valid_pixels) * 100 / total_pixels

            perc_counts_all_bands.append(perc_no_data)

        return perc_counts_all_bands


    def is_image_null(self, tile_arr):
        """
        Checks all bands in an image array to see if there is an entirely null value band. A tile (image array) with
        a single data band with all null values will be rejected in the main code using this code.

        :param tile_arr: Multi-band image array.

        :return: A boolean variable.
        """
        # reading each band separately and checking if an entire band is null or not.
        # If any of the band is entirely null, this function will immidiately return True and the tile will be skipped.
        # Note that, an image where all bands have at least one valid pixel will pass this filter.
        for num_band in range(0, tile_arr.shape[0]):
            single_arr = tile_arr[num_band]

            if np.all(single_arr == self.nodata_value):
                return True

        # If no bands are null, return False
        return False

    def save_tile(self, output_file, tile_arr, crs, transform, band_key_list):
        with rio.open(
                output_file,
                'w',
                driver='GTiff',
                height=tile_arr.shape[1],
                width=tile_arr.shape[2],
                dtype=tile_arr.dtype,
                count=tile_arr.shape[0],
                crs=crs,
                transform=transform,
                nodata=np.nan
        ) as dst:
            for band_id in range(tile_arr.shape[0]):
                dst.write(tile_arr[band_id], band_id + 1)
                dst.set_band_description(band_id + 1, band_key_list[band_id])



def train_val_test_split(target_data_csv, input_tile_dir, train_dir, val_dir, test_dir,
                         train_size=0.7, val_size=0.2, test_size=0.1,
                         random_state=42, skip_processing=False):
    """
    Splits the tiles into train, validation, and test datasets, along with their target values.

    :param target_data_csv: str. Path to the CSV file containing target values and tile numbers.
    :param input_tile_dir: str. Path of input tiles containing input variables.
    :param train_dir: str. Directory to save the training tiles and target csv.
    :param val_dir: str. Directory to save the validation tiles and target csv.
    :param test_dir: str. Directory to save the test tiles and target csv.
    :param train_size: float. Proportion of the dataset to include in the train split. Default is 0.7.
    :param val_size: float. Proportion of the dataset to include in the validation split. Default is 0.2.
    :param test_size: float. Proportion of the dataset to include in the test split. Default is 0.1.
    :param random_state: int. Random seed for reproducibility. Default is 42.
    :param skip_processing: Set to True to skip processing. Default is False.

    :return: None.
    """
    if not skip_processing:
        makedirs([train_dir, val_dir, test_dir])

        # ensuring the proportions sum to 1
        if not (train_size + val_size + test_size == 1.0):
            raise ValueError('The sum of train_size, val_size, and test_size must equal 1.0')

        # loading the target data CSV
        target_data = pd.read_csv(target_data_csv)

        # performing the split for train, validation, and test based on target values
        train_data, temp_data = train_test_split(target_data, train_size=train_size, random_state=random_state)
        val_data, test_data = train_test_split(temp_data, test_size=test_size / (val_size + test_size),
                                               random_state=random_state)

        # storing splitted train, val, and test target values as csv
        train_data_df = pd.DataFrame(train_data)
        train_data_df.to_csv(os.path.join(train_dir, 'y_train.csv'), index=False)

        val_data_df = pd.DataFrame(val_data)
        val_data_df.to_csv(os.path.join(val_dir, 'y_val.csv'), index=False)

        test_data_df = pd.DataFrame(test_data)
        test_data_df.to_csv(os.path.join(test_dir, 'y_test.csv'), index=False)

        # helper function to copy corresponding tiles to train, val, and test dir based on tile_no in target dataset
        def copy_tiles(splitted_target_data_df, copy_dir):
            for _, row in splitted_target_data_df.iterrows():
                tile_no = row['tile_no']
                matching_tiles = glob(os.path.join(input_tile_dir, f'*_{tile_no}*.tif'))
                if len(matching_tiles) != 1:
                    print(f'Skipping tile_no {tile_no}: found multiple matches.')

                tile_path = matching_tiles[0]
                shutil.copy(tile_path, os.path.join(copy_dir, os.path.basename(tile_path)))

        # helper function for cleaning the directories before copying tiles
        def clean_directory(dir_path):
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
            os.makedirs(dir_path, exist_ok=True)

        # copying tiles to respective train, val, and test directories
        clean_directory(train_dir)
        clean_directory(val_dir)
        clean_directory(test_dir)

        copy_tiles(train_data, train_dir)
        copy_tiles(val_data, val_dir)
        copy_tiles(test_data, test_dir)

    else:
        pass


def calc_scaling_statistics(train_dir, skip_processing=False):
    """
    Calculates the mean, standard deviation, min, and max for each band across all tiles in the training directory.

    :param train_dir: str. Path to the directory containing training tiles.
    :param skip_processing: Set to True to skip processing. Default is False.

    :return: Four dictionaries: mean_dict, std_dict, min_dict, and max_dict.
    """
    if not skip_processing:  # calculating the statistics
        # collecting all tiles in the training data directory
        all_tiles = glob(os.path.join(train_dir, f'*.tif'))

        # getting band descriptions and no data info
        bands = rio.open(all_tiles[0]).descriptions
        nodata = rio.open(all_tiles[0]).nodata

        # initializing dictionaries to store scaling statistics
        band_mean_dict = {band: [] for band in bands}
        band_std_dict = {band: [] for band in bands}
        band_min_dict = {band: [] for band in bands}
        band_max_dict = {band: [] for band in bands}

        # processing all tiles for individual bands
        for tile in all_tiles:
            dataset = rio.open(tile).read()  # reading all input bands in a tile

            # process for each band in the tile
            for band in bands:
                # extracting band index from 'bands' list
                # then, extracting corresponding array for that band and flattening
                band_idx = bands.index(band)
                band_arr = dataset[band_idx].flatten()

                # better NaN or nodata handling
                if nodata is not None:
                    if np.isnan(nodata):  # Handle NaN as nodata
                        band_arr = band_arr[~np.isnan(band_arr)]
                    else:  # Handle non-NaN nodata values
                        band_arr = band_arr[band_arr != nodata]

                # accumulating statistics for this band
                band_mean_dict[bands[band_idx]].extend(band_arr)
                band_std_dict[bands[band_idx]].extend(band_arr)
                band_min_dict[bands[band_idx]].extend(band_arr)
                band_max_dict[bands[band_idx]].extend(band_arr)

        # target statistics
        target_df = pd.read_csv(os.path.join(train_dir, 'y_train.csv'))
        target_df.dropna(inplace=True)
        target_val = np.array(target_df['value'].tolist())

        target_mean = np.nanmean(target_val)
        target_std = np.nanstd(target_val)
        target_min = np.nanmin(target_val)
        target_max = np.nanmax(target_val)

        # finalizing the statistics across all tiles for each band
        mean_dict = {}
        std_dict = {}
        min_dict = {}
        max_dict = {}

        for band in band_mean_dict.keys():
            data = np.array(band_mean_dict[band], dtype=np.float32)
            mean_dict[band] = np.nanmean(data)
            mean_dict['target'] = target_mean

        for band in band_std_dict.keys():
            data = np.array(band_std_dict[band], dtype=np.float32)
            std_dict[band] = np.nanstd(data)
            std_dict['target'] = target_std

        for band in band_min_dict.keys():
            data = np.array(band_min_dict[band], dtype=np.float32)
            min_dict[band] = np.nanmin(data)
            min_dict['target'] = target_min

        for band in band_max_dict.keys():
            data = np.array(band_max_dict[band], dtype=np.float32)
            max_dict[band] = np.nanmax(data)
            max_dict['target'] = target_max

        # saving dictionaries as pickle
        for dict_name, dictionary in zip(
                ['mean', 'std', 'min', 'max'], [mean_dict, std_dict, min_dict, max_dict]
        ):
            pickle_path = os.path.join(train_dir, f'{dict_name}.pkl')
            with open(pickle_path, 'wb') as f:
                pickle.dump(dictionary, f)

        return mean_dict, std_dict, min_dict, max_dict

    else:  # loading the saved statistics
        mean_dict = pickle.load(open(os.path.join(train_dir, 'mean.pkl'), 'rb'))
        std_dict = pickle.load(open(os.path.join(train_dir, 'std.pkl'), 'rb'))
        min_dict = pickle.load(open(os.path.join(train_dir, 'min.pkl'), 'rb'))
        max_dict = pickle.load(open(os.path.join(train_dir, 'max.pkl'), 'rb'))

        return mean_dict, std_dict, min_dict, max_dict


def standardize_train_val_test(input_dir, mean_dict, std_dict, split_type='train', skip_processing=False):
    """
    Standardizes multi-band raster tiles and target values for train, validation, or test datasets.
    The mean and std statistics used for standardizing comes from the train_set using the dictionaries generated by
    calc_scaling_statistics.

    :param input_dir: str. Path to the directory containing the input raster tiles and target value csv.
    :param mean_dict: dictionary. A dictionary containing mean values (from train_set) for each band and the target variable.
    :param std_dict: dictionary. A dictionary containing std values (from train_set) for each band and the target variable.
    :param split_type: str. Should be something from ['train', 'val', 'test'].
    :param skip_processing: boolean. Set to True to skip this step.

    :return: None.
    """
    if not skip_processing:
        # # standardizing tiles (multi-band input variables)

        # creating new directory for standardized datasets
        output_dir = os.path.join(input_dir, 'standardized')
        makedirs([output_dir])

        # collecting all tiles in the training data directory
        all_tiles = glob(os.path.join(input_dir, f'*.tif'))

        # getting band descriptions, crs, and transform
        file = rio.open(all_tiles[0])
        bands = file.descriptions
        file_crs = file.crs
        file_transform = file.transform

        for tile in all_tiles:
            data = rio.open(tile)
            data_arr = data.read()

            if data.count != len(bands):  # existing code in case number of band names and number of array don't match
                raise ValueError("Number of bands in metadata and number of array don't match")

            if not np.isnan(data.nodata):  # ensuring no data type as np.nan as this code will set 0 to no data position
                raise ValueError(f"Expected no data value to be NaN, but got {data.nodata}.")

            # initiating a new array (with all zeros) to store standardized bands
            standardized_arr = np.zeros_like(data_arr, dtype=np.float32)

            # performing standardization for each band
            for band in bands:
                mean_val = mean_dict[band]
                std_val = std_dict[band]

                # extracting band index from 'bands' list
                band_idx = bands.index(band)

                # standardizing
                band_arr = data_arr[band_idx]
                band_arr[~np.isnan(band_arr)] = (band_arr - mean_val) / std_val

                # saving band in the initiated zero array
                standardized_arr[band_idx] = band_arr

            # setting all no data (np.nan) values to zero
            standardized_arr[np.isnan(standardized_arr)] = 0

            # saving standardized tile as a new array
            new_tile_path = os.path.join(output_dir, os.path.basename(tile))
            with rio.open(
                    new_tile_path,
                    'w',
                    driver='GTiff',
                    height=standardized_arr.shape[1],
                    width=standardized_arr.shape[2],
                    count=standardized_arr.shape[0],
                    dtype=np.float32,
                    crs=file_crs,
                    transform=file_transform,
                    nodata=0
            ) as dst:
                dst.write(standardized_arr)

        # # standardizing target values
        if split_type not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split_type '{split_type}'. Must be one of 'train'/'val'/'test'")

        # reading target data csv
        target_df = pd.read_csv(os.path.join(input_dir, f'y_{split_type}.csv'))

        # extracting mean and std values from respective dictionaries
        mean_val = mean_dict['target']
        std_val = std_dict['target']

        # standardizing
        target_df['standardized_value'] = (target_df['value'] - mean_val) / std_val

        # saving standardized target values as csv
        target_df.to_csv(os.path.join(output_dir, f'y_{split_type}.csv'), index=False)

    else:
        pass
