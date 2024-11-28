# # Steps of implementation for readying datasets for the DL model

# create tiles (we will do odd size images, preferred size is 7*7 or 9*9 for now)
# randomly split train-test-validation data
# standardize or normalize input variables of training data. Might have to consider mean and std of all input variables (of training data)
# generally standardizing/normalizing both input features and target variables are good practice (normalizing target variable might hurt interpretability)
# use DataLoader to create batches with standardized data (DataLoader class in with the CNN script as that uses pyTorch)

import os
import numpy as np
import pandas as pd
from glob import glob
import rasterio as rio
from Codes.utils.system_ops import makedirs
from rasterio.windows import Window, transform
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


def make_multiband_datasets(list_of_temporal_var_dirs, list_of_static_var_dirs, band_key_list, output_dir,
                            years_list, skip_processing=False):
    """
    Make multi-band raster dataset from individual temporal/static attributes.

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


class make_multiband_tiles:
    """
        Processes a multi-band raster image into tiles and saves them as separate GeoTIFF files.

        **Caution**:
        1. The first band of the tile must be target band (train data). This class extracts the center pixel
        value of the first band as the target variable for each time and saves it separately, while the remaining
        bands are used as features (saved a tiled multi-band GeoTIFF) for the deep learning model.

        2. The `band_key_list` should only contain the names of the feature bands and exclude the target variable's name.
        """

    def __init__(self, tiff_path, band_key_list, tile_output_dir, target_data_output_csv,
                 tile_height=7, tile_width=7, nodata_threshold=50, skip_processing=False):
        """
        :param tiff_path (str): Path to the multi-band raster TIFF file to process.
        :param band_key_list (list): List of keys or descriptions for each band, used as metadata in output tiles.
                                     Make sure not to insert the target variable's name as band name because that band
                                     will be removed and processed separately.
        :param tile_output_dir (str): Directory where the processed tiles will be saved.
        :param target_data_output_csv (str): Path to save the target data (data used as target variable) as a CSV.
        :param tile_height (int): Height of each tile. Default set to 7.
        :param tile_width (int): Width of each tile. Default set to 7.
        :param nodata_threshold (int): Maximum percentage of nodata values allowed in a tile (default is 50).
        :param skip_processing (bool): If True, skips processing and initializes the class without performing operations.
                                       Default set to False.
        """
        if not skip_processing:
            print("**Caution**:")
            print("- The first band of the multi-band raster must represent the target variable.")
            print(
                "- The `band_key_list` should only contain the names of the feature bands and exclude the target variable's name.\n")

            makedirs([tile_output_dir])

            # initiating training data storage dictionary
            target_data = {'tile_no': [], 'target_value': []}

            # Opening a multi-band image file. The tile-ing operation will be done inside the opened image file.
            with rio.open(tiff_path) as tiff:

                # storing image width, height, and nodata for later use
                tiff_height = tiff.height
                tiff_width = tiff.width
                self.nodata = 0  # nodata will be zero for the processed tiles

                # getting year (+ month) info which will be later used in saving tile name
                if len(os.path.basename(tiff_path).split('.')[0].split('_')[-1]) == 4:  # check for annual data
                    year = os.path.basename(tiff_path).split('.')[0].split('_')[-1]
                    month = None

                else:  # check for monthly data
                    year = os.path.basename(tiff_path).split('.')[0].split('_')[-2]
                    month = os.path.basename(tiff_path).split('.')[0].split('_')[-1]

                # initiating tile number
                tile_no = 1

                # The 1st loop takes across width and the 2nd loop takes across height.
                # The 2 loops together creates a window which is used to create tile from the image.
                for i in range(0, tiff_width, tile_width):
                    for j in range(0, tiff_height, tile_height):
                        if (i + tile_width <= tiff_width) and (
                                j + tile_height <= tiff_height):  # a check to keep the window within the image

                            window = Window(col_off=i, row_off=j, width=tile_width,
                                            height=tile_height)  # the tile window

                            tile_arr = tiff.read(window=window)  # reading the image as an array for the tile window

                            tile_name = f'{year}_tile_{tile_no}' if month is None else f'{year}_{month}_tile_{tile_no}'  # tile name
                            print(f'processing tile - {tile_name}')

                            window_transform = transform(window, tiff.transform)  # tiled window's affine transformation

                            # we will only use the center pixel as the target value/variable
                            # So, only keeping the center pixel for band 1 and setting the other as nan
                            # if center pixel doesn't have a value discard the whole tile
                            # otherwise store the center value to the training data list
                            center_pixel_pos = tile_height // 2
                            center_value = tile_arr[0][center_pixel_pos, center_pixel_pos]

                            if center_value == -9999:
                                print(f'discarding tile {tile_name} due to null center value \n')
                                continue
                            else:
                                # storing the center value in the training_data dictionary
                                target_data['tile_no'].append(tile_no)
                                target_data['train_value'].append(center_value)

                            # remove the array containing the training data from the tile
                            tile_arr = tile_arr[1:]

                            # in case complete nodata tile detected, exiting the loop into next iteration
                            if make_multiband_tiles.is_image_null(tile_arr):
                                print(f'discarding tile {tile_name} due to null band \n')
                                continue

                            # a check to see if the tile is completely nodata. In case of an unexpected nodata tile
                            # passes through the if block, the assertion block will raise AssertionError
                            assert not make_multiband_tiles.is_image_null(
                                tile_arr), f'All nodata value in tile {tile_no}'

                            # setting all no data values to zero
                            tile_arr[tile_arr == -9999] = 0

                            # if the % zero values (nodata) is greater than a threshold, not including those tiles
                            if any(x > nodata_threshold for x in
                                   make_multiband_tiles.count_perc_nodata(tile_arr, nodata=0)):
                                print(f'discarding tile {tile_name} due >{nodata_threshold}% no data \n')
                                continue

                            # saving the tiled image
                            output_file = os.path.join(tile_output_dir, f'{tile_name}.tif')

                            with rio.open(
                                    output_file,
                                    'w',
                                    driver='GTiff',
                                    height=tile_arr.shape[1],
                                    width=tile_arr.shape[2],
                                    dtype=tile_arr.dtype,
                                    count=tile_arr.shape[0],
                                    crs=tiff.crs,
                                    transform=window_transform,
                                    nodata=self.nodata
                            ) as dst:

                                for id in range(0, tile_arr.shape[0]):  # looping for each band of the tiled array
                                    dst.write_band(id + 1, tile_arr[id])
                                    dst.set_band_description(id + 1, band_key_list[id])

                            tile_no += 1

            # saving training data as csv
            target_df = pd.DataFrame(target_data)
            target_df.to_csv(target_data_output_csv, index=False)

    @staticmethod
    def is_image_null(multiband_img_arr, nodata=-9999):
        """
        Checks all bands in an image array to see if there is an entirely null value band. A tile (image array) with
        a single data band with all null values will be rejected in the main code using this code.

        :param multiband_img_arr: Multi-band image array.
        :param nodata: No data value. Default set to -9999.

        :return: A boolean variable.
        """
        # initiating a variable is_img_null as False. If any of the band is entirely null, the function will
        # convert this variable to True. Otherwise, the function will return is_band_null as False.
        is_img_null = False

        # reading each band separately and checking if the entire band is null or not
        # if any of the band is entirely null, that tile won't be saved, as it has some missing attribute
        # Note that, an image where all bands have at least one valid pixel will pass this filter.
        for num_band in range(0, multiband_img_arr.shape[0]):
            single_arr = multiband_img_arr[num_band]

            if np.all(single_arr == nodata):
                is_img_null = True
            else:
                pass

        return is_img_null

    @staticmethod
    def count_perc_nodata(multiband_img_arr, nodata=-9999):
        """
        Calculates percent of nodata pixels in each band of an image array.

        :param multiband_img_arr: Multi-band image array.
        :param nodata: No data values. Default set to -9999. Can be zero also.

        :return: A list containing percent nodata pixel counts for all bands.
        """
        perc_counts_all_bands = []
        for num_band in range(0, multiband_img_arr.shape[0]):
            single_arr = multiband_img_arr[num_band]

            valid_pixels = np.count_nonzero(np.where(single_arr != nodata, 1, 0), keepdims=False)
            total_pixels = single_arr.shape[0] * single_arr.shape[1]
            nodata_count = total_pixels - valid_pixels

            perc_no_data = nodata_count * 100 / total_pixels

            perc_counts_all_bands.append(perc_no_data)

        return perc_counts_all_bands
