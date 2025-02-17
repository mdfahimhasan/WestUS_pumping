# Author : Md Fahim Hasan
# PhD Candidate
# Colorado State university
# Fahim.Hasan@colostate.edu

"""
Acknowledgment:
This script was developed based on the author's knowledge on satellite data processing and pre-processing of
machine/deep learning inputs. Assistance and insights have been taken from  ChatGPT, an AI model by OpenAI,
to improve  efficiency, accuracy, and readability of the script, considering the complex nature of this script.
"""

# # Steps of implementation for readying datasets for the DL model

# create tiles (we will do odd size images, preferred size is 7*7 or 9*9 for now)
# randomly split train-test-validation data
# standardize or normalize input variables of training data. Have to consider mean and std of all input variables (of training data)
# use DataLoader to create batches with standardized data (DataLoader class in with the CNN script as that uses pyTorch)


import os
import shutil
import numpy as np
import pandas as pd
from glob import glob
import rasterio as rio
from rasterio.windows import Window
from multiprocessing import Pool, Manager
from sklearn.model_selection import train_test_split

from Codes.utils.system_ops import makedirs, copy_file, clean_and_make_directory
from Codes.utils.raster_ops import read_raster_arr_object

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
    if month_range is None:  # # if the datasets are annual
        all_data_paths = []  # final list to append the data paths

        # 1st loop for each year and 2nd loop for each dataset
        for year in years_list:
            data_paths = []

            for var in list_of_temporal_var_dirs:
                data = glob(os.path.join(var, f'*{year}*.tif'))
                data_paths.extend(data)

            all_data_paths.append(data_paths)

    else:  # # if the datasets are monthly
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

    *** The output files can be arranged with temporal all_bands (each representing a particular time's dataset)
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

            # # good checks to know if right data is placed under the right band during multiband creation
            print(f'{layer=}')
            print(f'{layer_name=}')

            # writing each band
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

        makedirs([output_dir])

        global output_file_path

        # arranging the variables (their paths) in an order so that all input variables of a year/month will be in a
        # nested list inside the main output list
        data_paths_lists = org_vars(list_of_temporal_var_dirs=list_of_temporal_var_dirs,
                                    list_of_static_var_dirs=list_of_static_var_dirs,
                                    years_list=years_list)

        # # multi-band raster creation
        for paths in data_paths_lists:

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
    The `band_key_list` should only contain the names of the feature all_bands and exclude the target variable
    and stateID name.
    """

    def __init__(self, tiff_path_list, band_key_list,
                 interim_tile_output_dir, interim_target_data_output_csv,
                 final_tile_output_dir, final_target_data_output_csv,
                 tile_size=7, nodata_value=-9999, nodata_threshold=50, start_tile_no=1,
                 num_workers=20, skip_processing=False):
        """
        :param tiff_path_list (list): List of paths to the multi-band raster TIFF file to process.
        :param band_key_list (list): List of keys or descriptions for each band.
        :param interim_tile_output_dir (str): Interim directory where the processed tiles will be saved.
        :param final_tile_output_dir (str): Final directory where the processed tiles will be saved.
        :param interim_target_data_output_csv (str): Filepath to save the interim target training data as CSV.
        :param tile_size (int): Size of the square tile (e.g., 7x7). Must be odd to ensure a center pixel.
        :param nodata_value (int/float): NoData value in the raster (default: -9999).
        :param nodata_threshold (int): Maximum percentage of NoData values allowed in a tile (default: 50).
        :param start_tile_no (int): Initial value for the tile number. Default set to 1.
        :param num_workers: int. Number of parallel processes to use for multiprocessing. Default is 20.
        :param skip_processing (bool): If True, skips processing and initializes the class without performing operations.
        """
        self.interim_tile_output_dir = interim_tile_output_dir
        self.interim_target_data_output_csv = interim_target_data_output_csv
        self.final_tile_output_dir = final_tile_output_dir
        self.final_target_data_output_csv = final_target_data_output_csv
        self.tile_size = tile_size
        self.nodata_value = nodata_value
        self.nodata_threshold = nodata_threshold
        self.tile_no = start_tile_no
        self.num_workers = num_workers


        # implementing the tiling using multi-processing (multiprocess has been used to fasten processing speed)
        if not skip_processing:
            # removing old output files and making a new folder
            if os.path.exists(self.final_tile_output_dir):
                shutil.rmtree(self.final_tile_output_dir)

            makedirs([self.interim_tile_output_dir])

            for tiff_path in tiff_path_list:
                cumulative_tile_no = self.process_tiles(tiff_path, band_key_list)
                self.tile_no = cumulative_tile_no  # updating the self.tile_no after each multi-band raster iteration

            # final process to serialize tile_no (multiprocess sets tile_no based on total central pixel
            # processed in each chunk, resulting in large value of tile_no that need to be serialized)
            self._final_update_tile_no()

    def process_tiles(self, tiff_path, band_key_list):
        """
        Processes a multi-band raster image into tiles of training data and associated feature attributes.

        The function divides the raster into row chunks, processes each chunk in parallel using multiprocessing,
        and creates square tiles of data. It extracts the center pixel value from the first band (target data)
        and saves both the tiles and the associated target values.

        :param tiff_path: str. Path to the multi-band raster TIFF file to process.
        :param band_key_list: list. List of keys or descriptions for each band in the raster, excluding the target band.

        :return: None
            Saves the output tiles as multi-band GeoTIFF files in the specified directory (`interim_tile_output_dir`)
            and the target data (tile numbers and center pixel values) as a CSV file (`interim_target_data_output_csv`).

        **Details**:
        - The raster is divided into row chunks of 100 rows each (adjusted for tile size to avoid edge cases).
        - Each chunk is processed by a worker function (`_process_chunk_worker`) in parallel.
        - Shared multiprocessing objects:
            - `target_data_list`: A shared list to store target data (tile numbers and center pixel values).
        - Tiles are saved as GeoTIFF files, with band metadata modified to include the year extracted from `tiff_path`.
        - Target data is saved as a CSV file after all workers have completed processing.
        """

        # extracting year info from dta path and modifying band_key_list to reflect that in band name
        year = os.path.basename(tiff_path).split('.')[0].split('_')[-1]
        band_key_list_mod = [(i + '_' + year) for i in band_key_list]

        print(f'\ncreating multi-band tiles for {year}...')

        # opening a multi-band image file. The tile-ing operation will be done inside the opened image file.
        with rio.open(tiff_path) as tiff:
            tiff_height = tiff.height
            tile_radius = self.tile_size // 2
            row_chunks = [range(start, min(start + 100, tiff_height - tile_radius))
                          for start in range(tile_radius, tiff_height - tile_radius, 100)]  # Chunk size of 100 rows

            print(f'processing multi-band raster in {len(row_chunks)} chunks \n')

            # Manager() is used to create a shared list (`target_data_list`) for storing target data across worker processes.
            manager = Manager()
            target_data_list = manager.list()  # shared list of target data

            # assigning a unique starting tile number for each chunk using `cumulative_tile_no`.
            # This ensures unique tile IDs across different chunks and raster files.
            cumulative_tile_no = self.tile_no

            # # # # # # # # # # # # # # # preparing the input arguments for each worker process: # # # # # # # # # # # #
            config = {
                'tile_size': self.tile_size,
                'nodata_value': self.nodata_value,
                'interim_tile_output_dir': self.interim_tile_output_dir,
                'nodata_threshold': self.nodata_threshold
            }

            # initiating an empty list to store pool inputs
            pool_input =[]

            # processing data with multiprocessing
            for chunk_idx, chunk in enumerate(row_chunks):

                # assigning the starting tile number for the current chunk
                start_tile_no = cumulative_tile_no
                cumulative_tile_no += len(chunk) * (tiff.width - 2 * tile_radius)  # approximate number of tile in this chunk

                print(f'Chunk {chunk_idx} starts with tile_no {start_tile_no}')  # debugging
                pool_input.append((chunk, tiff_path, band_key_list_mod, start_tile_no, target_data_list, config))


            # # # # # # creating a multiprocessing pool with a specified number of worker processes  # # # # # # # # # #
            # This will control resource usage during parallel raster chunk processing.
            with Pool(processes=self.num_workers) as pool:
                pool.map(self._process_chunk_worker, pool_input)

            # saving the training data to a CSV file
            # ensure DataFrame has column names even if empty
            if len(target_data_list) > 0:
                target_df = pd.DataFrame(list(target_data_list))  # converting Manager.list() to python list >> DataFrame
            else:
                target_df = pd.DataFrame(columns=['tile_no', 'stateID', 'target_value'])  # Ensure correct column names

            target_df.to_csv(self.interim_target_data_output_csv, mode='a',
                             header=not os.path.exists(self.interim_target_data_output_csv),
                             index=False)

            # returning the updated 'cumulative_tile_no' which will be used to update the 'self.tile_no' in __init__()
            return cumulative_tile_no

    def _process_chunk_worker(self, args):
        """
        Processes a chunk of raster rows to create training tiles and save associated target data.

        :param args: These args are given/generated within the process_tiles() function.
                    - chunk: Specifies the range of rows to process for this worker.
                    - tiff_path: Path to the raster file so the worker can open and read it independently.
                    - band_key_list: List of band names or keys for saving tiles. It has been modified with 'year' name
                    - start_tile_no: The initial tile number for this chunk, assigned from `cumulative_tile_no`.
                    - target_data_list: Shared multiprocessing list for storing target data.
                    - config: Contains additional settings like tile_size, nodata_value, and interim_tile_output_dir.

        :return: None.
                 Results (e.g., tile numbers and target data) are saved in shared objects (tile_no, target_data_list)
                          and do not need to be returned.
        """
        chunk, tiff_path, band_key_list, start_tile_no, target_data_list, config = args

        current_tile_no = start_tile_no  # initializing the local tile number for this chunk

        with rio.open(tiff_path) as tiff:
            tile_radius = config['tile_size'] // 2

            # reading pumping and stateID array
            bands = tiff.descriptions  # list of band names

            pumping_band_idx = bands.index('pumping_mm') + 1   # +1 due to rasterio-based indexing
            training_band = tiff.read(pumping_band_idx)        # reading pumping_mm band (training data)

            stateID_idx = bands.index('stateID') + 1           # +1 due to rasterio-based indexing
            stateID_band = tiff.read(stateID_idx)              # reading stateID band

            # The first loop iterates across chunk (each chunk has 100 rows by default) and
            # the second loop across the width (columns) of the raster.
            # Both loops start at `tile_radius` (for row loop it's used during chunk creation) to ensure the tile remains
            # fully within the raster boundaries, avoiding edge cases.
            # Both loops end at `tiff_height - tile_radius` (for rows) and `tiff_width - tile_radius` (for columns)
            # to handle edge cases.
            # The loops move by 1 cell at a time, check if there is a valid training data value at the center pixel,
            # and create a tile around it if valid.
            for row in chunk:
                for col in range(tile_radius, tiff.width - tile_radius):
                    try:
                        # extracting stateID of the center pixel
                        stateID_val = stateID_band[row, col]

                        # skipping tile with NoData in the center pixel of the tile
                        center_train_value = training_band[row, col]
                        if center_train_value == self.nodata_value:
                            continue

                        # creating a window around the central pixel and reading the data
                        window = Window(col_off=col - tile_radius, row_off=row - tile_radius,
                                        width=self.tile_size, height=self.tile_size)

                        # keeping only the arrays except the pumping and stateID band
                        all_band_idxs = list(range(len(bands)))                      # 0-based indices
                        exclude_band_idxs = [pumping_band_idx - 1, stateID_idx - 1]  # -1 as the indices were 1-based
                        valid_band_idxs = [i + 1 for i in all_band_idxs
                                           if i not in exclude_band_idxs]            # +1 again to convert to 1-based indexing

                        # reading multi-band array for the window and with pumping and stateID all_bands excluded
                        tile_arr = tiff.read(valid_band_idxs, window=window)

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
                        window_transform = tiff.window_transform(window)

                        tile_name = f'tile_{current_tile_no}.tif'
                        output_file = os.path.join(config['interim_tile_output_dir'], tile_name)

                        try:
                            self.save_tile(output_file, tile_arr, crs, window_transform, band_key_list)

                        except Exception as e:
                            print(f'Skipping tile {tile_name} due to save error: {e}')
                            continue

                        # append target (training) data to target_data_list, along with tile_no and stateID
                        target_data_list.append({'tile_no': current_tile_no, 'stateID': stateID_val,
                                                 'target_value': center_train_value})
                        current_tile_no += 1

                    except Exception as e:
                        # Log and skip problematic tiles
                        print(f'Skipping tile at row {row}, col {col} due to read error: {e}')
                        continue

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
        Checks all all_bands in an image array to see if there is an entirely null value band. A tile (image array) with
        a single data band with all null values will be rejected in the main code using this code.

        :param tile_arr: Multi-band image array.

        :return: A boolean variable.
        """
        # reading each band separately and checking if an entire band is null or not.
        # If any of the band is entirely null, this function will immediately return True and the tile will be skipped.
        # Note that, an image where all all_bands have at least one valid pixel will pass this filter.
        for num_band in range(0, tile_arr.shape[0]):
            single_arr = tile_arr[num_band]

            if np.all(single_arr == self.nodata_value):
                return True

        # If no all_bands are null, return False
        return False

    @staticmethod
    def save_tile(output_file, tile_arr, crs, transform, band_key_list):
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

    def _final_update_tile_no(self):
        print('\nfinalizing tile_no for tiled rasters and target data...')
        all_tiles = glob(os.path.join(self.interim_tile_output_dir, '*tif'))
        old_tile_no = [os.path.basename(i).split('.')[0].split('_')[-1] for i in all_tiles]

        # generating new sequential tile numbers
        new_tile_no = list(range(1, len(old_tile_no) + 1))

        # creating a dictionary where old_tile_no is key and new_tile_no is value
        old_to_new_map = dict(zip(old_tile_no, new_tile_no))

        # copying all tile to the final directory after changing their names
        for file in all_tiles:
            old_no = os.path.basename(file).split('.')[0].split('_')[-1]
            new_no = old_to_new_map[old_no]
            new_file_name = f'tile_{new_no}'

            copy_file(input_dir_or_file=file, copy_dir=self.final_tile_output_dir,
                      rename=new_file_name)

        # replacing old tile no in target data csv with new tile no
        target_df = pd.read_csv(self.interim_target_data_output_csv)
        target_df['tile_no'] = target_df['tile_no'].astype(str).map(old_to_new_map)

        # replacing samples (very few) with stateID 3 - Nebraska and 1 -  Oklahoma. They belong to kansas
        # but came in Nebraska and Oklahoma during stateID raster creation (along state border)
        target_df.loc[target_df['stateID'] == 3, 'stateID'] = 12
        target_df.loc[target_df['stateID'] == 1, 'stateID'] = 12

        # save final csv
        target_df.to_csv(self.final_target_data_output_csv, index=False)


def copy_tiles(row, input_dir, copy_dir):
    """
   Copies a tile based on the tile number from the source to the destination directory.

   :param row: A row from the DataFrame containing 'tile_no'.
   :param input_dir: Path of the directory containing input tiles.
   :param copy_dir: Destination directory to copy the tiles.

   :return: None.
   """
    tile_no = int(row['tile_no'])
    matching_tiles = glob(os.path.join(input_dir, f'*_{tile_no}.*tif'))
    if len(matching_tiles) != 1:
        print(f'Skipping tile_no {tile_no}: found multiple matches.')

    tile_path = matching_tiles[0]
    shutil.copy(tile_path, os.path.join(copy_dir, os.path.basename(tile_path)))


# multiprocessing helper function for copying files
def copy_tiles_parallel(splitted_target_df, input_dir, copy_dir, num_workers):
    """
    Copy tiles to train, validation, and test data directory using multiprocessing.

    :param splitted_target_df: Train/validation/test dataframe.
    :param input_dir: Path of directory of all tiled data.
    :param copy_dir: Path to copy the tiles.
    :param num_workers: int. Number of parallel processes to use for multiprocessing. Default is 10.

    :return: None.
    """
    # preparing iterable of argument tuples for starmap
    args = [(row, input_dir, copy_dir) for _, row in splitted_target_df.iterrows()]

    # implementing copying using multiprocessing
    with Pool(processes=num_workers) as pool:
        pool.starmap(copy_tiles, args)


def train_val_test_split_tiles(target_data_csv, input_tile_dir, train_dir, val_dir, test_dir,
                               train_size=0.7, val_size=0.15, test_size=0.15,
                               random_state=42, num_workers=10,
                               stratify=True, skip_processing=False):
    """
    Splits the tiles into train, validation, and test datasets, along with their target values.

    :param target_data_csv: str. Path to the CSV file containing target values and tile numbers.
    :param input_tile_dir: str. Path of input tiles containing input variables.
    :param train_dir: str. Directory to save the training tiles and target csv.
    :param val_dir: str. Directory to save the validation tiles and target csv.
    :param test_dir: str. Directory to save the test tiles and target csv.
    :param train_size: float. Proportion of the dataset to include in the train split. Default is 0.7.
    :param val_size: float. Proportion of the dataset to include in the validation split. Default is 0.15.
    :param test_size: float. Proportion of the dataset to include in the test split. Default is 0.15.
    :param random_state: int. Random seed for reproducibility. Default is 42.
    :param num_workers: int. Number of parallel processes to use for multiprocessing. Default is 10.
    :param stratify: Whether to staritify based on 'stateID'. Default set to True to do stratified split.
    :param skip_processing: Set to True to skip processing. Default is False.

    :return: None.
    """
    if not skip_processing:
        print(f'making train-validation-test ({train_size * 100}-{val_size * 100}-{test_size * 100} %) splits....\n')

        makedirs([train_dir, val_dir, test_dir])

        # loading the target data CSV
        target_df = pd.read_csv(target_data_csv)

        # performing the split for train, validation, and test based on target values
        if stratify:
            train_data, temp_data = train_test_split(target_df, train_size=train_size,
                                                     stratify=target_df['stateID'], random_state=random_state)
            val_data, test_data = train_test_split(temp_data, test_size=test_size / (val_size + test_size),
                                                   stratify=temp_data['stateID'], random_state=random_state)
        else:
            train_data, temp_data = train_test_split(target_df, train_size=train_size, random_state=random_state)
            val_data, test_data = train_test_split(temp_data, test_size=test_size / (val_size + test_size),
                                                   random_state=random_state)

        # cleaning existing data from the directories
        clean_and_make_directory(train_dir)
        clean_and_make_directory(val_dir)
        clean_and_make_directory(test_dir)

        # storing splitted train, val, and test target values as csv
        train_data_df = pd.DataFrame(train_data)
        train_data_df.to_csv(os.path.join(train_dir, 'y_train.csv'), index=False)

        val_data_df = pd.DataFrame(val_data)
        val_data_df.to_csv(os.path.join(val_dir, 'y_val.csv'), index=False)

        test_data_df = pd.DataFrame(test_data)
        test_data_df.to_csv(os.path.join(test_dir, 'y_test.csv'), index=False)

        # copying tiles to respective train, val, and test directories
        copy_tiles_parallel(train_data, input_tile_dir, train_dir, num_workers)
        copy_tiles_parallel(val_data, input_tile_dir, val_dir, num_workers)
        copy_tiles_parallel(test_data, input_tile_dir, test_dir, num_workers)

    else:
        pass


def accumulate_band_values_each_tile(tile, bands, nodata):
    """
    Collects valid band values to calculate per-band statistics using multiprocessing.

    :param tile: Path to the tile file.
    :param bands: List of band descriptions.
    :param nodata: NoData value for the tile.

    :return: A dictionary containing per-band statistics (mean, std, min, max).
    """
    dataset = rio.open(tile).read()  # reading all input all_bands in a tile
    band_values = {band: [] for band in bands}

    # process for each band in the tile
    for band in bands:
        # extracting band index from 'all_bands' list
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
        if len(band_arr) > 0:  # to avoid empty arrays
            band_values[band].extend(band_arr)

    return band_values


def calc_scaling_statistics(train_dir, output_dir, num_workers=10, skip_processing=False):
    """
    Calculates the mean, standard deviation, min, and max for each band across all tiles in the training directory.

    :param train_dir: str. Path to the directory containing training tiles.
    :param output_dir: str. Path to the directory to save calculated statistics.
    :param num_workers: int. Number of parallel processes to use for multiprocessing. Default is 10.
    :param skip_processing: Set to True to skip processing. Default is False.

    :return: Four dictionaries: mean_csv, std_csv, min_dict, and max_dict.
    """
    if not skip_processing:  # calculating the statistics
        print('calculating statistics for data scaling...')

        clean_and_make_directory(output_dir)

        # collecting all tiles in the training data directory
        all_tiles = glob(os.path.join(train_dir, f'*.tif'))

        # getting band descriptions and no data info
        bands = rio.open(all_tiles[0]).descriptions
        bands = [band[0: band.rfind('_')] for band in bands]   # band.rfind('_') finds the index of the last '_' that separates the year attribute
        print(f'Band Descriptions: {bands} \n')

        nodata = rio.open(all_tiles[0]).nodata

        # Collecting individual band arrays separately in a dictionary
        # which will be used to calculate band statistics.
        # Using multiprocessing to parallelly process large number of tiles
        args = [(tile, bands, nodata) for tile in all_tiles]

        with Pool(processes=num_workers) as pool:
            result_dict = pool.starmap(accumulate_band_values_each_tile, args)

        # iterating overs results from multiprocessing
        aggregated_values = {band: [] for band in bands}
        for tiles_values in result_dict:
            # Each tile_values is a dictionary containing band names as keys
            # and the corresponding valid pixel values from that tile as a list.
            # Here, multiple dictionary are the results of multiprocessing, which
            # need to be merged across all tiles for each band.

            for band in bands:
                # Extending the aggregated list for the current band with values from this tile.
                # This ensures that all pixel values for each band, from all tiles, are merged
                # into a single list in the aggregated_values dictionary.
                # Without this second loop, we would only process the values from the last tile
                # in the results, which would result in incomplete statistics.
                aggregated_values[band].extend(tiles_values[band])

        # calculating statistics
        mean_dict = {band: np.nanmean(aggregated_values[band]) for band in bands}
        std_dict = {band: np.nanstd(aggregated_values[band]) for band in bands}
        min_dict = {band: np.nanmin(aggregated_values[band]) for band in bands}
        max_dict = {band: np.nanmax(aggregated_values[band]) for band in bands}

        # processing target statistics
        target_df = pd.read_csv(os.path.join(train_dir, 'y_train.csv'))
        target_df.dropna(inplace=True)
        target_val = np.array(target_df['target_value'].tolist())

        mean_dict['target'] = np.nanmean(target_val)
        std_dict['target'] = np.nanstd(target_val)
        min_dict['target'] = np.nanmin(target_val)
        max_dict['target'] = np.nanmax(target_val)

        # saving dictionaries as pickle
        for dict_name, dictionary in zip(
                ['mean', 'std', 'min', 'max'], [mean_dict, std_dict, min_dict, max_dict]
        ):
            df = pd.DataFrame(dictionary.items(), columns=['variable', 'value'])
            csv_path = os.path.join(output_dir, f'{dict_name}.csv')
            df.to_csv(csv_path, index=False)

        return mean_dict, std_dict, min_dict, max_dict

    else:  # loading the saved statistics
        mean_csv = pd.read_csv(os.path.join(output_dir, 'mean.csv'))
        std_csv = pd.read_csv(os.path.join(output_dir, 'std.csv'))
        min_csv = pd.read_csv(os.path.join(output_dir, 'min.csv'))
        max_csv = pd.read_csv(os.path.join(output_dir, 'max.csv'))

        mean_dict = dict(zip(mean_csv['variable'], mean_csv['value']))
        std_dict = dict(zip(std_csv['variable'], std_csv['value']))
        min_dict = dict(zip(min_csv['variable'], min_csv['value']))
        max_dict =dict(zip(max_csv['variable'], max_csv['value']))

        return mean_dict, std_dict, min_dict, max_dict


def standardize_single_tile(tile, exclude_bands_from_standardizing, mean_dict, std_dict, output_dir):
    """
    Standardizes multi-band raster tiles and target values for train, validation, or test datasets.
    The mean and std statistics used for standardizing comes from the train_set using the dictionaries generated by
    calc_scaling_statistics.

    :param tile: str. Path to the tile.
    :param exclude_bands_from_standardizing: List of band names to exclude from standardizing.
    :param mean_dict: dictionary. A dictionary containing mean values (from train_set) for each band and the target variable.
    :param std_dict: dictionary. A dictionary containing std values (from train_set) for each band and the target variable.
    :param output_dir: str. Path to the directory to save standardized outputs (tiles).

    :return: None.
    """
    # opening data for each tile, getting all_bands, crs, and affine transformation
    file = rio.open(tile)
    data_arr = file.read()

    file_crs = file.crs
    file_transform = file.transform
    bands = file.descriptions
    bands = [band[0: band.rfind('_')] for band in
             bands]  # band.rfind('_') finds the index of the last '_' that separates the year attribute


    if file.count != len(bands):  # exiting code in case number of band names and number of array don't match
        raise ValueError("Number of all_bands in metadata and number of array don't match")

    if not np.isnan(file.nodata):  # ensuring no data type as np.nan as this code will set 0 to no data position
        raise ValueError(f"Expected no data value to be NaN, but got {file.nodata}.")

    # initiating a new array (with all zeros) to store standardized all_bands
    standardized_arr = np.zeros_like(data_arr, dtype=np.float32)

    # performing standardization for each band
    # for irr_cropland array, skipping standardization as the data is already binary
    for band in bands:
        mean_val = mean_dict[band]
        std_val = std_dict[band]

        # extracting band index from 'all_bands' list
        band_idx = bands.index(band)

        # standardizing (except specifically listed all_bands, boolean arrays)
        if band not in exclude_bands_from_standardizing:
            band_arr = data_arr[band_idx]
            band_arr = np.where(~np.isnan(band_arr), (band_arr - mean_val) / std_val, band_arr)

        else:  # for specifically listed all_bands, use the original array
            band_arr = data_arr[band_idx]

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
        for idx, band_arr in enumerate(standardized_arr):
            dst.write(band_arr, idx+1)
            dst.set_band_description(idx+1, bands[idx])


def standardize_train_val_test(input_tile_dir, mean_dict, std_dict,  exclude_bands_from_standardizing, output_dir,
                               split_type='train', num_workers=10, skip_processing=False):
    """
    Standardizes multi-band raster tiles and target values for train, validation, or test datasets.
    The mean and std statistics used for standardizing comes from the train_set using the dictionaries generated by
    calc_scaling_statistics.

    :param input_tile_dir: str. Path to the directory containing the input raster tiles and target value csv.
    :param mean_dict: dictionary. A dictionary containing mean values (from train_set) for each band and the target variable.
    :param std_dict: dictionary. A dictionary containing std values (from train_set) for each band and the target variable.
    :param exclude_bands_from_standardizing: List of band names to exclude from standardizing.
    :param output_dir: str. Path to the directory to save standardized outputs (tiles).
    :param split_type: str. Should be something from ['train', 'val', 'test'].
    :param num_workers: int. Number of parallel processes to use for multiprocessing. Default is 10.
    :param skip_processing: boolean. Set to True to skip this step.

    :return: None.
    """
    if not skip_processing:
        print(f"standardizing '{split_type}' tiles... \n")

        # creating new directory for standardized datasets
        makedirs([output_dir])

        # collecting all tiles in the training data directory
        all_tiles = glob(os.path.join(input_tile_dir, f'*.tif'))

        # standardizing each tile with multiprocessing
        args = [(tile, exclude_bands_from_standardizing, mean_dict, std_dict, output_dir) for tile in all_tiles]

        with Pool(processes=num_workers) as pool:
            pool.starmap(standardize_single_tile, args)


        # reading target data csv
        target_df = pd.read_csv(glob(os.path.join(input_tile_dir, '*.csv'))[0])

        # extracting mean and std values from respective dictionaries
        mean_val = mean_dict['target']
        std_val = std_dict['target']

        # standardizing
        target_df['standardized_value'] = (target_df['target_value'] - mean_val) / std_val

        # saving standardized target values as csv
        target_df.to_csv(os.path.join(output_dir, f'y_{split_type}.csv'), index=False)

    else:
        pass


