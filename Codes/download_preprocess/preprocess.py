import os
from glob import glob

from Codes.utils.raster_ops import create_multiband_raster
from Codes.utils.system_ops import makedirs


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


def make_multiband_datasets(list_of_temporal_var_dirs, list_of_static_var_dirs, band_key_list, output_dir,
                            years_list, month_range=None):
    makedirs([output_dir])

    # arranging the variables (their paths) in an order so that all input variables of a year/month will be in a
    # nested list inside the main output list
    data_paths_lists = org_vars(list_of_temporal_var_dirs=list_of_temporal_var_dirs,
                                list_of_static_var_dirs=list_of_static_var_dirs,
                                years_list=years_list, month_range=month_range)

    global output_file_path
    for paths in data_paths_lists:
        # setting output name
        if len(os.path.basename(paths[0]).split('.')[0].split('_')[-1]) == 4:  # check for annual data
            year = os.path.basename(paths[0]).split('.')[0].split('_')[-1]
            output_file_path = os.path.join(output_dir, f'{year}.tif')

        elif len(os.path.basename(paths[0]).split('.')[0].split('_')[-1]) <= 2:  # check for monthly data
            year = os.path.basename(paths[0]).split('.')[0].split('_')[-2]
            month = os.path.basename(paths[0]).split('.')[0].split('_')[-1]
            output_file_path = os.path.join(output_dir, f'{year}_{month}.tif')

        elif not os.path.basename(paths[0]).split('.')[0].split('_')[0].isdigit():  # check if static data. basically checking if the last name is a digit or not.
                                                                                    # if not a digit, then the code enters this condition
            output_file_path = output_file_path

        create_multiband_raster(input_files_list=paths, band_key_list=band_key_list, output_file=output_file_path)
