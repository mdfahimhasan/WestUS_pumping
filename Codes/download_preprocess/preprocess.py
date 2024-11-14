import os
import re
import datetime
import numpy as np
from glob import glob
from osgeo import gdal
import rasterio as rio
from Codes.utils.system_ops import makedirs
from Codes.utils.raster_ops import read_raster_arr_object, write_array_to_raster, clip_resample_reproject_raster,\
                                   mean_rasters, sum_rasters

no_data_value = -9999
model_res = 0.01976293625031605786  # in deg, ~2 km
WestUS_shape = '../../Data_main/ref_shapes/WestUS_states.shp'
WestUS_raster = '../../Data_main/ref_rasters/Western_US_refraster_2km.tif'


def extract_month_from_GrowSeason_data(GS_data_dir, skip_processing=False):
    """
    Extract start and ending growing season months from growing season dataset (provided by Justin Huntington DRI;
    downloaded from GEE to google drive). The output datasets have 2 bands, containing start and end month info,
    respectively.

    :param GS_data_dir: Directory path of growing season dataset. The GEE-downloaded datasets are in the
                        'ee_exports' folder.
    :param skip_processing: Set to true if want to skip processing.

    :return: None.
    """

    def doy_to_month(year, doy):
        """
        Convert a day of year (DOY) to a month in a given year.

        :return: Month of the corresponding date.
        """
        if np.isnan(doy):  # Check if the DOY is NaN
            return np.nan

        # January 1st of the given year + timedelta of the DoY to extract month
        month = (datetime.datetime(year, 1, 1) + datetime.timedelta(int(doy) - 1)).month

        return month

    if not skip_processing:
        print('Processing growing season data...')

        # collecting GEE exported data files and making new directories for processing
        GS_data_files = glob(os.path.join(GS_data_dir, 'ee_exports', '*.tif'))
        interim_dir = os.path.join(GS_data_dir, 'interim')
        makedirs([interim_dir])

        # looping through each dataset, extracting start and end of the growing season months, saving as an array
        for data in GS_data_files:
            raster_name = os.path.basename(data)
            year = int(raster_name.split('_')[1].split('.')[0])

            # clipping and resampling the growing season data with the western US reference raster
            interim_raster = clip_resample_reproject_raster(input_raster=data,
                                                            input_shape=WestUS_shape,
                                                            raster_name=raster_name,
                                                            output_raster_dir=interim_dir,
                                                            clip=False, resample=False, clip_and_resample=True,
                                                            targetaligned=True, resample_algorithm='near',
                                                            use_ref_width_height=False, ref_raster=None,
                                                            resolution=model_res)

            # reading the start and end DoY of the growing season
            startDOY_arr, ras_file = read_raster_arr_object(interim_raster, band=1)
            endDOY_arr = read_raster_arr_object(interim_raster, band=2, get_file=False)

            # vectorizing the doy_to_month() function to apply on a numpy array
            vectorized_doy_to_date = np.vectorize(doy_to_month)

            # converting the start and end DoY to corresponding month
            start_months = vectorized_doy_to_date(year, startDOY_arr)
            end_months = vectorized_doy_to_date(year, endDOY_arr)

            # stacking the arrays together (single tif with 2 bands)
            GS_month_arr = np.stack((start_months, end_months), axis=0)

            # saving the array
            output_raster = os.path.join(GS_data_dir, raster_name)
            with rio.open(
                    output_raster,
                    'w',
                    driver='GTiff',
                    height=GS_month_arr.shape[1],
                    width=GS_month_arr.shape[2],
                    dtype=np.float32,
                    count=GS_month_arr.shape[0],
                    crs=ras_file.crs,
                    transform=ras_file.transform,
                    nodata=-9999
            ) as dst:
                dst.write(GS_month_arr)


def dynamic_gs_sum_var(year_list, growing_season_dir, monthly_input_dir, gs_output_dir,
                       sum_keyword, skip_processing=False):
    """
    Dynamically (spatio-temporally) sums any variable for dynamic growing seasons.

    :param year_list: List of years_list to process the data for.
    :param growing_season_dir: Directory path for growing season datasets.
    :param monthly_input_dir:  Directory path for monthly datasets.
    :param gs_output_dir:  Directory path (output) for summed growing season datasets.
    :param sum_keyword: Keyword str to add before the summed raster.
    :param skip_processing: Set to True if want to skip processing this step.

    :return:
    """
    if not skip_processing:
        print(f'Dynamically summing {sum_keyword} monthly datasets for growing season...')

        makedirs([gs_output_dir])

        # The regex r'_([0-9]{1,2})\.tif' extracts the month (1 or 2 digits; e.g., '_1.tif', '_12.tif')
        # from the filenames using the first group ([0-9]{1,2}).
        # The extracted month is then (inside the for loop in the sorting block) converted to an integer with int(group(1))
        # for proper sorting by month.
        month_pattern = re.compile(r'_([0-9]{1,2})\.tif')

        for year in year_list:
            # gathering and sorting the datasets by month (from 1 to 12)
            datasets = glob(os.path.join(monthly_input_dir, f'*{year}*.tif'))
            sorted_datasets = sorted(datasets, key=lambda x: int(
                month_pattern.search(x).group(1)))  # First capturing group (the month)

            # monthly array stacked in a single numpy array
            arrs_stck = np.stack([read_raster_arr_object(i, get_file=False) for i in sorted_datasets], axis=0)

            # gathering, reading, and stacking growing season array
            gs_data = glob(os.path.join(growing_season_dir, f'*{year}*.tif'))[0]
            start_gs_arr, ras_file = read_raster_arr_object(gs_data, band=1, get_file=True)  # band 1
            end_gs_arr = read_raster_arr_object(gs_data, band=2, get_file=False)  # band 2

            # We create a 1 pixel "kernel", representing months 1 to 12 (shape : 12, 1, 1).
            # Then it is broadcasted across the array and named as the kernel_mask.
            # The kernel_mask acts as a mask, and only sum peff values for months that are 'True'.
            kernel = np.arange(1, 13, 1).reshape(12, 1, 1)
            kernel_mask = (kernel >= start_gs_arr) & (kernel <= end_gs_arr)

            # sum monthly arrays over the valid months using the kernel_mask
            summed_arr = np.sum(arrs_stck * kernel_mask, axis=0)

            # saving the summed array
            output_name = f'{sum_keyword}_{year}.tif'
            output_path = os.path.join(gs_output_dir, output_name)
            with rio.open(
                    output_path,
                    'w',
                    driver='GTiff',
                    height=summed_arr.shape[0],
                    width=summed_arr.shape[1],
                    dtype=np.float32,
                    count=1,
                    crs=ras_file.crs,
                    transform=ras_file.transform,
                    nodata=-9999
            ) as dst:
                dst.write(summed_arr, 1)


def dynamic_gs_mean_var(year_list, growing_season_dir, monthly_input_dir, gs_output_dir,
                        mean_keyword, skip_processing=False):
    """
    Dynamically (spatio-temporally) averages any variable for dynamic growing seasons.

    :param year_list: List of years to process the data for.
    :param growing_season_dir: Directory path for growing season datasets.
    :param monthly_input_dir:  Directory path for monthly datasets.
    :param gs_output_dir:  Directory path (output) for averaged growing season datasets.
    :param mean_keyword: Keyword str to add before the averaged raster.
    :param skip_processing: Set to True if want to skip processing this step.

    :return:
    """
    if not skip_processing:
        print(f'Dynamically averaging {mean_keyword} monthly datasets for growing season...')

        makedirs([gs_output_dir])

        month_pattern = re.compile(r'_([0-9]{1,2})\.tif')

        for year in year_list:
            # gathering and sorting the datasets by month (from 1 to 12)
            datasets = glob(os.path.join(monthly_input_dir, f'*{year}*.tif'))
            sorted_datasets = sorted(datasets, key=lambda x: int(month_pattern.search(x).group(1)))

            # monthly array stacked in a single numpy array
            arrs_stck = np.stack([read_raster_arr_object(i, get_file=False) for i in sorted_datasets], axis=0)

            # gathering, reading, and stacking growing season array
            gs_data = glob(os.path.join(growing_season_dir, f'*{year}*.tif'))[0]
            start_gs_arr, ras_file = read_raster_arr_object(gs_data, band=1, get_file=True)
            end_gs_arr = read_raster_arr_object(gs_data, band=2, get_file=False)

            # We create a 1 pixel "kernel", representing months 1 to 12 (shape : 12, 1, 1).
            # Then it is broadcasted across the array and named as the kernel_mask.
            # The kernel_mask acts as a mask, and only sum peff values for months that are 'True'.
            kernel = np.arange(1, 13, 1).reshape(12, 1, 1)
            kernel_mask = (kernel >= start_gs_arr) & (kernel <= end_gs_arr)

            # Count the number of valid months in each pixel's growing season
            valid_month_count = np.sum(kernel_mask, axis=0)
            valid_month_count[valid_month_count == 0] = np.nan  # Avoid division by zero for non-growing season pixels

            # computing the mean over valid months
            summed_arr = np.sum(arrs_stck * kernel_mask, axis=0)
            mean_arr = summed_arr / valid_month_count

            # saving the mean array
            output_name = f'{mean_keyword}_{year}.tif'
            output_path = os.path.join(gs_output_dir, output_name)
            with rio.open(
                    output_path,
                    'w',
                    driver='GTiff',
                    height=mean_arr.shape[0],
                    width=mean_arr.shape[1],
                    dtype=np.float32,
                    count=1,
                    crs=ras_file.crs,
                    transform=ras_file.transform,
                    nodata=-9999
            ) as dst:
                dst.write(mean_arr, 1)


def convert_prism_data_to_tif(input_dir, output_dir, keyword='prism_precip'):
    """
    Convert prism rainfall/temperature datasets from .bil format to GeoTiff format.

    Download PRISM datasets directly from  'https://prism.oregonstate.edu/recent/'

    :param input_dir: Directory path of prism data in .bil format.
    :param output_dir: Directory path of converted (.tif) prism data.
    :param keyword: keyword to add before processed datasets.

    :return: None.
    """
    makedirs([output_dir])

    prism_datasets = glob(os.path.join(input_dir, '*.bil'))

    for data in prism_datasets:
        year_month = os.path.basename(data).split('_')[-2]
        output_name = keyword + '_' + year_month + '.tif'
        output_file = os.path.join(output_dir, output_name)
        gdal.Translate(destName=output_file, srcDS=data, format='GTiff', outputType=gdal.GDT_Float32,
                       outputSRS='EPSG:4269')


def process_prism_data(prism_bil_dir, prism_tif_dir, output_dir_prism_monthly, growing_season_dir, output_dir_prism_gs,
                       year_list=(1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
                                  2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020),
                       keyword='prism_precip', west_US_shape=WestUS_shape,
                       ref_raster=WestUS_raster, resolution=model_res, skip_processing=False):
    """
    Process (sum and mean to Western US extent) Prism Precipitation, Tmax, and Tmin data. The precipitation data is
    summed for all months in a year. The Tmax and Tmin data is averaged for growing season (April - October).

    :param prism_bil_dir: Directory file path of downloaded prism datasets in .bil format.
    :param prism_tif_dir: Directory file path of prism datasets converted to tif format.
    :param output_dir_prism_monthly: File path of directory to save monthly prism precipitation/temperature data for
                                     at Western US extent.
    :param growing_season_dir: Directory path for growing season datasets.
    :param output_dir_prism_gs: File path of directory to save summed/mean prism precipitation/temperature data for
                                growing season of each year at Western US extent.
    :param year_list: Tuple/list of year_list for which prism data was downloaded.
    :param keyword: keyword to add before processed datasets. Can take 'prism_precip', 'prism_tmax', 'prism_tmin'.
                    Default set to 'prism_precip'.
    :param west_US_shape: Filepath of Western US shapefile.
    :param ref_raster: Model reference raster filepath.
    :param resolution: Resolution used in the model. Default set to model_res = 0.02000000000000000736.
    :param skip_processing: Set to True if want to skip prism precip processing.

    :return: None.
    """
    if not skip_processing:
        interim_dir_for_monthly_data = os.path.join(output_dir_prism_monthly, 'interim_dir_for_monthly_data')
        makedirs([output_dir_prism_monthly, output_dir_prism_gs, interim_dir_for_monthly_data])

        convert_prism_data_to_tif(input_dir=prism_bil_dir, output_dir=prism_tif_dir, keyword=keyword)

        #########
        # # Code-block for saving monthly data for the Western US
        #########
        # Clipping Prism monthly datasets for Western US
        monthly_prism_tifs = glob(os.path.join(prism_tif_dir, '*.tif'))  # monthly prism datasets
        for data in monthly_prism_tifs:
            month = os.path.basename(data).split('.')[0][-2:]
            year = os.path.basename(data).split('.')[0].split('_')[2][:4]

            if month.startswith('0'):  # don't want to keep 0 in month for consistency will all datasets
                month = month[-1]

            if 'precip' in keyword:
                monthly_raster_name = f'prism_precip_{year}_{month}.tif'
            elif 'tmax' in keyword:
                monthly_raster_name = f'prism_tmax_{year}_{month}.tif'
            elif 'tmin' in keyword:
                monthly_raster_name = f'prism_tmin_{year}_{month}.tif'

            # the prism datasets are at 4km native resolution and directly clipping and resampling them from 4km
            # resolution creates misalignment of pixels from reference raster. So, first we are resampling CONUS
            # scale original datasets to 2km resolutions and then clipping them at reference raster (Western US) extent
            interim_monthly_raster = clip_resample_reproject_raster(input_raster=data,
                                                                    input_shape=west_US_shape,
                                                                    raster_name=monthly_raster_name, keyword=' ',
                                                                    output_raster_dir=interim_dir_for_monthly_data,
                                                                    clip=False, resample=True, clip_and_resample=False,
                                                                    targetaligned=True, resample_algorithm='near',
                                                                    use_ref_width_height=False, ref_raster=None,
                                                                    resolution=resolution)

            clip_resample_reproject_raster(input_raster=interim_monthly_raster,
                                           input_shape=west_US_shape,
                                           raster_name=monthly_raster_name, keyword=' ',
                                           output_raster_dir=output_dir_prism_monthly,
                                           clip=False, resample=False, clip_and_resample=True,
                                           targetaligned=True, resample_algorithm='near',
                                           use_ref_width_height=False, ref_raster=ref_raster,
                                           resolution=resolution)
        #########
        # # Code-block for summing monthly data for year_list by growing season for the Western US
        #########
        for year in year_list:  # first loop for year_list
            print(f'Processing {keyword} data for {year}...')

            if 'precip' in keyword:
                dynamic_gs_sum_var(year_list, growing_season_dir=growing_season_dir, monthly_input_dir=output_dir_prism_monthly,
                                   gs_output_dir=output_dir_prism_gs,
                                   sum_keyword=keyword, skip_processing=False)

            elif any(i in keyword for i in ['tmax', 'tmin']):
                dynamic_gs_mean_var(year_list, growing_season_dir=growing_season_dir,
                                    monthly_input_dir=output_dir_prism_monthly,
                                    gs_output_dir=output_dir_prism_gs,
                                    mean_keyword=keyword, skip_processing=False)

    else:
        pass


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

            data_paths_lists_with_nan.append(paths+[nan_raster_path])

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


def run_all_preprocessing(skip_process_GrowSeason_data=False,
                          skip_prism_processing=False):
    """
    Run all preprocessing steps.

    :param skip_process_GrowSeason_data: Set to True to skip processing growing season data.
    :param skip_prism_processing: Set True if want to skip prism (precipitation and temperature) data preprocessing.


    :return: None.
    """
    # process growing season data
    extract_month_from_GrowSeason_data(GS_data_dir='../../Data_main/rasters/Growing_season',
                                       skip_processing=skip_process_GrowSeason_data)

    # prism precipitation data processing
    process_prism_data(year_list=(1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
                                  2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020),
                       prism_bil_dir='../../Data_main/rasters/PRISM_Precip/bil_format',
                       prism_tif_dir='../../Data_main/rasters/PRISM_Precip/tif_format',
                       output_dir_prism_monthly='../../Data_main/rasters/PRISM_Precip/WestUS_monthly',
                       growing_season_dir='../../Data_main/rasters/Growing_season',
                       output_dir_prism_gs='../../Data_main/rasters/PRISM_Precip/WestUS_growing_season',
                       west_US_shape='../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_states.shp',
                       keyword='prism_precip', skip_processing=skip_prism_processing)

    # prism maximum temperature data processing
    process_prism_data(year_list=(1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
                                  2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020),
                       prism_bil_dir='../../Data_main/rasters/PRISM_Tmax/bil_format',
                       prism_tif_dir='../../Data_main/rasters/PRISM_Tmax/tif_format',
                       output_dir_prism_monthly='../../Data_main/rasters/PRISM_Tmax/WestUS_monthly',
                       growing_season_dir='../../Data_main/rasters/Growing_season',
                       output_dir_prism_gs='../../Data_main/rasters/PRISM_Precip/WestUS_growing_season',
                       west_US_shape='../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_states.shp',
                       keyword='prism_tmax', skip_processing=skip_prism_processing)
