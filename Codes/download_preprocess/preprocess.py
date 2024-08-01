import os
from glob import glob
from osgeo import gdal

from Codes.utils.raster_ops import create_multiband_raster, clip_resample_reproject_raster,\
    mean_rasters, sum_rasters
from Codes.utils.system_ops import makedirs

no_data_value = -9999
model_res = 0.01976293625031605786  # in deg, ~2 km
WestUS_shape = '../../Data_main/ref_shapes/WestUS_states.shp'
WestUS_raster = '../../Data_main/ref_rasters/Western_US_refraster_2km.tif'


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
    """
    Make multi-band raster dataset from individual temporal/static attributes.

    :param list_of_temporal_var_dirs: List of directories of temporal attributes/datasets.
    :param list_of_static_var_dirs: List of directories of static attributes/datasets.
    :param band_key_list: keyword to add as band names in the multi-band dataset.
    :param output_dir: Path of output dir where the multi-band datasets will be stored.
    :param years_list: List of years to process the datasets.
    :param month_range: Range of months to process the data for, e.g., (4, 10) for growing season months.
                        Default set to None to process annual datasets only.
    :return: None.
    """
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


def process_prism_data(prism_bil_dir, prism_tif_dir, output_dir_prism_monthly, output_dir_prism_yearly,
                       year_list=(1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
                                  2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020),
                       keyword='prism_precip',
                       west_US_shape='../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_states.shp',
                       ref_raster=WestUS_raster, resolution=model_res, skip_processing=False):
    """
    Process (sum and mean to Western US extent) Prism Precipitation, Tmax, and Tmin data. The precipitation data is
    summed for all months in a year. The Tmax and Tmin data is averaged for growing season (April - October).

    :param prism_bil_dir: Directory file path of downloaded prism datasets in .bil format.
    :param prism_tif_dir: Directory file path of prism datasets converted to tif format.
    :param output_dir_prism_monthly: File path of directory to save monthly prism precipitation/temperature data for
                                     at Western US extent.
    :param output_dir_prism_yearly: File path of directory to save summed/mean prism precipitation/temperature data for
                                    each year at Western US extent.
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
        makedirs([output_dir_prism_monthly, output_dir_prism_yearly, interim_dir_for_monthly_data])

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
                    prism_datasets = glob(os.path.join(output_dir_prism_monthly, f'*{year}*.tif'))  # monthly prism datasets for each year

                    # Summing precip for each calendar year
                    summed_output_for_year = os.path.join(output_dir_prism_yearly, f'prism_precip_{year}.tif')
                    sum_rasters(raster_list=prism_datasets, raster_dir=None, output_raster=summed_output_for_year,
                                ref_raster=prism_datasets[0])

                    # summing precip for each year's growing season only
                    prism_datasets = glob(os.path.join(output_dir_prism_monthly, f'*{year}_[4-9]*.tif')) + \
                                            glob(os.path.join(output_dir_prism_monthly, f'*{year}_10*.tif'))  # monthly growing season prism datasets for each year
                    summed_output_for_year = os.path.join(output_dir_prism_yearly, f'prism_precip_{year}.tif')
                    sum_rasters(raster_list=prism_datasets, raster_dir=None, output_raster=summed_output_for_year,
                                ref_raster=prism_datasets[0])

                    # summing precip for each water year
                    prism_datasets = glob(os.path.join(output_dir_prism_monthly, f'*{year - 1}_1[0-2].*tif')) + \
                                            glob(os.path.join(output_dir_prism_monthly, f'*{year}_[1-9].*tif'))

            elif any(i in keyword for i in ['tmax', 'tmin']):

                    prism_datasets = glob(os.path.join(output_dir_prism_monthly, f'*{year}_[4-9]*.tif')) + \
                                     glob(os.path.join(output_dir_prism_monthly, f'*{year}_10*.tif'))   # monthly growing season prism datasets for each year

                    # Calculating mean of rasters for growing season (April-October)
                    mean_output_for_year = os.path.join(output_dir_prism_yearly, f'{keyword}_{year}.tif')
                    mean_rasters(raster_list=prism_datasets, raster_dir=None, output_raster=mean_output_for_year,
                                 ref_raster=prism_datasets[0])

    else:
        pass

