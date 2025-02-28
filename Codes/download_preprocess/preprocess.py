# author : Md Fahim Hasan
# PhD Candidate
# Colorado State university
# Fahim.Hasan@colostate.edu

import os
import re
import sys
import datetime
import numpy as np
from glob import glob
from osgeo import gdal
import rasterio as rio

from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.utils.system_ops import makedirs
from Codes.utils.raster_ops import read_raster_arr_object, write_array_to_raster, \
    clip_resample_reproject_raster, shapefile_to_raster

no_data_value = -9999
model_res = 0.01976293625031605786  # in deg, ~2 km
WestUS_shape = '../../Data_main/ref_shapes/WestUS.shp'
WestUS_raster = '../../Data_main/ref_rasters/Western_US_refraster_2km.tif'


def extract_month_from_GrowSeason_data(GS_data_dir, skip_processing=False):
    """
    Extract start and ending growing season months from growing season dataset (provided by Justin Huntington DRI;
    downloaded from GEE to google drive). The output datasets have 2 all_bands, containing start and end month info,
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

            # stacking the arrays together (single tif with 2 all_bands)
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


def dynamic_gs_sum_of_variable(year_list, growing_season_dir, monthly_input_dir, gs_output_dir,
                               sum_keyword, skip_processing=False):
    """
    Dynamically (spatio-temporally) sums any variable for dynamic growing seasons.

    :param year_list: List of years_list to process the data for.
    :param growing_season_dir: Directory path for growing season datasets.
    :param monthly_input_dir:  Directory path for monthly datasets.
    :param gs_output_dir:  Directory path (output) for summed growing season datasets.
    :param sum_keyword: Keyword str to add before the summed raster.
    :param skip_processing: Set to True if want to skip processing this step.

    :return: None.
    """
    if not skip_processing:
        makedirs([gs_output_dir])

        # The regex r'_([0-9]{1,2})\.tif' extracts the month (1 or 2 digits; e.g., '_1.tif', '_12.tif')
        # from the filenames using the first group ([0-9]{1,2}).
        # The extracted month is then (inside the for loop in the sorting block) converted to an integer with int(group(1))
        # for proper sorting by month.
        month_pattern = re.compile(r'_([0-9]{1,2})\.tif')

        for year in year_list:
            print(f'Dynamically summing {sum_keyword} monthly datasets for growing season {year}...')

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


def dynamic_gs_mean_of_variable(year_list, growing_season_dir, monthly_input_dir, gs_output_dir,
                                mean_keyword, skip_processing=False):
    """
    Dynamically (spatio-temporally) averages any variable for dynamic growing seasons.

    :param year_list: List of years to process the data for.
    :param growing_season_dir: Directory path for growing season datasets.
    :param monthly_input_dir:  Directory path for monthly datasets.
    :param gs_output_dir:  Directory path (output) for averaged growing season datasets.
    :param mean_keyword: Keyword str to add before the averaged raster.
    :param skip_processing: Set to True if want to skip processing this step.

    :return: None.
    """
    if not skip_processing:
        makedirs([gs_output_dir])

        month_pattern = re.compile(r'_([0-9]{1,2})\.tif')

        for year in year_list:
            print(f'Dynamically averaging {mean_keyword} monthly datasets for growing season {year}...')

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
            valid_month_count = valid_month_count.astype('float')  # converting valid_month_count to float to allow np.nan assignment
            valid_month_count[valid_month_count == 0] = np.nan  # to avoid division by zero for non-growing season pixels

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
        if 'precip' in keyword:
            dynamic_gs_sum_of_variable(year_list, growing_season_dir=growing_season_dir, monthly_input_dir=output_dir_prism_monthly,
                                       gs_output_dir=output_dir_prism_gs,
                                       sum_keyword=keyword, skip_processing=False)

        elif any(i in keyword for i in ['tmax', 'tmin']):
            dynamic_gs_mean_of_variable(year_list, growing_season_dir=growing_season_dir,
                                        monthly_input_dir=output_dir_prism_monthly,
                                        gs_output_dir=output_dir_prism_gs,
                                        mean_keyword=keyword, skip_processing=False)

    else:
        pass


def create_stateID_raster(westUS_shp, output_dir, skip_processing=False):
    """
    Create a stateID reference raster.

    :param westUS_shp: Western US shapefile with the attribute 'stateID'.
    :param output_dir: Output directory to save the created raster.
    :param skip_processing: Set to True to skip this process.

    :return: None.
    """
    if not skip_processing:
        makedirs([output_dir])

        shapefile_to_raster(input_shape=westUS_shp, output_dir=output_dir, raster_name='stateID.tif',
                            burnvalue=None, use_attr=True,
                            attribute='stateID', add=None, ref_raster=WestUS_raster,
                            resolution=model_res, alltouched=False)

        print('created stateID reference raster...')

    else:
        pass


def create_HUC12_SW_irrigation_rasters(HUC12_SW_shape, output_dir, resolution=model_res,
                                       ref_raster=WestUS_raster, skip_processing=False):
    """
    Rasterize USGS HUC12 level SW dataset. The created rasters has similar SW irrigation values for each pixel in
    a HUC12. Units are in  MG (million gallon).

    :param HUC12_SW_shape: Shapefile of HUC12 SW irrigation shapefile.
    :param output_dir: Directory path to save outputs.
    :param resolution: Resolution. Default set to model resolution.
    :param ref_raster: Reference raster. Default set to Western US reference raster.
    :param skip_processing: Set to True to skip processing.

    :return: None.
    """
    if not skip_processing:
        print('rasterizing HUC12 SW irrigation datasets...')

        # making output directories
        interim_outdir = os.path.join(output_dir, 'interim')
        normalized_dir = os.path.join(output_dir, 'MinMax_normalized')
        makedirs([output_dir, interim_outdir, normalized_dir])

        years = list(range(2000, 2020 + 1))  # years from 2000 to 2020

        for year in years:

            column_to_rasterize = str(year)

            interim_raster = shapefile_to_raster(input_shape=HUC12_SW_shape, output_dir=interim_outdir,
                                                 raster_name=f'HUC12_SW_{year}.tif',
                                                 burnvalue=None, use_attr=True,
                                                 attribute=column_to_rasterize,
                                                 add=None, ref_raster=ref_raster,
                                                 resolution=resolution, alltouched=False)

            # the produced interim raster has no data values in a few HUCs
            # setting them to 0 to create a continuous raster
            ref_arr = read_raster_arr_object(ref_raster, get_file=False)

            interim_sw_arr, file = read_raster_arr_object(interim_raster)
            final_sw_arr = np.where(np.isnan(interim_sw_arr) & (ref_arr == 0), 0, interim_sw_arr)

            output_raster = os.path.join(output_dir, f'HUC12_SW_{year}.tif')
            write_array_to_raster(final_sw_arr, file, file.transform, output_raster)


            # min-max normalizing data (to represent relative use of surface water irrigation)
            min_sw = np.nanmin(final_sw_arr)
            max_sw = np.nanmax(final_sw_arr)

            normalized_sw_arr = np.where(~np.isnan(final_sw_arr), (final_sw_arr - min_sw) / (max_sw - min_sw), -9999)
            output_normal_raster = os.path.join(normalized_dir, f'HUC12_SW_{year}.tif')
            write_array_to_raster(normalized_sw_arr, file, file.transform, output_normal_raster)



def create_GW_use_perc_rasters(HUC12_GW_perc_shape, output_dir, resolution=model_res,
                               ref_raster=WestUS_raster, skip_processing=False):
    """
    Rasterize USGS HUC12 level GW use % (GW use % w.r.t. total water use) dataset.
    The created rasters has similar GW use % values for each pixel in
    a HUC12.

    :param HUC12_GW_perc_shape: Shapefile of HUC12 GW use % shapefile.
    :param output_dir: Directory path to save outputs.
    :param resolution: Resolution. Default set to model resolution.
    :param ref_raster: Reference raster. Default set to Western US reference raster.
    :param skip_processing: Set to True to skip processing.

    :return: None.
    """
    if not skip_processing:
        print('rasterizing HUC12 GW % datasets...')

        # making output directories
        interim_outdir = os.path.join(output_dir, 'interim')
        makedirs([output_dir, interim_outdir])

        years = list(range(2000, 2020 + 1))  # years from 2000 to 2020

        for year in years:

            column_to_rasterize = f'{year}_gw_%'

            interim_raster = shapefile_to_raster(input_shape=HUC12_GW_perc_shape,
                                                 output_dir=interim_outdir,
                                                 raster_name=f'HUC12_GW_perc_{year}.tif',
                                                 burnvalue=None, use_attr=True,
                                                 attribute=column_to_rasterize,
                                                 add=None, ref_raster=ref_raster,
                                                 resolution=resolution, alltouched=False)

            # the produced interim raster has no data values in a few HUCs
            # setting them to 0 to create a continuous raster
            ref_arr = read_raster_arr_object(ref_raster, get_file=False)

            interim_sw_arr, file = read_raster_arr_object(interim_raster)
            final_sw_arr = np.where(np.isnan(interim_sw_arr) & (ref_arr == 0), 0, interim_sw_arr)

            output_raster = os.path.join(output_dir, f'HUC12_GW_perc_{year}.tif')
            write_array_to_raster(final_sw_arr, file, file.transform, output_raster)


def process_and_OneHotEncode_Koppen_Geiger(koppen_geiger_raster, output_dir,
                                           ref_raster=WestUS_raster, skip_processing=False):
    """
    Process (clip + resample + reclassify + OneHotEncode) Kopen-Geiger climate data.

    :param koppen_geiger_raster: Filepath of raw Koppen-Geiger raster data.
    :param output_dir: Filepath of output dir.
    :param ref_raster: Western US reference raster.
    :param skip_processing: Set to True to skip this process.

    :return: None.
    """
    if not skip_processing:
        print('processing Koppen-Geiger climate data...')

        westUS_climate = \
            clip_resample_reproject_raster(input_raster=koppen_geiger_raster,
                                           input_shape=WestUS_shape,
                                           raster_name='Koppen_Geiger_westUS.tif',
                                           output_raster_dir=output_dir,
                                           clip=False, resample=False, clip_and_resample=True,
                                           targetaligned=True, resample_algorithm='near',
                                           use_ref_width_height=False, ref_raster=None,
                                           resolution=model_res)

        # One Hot Encoding
        climate_arr, raster_file = read_raster_arr_object(westUS_climate)

        # classification map
        classification_map = {'arid': [4, 5, 6],
                              'temperate_dry_summer': [7, 8, 9, 10],
                              'temperate_no_dry_summer': [14, 15, 16, 21, 25],
                              'cold': [17, 18, 19, 22, 26, 27]}

        # reclassifying categories and saving them as a new array (single)
        reclassified_arr = np.full_like(climate_arr, -9999)  # Default to -9999 (or NoData)
        reclassified_arr = np.where(np.isin(climate_arr, classification_map['arid']), 1, reclassified_arr)
        reclassified_arr = np.where(np.isin(climate_arr, classification_map['temperate_dry_summer']), 2, reclassified_arr)
        reclassified_arr = np.where(np.isin(climate_arr, classification_map['temperate_no_dry_summer']), 3, reclassified_arr)
        reclassified_arr = np.where(np.isin(climate_arr, classification_map['cold']), 4, reclassified_arr)

        output_reclassified_raster = os.path.join(output_dir, 'reclassified', 'Koppen_Geiger_westUS_classified.tif')
        write_array_to_raster(reclassified_arr, raster_file, raster_file.transform, output_reclassified_raster,
                              dtype='float32')

        # reference raster
        ref_arr = read_raster_arr_object(ref_raster, get_file=False)

        # reclassifying categories and saving them separately for each category
        for category, values in classification_map.items():
            perCategory_arr = np.where(np.isin(climate_arr, values), 1, 0)
            perCategory_arr[np.isnan(ref_arr)] = -9999           # setting -9999 outside of western Us

            output_perCategory_raster = os.path.join(output_dir, f'OneHotEncoded/{category}', f'{category}.tif')
            write_array_to_raster(perCategory_arr, raster_file, raster_file.transform, output_perCategory_raster,
                                  dtype='float32')
    else:
        pass


def process_netGWIrr_data(netGW_dir, output_dir, skip_processing=False):
    """
    Process consumptive groundwater irrigation dataset (replace 0 value with -9999/nodata).

    :param netGW_dir: Directory of consumptive groundwater irrigation dataset.
    :param output_dir: Output directory filepath.
    :param skip_processing: Set to True to skip this step.

    :return: None.
    """
    if not skip_processing:
        print('processing netGW data...')

        makedirs([output_dir])

        netGW_data = glob(os.path.join(netGW_dir, '*.tif'))

        for data in netGW_data:
            arr, file = read_raster_arr_object(data)

            # setting zero values as -9999 (nodata)
            arr[arr == 0] = -9999

            # saving data
            write_array_to_raster(arr, file, file.transform,
                                  output_path=os.path.join(output_dir, os.path.basename(data)))
    else:
        pass


def create_pixelID_raster(WestUS_refraster, output_dir, skip_processing=False):
    """
    Create pixelID raster. Each valid pixel has a unique pixelID value.

    :param WestUS_refraster: Western US reference raster.
    :param output_dir: Directory path to save the output raster.
    :param skip_processing: Set to True to skip processing.

    :return: None.
    """
    if not skip_processing:
        print('process PixelID raster')

        makedirs([output_dir])

        # the reference raster has zero in valid pixel and -9999 in invalid pixels
        # the valid pixel values are going to be replaced by an unique pixel id
        ref_arr, ref_file = read_raster_arr_object(WestUS_refraster)

        # converting into a binary array with valid values as 1
        binary_arr = np.where(ref_arr == 0, 1, 0).flatten()

        # valid pixel count
        valid_count = np.count_nonzero(binary_arr)
        pixelID = np.arange(1, valid_count+1, 1)

        # determining valid pixel positions with boolean
        mask_arr = binary_arr.astype(bool).flatten()

        # replacing valid positions with pixelID
        binary_arr[mask_arr] = pixelID
        binary_arr[binary_arr == 0] = -9999
        pixelID_arr = binary_arr.reshape(ref_file.shape)

        # saving the array
        output_path = os.path.join(output_dir, 'pixelID.tif')
        write_array_to_raster(pixelID_arr, ref_file, ref_file.transform,
                              output_path)


def run_all_preprocessing(skip_stateID_raster_creation=False,
                          skip_pixelID_raster_creation=False,
                          skip_process_GrowSeason_data=False,
                          skip_process_netGW=False,
                          skip_ET_processing=False,
                          skip_prism_precip_processing=False,
                          skip_prism_tmax_processing=False,
                          skip_gridmet_RET_processing=False,
                          skip_gridmet_precip_processing=False,
                          skip_gridmet_tmax_processing=False,
                          skip_gridmet_maxRH_processing=False,
                          skip_gridmet_minRH_processing=False,
                          skip_gridmet_windVel_processing=False,
                          skip_gridmet_shortRad_processing=False,
                          skip_gridmet_vpd_processing=False,
                          skip_daymet_sunHr_processing=False,
                          skip_HUC12_SW_processing=False,
                          skip_HUC12_GW_perc_processing=False,
                          skip_koppen_geiger_processing=False):
    """
    Run all preprocessing steps.

    :param skip_stateID_raster_creation: Set to True to skip stateID raster creation.
    :param skip_pixelID_raster_creation: Set to True to skip pixelID raster creation.
    :param skip_process_GrowSeason_data: Set to True to skip processing growing season data.
    :param skip_process_netGW: Set to True to skip consumptive groundwater irrigation dataset processing.
    :param skip_ET_processing: Set to True to skip processing grwing season ET data.
    :param skip_prism_precip_processing: Set True if want to skip prism precipitation data preprocessing.
    :param skip_prism_tmax_processing: Set True if want to skip prism temperature data preprocessing.
    :param skip_gridmet_RET_processing: Set to True to skip processing RET growing season data.
    :param skip_gridmet_precip_processing: Set to True to skip processing gridmet precip growing season data.
    :param skip_gridmet_tmax_processing: Set to True to skip processing gridmet max temperature growing season data.
    :param skip_gridmet_maxRH_processing: Set to True to skip processing gridmet max RH growing season data.
    :param skip_gridmet_minRH_processing: Set to True to skip processing gridmet min RH growing season data.
    :param skip_gridmet_windVel_processing: Set to True to skip processing gridmet wind velocity growing season data.
    :param skip_gridmet_shortRad_processing: Set to True to skip processing gridmet shortwave raditaion growing season data.
    :param skip_gridmet_vpd_processing: Set to True to skip processing gridmet vapor pressure deficit growing season data.
    :param skip_daymet_sunHr_processing: Set to True to skip processing daymet sun hour growing season data.
    :param skip_HUC12_SW_processing: Set to True to skip create SW irrigation dataset.
    :param skip_HUC12_GW_perc_processing: Set to True to skip create GW use % dataset.
    :param skip_koppen_geiger_processing: Set to False to skip Koppen Geigar climate data processing and One-Hot-Encoding.

    :return: None.
    """
    # create stateID raster
    create_stateID_raster(westUS_shp='../../Data_main/ref_shapes/WestUS_states.shp',
                          output_dir='../../Data_main/ref_rasters/stateID',
                          skip_processing=skip_stateID_raster_creation)

    # create PixelID raster
    create_pixelID_raster(WestUS_refraster=WestUS_raster,
                          output_dir='../../Data_main/ref_rasters/pixelID',
                          skip_processing=skip_pixelID_raster_creation)

    # process growing season data
    extract_month_from_GrowSeason_data(GS_data_dir='../../Data_main/rasters/Growing_season',
                                       skip_processing=skip_process_GrowSeason_data)

    # process netGW (consumptive groundwater irrigation) data
    process_netGWIrr_data(netGW_dir='../../Data_main/rasters/NetGW_irrigation/original',
                          output_dir='../../Data_main/rasters/NetGW_irrigation/WesternUS',
                          skip_processing=skip_process_netGW)

    # OpenET ensemble processing
    dynamic_gs_sum_of_variable(year_list=(2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
                                          2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020),
                               growing_season_dir='../../Data_main/rasters/Growing_season',
                               monthly_input_dir='../../Data_main/rasters/OpenET_ensemble/WestUS_monthly',
                               gs_output_dir='../../Data_main/rasters/OpenET_ensemble/WestUS_growing_season',
                               sum_keyword='OpenET', skip_processing=skip_ET_processing)

    # prism precipitation data processing
    process_prism_data(year_list=(1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
                                  2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020),
                       prism_bil_dir='../../Data_main/rasters/PRISM_Precip/bil_format',
                       prism_tif_dir='../../Data_main/rasters/PRISM_Precip/tif_format',
                       output_dir_prism_monthly='../../Data_main/rasters/PRISM_Precip/WestUS_monthly',
                       growing_season_dir='../../Data_main/rasters/Growing_season',
                       output_dir_prism_gs='../../Data_main/rasters/PRISM_Precip/WestUS_growing_season',
                       west_US_shape=WestUS_shape, keyword='prism_precip',
                       skip_processing=skip_prism_precip_processing)

    # prism maximum temperature data processing
    process_prism_data(year_list=(1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
                                  2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020),
                       prism_bil_dir='../../Data_main/rasters/PRISM_Tmax/bil_format',
                       prism_tif_dir='../../Data_main/rasters/PRISM_Tmax/tif_format',
                       output_dir_prism_monthly='../../Data_main/rasters/PRISM_Tmax/WestUS_monthly',
                       growing_season_dir='../../Data_main/rasters/Growing_season',
                       output_dir_prism_gs='../../Data_main/rasters/PRISM_Tmax/WestUS_growing_season',
                       west_US_shape=WestUS_shape, keyword='prism_tmax',
                       skip_processing=skip_prism_tmax_processing)

    # RET processing
    dynamic_gs_sum_of_variable(year_list=(2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
                                          2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020),
                               growing_season_dir='../../Data_main/rasters/Growing_season',
                               monthly_input_dir='../../Data_main/rasters/RET/WestUS_monthly',
                               gs_output_dir='../../Data_main/rasters/RET/WestUS_growing_season',
                               sum_keyword='RET', skip_processing=skip_gridmet_RET_processing)

    # GRIDMET precipitation data processing
    dynamic_gs_sum_of_variable(year_list=(2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
                                          2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020),
                               growing_season_dir='../../Data_main/rasters/Growing_season',
                               monthly_input_dir='../../Data_main/rasters/Precip/WestUS_monthly',
                               gs_output_dir='../../Data_main/rasters/Precip/WestUS_growing_season',
                               sum_keyword='Precip', skip_processing=skip_gridmet_precip_processing)

    # GRIDMET max temperature data processing
    dynamic_gs_mean_of_variable(year_list=(2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
                                          2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020),
                                growing_season_dir='../../Data_main/rasters/Growing_season',
                                monthly_input_dir='../../Data_main/rasters/Tmax/WestUS_monthly',
                                gs_output_dir='../../Data_main/rasters/Tmax/WestUS_growing_season',
                                mean_keyword='Tmax', skip_processing=skip_gridmet_tmax_processing)

    # GRIDMET max relative humidity data processing
    dynamic_gs_mean_of_variable(year_list=(2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
                                           2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020),
                                growing_season_dir='../../Data_main/rasters/Growing_season',
                                monthly_input_dir='../../Data_main/rasters/maxRH/WestUS_monthly',
                                gs_output_dir='../../Data_main/rasters/maxRH/WestUS_growing_season',
                                mean_keyword='maxRH', skip_processing=skip_gridmet_maxRH_processing)

    # GRIDMET min relative humidity data processing
    dynamic_gs_mean_of_variable(year_list=(2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
                                           2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020),
                                growing_season_dir='../../Data_main/rasters/Growing_season',
                                monthly_input_dir='../../Data_main/rasters/minRH/WestUS_monthly',
                                gs_output_dir='../../Data_main/rasters/minRH/WestUS_growing_season',
                                mean_keyword='minRH', skip_processing=skip_gridmet_minRH_processing)

    # GRIDMET wind velocity data processing
    dynamic_gs_mean_of_variable(year_list=(2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
                                           2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020),
                                growing_season_dir='../../Data_main/rasters/Growing_season',
                                monthly_input_dir='../../Data_main/rasters/windVel/WestUS_monthly',
                                gs_output_dir='../../Data_main/rasters/windVel/WestUS_growing_season',
                                mean_keyword='windVel', skip_processing=skip_gridmet_windVel_processing)

    # GRIDMET shortwave radiation data processing
    dynamic_gs_mean_of_variable(year_list=(2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
                                           2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020),
                                growing_season_dir='../../Data_main/rasters/Growing_season',
                                monthly_input_dir='../../Data_main/rasters/shortRad/WestUS_monthly',
                                gs_output_dir='../../Data_main/rasters/shortRad/WestUS_growing_season',
                                mean_keyword='shortRad', skip_processing=skip_gridmet_shortRad_processing)

    # GRIDMET vapor pressure deficit data processing
    dynamic_gs_mean_of_variable(year_list=(2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
                                           2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020),
                                growing_season_dir='../../Data_main/rasters/Growing_season',
                                monthly_input_dir='../../Data_main/rasters/vpd/WestUS_monthly',
                                gs_output_dir='../../Data_main/rasters/vpd/WestUS_growing_season',
                                mean_keyword='vpd', skip_processing=skip_gridmet_vpd_processing)

    # DAYMET sun hour data processing
    dynamic_gs_mean_of_variable(year_list=(2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
                                           2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020),
                                growing_season_dir='../../Data_main/rasters/Growing_season',
                                monthly_input_dir='../../Data_main/rasters/sunHr/WestUS_monthly',
                                gs_output_dir='../../Data_main/rasters/sunHr/WestUS_growing_season',
                                mean_keyword='sunHr', skip_processing=skip_daymet_sunHr_processing)

    # HUC12 SW rasterization
    create_HUC12_SW_irrigation_rasters(HUC12_SW_shape='../../Data_main/shapefiles/USGS_WaterUse/HUC12_WestUS_with_Annual_SW.shp',
                                       output_dir='../../Data_main/rasters/HUC12_SW',
                                       resolution=model_res, ref_raster=WestUS_raster,
                                       skip_processing=skip_HUC12_SW_processing)

    # HUC12 GW use % rasterization
    create_GW_use_perc_rasters(HUC12_GW_perc_shape='../../Data_main/shapefiles/USGS_WaterUse/HUC12_WestUS_with_GW_use_perc.shp',
                               output_dir='../../Data_main/rasters/HUC12_GW_perc', resolution=model_res,
                               ref_raster=WestUS_raster, skip_processing=skip_HUC12_GW_perc_processing)


    # Koppen_Geiger climate data processing
    process_and_OneHotEncode_Koppen_Geiger(
        koppen_geiger_raster='../../Data_main/rasters/Koppen_geiger/1991_2020/koppen_geiger_0p00833333.tif',
        output_dir='../../Data_main/rasters/Koppen_geiger',
        skip_processing=skip_koppen_geiger_processing)

