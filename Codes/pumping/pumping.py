# Author : Md Fahim Hasan
# PhD Candidate
# Colorado State university
# Fahim.Hasan@colostate.edu

# # The pumping data processed by this script requires some hand-filtering and processing after the process
# of raw data to point creation. After that some hand-filtering is performed and the final data is rasterized
# as pumping rasters

# # We also applied filtering out very low and high pumping values for Colorado and kansas.
# Removed pixels where pumping + surface water irrigation (if happens) + Peff < irrigated crop ET (under a threshold).
# For Arizona, we didn't do it as Arizona has large surface water irrigation, and we don't have an accurate
# surface water irrigation dataset (we only have a decent proxy from USGS HUC12 dataset).
# Moreover, both Kansas and Arizona are known to have high quality pumping data.

import os
import sys
import pickle
import numpy as np
import pandas as pd
from glob import glob
import geopandas as gpd
from pyproj import Transformer

from os.path import dirname, abspath

sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.utils.system_ops import makedirs
from Codes.utils.vector_ops import clip_vector
from Codes.utils.raster_ops import read_raster_arr_object, write_array_to_raster, \
    shapefile_to_raster, mask_raster_by_shape

no_data_value = -9999
model_res = 0.01976293625031605786  # in deg, ~2 km
WestUS_raster = '../../Data_main/ref_rasters/Western_US_refraster_2km.tif'


def get_well_coords_for_AZ(well_registry_shp, save_dict_path, skip_processing=False):
    """
    Processes the well registry shapefile to extract well coordinates or loads preprocessed data.

    This function reads a well registry shapefile, transforms UTM coordinates to latitude and longitude (EPSG:4269),
    and creates a dictionary mapping well registry IDs to their coordinates. The dictionary is saved to a specified
    path for future use.

    :param well_registry_shp: str. Path to the well registry shapefile containing UTM coordinates.
    :param save_dict_path: str. Path to save or load the well registry dictionary (mapping REGISTRY_I to [Lon, Lat]).
    :param skip_processing: bool. If True, skips processing the shapefile and loads the dictionary from
                            the specified path.

    :return: A dictionary where:
                        - Keys: Well registry IDs.
                        - Values: Lists of [longitude, latitude] coordinates.
    """
    if not skip_processing:
        wellReg_df = gpd.read_file(well_registry_shp)

        # converting coordinates to EPSG:4269
        lon_wellReg = wellReg_df['UTM_X_METE']
        lat_wellReg = wellReg_df['UTM_Y_METE']

        from_crs, to_crs = 'EPSG:26912', 'EPSG:4269'
        proj_transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
        transformed_lon, transformed_lat = proj_transformer.transform(lon_wellReg, lat_wellReg)

        wellReg_df['Lon'] = pd.Series(transformed_lon)
        wellReg_df['Lat'] = pd.Series(transformed_lat)

        # selecting required columns from the well_registry
        wellReg_df = wellReg_df[['REGISTRY_I', 'Lon', 'Lat']]

        # creating a dictionary where 'REGISTRY_I' is key and coordinates are values
        wellReg_dict = wellReg_df.set_index('REGISTRY_I')[['Lon', 'Lat']].apply(list, axis=1).to_dict()

        # saving dictionary
        pickle.dump(wellReg_dict, open(save_dict_path, mode='wb'))

        return wellReg_dict

    else:
        wellReg_dict = pickle.load(open(save_dict_path, mode='rb'))

        return wellReg_dict


def process_AZ_pumping_csv(raw_csv_dir, well_reg_shp, well_reg_dict, output_shp,
                           skip_process_Well_registry_file=False,
                           skip_process=False, **kwargs):
    """
    This function reads raw CSV files containing groundwater pumping data, filters the data based on specified
    criteria, matches well coordinates from a well registry shapefile, and outputs the processed data as both a
    CSV file and a shapefile.

    *** The function can run without the **kwargs. Wanted to try kwargs.

    :param raw_csv_dir: str. Directory containing raw CSV files for Arizona groundwater pumping data.
    :param well_reg_shp: str. Path to the well registry shapefile.
    :param well_reg_dict: str. Path to save or load the well registry dictionary (mapping Well IDs to coordinates).
    :param output_shp: str. Path to save the processed data to shapefile with groundwater pumping data.
    :param skip_process_Well_registry_file: bool. If True, skips reprocessing the well registry shapefile and loads the
                                            dictionary from the specified path. Default is False.
    :param skip_process: bool. If True, skips the entire processing workflow and does nothing. Default is False.
    :param kwargs: dict (optional). Can take dictionaries - selected_columns_from_csv=['Well Id', 'AMA INA', 'YEAR',
                                                                                      'Right Type', 'AF Pumped',
                                                                                      'Water Type'],
                                                          - filter_conditions={'Right Type': ['IRRIGATION USE',
                                                                                             'IRRIGATION DISTRICT (GW ONLY)'],
                                                                               'Water Type': 'GROUNDWATER'}
    :return: None.
    """
    if not skip_process:
        print('Processing pumping data for Arizona...')

        makedirs([os.path.dirname(output_shp)])

        # collecting all csvs
        pumping_csvs = glob(os.path.join(raw_csv_dir, '*GW_*.csv'))

        # compiling all csvs into one dataframe
        compiled_pump_df = pd.DataFrame()

        for csv in pumping_csvs:
            pump_df = pd.read_csv(csv)
            compiled_pump_df = pd.concat([compiled_pump_df, pump_df])

        # Default columns to select
        selected_columns = kwargs.get('selected_columns_from_csv', [
            'Well Id', 'AMA INA', 'YEAR',
            'Right Type', 'AF Pumped', 'Water Type'
        ])

        # selecting needed columns
        compiled_pump_df = compiled_pump_df[selected_columns]

        # applying filter conditions
        filter_conditions = kwargs.get('filter_conditions', {
            'Right Type': ['IRRIGATION USE', 'IRRIGATION DISTRICT (GW ONLY)',
                           'IRRIGATION DISTRICT (SW & GW)', 'BEST MANAGEMENT PRACTICES FARM UNIT'],
            'Water Type': 'GROUNDWATER'
        })

        # filtering for correct 'Right Type' and 'Water Type'
        if 'Right Type' in filter_conditions:
            compiled_pump_df = compiled_pump_df[compiled_pump_df['Right Type'].isin(filter_conditions['Right Type'])]

        if 'Water Type' in filter_conditions:
            compiled_pump_df = compiled_pump_df[compiled_pump_df['Water Type'] == filter_conditions['Water Type']]

        # converting Well ID to 6-digit str for matching with the well_registry
        modified_well_id = ['0' + str(well_id) if len(str(well_id)) == 5 else str(well_id)
                            for well_id in compiled_pump_df['Well Id']]

        compiled_pump_df['Well Id'] = modified_well_id

        # getting lat, lon info from well_registry data
        well_coords_dict = get_well_coords_for_AZ(well_reg_shp, well_reg_dict,
                                                  skip_processing=skip_process_Well_registry_file)

        # assigning lat-lon info in compiled pumping database
        compiled_pump_df['Lon'] = compiled_pump_df['Well Id'].map(lambda x: well_coords_dict.get(x, [None, None])[0])
        compiled_pump_df['Lat'] = compiled_pump_df['Well Id'].map(lambda x: well_coords_dict.get(x, [None, None])[1])

        # organizing IMA/AMA names and year
        rename_dict = {'DOUGLAS INA': 'DOUGLAS AMA', 'HARQUAHALA VALLEY INA': 'HARQUAHALA INA',
                       'JOSEPH CITY INA': 'JOSEPH CITY INA', 'PHOENIX AMA': 'PHOENIX AMA',
                       'PINAL AMA': 'PINAL AMA', 'PRESCOTT AMA': 'PRESCOTT AMA',
                       'SANTA CRUZ AMA': 'SANTA CRUZ AMA', 'TUCSON AMA': 'TUCSON AMA'}
        compiled_pump_df['AMA INA'] = compiled_pump_df['AMA INA'].map(rename_dict)
        compiled_pump_df = compiled_pump_df.rename(columns={'YEAR': 'Year', 'AF Pumped': 'AF_pumped'})

        # saving as a shapefile
        compiled_pump_gdf = gpd.GeoDataFrame(compiled_pump_df,
                                             geometry=gpd.points_from_xy(compiled_pump_df['Lon'],
                                                                         compiled_pump_df['Lat']))

        compiled_pump_gdf = compiled_pump_gdf.set_crs('EPSG:4269')
        compiled_pump_gdf.to_file(output_shp)
    else:
        pass


def process_KS_pumping_csv(raw_csv, output_pump_csv, output_pump_shp,
                           output_acres_csv, skip_process=False, **kwargs):
    """
    Processes Kansas groundwater pumping data and irrigated acreage data.

    This function reads raw CSV data containing Kansas groundwater pumping and irrigated acreage information,
    filters the data to include columns of interest, and reshapes the data from wide format to long format using
    pandas' `melt` method. The processed pumping data is saved as both a CSV file and a shapefile, and the
    irrigated acreage data is saved as a separate CSV file.

    :param raw_csv: str. Path to the raw groundwater pumping data CSV file.
    :param output_pump_csv: str. Path to save the processed pumping data as a CSV file.
    :param output_pump_shp:str. Path to save the processed pumping data as a shapefile.
    :param output_acres_csv:str. Path to save the processed irrigated acreage data as a CSV file.
    :param skip_process: bool. If True, skips the processing workflow and does nothing. Default is False.
    :param kwargs: dict (optional). Additional parameters:
                - `cols_of_interest` (list of str): Columns to retain from the raw CSV file. Default includes:
                                                    `lat_nad83`, `long_nad83`, `gmd``source`,
                                                     pumping data for years 2000-2023 (`AF_USED_*`),
                                                     and irrigated acreage data for years 2000-2020 (`ACRES_*`).
    :return: None.
    """
    if not skip_process:
        print('Processing pumping data for Kansas...')

        makedirs([os.path.dirname(output_pump_shp)])

        pumping_df = pd.read_csv(raw_csv)

        # selecting columns of interest
        # using kwargs so that we can incorporate additional columns if needed
        columns_of_interest = kwargs.get('cols_of_interest',
                                         ['long', 'lat', 'gmd', 'source', 'AF_USED_2000',
                                          'AF_USED_2001', 'AF_USED_2002', 'AF_USED_2003',
                                          'AF_USED_2004', 'AF_USED_2005', 'AF_USED_2006',
                                          'AF_USED_2007', 'AF_USED_2008', 'AF_USED_2009',
                                          'AF_USED_2010', 'AF_USED_2011', 'AF_USED_2012',
                                          'AF_USED_2013', 'AF_USED_2014', 'AF_USED_2015',
                                          'AF_USED_2016', 'AF_USED_2017', 'AF_USED_2018',
                                          'AF_USED_2019', 'AF_USED_2020', 'AF_USED_2021',
                                          'AF_USED_2022', 'AF_USED_2023', 'AF_USED_2024',
                                          'ACRES_2000', 'ACRES_2001', 'ACRES_2002', 'ACRES_2003', 'ACRES_2004',
                                          'ACRES_2005', 'ACRES_2006', 'ACRES_2007', 'ACRES_2008', 'ACRES_2009',
                                          'ACRES_2010', 'ACRES_2011', 'ACRES_2012', 'ACRES_2013', 'ACRES_2014',
                                          'ACRES_2015', 'ACRES_2016', 'ACRES_2017', 'ACRES_2018', 'ACRES_2019',
                                          'ACRES_2020', 'ACRES_2021', 'ACRES_2022', 'ACRES_2023', 'ACRES_2024'])

        pumping_df = pumping_df[columns_of_interest]

        # # using pandas melt to convert the wide format data to long format
        af_columns = pumping_df.filter(like='AF').columns
        acres_columns = pumping_df.filter(like='ACRES').columns

        # for Acre-ft
        af_melted = pumping_df.melt(id_vars=['lat', 'long', 'gmd', 'source'],
                                    value_vars=list(af_columns),
                                    var_name='var1', value_name='AF_pumped')

        af_melted['Year'] = af_melted['var1'].apply(lambda x: int(x[-4:]))
        af_melted = af_melted.drop(columns=['var1'])

        # for Acres
        acres_melted = pumping_df.melt(id_vars=['lat', 'long', 'gmd', 'source'],
                                       value_vars=list(acres_columns),
                                       var_name='var2', value_name='Acres')

        acres_melted['Year'] = acres_melted['var2'].apply(lambda x: int(x[-4:]))
        acres_melted = acres_melted.drop(columns=['var2'])

        # saving pumping csv and shapefile
        af_melted = af_melted.rename(columns={'lat': 'Lat', 'long': 'Lon'})
        af_melted.to_csv(output_pump_csv, index=False)

        af_melted_gdf = gpd.GeoDataFrame(af_melted,
                                         geometry=gpd.points_from_xy(af_melted['Lon'],
                                                                     af_melted['Lat']))
        af_melted_gdf = af_melted_gdf.set_crs('EPSG:4269')
        af_melted_gdf.to_file(output_pump_shp)

        # saving acres records
        acres_melted = acres_melted.rename(columns={'lat': 'Lat', 'long': 'Lon'})
        acres_melted.to_csv(output_acres_csv, index=False)

    else:
        pass


def process_CO_pumping_data(raw_data_dir, well_ID_shp, output_pump_shp, skip_process=False):
    """
   Processes Colorado groundwater pumping data.

   This function reads raw CSV data containing Colorado groundwater pumping information, merges them together, and
   filters the data to include columns of interest. The processed pumping data is saved as both a CSV file
   and a shapefile.

   ** Note that - after processing the data using this code, I filtered the dataset further for land use (removed wells
   in cities/mines with no nearby agricultural fields).

   ** The next step will be to select data in predominantly agricultural regions to make the dataset better.

   :param raw_data_dir: str. Path to the raw data directory.
    :param well_ID_shp: str. Well ID shapefile.
   :param output_pump_shp:str. Path to save the processed pumping data as a shapefile.
   :param skip_process: bool. If True, skips the processing workflow and does nothing. Default is False.

   :return: None.
   """
    if not skip_process:
        print('Processing pumping data for Colorado...')

        makedirs([os.path.dirname(output_pump_shp)])

        all_files = glob(os.path.join(raw_data_dir, '*.csv'))

        df_merged = pd.DataFrame()  # empty dataframe to store individual wells' data

        for i in all_files:
            try:
                df = pd.read_csv(i)

            except pd.errors.EmptyDataError:
                # Skip completely empty files
                continue

            if df.empty:
                continue

            else:
                df = df[['wdid', 'wcIdentifier', 'dataMeasDate', 'dataValue']]
                df = df.rename(columns={'wdid': 'WDID', 'dataMeasDate': 'Year', 'dataValue': 'AF_pumped'})

                df_merged = pd.concat([df_merged, df], axis=0)

        # converting WDID to str
        # some Well ID are 6 digits. Must need to make them 7 digits to retrieve the data correctly
        df_merged['WDID'] = df_merged['WDID'].astype(str).str.zfill(7)

        # adding lon - lat to aggregated shapefile
        well_id_gdf = gpd.read_file(well_ID_shp)

        compiled_gdf = df_merged.merge(well_id_gdf, how='left', on='WDID')

        # I found some very high pumping values, e.g., in a well (0207026) in south platte
        # has a value of 49153 AF. Dropping them.
        compiled_gdf = compiled_gdf[compiled_gdf['AF_pumped'] <= 5000]

        # converting to shapefile
        compiled_gdf = gpd.GeoDataFrame(compiled_gdf, geometry='geometry', crs='EPSG:4269')
        compiled_gdf.to_file(output_pump_shp)

        # saving as csv
        compiled_gdf.to_csv(os.path.join(os.path.dirname(output_pump_shp), 'pumping_CO_v0.csv'), index=False)

    else:
        pass


def process_UT_pumping_data(raw_csv, output_pump_shp, skip_process=False, **kwargs):
    """
   Preliminary processes Utah groundwater pumping data.

   This function reads raw CSV data containing Utah groundwater pumping information and
   filters the data to include columns of interest.

   ****** Note that - This filtered shapefile is not final. has to go through final hand-processing and adding Soheil's
   OpenET-disaggregated data before final datasets.******

   :param raw_csv: str. Path to the raw groundwater pumping data CSV file.
   :param output_pump_shp:str. Path to save the processed pumping data as a shapefile.
   :param skip_process: bool. If True, skips the processing workflow and does nothing. Default is False.
   :param kwargs: dict (optional). Can take dictionaries
                                        - selected_columns_from_csv=['Lat NAD83', 'Lon NAD83', 'System Name',
                                                                     'Source Name', 'Source Status', 'Source Type',
                                                                     'Diversion Type', 'Use Type',
                                                                     'Method of Measurement', 'Year', 'Total'],
                                        - filter_conditions={'Source Type': 'Well',
                                                             'Diversion Type': 'Withdrawal',
                                                             'Use Type': ['Agricultural', 'irrigation']}
   :return: None.
   """
    if not skip_process:
        print("preliminary processing pumping data for Utah...\n",
              "Note that - This filtered shapefile is not final (only preliminary \n"
              "filter for 'agricultural' & 'irrigation' user group.",
              "It has to go through final hand-processing and adding Soheil's",
              "OpenET-disaggregated data before the final dataset.")

        makedirs([os.path.dirname(output_pump_shp)])

        pumping_df = pd.read_csv(raw_csv)

        # selecting columns of interest
        columns_of_interest = kwargs.get('selected_columns_from_csv',
                                         ['Lat NAD83', 'Lon NAD83', 'System Name', 'Source Name',
                                          'Source Status', 'Source Type', 'Diversion Type',
                                          'Use Type', 'Method of Measurement', 'Year', 'Total'])
        pumping_df = pumping_df[columns_of_interest]

        # applying filter conditions
        filter_conditions = kwargs.get('filter_conditions', {
            'Source Type': 'Well',
            'Diversion Type': 'Withdrawal',
            'Use Type': ['Agricultural', 'irrigation']
        })

        pumping_df = pumping_df[(pumping_df['Source Type'] == filter_conditions['Source Type']) &
                                (pumping_df['Diversion Type'] == filter_conditions['Diversion Type']) &
                                (pumping_df['Use Type'].isin(filter_conditions['Use Type']))]

        # renaming columns
        pumping_df = pumping_df.rename(columns={'Lat NAD83': 'Lat', 'Lon NAD83': 'Lon',
                                                'System Name': 'System_N', 'Source Name': 'Source_N',
                                                'Source Status': 'Source_S', 'Source Type': 'Source_T',
                                                'Diversion Type': 'Diversion_T', 'Method of Measurement': 'Measured',
                                                'Total': 'AF_pumped'})

        # saving pumping shapefile
        pumping_gdf = gpd.GeoDataFrame(pumping_df,
                                       geometry=gpd.points_from_xy(pumping_df['Lon'],
                                                                   pumping_df['Lat']))

        pumping_gdf = pumping_gdf.set_crs('EPSG:4269')
        pumping_gdf.to_file(output_pump_shp)

    else:
        pass


def pumping_pts_to_raster_v1(state_code, years, pumping_pts_shp, pumping_attr_AF,
                             year_attr, output_dir,
                             ref_raster=WestUS_raster, resolution=model_res,
                             ET_dir=None, Peff_dir=None,
                             surface_irrig_dir=None, low_fraction=None, high_fraction=None,
                             skip_outlier_removal=False, skip_processing=False):
    """
    Convert point scale (shapefile) groundwater pumping estimates to rasters in AF and mm.
    For individual pixels (2km) sums up all the pumping values inside it.

    :param state_code: State code, such as 'AZ', 'CO', 'KS'.
    :param years: List of years_list to process data.
    :param pumping_pts_shp: Filepath of point shapefile with annual pumping estimates.
    :param pumping_attr_AF: Attribute in the point shapefile with pumping in AF values.
    :param year_attr: Attribute in the point shapefile with year.
    :param output_dir: Filepath of main output dir. Intermediate directories named 'pumping_AF_raster' and
                        'pumping_mm_raster' will be created automatically.
    :param ref_raster: Filepath of Western US reference raster.
    :param resolution: model resolution.
    :param ET_dir: Directory containing raster files of growing season iET data.
                   Default set to None.
    :param Peff_dir: Directory containing raster files of growing season Peff data.
                     Default set to None.
    :param surface_irrig_dir: Directory containing raster files of surface water irrigation data.
                              Default set to None.
                              Note that-this data was distributed into pixels from USGS HUC12 dataset and was used
                                         as a proxy for surface water irrigation in our effective precipitation paper.
    :param low_fraction: The minimum threshold for `(total_water / Irr_cropET)` ratio.
    :param high_fraction: The maximum threshold for `(total_water / Irr_cropET)` ratio.
    :param skip_outlier_removal: Set to True to skip filtering out pixels with low pumping values.
                                 Default set to False.
    :param skip_processing: Set to True to skip this process.

    :return: Raster directories' path with AF and mm pumping.
    """
    if not skip_processing:
        print(f'\nCreating pumping rasters for {state_code}...\n')

        # creating subdirectories
        annual_pump_shp_dir = os.path.join(output_dir, 'annual_pumping_shp')
        pumping_AF_dir = os.path.join(output_dir, 'pumping_AF')
        pumping_mm_dir = os.path.join(output_dir, 'pumping_mm')

        makedirs([annual_pump_shp_dir, pumping_AF_dir, pumping_mm_dir])

        # loading pumping shapefile
        pumping_gdf = gpd.read_file(pumping_pts_shp)

        # looping by year and processing pumping shapefile to raster
        for year in years:
            print(f'Converting pumping AF shapefile to mm raster for {year}...')

            # filtering pumping dataset by year (unit Acre-ft) and saving it
            gdf_filtered = pumping_gdf[pumping_gdf[year_attr] == year]
            annual_filtered_shp = os.path.join(annual_pump_shp_dir, f'pumping_{year}.shp')
            gdf_filtered.to_file(annual_filtered_shp)

            # converting yearly pumping point dataset into yearly AF raster.
            # all pumping inside a 2 km pixel will be summed
            # the generated raster is for the whole Western US with 0 values outside the basin
            output_AF_raster = f'pumping_AF_{year}.tif'
            pumping_AF_raster = shapefile_to_raster(input_shape=annual_filtered_shp,
                                                    output_dir=pumping_AF_dir,
                                                    raster_name=output_AF_raster, use_attr=True,
                                                    attribute=pumping_attr_AF, add=True,
                                                    ref_raster=ref_raster, resolution=resolution)

            # converting pumping unit from AF to mm
            # no pumping values are 0 here
            pumping_AF_arr, file = read_raster_arr_object(pumping_AF_raster)

            # area of a 2 km pixel
            # 2199 meter is the pixel size in latitude direction. 1746 is the pixel size in longitude direction.
            # For longitudinal dimension calculation, an average latitude of 37.42 deg was considered for the Western US.
            # Distance of 1 deg Longitude at the equator is = 111,320 m
            # 0.01976293625031605786 deg * 111320 * cos(37.42 * pi / 180) = 1746.53 m
            # 0.01976293625031605786 deg * 111320 = 2199.59 m
            area_mm2_single_pixel = (2199.59 * 1000) * (1746.53 * 1000)  # unit in mm2

            pumping_mm_arr = np.where(~np.isnan(pumping_AF_arr), pumping_AF_arr * 1233481837548 /
                                      area_mm2_single_pixel, -9999)

            if not skip_outlier_removal:
                pumping_mm_raster = os.path.join(pumping_mm_dir, 'original', f'pumping_mm_{year}.tif')
            else:
                pumping_mm_raster = os.path.join(pumping_mm_dir, f'pumping_mm_{year}.tif')

            write_array_to_raster(pumping_mm_arr, file, file.transform, pumping_mm_raster)

            # remove pumping values that are too low or to high
            if not skip_outlier_removal:
                if ET_dir is None or Peff_dir is None:
                    raise ValueError(
                        "To perform outlier removal, 'ET_dir' and 'Peff_dir'"
                        "must be provided. 'surface_irrig_dir' might also be needed for some regions. "
                        "Set 'skip_outlier_removal=True' to bypass this step.")

                pumping_mm_arr = \
                    filter_out_low_high_pumping_values_v1(year, pumping_mm_arr, ET_dir,
                                                          Peff_dir, low_fraction, high_fraction,
                                                          surface_irrig_dir,
                                                          skip_processing=skip_outlier_removal)

                # filling nan positions (0 values) with -9999
                pumping_mm_arr[pumping_mm_arr == 0] = -9999

                pumping_mm_raster = os.path.join(pumping_mm_dir, f'pumping_mm_{year}.tif')
                write_array_to_raster(pumping_mm_arr, file, file.transform, pumping_mm_raster)
    else:
        pass


def filter_out_low_high_pumping_values_v1(year, pumping_arr, ET_dir,
                                          Peff_dir, low_fraction, high_fraction,
                                          surface_irrig_dir=None, skip_processing=False):
    """
    Filters out high and low in-situ pumping values.

    This function processes annual pumping data by filtering out values that do not meet specific
    conditions based on the ratio of total water to evapotranspiration (ET) during the
    growing season. Invalid pumping values are set to zero in the returned array.

    **Filter Conditions**: (modified after Ott et al. (2024))
    - total water = pumping + peff + surface irrigation (if needed for specific region)
    - total water / ET >= low fraction (depending on regions the low fraction can be 0.75 - 0.85) and Peff not Nan
    - total water / ET <= high fraction (can eb around 1.5-1.6) and Peff not Nan

    :param year: The year for which the data is being processed.
    :param pumping_arr: The array of pumping values to filter.
    :param ET_dir: Directory containing raster files of growing season ET data.
    :param Peff_dir: Directory containing raster files of growing season Peff data.
    :param low_fraction: The minimum threshold for `(total_water / ET)` ratio.
    :param high_fraction: The maximum threshold for `(total_water / ET)` ratio.
    :param surface_irrig_dir: Directory containing raster files of surface water distribution data.
                              It might be needed in areas with high surface water use (like Arizona).
                              Default set to None - to avoid its use.
    :param skip_processing: If True, skip processing. Default is False.

    :return: Filtered pumping array with invalid values set to zero.
    """
    if not skip_processing:
        # reading growing season cropET data for irrigated croplands
        ET_data = glob(os.path.join(ET_dir, f'*{year}*.tif'))[0]
        ET_arr = read_raster_arr_object(ET_data, get_file=False)

        # reading growing season Peff data for irrigated croplands
        peff = glob(os.path.join(Peff_dir, f'*{year}*.tif'))[0]
        peff_arr = read_raster_arr_object(peff, get_file=False)

        # calculating total water
        if surface_irrig_dir is None:
            total_water = peff_arr + pumping_arr

        else:
            # Here we add surface water consumptive use to the filtering process.
            # Surface water irrigation (consumptive use) data is available up to 2020.
            # We can only use this filter from 2000 to 2020, for years after 2020, we skip using this filter

            if year < 2020:
                # reading annual surface water irrigation data (distributed using USGS HUC12 dataset)
                surf_irr = glob(os.path.join(surface_irrig_dir, f'*{year}*.tif'))[0]
                surf_irr_arr = read_raster_arr_object(surf_irr, get_file=False)

                total_water = peff_arr + pumping_arr + surf_irr_arr

            else:  # years after 2020
                total_water = peff_arr + pumping_arr

        # fraction of total_water / ET_arr (modified after Ott et al. (2024))
        ET_arr = np.where(ET_arr > 1e-6, ET_arr, np.nan)  # 1e-6 used as threshold to avoid division by very small value
        total_water = np.where(total_water > 0, total_water, np.nan)

        water_frac = np.where(~np.isnan(total_water) & ~np.isnan(ET_arr), total_water / ET_arr, -9999)

        # applying filter to identify valid pumping values
        mask = (water_frac >= low_fraction) & (water_frac <= high_fraction) & (ET_arr != -9999)
        pumping_arr = pumping_arr * mask  # invalid values are set to 0

        return pumping_arr

    else:
        pass


def pumping_pts_to_raster_v2_AZ(years, pumping_pts_shp, pumping_attr_AF,
                                year_attr, output_dir, lower_outlier_range, upper_outlier_range,
                                ref_raster=WestUS_raster, resolution=model_res,
                                skip_outlier_removal=False,
                                skip_store_unfiltered_raster=False,
                                unfiltered_pumping_shp=None,
                                skip_processing=False):
    """

                ************************* Only for Arizona *************************

    Convert point scale (shapefile) groundwater pumping estimates to rasters in AF and mm.
    For individual pixels (2km) sums up all the pumping values inside it. This is an alternate function of
    pumping_pts_to_raster_v1() and uses hard-coded lower and upper bounds to remove outliers in pumping data.

    :param years: List of years_list to process data.
    :param pumping_pts_shp: Filepath of point shapefile with annual pumping estimates.
    :param pumping_attr_AF: Attribute in the point shapefile with pumping in AF values.
    :param year_attr: Attribute in the point shapefile with year.
    :param output_dir: Filepath of main output dir. Intermediate directories named 'pumping_AF_raster' and
                        'pumping_mm_raster' will be created automatically.
    :param lower_outlier_range: Lower value for outlier removal.
    :param upper_outlier_range: Upper value for outlier removal.
    :param skip_processing: Set to True to skip this process.
    :param ref_raster: Filepath of Western US reference raster.
    :param resolution: model resolution.
    :param skip_outlier_removal: Set to True to skip filtering out pixels with low pumping values.
                                 Default set to False.
    :param skip_store_unfiltered_raster: Set to True if want to store unfiltered pumping raster/
    :param unfiltered_pumping_shp: Unfiltered pumping shapefile.

    :param skip_processing: Set to True to skip this process.

    :return: Raster directories' path with AF and mm pumping.
    """
    if not skip_processing:
        print(f'\nCreating pumping rasters for AZ...')

        # creating subdirectories
        annual_pump_shp_dir = os.path.join(output_dir, 'annual_pumping_shp')
        pumping_AF_dir = os.path.join(output_dir, 'pumping_AF')
        pumping_mm_dir = os.path.join(output_dir, 'pumping_mm')

        makedirs([annual_pump_shp_dir, pumping_AF_dir, pumping_mm_dir])

        # loading pumping shapefile
        pumping_gdf = gpd.read_file(pumping_pts_shp)

        # looping by year and processing pumping shapefile to raster
        for year in years:
            print(f'Converting pumping AF shapefile to mm raster for {year}...')

            # filtering pumping dataset by year (unit Acre-ft) and saving it
            gdf_filtered = pumping_gdf[pumping_gdf[year_attr] == year]
            annual_filtered_shp = os.path.join(annual_pump_shp_dir, f'pumping_{year}.shp')
            gdf_filtered.to_file(annual_filtered_shp)

            # converting yearly pumping point dataset into yearly AF raster.
            # all pumping inside a 2 km pixel will be summed
            # the generated raster is for the whole Western US with 0 values outside the basin
            output_AF_raster = f'pumping_AF_{year}.tif'
            pumping_AF_raster = shapefile_to_raster(input_shape=annual_filtered_shp,
                                                    output_dir=pumping_AF_dir,
                                                    raster_name=output_AF_raster, use_attr=True,
                                                    attribute=pumping_attr_AF, add=True,
                                                    ref_raster=ref_raster, resolution=resolution)

            # converting pumping unit from AF to mm
            # no pumping values are 0 here
            pumping_AF_arr, file = read_raster_arr_object(pumping_AF_raster)

            # area of a 2 km pixel
            # 2199 meter is the pixel size in latitude direction. 1746 is the pixel size in longitude direction.
            # For longitudinal dimension calculation, an average latitude of 37.42 deg was considered for the Western US.
            # Distance of 1 deg Longitude at the equator is = 111,320 m
            # 0.01976293625031605786 deg * 111320 * cos(37.42 * pi / 180) = 1746.53 m
            # 0.01976293625031605786 deg * 111320 = 2199.59 m
            area_mm2_single_pixel = (2199.59 * 1000) * (1746.53 * 1000)  # unit in mm2

            pumping_mm_arr = np.where(~np.isnan(pumping_AF_arr), pumping_AF_arr * 1233481837548 /
                                      area_mm2_single_pixel, -9999)

            # remove pumping values that are too low or to high
            if not skip_outlier_removal:
                pumping_mm_arr = np.where(
                    (pumping_mm_arr >= lower_outlier_range) & (pumping_mm_arr <= upper_outlier_range),
                    pumping_mm_arr, -9999)

                pumping_mm_raster = os.path.join(pumping_mm_dir, f'pumping_mm_{year}.tif')
                write_array_to_raster(pumping_mm_arr, file, file.transform, pumping_mm_raster)

            if not skip_store_unfiltered_raster:
                # creating subdirectories
                annual_pump_shp_dir = os.path.join(output_dir, 'annual_pumping_shp_Original')
                pumping_AF_dir = os.path.join(output_dir, 'pumping_AF_Original')
                pumping_mm_dir = os.path.join(output_dir, 'pumping_mm/Original')

                makedirs([annual_pump_shp_dir, pumping_AF_dir, pumping_mm_dir])

                # loading pumping shapefile
                if unfiltered_pumping_shp is None:
                    raise ValueError("'unfiltered_pumping_data' can not be 'None'")

                pumping_gdf = gpd.read_file(unfiltered_pumping_shp)

                # looping by year and processing pumping shapefile to raster
                for year in years:
                    # filtering pumping dataset by year (unit Acre-ft) and saving it
                    gdf_filtered = pumping_gdf[pumping_gdf[year_attr] == year]
                    annual_filtered_shp = os.path.join(annual_pump_shp_dir, f'pumping_{year}.shp')
                    gdf_filtered.to_file(annual_filtered_shp)

                    # converting yearly pumping point dataset into yearly AF raster.
                    # all pumping inside a 2 km pixel will be summed
                    # the generated raster is for the whole Western US with 0 values outside the basin
                    output_AF_raster = f'pumping_AF_{year}.tif'
                    pumping_AF_raster = shapefile_to_raster(input_shape=annual_filtered_shp,
                                                            output_dir=pumping_AF_dir,
                                                            raster_name=output_AF_raster, use_attr=True,
                                                            attribute=pumping_attr_AF, add=True,
                                                            ref_raster=ref_raster, resolution=resolution)

                    # converting pumping unit from AF to mm
                    # no pumping values are 0 here
                    pumping_AF_arr, file = read_raster_arr_object(pumping_AF_raster)

                    # area of a 2 km pixel
                    # 2199 meter is the pixel size in latitude direction. 1746 is the pixel size in longitude direction.
                    # For longitudinal dimension calculation, an average latitude of 37.42 deg was considered for the Western US.
                    # Distance of 1 deg Longitude at the equator is = 111,320 m
                    # 0.01976293625031605786 deg * 111320 * cos(37.42 * pi / 180) = 1746.53 m
                    # 0.01976293625031605786 deg * 111320 = 2199.59 m
                    area_mm2_single_pixel = (2199.59 * 1000) * (1746.53 * 1000)  # unit in mm2

                    pumping_mm_arr = np.where(~np.isnan(pumping_AF_arr), pumping_AF_arr * 1233481837548 /
                                              area_mm2_single_pixel, -9999)

                    pumping_mm_raster = os.path.join(pumping_mm_dir, f'pumping_mm_{year}.tif')
                    write_array_to_raster(pumping_mm_arr, file, file.transform, pumping_mm_raster)
    else:
        pass


def create_pod_pumping_DV(
        pumping_csv: str,
        pod_shp: str,
        output_dir: str,
        year_col: str = 'year',
        pumping_col: str = 'pumping_AF',
        pod_col_csv: str = 'all_app',
        pod_col_shp: str = 'app',
        skip_processing=False):
    """
    Create individual POD pumping rows for each year for Diamond Valley in Nevada.

    Args:
        pumping_csv (str): Path to the CSV file containing pumping data.
        pod_shp (str): Path to the shapefile containing POD data.
        output_dir (str): Directory to save the output files.
        year_col (str): Column name for the year in the pumping data.
        pumping_col (str): Column name for the pumping values in the pumping data csv.
        pod_col_csv (str): Column name for the POD identifier in the pumping data csv.
        pod_col_shp (str): Column name for the POD identifier in the POD shapefile.
        skip_processing: Set to True to skip this process.

    Returns:
        None.
    """
    if not skip_processing:
        makedirs([output_dir])

        pumping_df = pd.read_csv(pumping_csv)
        pod_pumping_df = pd.DataFrame()
        for _, row in pumping_df.iterrows():
            pod_list = row[pod_col_csv].split('_')

            # distribute pumping volume equally among multiple PODs
            pumping_vol_per_pod = row[pumping_col] / len(pod_list)

            for pod in pod_list:
                temp_df = pd.DataFrame({
                    pod_col_shp: [str(pod)],
                    year_col: [row[year_col]],
                    'AF_pumped': [pumping_vol_per_pod]
                })

                pod_pumping_df = pd.concat([pod_pumping_df, temp_df], ignore_index=True)

        pod_gdf = gpd.read_file(pod_shp).to_crs(epsg=4269)
        pod_gdf[pod_col_shp] = pod_gdf[pod_col_shp].astype(str)
        pod_pumping_gdf = gpd.GeoDataFrame(pod_pumping_df.merge(pod_gdf[[pod_col_shp, 'geometry']], on=pod_col_shp))
        pod_pumping_gdf.to_csv(f'{output_dir}/pod_pumping_data.csv', index=False)

        output_shp = os.path.join(output_dir, 'pod_pumping.shp')
        pod_pumping_gdf.to_file(output_shp)

        return output_shp

    else:
        os.path.join(output_dir, 'pod_pumping.shp')


def combine_pumping_rasters(years, years_no_data_dict,
                            KS_dir, CO_dir, AZ_dir, NV_dir,
                            output_dir,
                            basins_exclude_from_training=None,
                            irr_cropland_dir=None,
                            ref_raster=WestUS_raster,
                            skip_processing=False):
    """
    Combines multiple pumping rasters into a final pumping raster by replacing zero values
    in a reference raster with corresponding pumping data. Sets remaining zero values to -9999.


    :param years: List/Tuple of years to process data for.
    :param years_no_data_dict: A dictionary with information of what years of pumping data not available/processed for
                           KS, CO, and AZ ---
                           {'KS': [],
                           'CO': list(range(2000, 2011)),
                           'AZ': [],
                           'NV': list(range(2000, 2018)) + [2023]}
    :param KS_dir: Annual pumping data directory (rasters) for Kansas.
    :param CO_dir: Annual pumping data directory (rasters) for Colorado.
    :param AZ_dir: Annual pumping data directory (rasters) for Arizona.
    :param NV_dir: Annual pumping data directory (rasters) for Nevada.
    :param irr_cropland_dir: Annual irrigated cropland directory to apply irrigated cropland filter.
                             Default set to None to skip filtering by irrigated cropland raster.
    :param output_dir: Path to save the combined pumping rasters.
    :param basins_exclude_from_training: List of basin code to exclude from training. Default set to None to use all basins.
                                     Must be from :
                                     ['gmd4', 'gmd3', 'rpb', 'spb', 'ar', 'slv', 'hqr',
                                     'doug', 'phx', 'pnl', 'scruz', 'dv']
    :param ref_raster: Filepath of reference raster.
    :param skip_processing: Set to True to skip this process. Default set to False.

    :return: None.
    """
    if not skip_processing:
        print("\nCombining states' pumping data to create Western US-wide pumping raster... \n"
              "This will be used as a training data for the model...")

        # creating output directory
        makedirs([output_dir])

        # processing the output annual pumping raster for each year
        for year in years:
            # reading pumping data for each year for given states
            if year not in years_no_data_dict['KS']:
                KS_data = glob(os.path.join(KS_dir, f'*{year}*.tif'))[0]
                KS_arr = read_raster_arr_object(KS_data, get_file=False)
            else:
                KS_arr = None

            if year not in years_no_data_dict['CO']:
                CO_data = glob(os.path.join(CO_dir, f'*{year}*.tif'))[0]
                CO_arr = read_raster_arr_object(CO_data, get_file=False)
            else:
                CO_arr = None

            if year not in years_no_data_dict['AZ']:
                AZ_data = glob(os.path.join(AZ_dir, f'*{year}*.tif'))[0]
                AZ_arr = read_raster_arr_object(AZ_data, get_file=False)
            else:
                AZ_arr = None

            if year not in years_no_data_dict['NV']:
                NV_data = glob(os.path.join(NV_dir, f'*{year}*.tif'))[0]
                NV_arr = read_raster_arr_object(NV_data, get_file=False)
            else:
                NV_arr = None

            # initializing a zero array to store pumping data
            ref_arr, ref_file = read_raster_arr_object(ref_raster)
            pump_arr = np.zeros_like(ref_arr)

            # replacing zero value of reference raster with pumping value
            if KS_arr is not None:
                pump_arr = np.where(KS_arr > 0, KS_arr, pump_arr)
            if CO_arr is not None:
                pump_arr = np.where(CO_arr > 0, CO_arr, pump_arr)
            if AZ_arr is not None:
                pump_arr = np.where(AZ_arr > 0, AZ_arr, pump_arr)
            if NV_arr is not None:
                pump_arr = np.where(NV_arr > 0, NV_arr, pump_arr)

            # setting nan (-9999) to pixels where there are no pumping values.
            # some of them might get modified to zero if 'add_zero_pumping_in_training' is used
            pump_arr = np.where(pump_arr > 0, pump_arr, -9999)

            if irr_cropland_dir is not None:
                # irrigated cropland data for the year
                irr_cropland = glob(os.path.join(irr_cropland_dir, f'*{year}*.tif'))[0]
                irr_arr = read_raster_arr_object(irr_cropland, get_file=False)

                # Setting nan value for pixels that are not irrigated per irrigated cropland data
                pump_arr = np.where(np.isnan(irr_arr), -9999, pump_arr)

            # saving interim pumping array as raster
            # in case, basins_exclude_from_training = None, this is the final output raster
            interim_raster = os.path.join(output_dir, f'pumping_{year}_interim.tif')
            write_array_to_raster(pump_arr, ref_file, ref_file.transform, interim_raster)

            # this is a provision to exclude particular basins out of the training data
            basin_shp_dict = {
                'gmd4': '../../Data_main/shapefiles/Basins_of_interest/GMD4.shp',
                'gmd3': '../../Data_main/shapefiles/Basins_of_interest/GMD3.shp',
                'rpb': '../../Data_main/shapefiles/Basins_of_interest/Republican_Basin.shp',
                'spb': '../../Data_main/shapefiles/Basins_of_interest/South_Platte_Basin.shp',
                'ar': '../../Data_main/shapefiles/Basins_of_interest/Arkansas_Basin.shp',
                'slv': '../../Data_main/shapefiles/Basins_of_interest/Rio_Grande_Basin.shp',
                'hqr': '../../Data_main/shapefiles/Basins_of_interest/Harquahala_INA.shp',
                'doug': '../../Data_main/shapefiles/Basins_of_interest/Douglas_AMA.shp',
                'phx': '../../Data_main/shapefiles/Basins_of_interest/Phoenix_AMA.shp',
                'pnl': '../../Data_main/shapefiles/Basins_of_interest/Pinal_AMA.shp',
                'tucs': '../../Data_main/shapefiles/Basins_of_interest/Tucson_AMA.shp',
                'scruz': '../../Data_main/shapefiles/Basins_of_interest/SantaCruz_AMA.shp',
                'dv': '../../Data_main/shapefiles/Basins_of_interest/Diamond_Valley_Basin.shp'
            }

            if basins_exclude_from_training is not None:
                for basin in basins_exclude_from_training:

                    if basin not in basin_shp_dict.keys():
                        print(f'Basin name must be from this list - \n'
                              f'{basin_shp_dict.keys()}')

                    mask_raster_by_shape(input_raster=interim_raster,
                                         input_shape=basin_shp_dict[basin],
                                         output_dir=output_dir,
                                         raster_name=f'pumping_{year}.tif',
                                         crop=False, invert=True, filled=False,
                                         nodata=no_data_value)

                    # remove the 1st interim raster
                    os.remove(interim_raster)

                    # making the masked raster as new interim raster
                    masked_raster = os.path.join(output_dir, f'pumping_{year}.tif')
                    interim_raster = os.path.join(output_dir, f'pumping_{year}_interim.tif')
                    os.rename(masked_raster, interim_raster)

                # the final rasters name became interim raster, naming it back
                os.rename(interim_raster, os.path.join(output_dir, f'pumping_{year}.tif'))

    else:
        pass


def create_consumptive_gw_use_raster(Irr_eff_dir, combined_pumping_dir,
                                     output_dir, skip_processing=False):
    """
    Creates consumptive groundwater use raster from compiled pumping data and
    USGS HUC12 irrigation efficiency data.

    :param Irr_eff_dir: Rasterized USGS HUC12 irrigation efficiency raster.
    :param combined_pumping_dir: Combiner pumping data.
    :param output_dir: Path of output directory.
    :param skip_processing: Set True to skip this process.

    :return: None.
    """
    if not skip_processing:
        makedirs([output_dir])

        for year in range(2000, 2024):

            Irr_eff_data = glob(os.path.join(Irr_eff_dir, f'*{year}*.tif'))[0]
            Irr_eff, file = read_raster_arr_object(Irr_eff_data)

            arr_shape = Irr_eff.shape
            Irr_eff = Irr_eff.flatten()

            pumping_data = glob(os.path.join(combined_pumping_dir, f'*{year}.tif'))[0]
            pumping = read_raster_arr_object(pumping_data, get_file=False).flatten()

            # consumptive groundwater use
            consmp_gw = np.where((~np.isnan(Irr_eff)) & (~np.isnan(pumping)), pumping * Irr_eff, -9999)
            consmp_gw = consmp_gw.reshape(arr_shape)

            write_array_to_raster(consmp_gw, file, file.transform, os.path.join(output_dir, f'pumping_{year}.tif'))

    else:
        pass


def main(skip_process_AZ_pumping,
         skip_irrig_zone_filter_AZ,
         skip_make_AZ_pumping_raster,
         skip_process_KS_pumping,
         skip_make_KS_pumping_raster,
         skip_process_CO_pumping,
         skip_make_CO_pumping_raster,
         skip_NV_dist_pumping_to_pod,
         skip_make_NV_pumping_raster,
         skip_combine_pumping_rasters):
    ####################################################################################################################
    # # Arizona
    ####################################################################################################################
    # compile from raw data
    process_AZ_pumping_csv(
        raw_csv_dir='../../Data_main/Pumping/Arizona/raw',
        well_reg_shp='../../Data_main/pumping/Arizona/Well_Registry/WellRegistry.shp',
        well_reg_dict='../../Data_main/pumping/Arizona/Well_coords_dict.pkl',
        output_shp='../../Data_main/pumping/Arizona/Final/pumping_AZ_v0.shp',
        skip_process_Well_registry_file=True,
        skip_process=skip_process_AZ_pumping,
        selected_columns_from_csv=['Well Id', 'AMA INA', 'YEAR', 'Right Type',
                                   'AF Pumped', 'Water Type'],
        filter_conditions={
            'Right Type': ['IRRIGATION USE', 'IRRIGATION DISTRICT (GW ONLY)',
                           'IRRIGATION DISTRICT (SW & GW)', 'BEST MANAGEMENT PRACTICES FARM UNIT'],
            'Water Type': 'GROUNDWATER'
        }
    )

    # clip for major irrigated zones
    if not skip_irrig_zone_filter_AZ:
        clip_vector(input_shapefile='../../Data_main/pumping/Arizona/Final/pumping_AZ_v0.shp',
                    mask_shapefile='../../Data_main/pumping/Arizona/major_irrig_zones/major_irrig_zones_AZ.shp',
                    output_shapefile='../../Data_main/pumping/Arizona/Final/pumping_AZ_v1.shp',
                    change_crs=None, create_zero_buffer=False)

    # make 2 km rasters
    # up to 2023, as Peff data (used for filtering) is available up to 2023
    pumping_pts_to_raster_v2_AZ(years=list(range(2000, 2024)),
                                pumping_pts_shp='../../Data_main/pumping/Arizona/Final/pumping_AZ_v1.shp',  # using 'v1'
                                pumping_attr_AF='AF_pumped',
                                year_attr='Year',
                                output_dir='../../Data_main/pumping/rasters/Arizona',
                                skip_outlier_removal=False,
                                lower_outlier_range=0,  # not setting a low threshold
                                upper_outlier_range=1500,  # based on ET and netGW_Irr analysis
                                ref_raster=WestUS_raster, resolution=model_res,
                                skip_processing=skip_make_AZ_pumping_raster,
                                skip_store_unfiltered_raster=False,
                                unfiltered_pumping_shp='../../Data_main/pumping/Arizona/Final/pumping_AZ_v0.shp')  # using 'v0'

    ####################################################################################################################
    # # Kansas
    ####################################################################################################################

    # # This pumping data didn't have to be filtered due to high quality
    process_KS_pumping_csv(raw_csv='../../Data_main/pumping/Kansas/csv/pumping_KS.csv',
                           output_pump_csv='../../Data_main/pumping/Kansas/Final/pumping_KS.csv',
                           output_pump_shp='../../Data_main/pumping/Kansas/Final/pumping_KS.shp',
                           output_acres_csv='../../Data_main/pumping/Kansas/pumping_acres_KS.csv',
                           skip_process=skip_process_KS_pumping)

    # make 2 km rasters
    # up to 2023 as Peff data (used for filtering) is available up to 2023
    pumping_pts_to_raster_v1(state_code='KS', years=list(range(2000, 2024)),
                             pumping_pts_shp='../../Data_main/pumping/Kansas/Final/pumping_Ks.shp',
                             pumping_attr_AF='AF_pumped',
                             year_attr='Year',
                             output_dir='../../Data_main/pumping/rasters/Kansas',
                             ref_raster=WestUS_raster, resolution=model_res,
                             skip_processing=skip_make_KS_pumping_raster,
                             ET_dir='../../Data_main/rasters/OpenET_ensemble/WestUS_growing_season',
                             Peff_dir='../../Data_main/rasters/Effective_precip_prediction_WestUS/v19_grow_season_scaled',
                             surface_irrig_dir=None,
                             low_fraction=0.7,
                             high_fraction=1.5,
                             skip_outlier_removal=False)  # implementing low-high pumping value removal in KS

    ####################################################################################################################
    # # Colorado
    ####################################################################################################################
    # compile from raw data (v0)
    # already processed for major_irrig_zones
    process_CO_pumping_data(raw_data_dir='../../Data_main/pumping/Colorado/raw/pumping/all_data',
                            well_ID_shp='../../Data_main/pumping/Colorado/raw/pumping/Well_ID/Well_ID_CO_v1.shp',
                            # this is clipped for 'major_irrig_zones'
                            output_pump_shp='../../Data_main/pumping/Colorado/Final/pumping_CO_v0.shp',
                            skip_process=skip_process_CO_pumping)

    # make 2 km rasters
    # from 2011 up to 2023, as Peff data (used for filtering) is available up to 2023
    # Pumping data for CO before 2010 isn't of good quality
    pumping_pts_to_raster_v1(state_code='CO', years=list(range(2011, 2024)),
                             pumping_pts_shp='../../Data_main/pumping/Colorado/Final/pumping_CO_v0.shp',
                             pumping_attr_AF='AF_pumped',
                             year_attr='Year',
                             output_dir='../../Data_main/pumping/rasters/Colorado',
                             ref_raster=WestUS_raster, resolution=model_res,
                             skip_processing=skip_make_CO_pumping_raster,
                             ET_dir='../../Data_main/rasters/OpenET_ensemble/WestUS_growing_season',
                             Peff_dir='../../Data_main/rasters/Effective_precip_prediction_WestUS/v19_grow_season_scaled',
                             surface_irrig_dir='../../Data_main/rasters/SW_irrigation',
                             low_fraction=0.7,
                             high_fraction=1.5,
                             skip_outlier_removal=False)  # implementing low-high pumping value removal in CO

    ####################################################################################################################
    # # Nevada (Diamond Valley)
    ####################################################################################################################

    # connecting pod to pumping database for distributing
    pod_pumping_shp = create_pod_pumping_DV(
        pumping_csv='../../Data_main/pumping/Nevada/raw/Diamond Valley/joined_data/dv_joined_et_pumping_data_all.csv',
        pod_shp='../../Data_main/pumping/Nevada/raw/Diamond Valley/pod/dv_pod.shp',
        output_dir='../../Data_main/pumping/Nevada/Final',
        year_col='year',
        pumping_col='pumping_AF',
        pod_col_csv='all_app',
        pod_col_shp='app',
        skip_processing=skip_NV_dist_pumping_to_pod)

    # make 2 km rasters
    pumping_pts_to_raster_v1(state_code='NV', years=list(range(2018, 2023)),
                             pumping_pts_shp=pod_pumping_shp,
                             pumping_attr_AF='AF_pumped',
                             year_attr='year',
                             output_dir='../../Data_main/pumping/rasters/Nevada',
                             ref_raster=WestUS_raster, resolution=model_res,
                             skip_processing=skip_make_NV_pumping_raster,
                             ET_dir=None, Peff_dir=None, surface_irrig_dir=None,
                             skip_outlier_removal=True)  # implementing low-high pumping value removal in KS

    ####################################################################################################################
    # # Combine to pumping raster for training
    ####################################################################################################################
    # Combine states' pumping data into a combined pumping raster for Western US

    # combine the original pumping rasters where filtering by threshold wasn't applied.
    # useful for basin scale comparison with model outputs.
    combine_pumping_rasters(years=list(range(2000, 2024)),
                            years_no_data_dict={'KS': [],
                                                'CO': list(range(2000, 2011)),
                                                'AZ': [],
                                                'NV': list(range(2000, 2018)) + [2023]},
                            basins_exclude_from_training=None,
                            KS_dir='../../Data_main/pumping/rasters/Kansas/pumping_mm/Original',
                            CO_dir='../../Data_main/pumping/rasters/Colorado/pumping_mm/Original',
                            AZ_dir='../../Data_main/pumping/rasters/Arizona/pumping_mm/Original',
                            NV_dir='../../Data_main/pumping/rasters/Nevada/pumping_mm',
                            irr_cropland_dir=None,
                            output_dir='../../Data_main/pumping/rasters/WestUS_pumping//Original',
                            skip_processing=skip_combine_pumping_rasters)

    # combine the filtered pumping rasters where filtering through threshold was applied.
    # Either this or consumptive groundwater use data will be used for model training.
    combine_pumping_rasters(years=list(range(2000, 2024)),
                            years_no_data_dict={'KS': [],
                                                'CO': list(range(2000, 2011)),
                                                'AZ': [],
                                                'NV': list(range(2000, 2018)) + [2023]},
                            basins_exclude_from_training=['spb', 'ar',
                                                          'pnl', 'phx'],
                            KS_dir='../../Data_main/pumping/rasters/Kansas/pumping_mm',
                            CO_dir='../../Data_main/pumping/rasters/Colorado/pumping_mm',
                            AZ_dir='../../Data_main/pumping/rasters/Arizona/pumping_mm',
                            NV_dir='../../Data_main/pumping/rasters/Nevada/pumping_mm',
                            irr_cropland_dir='../../Data_main/rasters/Irrigated_cropland',
                            output_dir='../../Data_main/pumping/rasters/WestUS_pumping',
                            skip_processing=skip_combine_pumping_rasters)

    # create consumptive groundwater use raster
    # Either this or pumping data will be used for model training.
    create_consumptive_gw_use_raster(Irr_eff_dir='../../Data_main/rasters/HUC12_Irr_Eff',
                                     combined_pumping_dir='../../Data_main/pumping/rasters/WestUS_pumping',
                                     output_dir='../../Data_main/pumping/rasters/WestUS_consumptive_gw',
                                     skip_processing=False)


if __name__ == '__main__':
    main(
        skip_process_AZ_pumping=True,  #######
        skip_irrig_zone_filter_AZ=True,  #######
        skip_make_AZ_pumping_raster=True,  #######
        skip_process_KS_pumping=True,  #######
        skip_make_KS_pumping_raster=False,  #######
        skip_process_CO_pumping=True,  #######
        skip_make_CO_pumping_raster=False,  #######
        skip_NV_dist_pumping_to_pod=True,  #######
        skip_make_NV_pumping_raster=True,  #######
        skip_combine_pumping_rasters=False  #######
    )