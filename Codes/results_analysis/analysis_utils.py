# Author : Md Fahim Hasan
# PhD Candidate
# Colorado State university
# Fahim.Hasan@colostate.edu

import os
import numpy as np
import pandas as pd
from glob import glob
import geopandas as gpd

import sys
from os.path import dirname, abspath

sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.utils.system_ops import makedirs
from Codes.utils.vector_ops import clip_vector
from Codes.utils.raster_ops import read_raster_arr_object, \
    mask_raster_by_shape, make_lat_lon_array_from_raster

no_data_value = -9999
model_res = 0.01976293625031605786  # in deg, ~2 km
WestUS_shape = '../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_states.shp'
WestUS_raster = '../../Data_main/reference_rasters/Western_US_refraster_2km.tif'
GEE_merging_refraster_large_grids = '../../Data_main/reference_rasters/GEE_merging_refraster_larger_grids.tif'


def clip_pumping_for_basin(years, basin_shp, predicted_pumping_dir, output_dir,
                           actual_pumping_dir=None,
                           skip_processing=False):
    """
    Clip predicted (and optionally actual) pumping rasters to the basin boundary.

    :param years: list[int]
        List of years to process (e.g., [2015, 2016, 2017]).
    :param basin_shp: str
        Filepath to the basin shapefile used for clipping.
    :param predicted_pumping_dir: str
        Directory containing annual predicted pumping rasters.
    :param output_dir: str
        Output directory where clipped rasters will be saved under subfolders `predicted/` and `actual/`.
    :param actual_pumping_dir: str or None
        Directory containing annual actual/in-situ pumping rasters. If None, actual pumping will not be processed.
    :param skip_processing: bool
        If True, skips processing and returns the expected output paths directly.

    :return: tuple[str, str]
        Tuple of (clipped predicted pumping dir, clipped actual pumping dir).
    """
    if not skip_processing:
        basin_predicted_output_dir = os.path.join(output_dir, 'predicted')
        basin_actual_output_dir = os.path.join(output_dir, 'actual')

        makedirs([basin_predicted_output_dir, basin_actual_output_dir])

        for year in years:
            print(f'Clipping pumping prediction for {year}...')

            # predicted pumping raster
            predicted_pumping_raster = glob(os.path.join(predicted_pumping_dir, f'*{year}*.tif'))[0]

            mask_raster_by_shape(input_raster=predicted_pumping_raster,
                                 input_shape=basin_shp, crop=True,
                                 output_dir=basin_predicted_output_dir,
                                 raster_name=f'pumping_{year}.tif')

            if actual_pumping_dir is not None:
                # actual pumping raster
                actual_pumping_raster = glob(os.path.join(actual_pumping_dir, f'*{year}*.tif'))[0]

                mask_raster_by_shape(input_raster=actual_pumping_raster,
                                     input_shape=basin_shp, crop=True,
                                     output_dir=basin_actual_output_dir,
                                     raster_name=f'pumping_{year}.tif')

        return basin_predicted_output_dir, basin_actual_output_dir

    else:
        basin_predicted_output_dir = os.path.join(output_dir, 'predicted')
        basin_actual_output_dir = os.path.join(output_dir, 'actual')

        return basin_predicted_output_dir, basin_actual_output_dir


def compile_pixelwise_basin_df(years, basin_predicted_pumping_dir,
                               output_csv, basin_insitu_pumping_dir=None,
                               skip_processing=False):
    """
    Compile pixel-wise predicted and in-situ pumping values with latitude and longitude info.

    :param years: list[int]
        List of years to process.
    :param basin_predicted_pumping_dir: str
        Directory containing clipped predicted pumping rasters.
    :param output_csv: str
        Path to save the resulting pixel-wise CSV file.
    :param basin_insitu_pumping_dir: str
        Directory containing clipped actual/in-situ pumping rasters. Default set to None to not collect actual pumping data.
    :param skip_processing: bool
        If True, skips processing and assumes the CSV already exists.

    :return: str
        Filepath to the saved pixel-wise pumping CSV file.
    """
    if not skip_processing:
        makedirs([os.path.dirname(output_csv)])

        print(f'\nCompiling pixel-scale predicted and in-situ pumping dataframe...')

        # empty dictionary with to store data
        if basin_insitu_pumping_dir is not None:
            extract_dict = {'year': [], 'pred_pumping_mm': [], 'actual_pumping_mm': [],
                            'lat': [], 'lon': []}

        else:
            extract_dict = {'year': [], 'pred_pumping_mm': [],
                            'lat': [], 'lon': []}

        # lopping through each year and storing data in a list
        for year in years:
            # loading predicted and actual pumping array
            pred_pumping_data = glob(os.path.join(basin_predicted_pumping_dir, f'*{year}*.tif'))[0]
            pred_pumping_arr = read_raster_arr_object(pred_pumping_data, get_file=False).flatten()

            # replacing nodata of as zero.
            # this will help gather predicted pumping data even if actual pumping is zero and vice versa.
            pred_pumping_arr[np.isnan(pred_pumping_arr)] = 0

            # creating lat/lon array
            lon_arr, lat_arr = make_lat_lon_array_from_raster(pred_pumping_data)
            lon_arr = lon_arr.flatten()
            lat_arr = lat_arr.flatten()

            # adding data to dictionary for storing
            year_list = [year] * len(pred_pumping_arr)

            extract_dict['year'].extend(year_list)
            extract_dict['pred_pumping_mm'].extend(list(pred_pumping_arr))
            extract_dict['lon'].extend(list(lon_arr))
            extract_dict['lat'].extend(list(lat_arr))

            # separate block to compile in-situ pumping data to csv (basin_insitu_pumping_dir != None)
            if basin_insitu_pumping_dir is not None:
                # loading data
                actual_pumping_data = glob(os.path.join(basin_insitu_pumping_dir, f'*{year}*.tif'))[0]
                actual_pumping_arr = read_raster_arr_object(actual_pumping_data, get_file=False).flatten()

                # replacing nodata of as zero.
                pred_pumping_arr[np.isnan(pred_pumping_arr)] = 0
                actual_pumping_arr[np.isnan(actual_pumping_arr)] = 0

                extract_dict['actual_pumping_mm'].extend(list(actual_pumping_arr))

        # converting dictionary to dataframe and saving to csv
        df = pd.DataFrame(extract_dict)

        # Noticed some very high predicted value in the deep learning model prediction.
        # This happens in a very small number of pixels (10 may be in a year). Setting them as zero.
        # This issue doesn't happen in the ML model.
        df.loc[df['pred_pumping_mm'] > 2000] = 0

        # dropping nan value
        df = df.dropna()
        df.to_csv(output_csv, index=False)

        return output_csv


def compile_basinscale_annual_df(basin_code, pixelscale_csv, output_csv,
                                 actual_pumping_mm_col='actual_pumping_mm',
                                 pred_pumping_mm_col='pred_pumping_mm',
                                 skip_processing=False):
    """
    Compile basin-scale annual groundwater pumping based on overlapping pixel-wise predicted and actual values.

    ** Only considering pixels that have both predicted and actual pumping. Otherwise, predicted
    basin-scale average pumping will be much higher.

    :param basin_code: str
        Basin code used to look up total basin area (e.g., 'gmd3', 'hqr', 'spb').
    :param pixelscale_csv: str
        Filepath to pixelwise pumping CSV (from compile_pixelwise_basin_df).
    :param output_csv: str
        Path to save the basin-scale annual pumping CSV.
    :param actual_pumping_mm_col: str
        Column name containing the actual pumping values. Can be set to None.
    :param pred_pumping_mm_col: str
        Column name containing the predicted pumping values.
    :param skip_processing: bool
        If True, skips processing and assumes output CSV already exists.

    :return: str
        Filepath to the saved annual basin-scale pumping CSV.
    """
    if not skip_processing:
        makedirs([os.path.dirname(output_csv)])

        print(f'\nCompiling basin-scale predicted and in-situ pumping dataframe...')

        basin_area_dict = {
            'gmd4': 12737667189.642 * (1000 * 1000),  # in mm2
            'gmd3': 21820149683.491 * (1000 * 1000),  # in mm2
            'rpb': 22753400088.854 * (1000 * 1000),  # in mm2
            'spb': 7189173178.537 * (1000 * 1000),  # in mm2
            'ar': 2066332947.827 * (1000 * 1000),  # in mm2
            'slv': 19550637876.463 * (1000 * 1000),  # in mm2
            'hqr': 1982641859.510 * (1000 * 1000),  # in mm2
            'doug': 2459122191.981 * (1000 * 1000),  # in mm2
            'phx': 13956271593.171 * (1000 * 1000),  # in mm2
            'pnl': 10614105343.818 * (1000 * 1000),  # in mm2
            'tucs': 10027359710.224 * (1000 * 1000),  # in mm2
            'scruz': 1855467461.919 * (1000 * 1000),  # in mm2
            'dv': 1933578136.225 * (1000 * 1000),  # in mm2
            'cv': 52592337551.544 * (1000 * 1000),  # in mm2
            'srb': 39974554889.422 * (1000 * 1000),  # in mm2
        }

        # loading dataframe
        pixel_df = pd.read_csv(pixelscale_csv)

        if any(i in pixel_df.columns for i in ['lat', 'lon']):
            pixel_df = pixel_df.drop(columns=['lat', 'lon'])

        # area of a 2 km pixel
        # 2199 meter is the pixel size in latitude direction. 1746 is the pixel size in longitude direction.
        # For longitudinal dimension calculation, an average latitude of 37.42 deg was considered for the Western US.
        # Distance of 1 deg Longitude at the equator is = 111,320 m
        # 0.01976293625031605786 deg * 111320 * cos(37.42 * pi / 180) = 1746.53 m
        # 0.01976293625031605786 deg * 111320 = 2199.59 m
        area_mm2_single_pixel = (2199.59 * 1000) * (1746.53 * 1000)  # unit in mm2

        # annual sum using groupby()
        yearly_df = pixel_df.groupby('year').sum().reset_index()

        # calculating total volume
        yearly_df['pred_pumping_m3'] = yearly_df[pred_pumping_mm_col] * area_mm2_single_pixel / 1e9

        yearly_df['pred_pumping_AF'] = yearly_df['pred_pumping_m3'] / 1233.48

        # calculating area-averaged mean actual + predicted pumping (in mm)
        # AF >> mm3 >> mean mm
        yearly_df['mean pred_pumping_mm'] = yearly_df['pred_pumping_AF'] * 1233481837547.5 / basin_area_dict[basin_code]

        yearly_df['basin_code'] = basin_code

        if actual_pumping_mm_col is not None:
            yearly_df['actual_pumping_m3'] = yearly_df[actual_pumping_mm_col] * area_mm2_single_pixel / 1e9
            yearly_df['actual_pumping_AF'] = yearly_df['actual_pumping_m3'] / 1233.48
            yearly_df['mean actual_pumping_mm'] = yearly_df['actual_pumping_AF'] * 1233481837547.5 / basin_area_dict[
                basin_code]

        # saving
        yearly_df.to_csv(output_csv, index=False)

        return output_csv


def compile_basinscale_annual_df_UT(basin_code, years, basin_shp, predicted_pumping_dir,
                                    basin_predicted_pumping_dir,
                                    insitu_data_csv, output_csv,
                                    skip_processing=False):
    """
    Compile basin-scale annual predicted and in-situ pumping for Parowan, Beryl, anc Cedar Valley, Utah.

    :param basin_code: str
        Basin code used to look up total basin area (e.g., 'pv', 'brl').
    :param years: list[int]
        List of years to process.
    :param basin_shp: str
        Shapefile path of the basin boundary.
    :param predicted_pumping_dir: str
        Directory containing predicted pumping rasters.
    :param basin_predicted_pumping_dir: str
        Directory containing predicted pumping raster files.
    :param insitu_data_csv: str
        CSV file with UTAH Water Balance model estimated pumping (in AF).
    :param output_csv: str
        Path to save the compiled annual CSV.
    :param skip_processing: bool
        If True, skips processing and assumes output already exists.

    :return: str
        Filepath of the saved annual basin-scale pumping CSV.
    """
    if not skip_processing:
        makedirs([os.path.dirname(output_csv)])

        print(f'\nCompiling basin-scale predicted and in-situ pumping dataframe...')

        # Step 1: Clip predicted pumping rasters to basin
        basin_predicted_pumping_dir, basin_actual_pumping_dir = \
            clip_pumping_for_basin(years=years,
                                   basin_shp=basin_shp,
                                   predicted_pumping_dir=predicted_pumping_dir,
                                   actual_pumping_dir=None,
                                   output_dir=basin_predicted_pumping_dir,
                                   skip_processing=False)

        # Step 2: Compiling data

        # empty dictionary to store data
        extract_dict = {'year': [], 'pred_pumping_mm': []}

        # lopping through each year and storing data in a list
        for year in years:
            pred_pumping_data = glob(os.path.join(basin_predicted_pumping_dir, f'*{year}*.tif'))[0]
            pred_pumping_arr = read_raster_arr_object(pred_pumping_data, get_file=False).flatten()

            year_list = [year] * len(pred_pumping_arr)

            extract_dict['year'].extend(year_list)
            extract_dict['pred_pumping_mm'].extend(list(pred_pumping_arr))

        # converting to a Dataframe
        pixel_df = pd.DataFrame(extract_dict)

        # area of a 2 km pixel
        # 2199 meter is the pixel size in latitude direction. 1746 is the pixel size in longitude direction.
        # For longitudinal dimension calculation, an average latitude of 37.42 deg was considered for the Western US.
        # Distance of 1 deg Longitude at the equator is = 111,320 m
        # 0.01976293625031605786 deg * 111320 * cos(37.42 * pi / 180) = 1746.53 m
        # 0.01976293625031605786 deg * 111320 = 2199.59 m
        area_mm2_single_pixel = (2199.59 * 1000) * (1746.53 * 1000)  # unit in mm2

        # annual sum using groupby()
        yearly_df = pixel_df.groupby('year').sum().reset_index()

        # loading UTAH Water Balance Model estimated pumping data
        insitu_df = pd.read_csv(insitu_data_csv)
        insitu_df = insitu_df[['Year', 'Irrigation_AF']]
        insitu_df = insitu_df.rename(columns={'Year': 'year'})

        # merging both dataframes
        yearly_df = pd.merge(yearly_df, insitu_df, how='outer', on='year')
        yearly_df = yearly_df[yearly_df['year'] >= 2000]

        # calculating total volume
        yearly_df['pred_pumping_m3'] = yearly_df['pred_pumping_mm'] * area_mm2_single_pixel / 1e9
        yearly_df['pred_pumping_AF'] = yearly_df['pred_pumping_m3'] / 1233.48

        yearly_df['actual_pumping_m3'] = yearly_df['Irrigation_AF'] * 1233.48

        yearly_df = yearly_df.rename(columns={'Irrigation_AF': 'actual_pumping_AF'})

        # calculating basin average pumping
        basin_area_dict = {
            'pv': 1339578824.848 * (1000 * 1000),    # in mm2
            'cdr': 1433689295.822 * (1000 * 1000),   # in mm2
            'brl': 4978562228.985 * (1000 * 1000),   # in mm2
        }

        # calculating area-averaged mean actual + predicted pumping (in mm)
        # AF >> mm3 >> mean mm
        yearly_df['mean pred_pumping_mm'] = yearly_df['pred_pumping_AF'] * 1233481837547.5 / basin_area_dict[basin_code]
        yearly_df['mean actual_pumping_mm'] = yearly_df['actual_pumping_AF'] * 1233481837547.5 / basin_area_dict[basin_code]

        # saving
        yearly_df.to_csv(output_csv, index=False)

        return output_csv


def compile_basinscale_annual_df_Diamond(years, basin_predicted_pumping_dir,
                                         insitu_data_csv, output_csv,
                                         skip_processing=False):
    """
    Compile basin-scale annual predicted and in-situ pumping for Diamond Valley, Nevada.

    :param years: list[int]
        List of years to process.
    :param basin_predicted_pumping_dir: str
        Directory containing predicted pumping raster files.
    :param insitu_data_csv: str
        CSV file with in-situ measured field-level pumping (in AF), to be aggregated.
    :param output_csv: str
        Path to save the compiled annual CSV.
    :param skip_processing: bool
        If True, skips processing and assumes output already exists.

    :return: str
    """
    if not skip_processing:
        makedirs([os.path.dirname(output_csv)])

        print(f'\nCompiling basin-scale predicted and in-situ pumping dataframe...')

        # empty dictionary with to store data
        extract_dict = {'year': [], 'pred_pumping_mm': []}

        # lopping through each year and storing data in a list
        for year in years:
            pred_pumping_data = glob(os.path.join(basin_predicted_pumping_dir, f'*{year}*.tif'))[0]
            pred_pumping_arr = read_raster_arr_object(pred_pumping_data, get_file=False).flatten()

            year_list = [year] * len(pred_pumping_arr)

            extract_dict['year'].extend(year_list)
            extract_dict['pred_pumping_mm'].extend(list(pred_pumping_arr))

        # converting to a Dataframe
        pixel_df = pd.DataFrame(extract_dict)

        # area of a 2 km pixel
        # 2199 meter is the pixel size in latitude direction. 1746 is the pixel size in longitude direction.
        # For longitudinal dimension calculation, an average latitude of 37.42 deg was considered for the Western US.
        # Distance of 1 deg Longitude at the equator is = 111,320 m
        # 0.01976293625031605786 deg * 111320 * cos(37.42 * pi / 180) = 1746.53 m
        # 0.01976293625031605786 deg * 111320 = 2199.59 m
        area_mm2_single_pixel = (2199.59 * 1000) * (1746.53 * 1000)  # unit in mm2

        # annual sum using groupby()
        yearly_df = pixel_df.groupby('year').sum().reset_index()

        # loading DRI-provided pumping data for Diamond Valley
        insitu_df = pd.read_csv(insitu_data_csv)
        insitu_df = insitu_df[['year', 'pumping_AF']]
        insitu_df = insitu_df.groupby('year').sum().reset_index()  # the original data is for individual field

        # merging both dataframes
        yearly_df = pd.merge(yearly_df, insitu_df, how='outer', left_on='year', right_on='year')

        # calculating total volume
        yearly_df['pred_pumping_m3'] = yearly_df['pred_pumping_mm'] * area_mm2_single_pixel / 1e9
        yearly_df['pred_pumping_AF'] = yearly_df['pred_pumping_m3'] / 1233.48

        yearly_df['actual_pumping_m3'] = yearly_df['pumping_AF'] * 1233.48

        yearly_df = yearly_df.rename(columns={'pumping_AF': 'actual_pumping_AF'})

        # calculating basin average pumping
        basin_area_dict = {
            'dv': 1933578136.225 * (1000 * 1000),  # in mm2
        }

        # calculating area-averaged mean actual + predicted pumping (in mm)
        # AF >> mm3 >> mean mm
        yearly_df['mean pred_pumping_mm'] = yearly_df['pred_pumping_AF'] * 1233481837547.5 / basin_area_dict['dv']
        yearly_df['mean actual_pumping_mm'] = yearly_df['actual_pumping_AF'] * 1233481837547.5 / basin_area_dict['dv']

        # saving
        yearly_df.to_csv(output_csv, index=False)

        return output_csv


def compile_basin_predicted_actual_pumping_KS_CO_AZ(basin_code, years, basin_shp,
                                                    predicted_pumping_dir, actual_pumping_dir, output_dir,
                                                    skip_clip_to_basins=False,
                                                    skip_pixelscale_compilation=False,
                                                    skip_basinscale_compilation=False):
    """
    Run the pumping analysis pipeline (clip + pixelwise + annual summary) for Kansas, Colorado, and Arizona basins.

    :param basin_code: str
        Basin code (e.g., 'rpb', 'gmd3', 'phx').
    :param years: list[int]
        List of years to process.
    :param basin_shp: str
        Shapefile path of the basin boundary.
    :param predicted_pumping_dir: str
        Directory containing predicted pumping rasters.
    :param actual_pumping_dir: str
        Directory containing actual/in-situ pumping rasters (in mm).
    :param output_dir: str
        Output directory for saving clipped rasters and CSVs.
    :param skip_clip_to_basins: bool
        If True, skips the raster clipping step.
    :param skip_pixelscale_compilation: bool
        If True, skips generating pixelwise CSV.
    :param skip_basinscale_compilation: bool
        If True, skips generating annual basin-scale CSV.

    :return: None
    """
    # Step 1: Clip predicted pumping rasters to basin
    basin_predicted_pumping_dir, basin_actual_pumping_dir = \
        clip_pumping_for_basin(years=years,
                               basin_shp=basin_shp,
                               predicted_pumping_dir=predicted_pumping_dir,
                               actual_pumping_dir=actual_pumping_dir,
                               output_dir=output_dir, skip_processing=skip_clip_to_basins)

    # Step 2: Compile pixelwise pumping CSV
    pixelwise_csv = os.path.join(output_dir, f'pixelwise_pumping_{basin_code}.csv')
    compile_pixelwise_basin_df(years=years,
                               basin_predicted_pumping_dir=basin_predicted_pumping_dir,
                               basin_insitu_pumping_dir=basin_actual_pumping_dir,
                               output_csv=pixelwise_csv,
                               skip_processing=skip_pixelscale_compilation)

    # Step 3: Compile basin-scale pumping CSV
    basinscale_csv = os.path.join(output_dir, f'basinscale_pumping_{basin_code}.csv')
    compile_basinscale_annual_df(basin_code=basin_code, pixelscale_csv=pixelwise_csv,
                                 output_csv=basinscale_csv,
                                 skip_processing=skip_basinscale_compilation)


def aggregate_USGS_pumping_annual_csv(years, usgs_GW_shp_for_basin, convert_to_crs, output_csv):
    """
    Aggregate USGS HUC12-scale GW pumping estimates to a annual scale for a basin of interest.

    :param years: List of years_list to process the data for.
    :param usgs_GW_shp_for_basin: USGS HUC12-level shapefile (for the basin) with annual GW pumping estimates in AF.
    :param convert_to_crs: For estimating HUC12 areas inside the basin of interest, use a projected crs.
    :param output_csv: Filepath of output csv with USGS annual GW pumping estimates for a basin of interest.

    :return: Filepath of output csv.
    """
    print(f'aggregating annual USGS GW irrigation vs pumping ...')

    # converting integer years_list to str
    years = [str(y) for y in years if y < 2021]

    # read USGS dataset
    usgs_df = gpd.read_file(usgs_GW_shp_for_basin)

    # estimating area of each huc12 that is inside the basin
    usgs_df = usgs_df.to_crs(convert_to_crs)
    usgs_df['clipped_area_sqkm'] = usgs_df['geometry'].area / (1000 * 1000)  # unit in sqkm
    usgs_df['areasqkm'] = usgs_df['areasqkm'].astype('float')

    # using an area filter
    # if clipped_area_sqkm < 20% of areasqkm, removing that from GW_AF calculation
    # if clipped_area_sqkm within 5% of the areasqkm, consider it fully
    # otherwise use the ratio - clipped_area_sqkm/areasqkm to scale GW_AF
    area_ratio = []
    for idx, row in usgs_df.iterrows():
        if row['clipped_area_sqkm'] <= 0.20 * row['areasqkm']:
            area_ratio.append(0)
        elif row['clipped_area_sqkm'] >= 0.95 * row['areasqkm']:
            area_ratio.append(1)
        else:
            area_ratio.append(row['clipped_area_sqkm'] / row['areasqkm'])

    # multiplying by the area ratio to discard little/no coverage huc12s
    usgs_df['area_ratio'] = area_ratio
    usgs_df = usgs_df[years].mul(usgs_df['area_ratio'], axis=0)

    # transposing to bring years_list in a columns
    usgs_df_T = usgs_df.T
    usgs_df_T['year'] = usgs_df_T.index
    usgs_df_T['year'] = usgs_df_T['year'].astype(
        int)  # the 'year' needs to be converted to int for following merging operation
    usgs_df_T = usgs_df_T.reset_index(drop=True)

    usgs_df_T['USGS_AF'] = usgs_df_T.drop('year', axis=1).sum(axis=1)
    usgs_df_T = usgs_df_T[['year', 'USGS_AF']]

    usgs_df_T.to_csv(output_csv, index=False)


def aggregate_usgs_and_predicted_pumping_to_annualCSV_CA_ID(annual_predicted_pumping_csv, annual_usgs_GW_csv,
                                                            area_basin_mm2, output_annual_csv):
    """
    *** used for Central Valley, CA and Snake River Basin, ID  ***

    Aggregate annual predicted and usgs pumping estimates for a basin
    to an annual csv.

    :param annual_predicted_pumping_csv: Filepath of csv holding annual predicted pumping data for a basin.
    :param annual_usgs_GW_csv: USGS annual pumping estimates' csv for the basin.
    :param area_basin_mm2: Area of the basin in mm2.
    :param output_annual_csv: Filepath of output csv.

    :return: None.
    """
    print('Aggregating annual pumping and USGS estimated pumping to a csv...')

    # loading dataframe with annual netGW estimates
    annual_pred_pumping_df = pd.read_csv(annual_predicted_pumping_csv)

    # loading USGS annual pumping estimates data
    usgs_df = pd.read_csv(annual_usgs_GW_csv)

    # merging netGW + USGS pumping estimates together
    yearly_df = annual_pred_pumping_df.merge(usgs_df, on='year', how='outer')

    # calculating m3 values
    yearly_df['pred_pumping_m3'] = yearly_df['pred_pumping_AF'] * 1233.48
    yearly_df['USGS_m3'] = yearly_df['USGS_AF'] * 1233.48

    # calculating mean netGW + mean USGS pumping (in mm)
    area_mm2_single_pixel = (2193 * 1000) * (2193 * 1000)  # unit in mm2
    yearly_df['mean pred_pumping_mm'] = yearly_df['pred_pumping_AF'] * 1233481837547.5 / area_basin_mm2
    yearly_df['mean USGS_mm'] = yearly_df['USGS_AF'] * 1233481837547.5 / area_basin_mm2  # AF >> mm3 >> mean mm

    # saving final csv
    yearly_df.to_csv(output_annual_csv, index=False)


# def run_annual_csv_processing_CA_ID(years, basin_code, basin_shp, model_predicted_pumping_dir,
#                                     main_output_dir, usgs_westUS_GW_shp,
#                                     skip_processing=False):
#     """
#     Run processes to compile a basins' netGW, pumping, and USGS pumping data at annual scale in a csv for
#     Central Valley, CA and Snake River Basin, ID.
#
#     :param years: List of years_list to process data.
#     :param basin_code: Basin keyword to get area and save processed datasets. Must be one of the following-
#                         ['cv', 'srb]
#     :param basin_shp: Filepath of basin shapefile.
#     :param model_predicted_pumping_dir: ML/ANN predicted Western US pumping directory.
#     :param main_output_dir: Filepath of main output directory to store processed data for a basin.
#     :param usgs_westUS_GW_shp: USGS HUC12-level shapefile (for the Western US) with annual GW pumping estimates in AF.
#     :param skip_processing: Set to True to skip the processing.
#
#     :return: None.
#     """
#     if not skip_processing:
#         # area of basins
#         basin_area_dict = {
#             'cv': 52592337551.544 * (1000 * 1000),  # in mm2
#             'srb': 39974554889.422 * (1000 * 1000),  # in mm2
#
#         }
#
#         # # creating output directories for different processes
#         basin_dir = os.path.join(main_output_dir, f'{basin_code}')
#         usgs_basin_GW_dir = os.path.join(main_output_dir, f'{basin_code}/USGS_GW_irr')
#         usgs_basin_GW_shp = os.path.join(main_output_dir, f'{basin_code}/USGS_GW_irr', 'USGS_GW_irr.shp')
#         makedirs([basin_dir, usgs_basin_GW_dir])
#
#         # # # # #  STEP 1 # # # # #
#         # # Clip pumping for the basin
#         print('# # # # #  STEP 1 # # # # #')
#         clip_pumping_for_basin(years=years, basin_shp=basin_shp,
#                                predicted_pumping_dir=model_predicted_pumping_dir,
#                                output_dir=basin_dir,
#                                actual_pumping_dir=None,
#                                skip_processing=False)
#
#         # # # # #  STEP 2 # # # # #
#         # # Compile pixelwise pumping to annual total dataframe
#         print('# # # # #  STEP 2 # # # # #')
#
#         pixelwise_csv = os.path.join(main_output_dir, f'{basin_code}/pixelwise_pumping_{basin_code}.csv')
#         compile_pixelwise_basin_df(years=years, basin_predicted_pumping_dir=os.path.join(basin_dir, 'predicted'),
#                                    output_csv=pixelwise_csv,
#                                    basin_insitu_pumping_dir=None,
#                                    skip_processing=False)
#
#         basinscale_csv = os.path.join(main_output_dir, f'{basin_code}/basinscale_pumping_{basin_code}_interim.csv')
#         compile_basinscale_annual_df(basin_code=basin_code, pixelscale_csv=pixelwise_csv,
#                                      output_csv=basinscale_csv,
#                                      pred_pumping_mm_col='pred_pumping_mm',
#                                      actual_pumping_mm_col=None,
#                                      skip_processing=False)
#
#         # # # # #  STEP 3 # # # # #
#         # Clip USGS HUC12-scale basins with NHM predicted GW pumping estimates
#         print('# # # # #  STEP 3 # # # # #', '\n', 'Clipping HUC12-scale basins with USGS GW pumping data...')
#
#         clip_vector(input_shapefile=usgs_westUS_GW_shp, mask_shapefile=basin_shp,
#                     output_shapefile=usgs_basin_GW_shp, create_zero_buffer=False,
#                     change_crs='EPSG:4269')  # the conversion to EPSG 4269 is needed as all basin shapefiles are in this crs
#
#         # # # # #  STEP 4 # # # # #
#         # # Aggregate USGS HUC12-scale GW pumping estimates to an annual scale for the basin of interest
#         print('# # # # #  STEP 4 # # # # #')
#
#         usgs_annual_csv = os.path.join(main_output_dir, f'{basin_code}/usgs_annual.csv')
#         aggregate_USGS_pumping_annual_csv(years=years, usgs_GW_shp_for_basin=usgs_basin_GW_shp,
#                                           convert_to_crs='EPSG:3857',
#                                           output_csv=usgs_annual_csv)
#
#         # # # # #  STEP 5 # # # # #
#         # # Compile the basin's total annual netGW and USGS pumping to a common csv
#         print('# # # # #  STEP 5 # # # # #')
#
#         final_basinscale_csv = os.path.join(main_output_dir, f'{basin_code}/basinscale_pumping_{basin_code}.csv')
#         aggregate_usgs_and_predicted_pumping_to_annualCSV_CA_ID(annual_predicted_pumping_csv=basinscale_csv,
#                                                                 annual_usgs_GW_csv=usgs_annual_csv,
#                                                                 area_basin_mm2=basin_area_dict[basin_code],
#                                                                 output_annual_csv=final_basinscale_csv)
#         # os.remove(basinscale_csv)
#
#     else:
#         pass


def count_num_irrigated_pixels(basin_pixel_year_dicts, output_csv):
    """
    Count the number of irrigated pixels for each basin and year based on
    predicted and reported groundwater pumping data.

    This function iterates through a dictionary of basin configurations,
    each containing a list of years and a CSV file path with pixel-level
    pumping estimates. For each basin and year, it counts how many pixels
    have positive pumping values in both the predicted and actual (reported)
    columns, then writes the results to an output CSV file.

    Parameters
    ----------
    basin_pixel_year_dicts : dict
        Dictionary containing basins as keys. Each basin entry must include:
            - 'year' : list of int
                List of years to analyze.
            - 'path' : str
                File path to the CSV file containing pixelwise pumping data.
        Example:
            {
                'gmd4': {
                    'year': [2000, 2001, 2002],
                    'path': '../../Data/pumping_gmd4.csv'
                },
                'scruz': {
                    'year': [2011, 2012, 2013],
                    'path': '../../Data/pumping_scruz.csv'
                }
            }

    output_csv : str
        File path where the resulting summary table (basin, year, counts)
        will be saved as a CSV.

    Raises
    ------
    ValueError
        If any basin dictionary is missing the keys 'year' or 'path'.

    Output
    ------
    A CSV file saved at `output_csv` with the following columns:
        - 'basin' : basin name
        - 'year' : year analyzed
        - 'irr_count_RS' : number of pixels with predicted pumping > 0
        - 'irr_count_reported' : number of pixels with reported pumping > 0
    """
    for basin, info in basin_pixel_year_dicts.items():
        if 'year' not in info or 'path' not in info:
            raise ValueError(f"Basin dict - '{basin}' must contain both 'year' and 'path' keys.")

    # empty dictionary to store data
    count_dict = {'basin': [],
                  'year': [],
                  'irr_count_RS': [],
                  'irr_count_reported': []}

    for basin in basin_pixel_year_dicts.keys():
        df = pd.read_csv(basin_pixel_year_dicts[basin]['path'])

        for yr in basin_pixel_year_dicts[basin]['year']:
            df_yr = df[df['year'] == yr]

            irr_pixels_RS = (
                    df_yr['pred_pumping_mm'] > 0).sum()  # irrigated pixels in predicted data, using IrrMappe/LANID
            irr_pixels_reported = (df_yr['actual_pumping_mm'] > 0).sum()  # irrigated pixels in actual (reported) data

            count_dict['basin'].append(basin)
            count_dict['year'].append(yr)
            count_dict['irr_count_RS'].append(irr_pixels_RS)
            count_dict['irr_count_reported'].append(irr_pixels_reported)

    # store results in dataframe
    count_df = pd.DataFrame(count_dict)

    count_df.to_csv(output_csv, index=False)


def compile_prediction_CI(basin_code, years, basin_shp,
                          prediction_CI_dir, basin_output_dir):
    """
    Collect and compile 95% confidence intervals of predictions.

    :param basin_code: str
        Basin code (e.g., 'rpb', 'gmd3', 'phx').
    :param years: list[int]
        List of years to process.
    :param basin_shp: str
        Shapefile path of the basin boundary.
    :param prediction_CI_dir: str
        Path of directory containing the low and high CI prediction rasters.
    :param basin_output_dir: str
        Path of directory containing the clipped basin-scale low and high CI prediction rasters.

    :return: None
    """
    ####################################################################################################################
    # Step 1: Clip CIs pumping rasters to basin
    ####################################################################################################################
    CIs_pumping_raster = glob(os.path.join(prediction_CI_dir, f'*.tif'))    # all low and high CI

    for each in CIs_pumping_raster:
        mask_raster_by_shape(input_raster=each,
                             input_shape=basin_shp, crop=True,
                             output_dir=basin_output_dir,
                             raster_name=os.path.basename(each))

    ####################################################################################################################
    # Step 2: Compile pixelwise CSV
    ####################################################################################################################
    extract_dict = {'year': [], 'high_CI': [], 'low_CI': []}

    for year in years:
        # loading low and high CI array
        low_arr = read_raster_arr_object(os.path.join(basin_output_dir, f'low_{year}.tif'), get_file=False).flatten()
        high_arr = read_raster_arr_object(os.path.join(basin_output_dir, f'high_{year}.tif'), get_file=False).flatten()

        # replacing nodata of as zero
        # this will help gather predicted pumping data even if actual pumping is zero and vice versa.
        low_arr[np.isnan(low_arr)] = 0
        high_arr[np.isnan(high_arr)] = 0

        # adding data to dictionary for storing
        year_list = [year] * len(low_arr)

        extract_dict['year'].extend(year_list)
        extract_dict['low_CI'].extend(list(low_arr))
        extract_dict['high_CI'].extend(list(high_arr))

    # converting dictionary to dataframe and saving to csv
    df = pd.DataFrame(extract_dict)

    # dropping nan value and save
    df = df.dropna()

    pixelwise_csv = os.path.join(os.path.dirname(basin_output_dir), f'pixelwise_CI_{basin_code}.csv')
    df.to_csv(pixelwise_csv, index=False)

    ####################################################################################################################
    # Step 3: Compile basin-scale pumping CSV
    ####################################################################################################################

    basin_area_dict = {
        'gmd4': 12737667189.642 * (1000 * 1000),  # in mm2
        'gmd3': 21820149683.491 * (1000 * 1000),  # in mm2
        'rpb': 22753400088.854 * (1000 * 1000),  # in mm2
        # 'spb': 48937250302.991 * (1000 * 1000),     # in mm2
        # 'ar': 73247891962.490 * (1000 * 1000),      # in mm2
        'spb': 7189173178.537 * (1000 * 1000),  # in mm2
        'ar': 29106942834.344 * (1000 * 1000),  # in mm2
        'slv': 19550637876.463 * (1000 * 1000),  # in mm2
        'hqr': 1982641859.510 * (1000 * 1000),  # in mm2
        'doug': 2459122191.981 * (1000 * 1000),  # in mm2
        'phx': 13956271593.171 * (1000 * 1000),  # in mm2
        'pnl': 10614105343.818 * (1000 * 1000),  # in mm2
        'tucs': 10027359710.224 * (1000 * 1000),  # in mm2
        'scruz': 1855467461.919 * (1000 * 1000),  # in mm2
        'dv': 1933578136.225 * (1000 * 1000),  # in mm2
        'cv': 52592337551.544 * (1000 * 1000),  # in mm2
        'srb': 39974554889.422 * (1000 * 1000),  # in mm2
        'pv': 1339578824.848 * (1000 * 1000),  # in mm2
        'cdr': 1433689295.822 * (1000 * 1000),  # in mm2
        'brl': 4978562228.985 * (1000 * 1000)  # in mm2
    }

    # loading dataframe
    pixel_df = pd.read_csv(pixelwise_csv)

    # area of a 2 km pixel
    area_mm2_single_pixel = (2199.59 * 1000) * (1746.53 * 1000)  # unit in mm2

    # annual sum using groupby()
    yearly_df = pixel_df.groupby('year').sum().reset_index()

    # calculating total volume
    yearly_df['low_m3'] = yearly_df['low_CI'] * area_mm2_single_pixel / 1e9
    yearly_df['low_AF'] = yearly_df['low_m3'] / 1233.48

    yearly_df['high_m3'] = yearly_df['high_CI'] * area_mm2_single_pixel / 1e9
    yearly_df['high_AF'] = yearly_df['high_m3'] / 1233.48

    # calculating area-averaged mean actual + predicted pumping (in mm)
    # AF >> mm3 >> mean mm
    yearly_df['mean low_mm'] = yearly_df['low_AF'] * 1233481837547.5 / basin_area_dict[basin_code]
    yearly_df['mean high_mm'] = yearly_df['high_AF'] * 1233481837547.5 / basin_area_dict[basin_code]

    yearly_df['basin_code'] = basin_code

    yearly_df = yearly_df[['year', 'basin_code', 'mean low_mm', 'mean high_mm']]

    # saving
    basinscale_csv = os.path.join(os.path.dirname(basin_output_dir), f'basinscale_CI_{basin_code}.csv')
    yearly_df.to_csv(basinscale_csv, index=False)


def compile_annual_pumping_all_basins(annual_csv_list, output_csv,
                                      collect_CIs=True, CI_csv_list=None):
    """
    Combine annual basin-scale pumping CSVs into a single summary file across all basins.

    :param annual_csv_list: list[str]
        List of annual basin-scale pumping CSV filepaths (one per basin).
    :param output_csv: str
        Filepath to save the combined dataset for all basins.
    :param collect_CIs : bool
                    Whether to include the processing and compilation of uncertainty (confidence interval)
                    rasters and CSVs. Only applicable when `model='ML'`.
    :param CI_csv_list:  list[str]
        List of annual basin-scale pumping confidence interval CSV filepaths (one per basin).

    :return: None
    """
    # making the output directory if not available
    makedirs([os.path.dirname(output_csv)])

    # empty dataframe to store the results
    compiled_annual_df = pd.DataFrame()

    # basin name dict
    basin_name_dict = {'gmd4': 'GMD4, KS', 'gmd3': 'GMD3, KS',
                       'rpb': 'Republican River Basin, CO',
                       'spb': 'South Platte River Basin, CO',
                       'ar': 'Arkansas River Basin, CO',
                       'slv': 'San Luis Valley, CO',
                       'hqr': 'Harquahala INA, AZ',
                       'doug': 'Douglas AMA, AZ',
                       'phx': 'Phoenix AMA, AZ',
                       'pnl': 'Pinal AMA, AZ',
                       'tucs': 'Tucson AMA, AZ',
                       'scruz': 'Santa Cruz AMA, AZ',
                       'dv': 'Diamond Valley, NV',
                       'pv': 'Parowan Valley, UT',
                       'cdr': 'Cedar Valley, UT',
                       'brl': 'Beryl-Enterprise, UT'
                       }

    for csv in annual_csv_list:
        df = pd.read_csv(csv)

        basin_code = os.path.basename(csv).split('_')[-1].split('.')[0]
        df['basin_code'] = basin_code
        df['basin'] = [basin_name_dict[basin_code] for i in range(len(df))]
        compiled_annual_df = pd.concat([compiled_annual_df, df])

    compiled_annual_df = compiled_annual_df[['year', 'pred_pumping_m3', 'actual_pumping_m3',
                                             'pred_pumping_AF', 'actual_pumping_AF',
                                             'mean pred_pumping_mm', 'mean actual_pumping_mm',
                                             'basin_code', 'basin']]

    # setting all zero values to np.nan
    compiled_annual_df = compiled_annual_df.replace({0: np.nan})

    if collect_CIs and (CI_csv_list is not None):
        compiled_CI_df = pd.DataFrame()

        for csv in CI_csv_list:
            df = pd.read_csv(csv)
            compiled_CI_df = pd.concat([compiled_CI_df, df])

        # merge with main compiled_annual_df
        compiled_annual_df = compiled_annual_df.merge(compiled_CI_df, on=['year', 'basin_code'])

    compiled_annual_df.to_csv(output_csv, index=False)






