import os
from glob import glob
from download import download_all_gee_data
from download_openET import download_all_openET_datasets
from Codes.download_preprocess.preprocess import run_all_preprocessing, make_multiband_datasets
from Codes.download_preprocess.tile import make_tiles

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # Data download
skip_download_gee_data = False
skip_download_OpenET_data = True

gee_data_list = ['Landsat5_NDVI', 'Landsat8_NDVI', 'Landsat5_NDMI',
                 'Landsat8_NDMI', 'Landsat5_GCVI', 'Landsat8_GCVI', 'GRIDMET_RET',
                 'Field_capacity', 'Sand_content', 'Clay_content']

openET_data_list = ['Irrig_crop_OpenET_IrrMapper',
                    'Irrig_crop_OpenET_LANID',
                    'Irrigation_Frac_IrrMapper',
                    'Irrigation_Frac_LANID']

years = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007,
         2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
         2016, 2017, 2018, 2019, 2020]
months = (1, 12)

data_download_dir = '../../Data_main/rasters'
gee_grid_shape_large = '../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_gee_grid.shp'
gee_grid_shape_for30m_IrrMapper = '../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_gee_grid_for30m_IrrMapper.shp'
gee_grid_shape_for30m_LANID = '../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_gee_grid_for30m_LANID.shp'

download_all_gee_data(data_list=gee_data_list, download_dir=data_download_dir,
                      year_list=years, month_range=months,
                      skip_download=skip_download_gee_data)

download_all_openET_datasets(year_list=years, month_range=months,
                             openET_data_list=openET_data_list,
                             data_download_dir=data_download_dir,
                             grid_shape_for_2km_ensemble=None,
                             grid_shape_for30m_irrmapper=gee_grid_shape_for30m_IrrMapper,
                             grid_shape_for30m_lanid=gee_grid_shape_for30m_LANID,
                             skip_download_OpenET_data=skip_download_OpenET_data)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # data preprocess (preprocess + multiband creation + tile creation)

# multi-band creation (western US scale)
temporal_vars = ['../../Data_main/rasters/NetGW_irrigation/WesternUS',
                 '../../Data_main/rasters/Effective_precip_prediction_WestUS/v11_grow_season',
                 '../../Data_main/rasters/RET/WestUS_yearly',
                 '../../Data_main/rasters/PRISM_Precip/WestUS',
                 '../../Data_main/rasters/Irrigated_cropET/WestUS_grow_season',
                 '../../Data_main/rasters/GCVI/WestUS_yearly',
                 '../../Data_main/rasters/NDVI/WestUS_yearly',
                 '../../Data_main/rasters/NDMI/WestUS_yearly']
static_vars = ['../../Data_main/rasters/Sand_content/WestUS',
               '../../Data_main/rasters/Field_capacity/WestUS']
band_key_list = ['netGWIrr',
                 'peff', 'ret', 'precip', 'irrcropET',
                 'GCVI', 'NDVI', 'NDMI', 'sand', 'fc',
                 'nanmask']

westUS_multiband_dir = '../../Data_main/rasters/multibands/westUS'

years = list(range(2000, 2019 + 1))
skip_create_multiband_raster = True

run_all_preprocessing(skip_process_GrowSeason_data=False,
                      skip_prism_processing=False)

make_multiband_datasets(list_of_temporal_var_dirs=temporal_vars, list_of_static_var_dirs=static_vars,
                        band_key_list=band_key_list,
                        output_dir=westUS_multiband_dir,
                        years_list=years, skip_processing=skip_create_multiband_raster)

# tile creation
multiband_tile_dir = '../../Data_main/rasters/multibands/tiles'
multiband_rasters = glob(os.path.join(westUS_multiband_dir, '*.tif'))
skip_create_tile = False
print('creating multi-band tiles...')

for multi_ras in multiband_rasters:

    make_tiles(tiff_path=multi_ras, tile_output_dir=multiband_tile_dir, band_key_list=band_key_list,
               skip_processing=skip_create_tile)