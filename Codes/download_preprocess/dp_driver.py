from download import download_all_gee_data
from Codes.download_preprocess.preprocess import make_multiband_datasets

# # Data download
skip_download_gee_data = False

gee_data_list = ['Landsat5_NDVI', 'Landsat8_NDVI',
                 'Landsat5_NDMI', 'Landsat8_NDMI',
                 'Landsat5_GCVI', 'Landsat8_GCVI',
                 'GRIDMET_RET',
                 'Field_capacity', 'Bulk_density', 'Sand_content', 'Clay_content']

years = [
    #2000,
    2001, 2002, 2003, 2004, 2005, 2006, 2007,
         2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
         2016, 2017, 2018, 2019, 2020]
months = (4, 10)
data_download_dir = '../../Data_main/rasters'

download_all_gee_data(data_list=gee_data_list, download_dir=data_download_dir,
                      year_list=years, month_range=months,
                      skip_download=skip_download_gee_data)

# temporal_vars = ['../Data_main/rasters/Effective_precip_prediction_WestUS/v11_grow_season',
#                  '../Data_main/rasters/PRISM_Precip/WestUS']
# static_vars = ['../Data_main/rasters/Sand_content/WestUS']
# years = list(range(2000, 2021))
#
# make_multiband_datasets(list_of_temporal_var_dirs=temporal_vars, list_of_static_var_dirs=static_vars,
#                         band_key_list=['peff', 'precip', 'sand'],
#                         output_dir='../Data_main/rasters/multibands',
#                         years_list=years, month_range=None)