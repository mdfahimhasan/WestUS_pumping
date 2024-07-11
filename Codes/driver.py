from download_preprocess.preprocess import make_multiband_datasets

temporal_vars = ['../Data_main/rasters/Effective_precip_prediction_WestUS/v11_grow_season',
                 '../Data_main/rasters/PRISM_Precip/WestUS']
static_vars = ['../Data_main/rasters/Sand_content/WestUS']
years = list(range(2000, 2021))

make_multiband_datasets(list_of_temporal_var_dirs=temporal_vars, list_of_static_var_dirs=static_vars,
                        band_key_list=['peff', 'precip', 'sand'],
                        output_dir='../Data_main/rasters/multibands',
                        years_list=years, month_range=None)