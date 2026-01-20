import sys
from os.path import dirname, abspath

sys.path.insert(0, dirname(abspath(__file__)))

from Codes.results_analysis.analysis_utils import (compile_basin_predicted_actual_pumping_KS_CO_AZ,
                                                   compile_annual_pumping_all_basins)

no_data_value = -9999
model_res = 0.01976293625031605786  # in deg, ~2 km

YEARS = list(range(2000, 2024))

# Compiling basin-scale pumping (predicted + actual) to annual dataframe
ML_basin_configs = {
    'gmd4': {'year': list(range(2000, 2024)),
             'shape': '../../Data_main/shapefiles/Basins_of_interest/GMD4.shp',
             'pred_pump_dir': f'../../Data_main/rasters/ML_LOBO/GMD4',
             'type': 'ks_co_az',
             'skip': False},
    'gmd3': {'year': list(range(2000, 2024)),
             'shape': '../../Data_main/shapefiles/Basins_of_interest/GMD3.shp',
             'pred_pump_dir': f'../../Data_main/rasters/ML_LOBO/GMD3',
             'type': 'ks_co_az',
             'skip': False},
    'rpb': {'year': list(range(2000, 2024)),
            'shape': '../../Data_main/shapefiles/Basins_of_interest/Republican_Basin.shp',
            'pred_pump_dir': f'../../Data_main/rasters/ML_LOBO/RPB',
            'type': 'ks_co_az',
            'skip': False},
    'slv': {'year': list(range(2000, 2024)),
            'shape': '../../Data_main/shapefiles/Basins_of_interest/Rio_Grande_Basin.shp',
            'pred_pump_dir': f'../../Data_main/rasters/ML_LOBO/SLV',
            'actual_pump_dir': f'../../Data_main/pumping/rasters/WestUS_pumping/Original',
            'type': 'ks_co_az',
            'skip': False},
    'doug': {'year': list(range(2000, 2024)),
             'shape': '../../Data_main/shapefiles/Basins_of_interest/Douglas_AMA.shp',
             'pred_pump_dir': f'../../Data_main/rasters/ML_LOBO/DOUG',
             'actual_pump_dir': f'../../Data_main/pumping/rasters/WestUS_pumping/Original',
             'type': 'ks_co_az',
             'skip': False},
    'hqr': {'year': list(range(2000, 2024)),
            'shape': '../../Data_main/shapefiles/Basins_of_interest/Harquahala_INA.shp',
            'pred_pump_dir': f'../../Data_main/rasters/ML_LOBO/HQR',
            'actual_pump_dir': f'../../Data_main/pumping/rasters/WestUS_pumping/Original',
            'type': 'ks_co_az',
            'skip': False},
    'scruz': {'year': list(range(2000, 2024)),
              'shape': '../../Data_main/shapefiles/Basins_of_interest/SantaCruz_AMA.shp',
              'pred_pump_dir': f'../../Data_main/rasters/ML_LOBO/SCRUZ',
              'actual_pump_dir': f'../../Data_main/pumping/rasters/WestUS_pumping/Original',
              'type': 'ks_co_az',
              'skip': False},
    'dv': {'year': list(range(2000, 2024)),
           'shape': '../../Data_main/shapefiles/Basins_of_interest/Diamond_Valley_Basin.shp',
           'pred_pump_dir': f'../../Data_main/rasters/ML_LOBO/DV',
           'actual_csv': '../../Data_main/pumping/Nevada/raw/Diamond Valley/joined_data/dv_joined_et_pumping_data_all.csv',
           'type': 'special',
           'skip': False},
}


def process_all_basins(config_dict):
    for basin, cfg in config_dict.items():
        if cfg['skip']:
            continue

        print(f"\nProcessing basin: {basin.upper()}")
        print('-----------------------------------------\n')
        actual_pumping_dir = f'../../Data_main/pumping/rasters/WestUS_pumping/Original'
        output_dir = f'../../Model_run/basin_comparison_results/ML_LOBO/{basin}'

        compile_basin_predicted_actual_pumping_KS_CO_AZ(
            basin_code=basin, years=cfg['year'], basin_shp=cfg['shape'],
            predicted_pumping_dir=cfg['pred_pump_dir'], actual_pumping_dir=actual_pumping_dir,
            output_dir=output_dir,
            skip_clip_to_basins=False,
            skip_pixelscale_compilation=False,  # ~~~~~~~~~  note - check if replacing nan with zero is turned on/off
            skip_basinscale_compilation=False)

    else:
        pass


def compile_all_basin_summaries(config_dict, output_csv):
    basin_annual_csvs = [
        f'../../Model_run/basin_comparison_results/ML_LOBO/{b}/basinscale_pumping_{b}.csv'
        for b in config_dict.keys()
    ]
    compile_annual_pumping_all_basins(
        annual_csv_list=basin_annual_csvs,
        output_csv=output_csv)


def main(config_dict, output_csv):
    process_all_basins(config_dict)
    compile_all_basin_summaries(config_dict, output_csv)


if __name__ == '__main__':
    print(
        "# **** Note (from Ryan): Here, we use the original (unfiltered) pumping rasters as the actual (reference) for comparison.\n"
        "This approach is intentional because our goal is to develop a pumping raster product that remains valid even when actual \n"
        "pumping data are unavailable. Therefore, for years where actual data exist, we include all available pumping values in\n"
        "the basin-scale comparison with predicted pumping, so that the model can be trusted in years when direct comparison\n"
        "is not possible.\n"
    )

    # compile individual basin's LOBO results (annual) to individual csv + compiling all data in a single csv
    skip_compile_ML_LOBO = False         ######################

    # model version
    model_version_ML = 'v11'              ######################

    if not skip_compile_ML_LOBO:
        print('\n-----------------------------------------')
        print('Compiling LOBO results from ML model')
        print('-----------------------------------------')
        main(config_dict=ML_basin_configs,
             output_csv=f'../../Model_run/basin_comparison_results/annual_pumping_ML_LOBO_{model_version_ML}.csv')
