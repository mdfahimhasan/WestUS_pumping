import os
import sys
from os.path import dirname, abspath

sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.results_analysis.analysis_utils import (
    compile_basin_predicted_actual_pumping_KS_CO_AZ,
    compile_basinscale_annual_df_UT,
    compile_prediction_CI,
    compile_annual_pumping_all_basins
)

# # # Compiling basin-scale pumping (predicted + actual) to annual dataframe

basin_configs = {
    'gmd4': {'year': list(range(2000, 2024)),
             'shp': '../../Data_main/shapefiles/Basins_of_interest/GMD4.shp',
             'type': 'ks_co_az', 'skip': False},
    'gmd3': {'year': list(range(2000, 2024)),
             'shp': '../../Data_main/shapefiles/Basins_of_interest/GMD3.shp',
             'type': 'ks_co_az', 'skip': False},
    'rpb': {'year': list(range(2000, 2024)),
            'shp': '../../Data_main/shapefiles/Basins_of_interest/Republican_Basin.shp',
            'type': 'ks_co_az', 'skip': False},
    'spb': {'year': list(range(2000, 2024)),
            'shp': '../../Data_main/shapefiles/Basins_of_interest/South_Platte_Basin_comparison_extent.shp',
            'type': 'ks_co_az', 'skip': False},
    'ar': {'year': list(range(2000, 2024)),
           'shp': '../../Data_main/shapefiles/Basins_of_interest/Arkansas_Basin_comparison_extent.shp',
           'type': 'ks_co_az', 'skip': False},
    'slv': {'year': list(range(2000, 2024)),
            'shp': '../../Data_main/shapefiles/Basins_of_interest/Rio_Grande_Basin.shp',
            'type': 'ks_co_az', 'skip': False},
    'hqr': {'year': list(range(2000, 2024)),
            'shp': '../../Data_main/shapefiles/Basins_of_interest/Harquahala_INA.shp',
            'type': 'ks_co_az', 'skip': False},
    'doug': {'year': list(range(2000, 2024)),
             'shp': '../../Data_main/shapefiles/Basins_of_interest/Douglas_AMA.shp',
             'type': 'ks_co_az', 'skip': False},
    'phx': {'year': list(range(2000, 2024)),
            'shp': '../../Data_main/shapefiles/Basins_of_interest/Phoenix_AMA.shp',
            'type': 'ks_co_az', 'skip': False},
    'pnl': {'year': list(range(2000, 2024)),
            'shp': '../../Data_main/shapefiles/Basins_of_interest/Pinal_AMA.shp',
            'type': 'ks_co_az', 'skip': False},
    'scruz': {'year': list(range(2000, 2024)),
              'shp': '../../Data_main/shapefiles/Basins_of_interest/SantaCruz_AMA.shp',
              'type': 'ks_co_az', 'skip': False},
    'dv': {'year': list(range(2000, 2024)),
           'shp': '../../Data_main/shapefiles/Basins_of_interest/Diamond_Valley_Basin.shp',
           'type': 'ks_co_az', 'skip': False},
    'pv': {'year': list(range(2000, 2024)),
           'shp': '../../Data_main/shapefiles/Basins_of_interest/Parowan_Valley.shp',
           'actual': '../../Data_main/pumping/Utah/from Soheil/processed_Soheil/gw_withdrawals_parowan.csv',
           'type': 'ut', 'skip': False},
    'cdr': {'year': list(range(2000, 2024)),
            'shp': '../../Data_main/shapefiles/Basins_of_interest/Cedar_Valley.shp',
            'actual': '../../Data_main/pumping/Utah/from Soheil/processed_Soheil/gw_withdrawals_cedar.csv',
            'type': 'ut', 'skip': False},
    'brl': {'year': list(range(2000, 2024)),
            'shp': '../../Data_main/shapefiles/Basins_of_interest/Beryl_Valley.shp',
            'actual': '../../Data_main/pumping/Utah/from Soheil/processed_Soheil/gw_withdrawals_beryl.csv',
            'type': 'ut', 'skip': False},
}


def process_all_basins(model_version, model_prediction_dir, output_dir,
                       model='ML', process_CIs=False):
    """
    Processes groundwater pumping predictions and related outputs for all configured basins.

    This function iterates over each basin defined in `basin_configs` and performs a series of
    operations. It compiles:
      - Pixel-scale and basin-scale predicted vs. actual pumping datasets
      - Scaled pumping predictions (for basins where surface water irrigation dominates)
      - Uncertainty bounds (lower and upper confidence intervals) for ML-based models

    The function supports multiple model types ("ML", "ANN") and optional collection
    of uncertainty rasters (CIs). Basin-specific processing logic is determined by
    the `type` field in `basin_configs`.

    Parameters
    ----------
    model_version : str
        Version name or identifier for the machine learning model used (e.g., "v10").
        Used to locate corresponding uncertainty raster directories when `collect_CIs=True`.

    model_prediction_dir : str
        Path to the directory containing Western US-wide predicted groundwater pumping rasters for all years.

    output_dir : str
        Directory where compiled outputs (e.g., pixelwise CSVs, basin-scale summaries) will be saved.
        Basin-level subfolders are automatically created within output_dir.

    model : {'ML', 'ANN'}
        Specifies the model type. Determines which processing pipelines are executed.
        Defaults to 'ML'.

    process_CIs : bool, optional
        If True and `model='ML'`, collects and processes uncertainty (95% confidence interval)
        rasters corresponding to the machine learning predictions. Default is False.

    Raises
    ------
    ValueError
        If `model` is not one of {'ML', 'ANN'}.
        If `collect_CIs=True` but `model` is not 'ML' (since CI predictions exist only for ML models).

    Notes
    -----
    - Basin configurations (e.g., shapefiles, year ranges, flags) are read from the global `basin_configs` dict.
    - For basins tagged as `type='ks_co_az'`, the function uses
      `compile_basin_predicted_actual_pumping_KS_CO_AZ()`.
    - For Utah-type basins (`type='ut'`), it uses `compile_basinscale_annual_df_Parowan()`.
    - Scaled predictions (based on groundwater percentage data) are generated only for
      selected basins ('spb', 'ar', 'pnl', 'phx').
    - Compilation of CI predictions are executed for 'ML' model results only.
    """

    if model.lower() not in ['ml', 'ann']:
        ValueError("Model must me either 'ML' or 'ANN'")

    for basin, cfg in basin_configs.items():
        if cfg['skip']:
            continue

        print('\n-----------------------------------------')
        print(f"Processing basin: {basin.upper()}")
        print('-----------------------------------------')
        actual_pumping_dir = f'../../Data_main/pumping/rasters/WestUS_pumping/Original'

        # original prediction

        if cfg['type'] == 'ks_co_az':

            compile_basin_predicted_actual_pumping_KS_CO_AZ(
                basin_code=basin, years=cfg['year'], basin_shp=cfg['shp'],
                predicted_pumping_dir=model_prediction_dir,
                actual_pumping_dir=actual_pumping_dir,
                output_dir=os.path.join(output_dir, basin),
                skip_clip_to_basins=False,
                skip_pixelscale_compilation=False,
                skip_basinscale_compilation=False)

        elif cfg['type'] == 'ut':

            compile_basinscale_annual_df_UT(basin_code=basin,
                                            years=cfg['year'],
                                            basin_shp=cfg['shp'],
                                            predicted_pumping_dir=model_prediction_dir,
                                            basin_predicted_pumping_dir=os.path.join(output_dir, f'{basin}'),
                                            insitu_data_csv=cfg['actual'],
                                            output_csv=os.path.join(output_dir, basin,
                                                                    f'basinscale_pumping_{basin}.csv'),
                                            skip_processing=False)

        ################################################################################################################
        # This block collects lower and upper CI of pumping predictions
        ################################################################################################################
        if process_CIs and (model.lower() == 'ml'):
            print('\nCollecting CI predictions...')

            compile_prediction_CI(basin_code=basin, years=cfg['year'],
                                  basin_shp=cfg['shp'],
                                  prediction_CI_dir=f'../../Data_main/rasters/pumping_prediction/ML_uncertainty/{model_version}',
                                  basin_output_dir=os.path.join(output_dir, f'{basin}/low_high_CI'))

        elif process_CIs and (model.lower() != 'ml'):
            raise ValueError('\nCI predictions are available for "ML" model only')


def compile_all_basin_summaries(basinwise_output_dir, compiled_output_csv,
                                model='ML', collect_CIs=True):
    """
    Compiles and merges annual basin-scale groundwater pumping summaries across all basins.

    This function consolidates basin-level groundwater pumping outputs into a single
    annual summary CSV.

    Parameters
    ----------
    basinwise_output_dir : str
        Path to the directory containing basin-level output sub-folders.
        Each sub-folder must include a CSV file named
        'basinscale_pumping_<basin>.csv'.

    compiled_output_csv : str
        Filepath of the output CSV where the merged, basin-wide annual summary
        will be written. Intermediate temporary CSVs (original and scaled)
        are deleted after merging.

    model : {'ML', 'ANN'}, optional
        Specifies the model type. Determines whether scaled predictions and
        uncertainty compilation are performed. Default is 'ML'.

    collect_CIs : bool
            Whether to include the processing and compilation of uncertainty (confidence interval)
            rasters and CSVs. Only applicable when `model='ML'`.

    Returns
    -------
    None
        The merged annual summary CSV is written to disk.
    """
    if model.lower() not in ['ml', 'ann']:
        ValueError("Model must me either 'ML' or 'ANN'")

    basin_annual_csvs = [
        os.path.join(basinwise_output_dir, f'{b}/basinscale_pumping_{b}.csv')
        for b in basin_configs.keys()
    ]

    if collect_CIs:
        csv_list_CI = [
            os.path.join(basinwise_output_dir, f'{b}/basinscale_CI_{b}.csv')
            for b in basin_configs.keys()
        ]
    else:
        csv_list_CI = None

    compile_annual_pumping_all_basins(annual_csv_list=basin_annual_csvs, output_csv=compiled_output_csv,
                                      collect_CIs=collect_CIs, CI_csv_list=csv_list_CI)


def main(model_version, model_prediction_dir, basinwise_output_dir, compiled_output_csv, model, collect_CIs):
    """
    Main driver function for processing and compiling groundwater pumping predictions across all basins.

    This function coordinates two major tasks:
      1. Executes basin-level pumping prediction workflows using `process_all_basins()`
      2. Aggregates the basin-level results into a unified summary CSV file using
         `compile_all_basin_summaries()`.

    Parameters
    ----------
    model_version : str
        Version tag or identifier of the prediction model (e.g., 'v10').
        Used to locate the correct model outputs and uncertainty directories.

    model_prediction_dir : str
        Path to the directory containing per-year model prediction rasters
        (e.g., annual groundwater pumping predictions for all basins).

    basinwise_output_dir : str
        Output directory where basin-specific prediction and summary files will be saved.
        Each basin will have its own subfolder within this directory.

    compiled_output_csv : str
        Filepath of the final aggregated CSV that will store all compiled basin-level summaries
        (e.g., one combined annual summary table).

    model : {'ML', 'ANN'}
        Type of predictive model used. Determines which pipelines are executed downstream.

    collect_CIs : bool
        Whether to include the processing and compilation of uncertainty (confidence interval)
        rasters and CSVs. Only applicable when `model='ML'`.

    Notes
    -----
    - This function acts as a wrapper to streamline the full basin processing workflow.
    - It should be executed once per model version to regenerate all predictions and summary files.
    - CI (uncertainty) outputs are generated only for machine learning models (`model='ML'`).

    """
    # for actual prediction
    process_all_basins(model_version, model_prediction_dir, basinwise_output_dir, model, process_CIs=collect_CIs)

    # compiling annual csv
    compile_all_basin_summaries(basinwise_output_dir, compiled_output_csv, model, collect_CIs=collect_CIs,)


########################################################################################################################
# Execution
########################################################################################################################

if __name__ == '__main__':
    print(
        "\n*********************************************************************************************************************",
        "\n### Note (from Ryan): Here, we use the original (unfiltered) pumping rasters as the actual (reference) for comparison.\n"
        "This approach is intentional because our goal is to develop a pumping raster product that remains valid even when actual \n"
        "pumping data are unavailable. Therefore, for years where actual data exist, we include all available pumping values in\n"
        "the basin-scale comparison with predicted pumping, so that the model can be trusted in years when direct comparison\n"
        "is not possible.\n"
        "*********************************************************************************************************************"
    )
    # model version
    model_version_ML = 'v11'    ######################

    # compile individual basin's prediction results to individual csv + compiling all data in a single csv
    skip_compile_ML = False     ######################
    collect_CIs_for_ML = True   ######################

    if not skip_compile_ML:
        print('\n-----------------------------------------')
        print('Compiling results from ML model')
        print('-----------------------------------------')

        main(model_version=model_version_ML, model='ML',
             model_prediction_dir=f'../../Data_main/rasters/pumping_prediction/ML/{model_version_ML}/WestUS_pumping',
             basinwise_output_dir=f'../../Model_run/basin_comparison_results/ML',
             compiled_output_csv=f'../../Model_run/basin_comparison_results/annual_pumping_ML_{model_version_ML}.csv',
             collect_CIs=collect_CIs_for_ML)
