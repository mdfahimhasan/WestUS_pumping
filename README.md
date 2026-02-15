# High-resolution groundwater pumping estimation for the Western United States

Efforts to monitor groundwater pumping for irrigation in the Western United States (Western US) are hindered by the lack of comprehensive pumping records. While previous studies have developed region-specific machine learning models using limited datasets, these models are often not transferable across regions, and a groundwater pumping dataset that goes beyond local and state boundaries remains missing. In this study, we develop a regional-scale, data-driven machine learning framework to address these limitations by integrating remote sensing datasets and in-situ pumping records from Arizona, Colorado, Kansas, and Nevada. Using gridded hydroclimatic and land use variables, including effective precipitation, fraction of irrigated croplands, and evapotranspiration, the model generates spatially continuous, high-resolution (2 km, annual) historical groundwater pumping estimates from 2000 to 2023 for groundwater-dominated basins of the Western US, while predicting total irrigation in conjunctive basins. The model demonstrates good predictive performance under randomized split, with an R2 =0.62, NRMSE = 0.50, NMAE = 0.34, and PBIAS = 8.59% on the test set. Model validation over groundwater-dominated and conjunctive basins across the region shows satisfactory results. In addition, comparisons using spatial holdout analysis and power consumption-based pumping records in multiple basins indicate strong generalization capacity and spatial transferability within the study region. Our assessment identifies limited availability of in-situ pumping records and lack of surface water irrigation datasets as the primary constraints for further advancing such regional-scale frameworks. Overall, the findings highlight that regional transferability of machine learning models for predicting groundwater irrigation is achievable but contingent on holistic representation of the hydrologic system. 

## Predicted pumping and total irrigation maps
<img src="readme_figs/model_prediciton.png" height="900"/>

## Citations
- Hasan, M. F., Smith, R. G., Davenport, F. V., Majumdar, S. (2026). Extending Historical Groundwater Pumping Estimates for Major Agricultural Basins of the Western United States with Machine Learning and Satellite Products. In Prep. for Journal of Hydrology.

## Organizations
<img src="readme_figs/CSU-Signature-C-357-617.png" height="90"/> <img src="readme_figs/Official-DRI-Logo-for-Web.png" height="80"/>

## Funding
<img src="readme_figs/NASA-Logo-Large.png" height="80"/>

## Running the repository

### Repository structure
The repository has six main modules described as follows-

__1. utils -__ Utility scripts for core operations across the repository:
- `raster_ops.py` - Raster processing (read/write arrays, clipping, resampling, masking)
- `vector_ops.py` - Vector operations (buffering, clipping shapefiles, coordinate transformations)
- `stats_ops.py` - Statistical metrics (RMSE, MAE, R², NRMSE, PBIAS calculations)
- `ML_ops.py` - Machine learning operations using LightGBM (data preparation, training, hyperparameter tuning via Hyperopt, SHAP analysis, prediction)
- `DL_ops.py` - Deep learning operations using PyTorch (DataLoader, ANN model architecture, training with Optuna optimization)
- `plots.py` - Visualization utilities
- `system_ops.py` - File system operations

__2. download_preprocess -__ Scripts for data acquisition and preprocessing:
- `download.py` - Functions to download data from Google Earth Engine (GRIDMET, DAYMET products)
- `download_openET.py` - Functions to download OpenET and irrigation fraction datasets (IrrMapper, LANID)
- `preprocess.py` - Data preprocessing and compilation functions
- `dp_driver.py` - Main driver script that executes functionalities in `download.py`, `download_openET.py`, and `preprocess.py` to download and preprocess all datasets

__3. pumping -__ Pumping data processing module:
- `pumping.py` - Processes, filters, and rasterizes in-situ pumping records from Arizona, Colorado, Kansas, and Nevada. Includes well coordinate transformation, data quality filtering, and rasterization. Output serves as training data for the ML model.

__4. models -__ Core machine learning module:
- `ml_driver.py` - Main ML driver for model training, testing, and prediction using LightGBM DART
- `ml_driver_LOBO.py` - Leave-One-Basin-Out (LOBO) cross-validation driver for spatial transferability assessment
- `ml_uncertainty.py` - Bootstrap-based uncertainty quantification and confidence interval estimation
- `ann_*.py` - Artificial Neural Network model scripts (experimental; not used in final manuscript)
- Associated `.sh` scripts for HPC job submission

__5. EDA -__ Exploratory Data Analysis notebooks:
- `pumping_irrFrac_ET.ipynb` - Analysis of pumping, irrigation fraction, and evapotranspiration relationships
- `RS_vs_NASS_acres.ipynb` - Comparison of remote sensing-derived vs NASS irrigated acreage
- `valueRanges_pumping_netGW.ipynb` - Value range analysis for pumping and net groundwater data
- `allState_vs_GMD3_dist.ipynb` - Distribution comparison analysis

__6. results_analysis -__ Model evaluation and results compilation:
- `basin_compile.py` / `basin_compile_LOBO.py` - Compile basin-scale predicted and actual pumping data
- `analysis_utils.py` - Utility functions for results analysis
- `model_diagnosis.ipynb` - Model performance diagnostics
- `pumping_basin_compare_with_LOBO_v11.ipynb` - Basin-level comparison with LOBO validation
- Various notebooks for water balance analysis and result visualization

The __utils__ module does not require direct execution. Other modules should be executed using their respective driver files. __Please reach out to the authors for additional support in running this repository.__

### Execution workflow
For full model implementation, execute modules in the following order:
1. **download_preprocess** → Run `dp_driver.py` to download and preprocess all input datasets
2. **pumping** → Run `pumping.py` to process and rasterize in-situ pumping records (training data)
3. **models** → Run `ml_driver.py` for model training/prediction or `ml_driver_LOBO.py` for spatial validation
4. **results_analysis** → Use notebooks and scripts to analyze model outputs

### Dependencies
__conda environment:__ A _conda environment_, set up using [Anaconda](https://www.anaconda.com/products/individual) with Python 3.9, has been used to implement this repository. The `yml_files_env` folder contains `.yml` files to set up similar conda environments for both Linux and Windows.

__Key packages:__
- `lightgbm` - LightGBM DART regressor for ML modeling
- `hyperopt` - Bayesian hyperparameter optimization  
- `shap` - Model interpretability and feature importance
- `rasterio`, `gdal` - Geospatial raster operations
- `geopandas` - Vector data processing
- `torch` - PyTorch for deep learning (experimental ANN)
- `optuna` - Neural network hyperparameter tuning
- `earthengine-api` - Google Earth Engine data access

## Data availability
Will be added shortly.



