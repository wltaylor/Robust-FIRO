# Robust-FIRO
Files supporting the paper "Comparing the robustness of forecast-informed reservoir operating policies under forecast uncertainty and hydrologic extremes". The experiment simulates three FIRO models against scaled versions of four floods (1986, 1997, 2006, 2017) at Oroville reservoir, California, analyzing how each model performs by measuring the volume of flood pool required to safely manage the event.

### Requirements: numpy, pandas, xarray, scipy, cvxpy

# Steps:
### Load Required Data:
- Forecast arrays need to be loaded into the data folder. There should be a file for HEFS (Qf-hefs.nc), HEFS 1986 (Qf_hefs86.nc), Perfect forecasts (perfect_forecast.nc) and the 100 synthetic forecasts (Qf-syn_pcnt=0.99_ORDC1_5fold-test.nc). Note to Jon: my Box account will cease to exist after I graduate, do you want to host the data? I could also upload it to Zenodo.
- Historical medians .csv file is already in the data folder. This is used for the Baseline MPC policy.
- Observed flows .csv file is already in the data folder. All models use this file.

### Train the EFO Policies:
- The main experiment EFO policies are trained with 'train_EFO.py'. There are code blocks for the HEFS-trained EFO (10 random seeds), HEFS-trained EFO with 1997 held out (10 random seeds), 100 synthetic ensembles, and 100 synthetic ensembles with 1997 held out.
- The pre-trained EFO policies are saved in 'results/EFO_policies/' as .csv files, with separately trained policies for each weight value.
- HEFS-trained EFO are saved in the format: 'EFO_risk_thresholds_weight{weight}_seed{i}.csv'
- HEFS-trained EFO with 1997 held out are saved in the format: 'EFO_risk_thresholds_no1997_weight{weight}_seed{i}.csv'
- After training the HEFS policies with the ten random seeds, the median policy for both the fully trained and 1997 held out case is created with the file 'find_median_EFO.py'. The resulting median policies are used in the simulation step and are saved in the format: 'EFO_risk_thresholds_weight{weight}.csv', and 'EFO_risk_thresholds_no1997_weight{weight}.csv', notably the same as the random seed policies but without a numeric indicator for seed.
- There are 100 synthetic EFO policies. They are saved in the format: 'synthetic_risk_thresholds_weight{weight}_{i}.csv'
- There are 100 synthetic EFO with 1997 held out policies. They are saved in the format: 'synthetic_risk_thresholds_no1997_weight{weight}_{i}.csv'
- After training the synthetic policies, the median policy for both the fully trained and 1997 held out case is created with the file 'find_median_EFO.py'. The resulting median policies are used in the simulation step and are saved in the format: 'synthetic_risk_thresholds_weight{weight}.csv', and 'synthetic_risk_thresholds_no1997_weight{weight}.csv', notably the same as the 100 ensemble policies but without a numeric indicator.
- The file 'train_EFO_annually.py' trains 10 random seeds for each year individually, creating the policies shown in SI Figure 20. The policies are saved in the format: 'EFO_risk_thresholds_{year}.csv'. All ten seeds are saved in the same .csv for each year.
- The file 'train_EFO_scaled.py' trains EFO policies using scaled inflow and forecast data, creating the policies shown in SI Figure 25. The policies are saved in the format: 'EFO_risk_thresholds_scale{scale}_weight{weight}_seed{i}.csv'. The scaling factors for each event are based on the values found for the 1997 flood.

### Simulate the models:
- Each flood has a corresponding file to simulate the models, named 'estimate_required_storage_{year}.py'
- 'model.py' contains the functions to run the EFO models and the Cumulative Method. 
- The first block of code includes a list of the scaling factors for that particular event. There is also a weight scalar input that adjusts the drawdown weight value for all relevant models (MPC and EFO variants).
- The MPC simulations are run for the baseline, perfect, HEFS, and synthetic forecasts. These are saved in one .csv file per scaled event, in the results folder, formatted: 'results/storage_estimates_{year}/res_df_mpc{event}_weight{weight}.csv'. These use the class defined in 'MPCProblem.py'.
- Next are the HEFS-trained EFO simulations. The median HEFS trained EFO policy is loaded and used to perform the simulations with HEFS forecasts and the 100 synthetic ensembles. The median HEFS trained EFO policy with 1997 held out is also loaded and used to simulate the HEFS forecasts and the 100 synthetic ensembles. The storage and release values are saved into one .csv file formatted: 'results/storage_estimates_{year}/res_dfs_hefs_{event}_weight{weight}.csv'.
- Next are the synthetic-trained EFO policies. For both the fully-trained and held out case, the median synthetic policy is simulated with the HEFS forecast. Then, the script loops through all 100 synthetic ensembles and tests their optimized policy against a randomly selected synthetic ensemble, ensuring that it is not the same synthetic ensemble that the policy was trained on (this is accomplished with the 'choose_excluding' function defined at the beginning of the script). The resulting storage and release values are saved in the format: 'results/storage_estimates_{year}/res_df_syn_train_{event}_weight{weight}.csv' and 'results/storage_estimates_{year}/res_df_syn_train_no1997_{event}_weight{weight}.csv'.
- Finally, the Cumulative Method is simulated in the same manner, first testing the model against HEFS and then looping through the 100 synthetic ensembles. The results are saved in the format: 'results/storage_estimates_{year}/res_df_cumulative_{event}.csv'. Since the Cumulative Method is not affected by a weighting value it does not have a weight label in the results name.

### Create figures:
- 
