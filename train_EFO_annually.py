import numpy as np
import pandas as pd
import xarray as xr
import model
from scipy.optimize import differential_evolution as DE
from time import localtime, strftime
from numba import njit
from util import *

#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
# Scalar variables
kcfs_to_tafd = 2.29568411*10**-5 * 86400
K = 3524 # TAF
Rmax = 150 * kcfs_to_tafd # estimate - from MBK
ramping_rate = 30868/1000 * kcfs_to_tafd # cfs to kcfs to tafd

sd = '1990-10-01' 
ed = '2019-08-31'

opt_forecast = 'hefs' # either 'hefs' for actual HEFS hindcast or 'syn' for a synthetic sample
syn_samp = 1
gen_path = 'r-gen'
policy = 'firo' # either 'firo', 'perfect', or 'held_out_1997'

#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
# Read in data
Q,Qf,dowy,df_idx = model.extract(sd,ed,forecast_type=opt_forecast,gen_path=gen_path,syn_sample=syn_samp)
tocs = model.get_tocs(dowy)
Qf = Qf[:,:,:14] # just use 14 lead days
Qf_summed = np.cumsum(Qf, axis=2)
Qf_summed_sorted = np.sort(Qf_summed, axis=1)
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
# Functions

@njit
def create_risk_curve(x):
	# convert 0-1 values to non-decreasing risk curve
    x_copy = np.copy(x)
    for i in range(1, len(x_copy)):
        x_copy[i] = x_copy[i-1] + (1 - x_copy[i-1]) * x_copy[i]
    return x_copy

@njit
def opt_wrapper_explicit(x,firo_pool,Q,Qf,dowy,tocs, weight):
    
    risk_thresholds = create_risk_curve(x)
    ix = ((1 - risk_thresholds[:14]) * (41 - 1)).astype(np.int32)
    S, R, spill = model.simulate_EFO(firo_pool, ix=ix, Q=Q, Qf=Qf, dowy=dowy, tocs=tocs, S_start=0)

    obj = model.magnitude_objective(S, R, Rmax, spill, weight)
    return obj

#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
# optimization
# mask so that each year is training individually
df = pd.read_csv('data/observed_flows.csv', parse_dates=True)
df['ORDC1'] = df['ORDC1']*kcfs_to_tafd
df['date'] = pd.to_datetime(df['Date'])
df.sort_values(by='date', inplace=True)
df = df.loc[(df['date'] >= sd) & (df['date'] <= ed)]
df['water_year'] = df['date'].apply(lambda x: x.year + 1 if x.month >= 10 else x.year)
unique_water_years = df['water_year'].unique()
water_years = df['water_year'].values
unique_water_years = np.unique(water_years)

for year in unique_water_years:
	print(year)

	# create the mask
	mask = (df.water_year == year)

	# apply the mask to the data
	Q_mask = Q[mask]
	Qf_summed_sorted_mask = Qf_summed_sorted[mask]
	dowy_mask = dowy[mask]
	tocs_mask = tocs[mask]

	# optimize
	firo_pool = 0
	bounds = [(0,1)]*14
	policies = pd.DataFrame()

	for i in range(0,9):
		opt = DE(lambda x: opt_wrapper_explicit(x, firo_pool, Q_mask, Qf_summed_sorted_mask, dowy_mask, tocs_mask, weight=0.0001), bounds = bounds, disp=True, polish=True, maxiter=1000, seed=i)
		print(opt)
		# save policy
		policies = pd.concat([policies, pd.DataFrame([opt.x])], ignore_index=True)

	policies.to_csv(f'results/EFO_policies/EFO_risk_thresholds_{year}.csv', index=False)





