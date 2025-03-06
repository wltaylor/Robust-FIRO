import numpy as np
import matplotlib.pyplot as plt
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
#ramping_rate = 30868/1000 * kcfs_to_tafd # cfs to kcfs to tafd
ramping_rate = 120 * kcfs_to_tafd # cfs to kcfs to tafd

sd = '1990-10-01' 
ed = '2019-08-31'

opt_forecast = 'hefs' # either 'hefs' for actual HEFS hindcast or 'syn' for a synthetic sample
syn_samp = 1
gen_path = 'r-gen'
weight = 0.5

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
def opt_wrapper(x):  # x is an array of the decision variables
    
    risk_thresholds = create_risk_curve(x)
    ix = ((1 - risk_thresholds[:14]) * (41 - 1)).astype(np.int32)
    S, R, spill = model.simulate_EFO(firo_pool=0, ix=ix, Q=Q, Qf=Qf_summed_sorted, dowy=dowy, tocs=tocs)

    obj = model.magnitude_objective(S, R, Rmax, spill)
    #print(obj)
  
    return obj

@njit
def opt_wrapper_with_weight(x,weight):  # x is an array of the decision variables
    
    risk_thresholds = create_risk_curve(x)
    ix = ((1 - risk_thresholds[:14]) * (41 - 1)).astype(np.int32)
    S, R, spill = model.simulate_EFO(firo_pool=0, ix=ix, Q=Q, Qf=Qf_summed_sorted, dowy=dowy, tocs=tocs, S_start=0)

    obj = model.magnitude_objective(S, R, Rmax, spill, weight)
    #print(obj)
  
    return obj

@njit
def opt_wrapper_explicit(x,firo_pool,Q,Qf,dowy,tocs, weight):
    
    risk_thresholds = create_risk_curve(x)
    ix = ((1 - risk_thresholds[:14]) * (41 - 1)).astype(np.int32)
    S, R, spill = model.simulate_EFO(firo_pool, ix=ix, Q=Q, Qf=Qf, dowy=dowy, tocs=tocs, S_start=0)

    obj = model.magnitude_objective(S, R, Rmax, spill, weight)
    return obj

#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
# Default case optimization
# Q,Qf,dowy,df_idx = model.extract(sd,ed,forecast_type=opt_forecast,gen_path=gen_path,syn_sample=syn_samp)
# tocs = model.get_tocs(dowy)
# Qf = Qf[:,:,:14] # just use 14 lead days

# Qf_summed = np.cumsum(Qf, axis=2)
# Qf_summed_sorted = np.sort(Qf_summed, axis = 1)

# weight = 0.0001
# bounds = [(0,1)]*14 # decision variable bounds
# opt = DE(lambda x: opt_wrapper_with_weight(x, weight), bounds = bounds, disp = True, maxiter = 1000, seed = 0, polish = True)
# print(opt)

# # what does the risk curve look like?
# plt.plot(create_risk_curve(opt.x))
# plt.xlabel('Lead time (days)')
# plt.ylabel('Risk %')
# plt.ylim(0,1)
# plt.show()
#%%
#save trained parameters
# if policy == 'perfect':
#     np.savetxt('results/perfect_risk_thresholds.csv', opt.x, delimiter=",")
# if policy == 'firo':
#     np.savetxt('results/EFO_risk_thresholds.csv', opt.x, delimiter=",")
# if policy == 'held_out_1997':
#     np.savetxt('results/EFO_risk_thresholds_no1997.csv', opt.x, delimiter=",")


# sum and sort the forecasts    
# Qf_summed = np.cumsum(Qf, axis=2)
# Qf_summed_sorted = np.sort(Qf_summed, axis = 1)
# firo_pool = 0
# bounds = [(0,1)]*14 # decision variable bounds
# #opt = DE(lambda x: opt_wrapper_explicit(x, firo_pool, Q_masked, Qf_summed_sorted, dowy_masked, tocs_masked), bounds = bounds, disp = True, maxiter = 1000, seed = 0, polish = True)
# opt = DE(lambda x: opt_wrapper_explicit(x, firo_pool, Q, Qf_summed_sorted, dowy, tocs), bounds = bounds, disp = True, maxiter = 1000, seed = 0, polish = True)

# print(opt)

# # what does the risk curve look like?
# plt.plot(create_risk_curve(opt.x))
# plt.xlabel('Lead time (days)')
# plt.ylabel('Risk %')
# plt.ylim(0,1)
# plt.show()
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
# Random seed comparison (HEFS, fully trained)
Q,Qf,dowy,df_idx = model.extract(sd,ed,forecast_type=opt_forecast,gen_path=gen_path,syn_sample=syn_samp)
tocs = model.get_tocs(dowy)
Qf = Qf[:,:,:14] # just use 14 lead days

Qf_summed = np.cumsum(Qf, axis=2)
Qf_summed_sorted = np.sort(Qf_summed, axis = 1)
full_curves = []
firo_pool = 0
bounds = [(0,1)]*14 # decision variable bounds
for i in range(0,10):
    opt = DE(lambda x: opt_wrapper_explicit(x, firo_pool, Q, Qf_summed_sorted, dowy, tocs, weight), bounds = bounds, disp = True, maxiter = 1000, seed = i, polish = True)

    print(i)
    full_curves.append(create_risk_curve(opt.x))

    # save trained parameters
    np.savetxt(f'results/EFO_policies/EFO_risk_thresholds_weight{weight}_seed{i}.csv', opt.x, delimiter=",")

#%%
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
# Random seed comparison (HEFS, 1997 held out)

Q,Qf,dowy,df_idx = model.extract(sd,ed,forecast_type=opt_forecast,gen_path=gen_path,syn_sample=syn_samp)
tocs = model.get_tocs(dowy)
Qf = Qf[:,:,:14] # just use 14 lead days

# create a mask for WY 1997
df = pd.read_csv('data/observed_flows.csv', parse_dates=True)
df['ORDC1'] = df['ORDC1']*kcfs_to_tafd
df['date'] = pd.to_datetime(df['Date'])
df.sort_values(by='date', inplace=True)
df = df.loc[(df['date'] >= sd) & (df['date'] <= ed)]
df['water_year'] = df['date'].apply(lambda x: x.year + 1 if x.month >= 10 else x.year)
unique_water_years = df['water_year'].unique()
water_years = df['water_year'].values
unique_water_years = np.unique(water_years)
mask = (df.water_year != 1997)

# replace the data with the masked version
Q_masked = Q[mask]
Qf_masked = Qf[mask]
dowy_masked = dowy[mask]
df_idx_masked = df_idx[mask]
tocs_masked = tocs[mask]

Qf_summed = np.cumsum(Qf_masked, axis=2)
Qf_summed_sorted = np.sort(Qf_summed, axis = 1)
full_curves = []
firo_pool = 0
bounds = [(0,1)]*14 # decision variable bounds
for i in range(0,10):
    opt = DE(lambda x: opt_wrapper_explicit(x, firo_pool, Q_masked, Qf_summed_sorted, dowy_masked, tocs_masked, weight), bounds = bounds, disp = True, maxiter = 1000, seed = i, polish = True)

    print(i)
    full_curves.append(create_risk_curve(opt.x))

    # save trained parameters
    np.savetxt(f'results/EFO_policies/EFO_risk_thresholds_no1997_weight{weight}_seed{i}.csv', opt.x, delimiter=",")

#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
#%% train on synthetic ensembles 1-100
firo_pool = 0
Q,_,dowy,df_idx = model.extract(sd,ed,forecast_type=opt_forecast,gen_path=gen_path,syn_sample=syn_samp)
tocs = model.get_tocs(dowy)

for i in range(0,100):
    print(i)
    opt_forecast = 'syn' # either 'hefs' for actual HEFS hindcast or 'syn' for a synthetic sample
    syn_samp = i
    gen_path = 'r-gen'
    policy = 'firo' # either 'firo', 'perfect', or 'held_out_1997'

    NL=14
    ds_syn = xr.open_dataset('data/Qf-syn_pcnt=0.99_ORDC1_5fold-test.nc')
    da_syn = ds_syn['syn'].sel(site=2, lead=slice(0,NL-1), date=slice(sd, ed))
    num_syn = ds_syn['syn'].shape[-1]    
    Qf = da_syn.sel(ensemble=i).values * kcfs_to_tafd
    Qf_summed = np.cumsum(Qf, axis=2)
    Qf_summed_sorted = np.sort(Qf_summed, axis = 1)
    
    bounds = [(0,1)]*14 
    opt = DE(lambda x: opt_wrapper_explicit(x, firo_pool, Q, Qf_summed_sorted, dowy, tocs, weight), bounds = bounds, disp = True, maxiter = 1000, seed = 0, polish = True)

    np.savetxt(f'results/EFO_policies/synthetic_risk_thresholds_weight{weight}_{i}.csv', opt.x, delimiter=",")

#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
#%% train on synthetic ensembles 1-100 with 1997 held out
firo_pool = 0

ds_syn = xr.open_dataset('data/Qf-syn_pcnt=0.99_ORDC1_5fold-test.nc')
NL=14
da_syn = ds_syn['syn'].sel(site=2, lead=slice(0,NL-1), date=slice(sd, ed))
num_syn = ds_syn['syn'].shape[-1]    

Q,Qf,dowy,df_idx = model.extract(sd,ed,forecast_type=opt_forecast,gen_path=gen_path,syn_sample=syn_samp)
tocs = model.get_tocs(dowy)

# create a mask
df = pd.read_csv('data/observed_flows.csv', parse_dates=True)
df['ORDC1'] = df['ORDC1']*kcfs_to_tafd
df['date'] = pd.to_datetime(df['Date'])
df.sort_values(by='date', inplace=True)
df = df.loc[(df['date'] >= sd) & (df['date'] <= ed)]
df['water_year'] = df['date'].apply(lambda x: x.year + 1 if x.month >= 10 else x.year)
unique_water_years = df['water_year'].unique()
water_years = df['water_year'].values
unique_water_years = np.unique(water_years)
mask = (df.water_year != 1997)

# replace the data with the masked version
Q = Q[mask]
dowy = dowy[mask]
df_idx = df_idx[mask]
tocs = tocs[mask]

for i in range(0,100):
    print(i)
    Qf = da_syn.sel(ensemble=i).values * kcfs_to_tafd
    Qf_summed = np.cumsum(Qf, axis=2)
    Qf_summed_sorted = np.sort(Qf_summed, axis = 1)
    Qf_summed_sorted = Qf_summed_sorted[mask]
    bounds = [(0,1)]*14 
    opt = DE(lambda x: opt_wrapper_explicit(x, firo_pool, Q, Qf_summed_sorted, dowy, tocs, weight), bounds = bounds, disp = True, maxiter = 1000, seed = 0, polish = True)
    np.savetxt(f'results/EFO_policies/synthetic_risk_thresholds_no1997_weight{weight}_{i}.csv', opt.x, delimiter=",")

