#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 12:34:48 2024

@author: williamtaylor
"""

import numpy as np 
import matplotlib.pyplot as plt
plt.style.use('default')
import pandas as pd
import xarray as xr
import model
import ecrps_functions
import matplotlib.patches as mpatches
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

kcfs_to_tafd = 2.29568411*10**-5 * 86400
Rmax = 150 * kcfs_to_tafd
ramp_rate = 120 * kcfs_to_tafd # limit in either direction, per day 
# this ramping value does not change the results much
full_output = True
run_synth = True
site = 'ORDC1'
NL = 14 # could use up to 30
start_date = '1986-02-02' 
end_date = '1986-02-16' # model runs until (end_date - NL)
K = 3524
#scaling_factor = 1.5030 # 300 year
scaling_factor = 1

df = pd.read_csv('data/observed_flows.csv', index_col=0, parse_dates=True)[start_date:end_date]
Q = df[site].values * kcfs_to_tafd * scaling_factor

def calculate_forecast_bias(Q, Qf, percentile):
    """
    Calculate the forecast bias as a percentage of observed inflow.

    Parameters:
    Q (numpy.ndarray): Observed inflows with shape (days,).
    Qf (numpy.ndarray): Forecasted inflows with shape (days, ensemble, lead).

    Returns:
    numpy.ndarray: Mean bias (%) for each lead day.
    """
    lead_days = Qf.shape[2]  # Number of lead days
    max_valid_days = Q.shape[0] - lead_days  # Ensure forecast and observed lengths match
    bias = np.zeros((max_valid_days, lead_days))  # Bias shape: [valid_days, lead]

    # Calculate the mean forecasted inflow across ensembles (axis 1)
    Qf_mean = np.percentile(Qf, percentile, axis=1)

    # For each lead day, calculate the bias using the corresponding future observed inflow
    for lead in range(lead_days):
        # Compare the forecast for day i with the observed inflow for day i+1
        bias[:, lead] = (Qf_mean[:max_valid_days, lead] - Q[lead+1:max_valid_days+lead+1]) / Q[lead+1:max_valid_days+lead+1] * 100

    # Calculate the mean bias across all days for each lead time
    mean_bias = np.mean(bias, axis=0)
    
    return mean_bias

def peak_bias(Q, Qf):
    """
    Calculate the forecast bias for only the peak inflow, as a percentage of the observed inflow
    
    Parameters:
    Q (numpy.ndarray): Observed inflows with shape (days).
    Qf (numpy.ndarray): Forecasted inflows with shape (days, trace, lead).
    
    Returns:
    numpy.ndarray: bias (%) for each lead day
    """
    bias_list = []
    for i in range(0, Qf.shape[2]):
        lead = i
        mean = np.mean(Qf[:-lead-1, :, lead], axis=1)
        bias_lead = (mean[-1] - Q[-1])/(Q[-1])*100
        bias_list.append(bias_lead)
    
    return np.asarray(bias_list)
    
#%% forecast bias and ecrps
years = {
    1986: {'start': '1986-01-29', 'end': '1986-03-10', 'scale': 1.91, 'plot_end': '1986-02-25', 'bias_start':'1986-02-02', 'bias_end':'1986-02-16'}, 
    1997: {'start': '1996-12-15', 'end': '1997-01-30', 'scale': 1.3374, 'plot_end': '1997-01-12', 'bias_start':'1996-12-19', 'bias_end':'1997-01-02'}, 
    2006: {'start': '2005-12-15', 'end': '2006-01-30', 'scale': 2.59, 'plot_end': '2006-01-15', 'bias_start':'2005-12-18', 'bias_end':'2006-01-01'}, 
    2017: {'start': '2017-01-20', 'end': '2017-03-10', 'scale': 2.33, 'plot_end': '2017-02-27', 'bias_start':'2017-01-27', 'bias_end':'2017-02-10'}, 
    }

year_scores = {1986:{}, 1997:{}, 2006:{}, 2017:{}}

leads = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
for i,year in enumerate(years):
    opt_forecast = 'hefs'
    gen_path = 'hefs-86' if year == 1986 else 'r-gen'
    syn_samp = 1 # placeholder
    Q,Qf_hefs,dowy,df_idx = model.extract(years[year]['start'], 
                                          years[year]['end'], 
                                          forecast_type=opt_forecast, 
                                          gen_path=gen_path, 
                                          syn_sample=syn_samp)
    # scale Q and Hefs
    Q = Q*years[year]['scale']
    Qf_hefs = Qf_hefs*years[year]['scale']

    hefs_cleaned = ecrps_functions.onesamp_forecast_rearrange(Qf_hefs)

    hefs_scores = []
    for j,lead in enumerate(leads):
        score = ecrps_functions.onesamp_ecrps(hefs_cleaned[:,:,lead], Q, (0.50,1), forc_sort=False)[1]
        hefs_scores.append(score)
    year_scores[year]['HEFS'] = hefs_scores
    
    ds_syn = xr.open_dataset('data/Qf-syn_pcnt=0.99_ORDC1_5fold-test.nc') 
    da_syn = ds_syn['syn'].sel(site=2, lead=slice(0,13), date=slice(years[year]['start'], years[year]['end']))
    da_syn = da_syn.transpose('ensemble','date','trace','lead')
    da_syn = da_syn * kcfs_to_tafd * years[year]['scale']
    syn_cleaned = ecrps_functions.multisamp_forecast_rearrange(da_syn)
    
    syn_scores = []
    for j,lead in enumerate(leads):
        score = ecrps_functions.multisamp_ecrps(syn_cleaned[:,:,:,lead], Q, (0.50,1), par=False, forc_sort=True)
        syn_scores.append(score)
    year_scores[year]['syn'] = syn_scores

bias_scores = {1986:{}, 1997:{}, 2006:{}, 2017:{}}
leads = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]

for i,year in enumerate(years):
    opt_forecast = 'hefs'
    gen_path = 'hefs-86' if year == 1986 else 'r-gen'
    syn_samp = 1 # placeholder
    Q,Qf_hefs,dowy,df_idx = model.extract(years[year]['bias_start'], 
                                          years[year]['bias_end'], 
                                          forecast_type=opt_forecast, 
                                          gen_path=gen_path, 
                                          syn_sample=syn_samp)
    Qf_hefs = Qf_hefs[:,:,:14]
    # scale Q and Hefs
    #Q = Q*years[year]['scale']
    #Qf_hefs = Qf_hefs*years[year]['scale']
    score = peak_bias(Q, Qf_hefs)
    bias_scores[year]['HEFS'] = score

    ds_syn = xr.open_dataset('data/Qf-syn_pcnt=0.99_ORDC1_5fold-test.nc') 
    da_syn = ds_syn['syn'].sel(site=2, lead=slice(0,13), date=slice(years[year]['bias_start'], years[year]['bias_end']))
    da_syn = da_syn.transpose('ensemble','date','trace','lead')
    da_syn = da_syn * kcfs_to_tafd #* years[year]['scale']

    syn_scores = np.zeros((100,14))    
    for i in range(0,100):
        Qf_syn = da_syn.sel(ensemble=i).values
        syn_scores[i,:] = peak_bias(Q, Qf_syn)
    
    bias_scores[year]['syn'] = syn_scores

#%%
fig, ax = plt.subplots(2,4,figsize=(12,6))
offsets = np.linspace(-1,1,14)
leads = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
lead_labels = np.arange(1,15,1)
alpha_value = 0.5
for i, year in enumerate(years):
    for j, lead in enumerate(leads):
        # Scatter plot for HEFS bias
        ax[0, i].scatter(offsets[j], bias_scores[year]['HEFS'][lead], marker='*', c='black', s=125, zorder=5)
        # Boxplot for synthetic forecast bias
        ax[0, i].boxplot(bias_scores[year]['syn'][:, lead], positions=[offsets[j]], widths=0.1,
                         boxprops=dict(color='blue', alpha=alpha_value), 
                         capprops=dict(color='blue', alpha=alpha_value), 
                         whiskerprops=dict(color='blue', alpha=alpha_value), 
                         flierprops=dict(markeredgecolor='blue', alpha=alpha_value),
                         medianprops=dict(color='blue', alpha=alpha_value), zorder=4)

for i, year in enumerate(years):
    for j, lead in enumerate(leads):
        # Scatter plot for HEFS ECRPS
        ECRPS = year_scores[year]['HEFS'][lead]
        ax[1, i].scatter(offsets[j], ECRPS, c='black', marker="*", s=125, zorder=5)
        # Boxplot for synthetic forecast ECRPS
        ECRPS_syn = year_scores[year]['syn'][lead]
        ax[1, i].boxplot(ECRPS_syn, positions=[offsets[j]], widths=0.1, 
                         boxprops=dict(color='blue', alpha=alpha_value), 
                         capprops=dict(color='blue', alpha=alpha_value), 
                         whiskerprops=dict(color='blue', alpha=alpha_value), 
                         flierprops=dict(markeredgecolor='blue', alpha=alpha_value),
                         medianprops=dict(color='blue', alpha=alpha_value), zorder=4)

    ax[0,i].set_ylim(-100,50)
    ax[0,i].set_xlim(-1.1,1.1)  
    ax[1,i].set_ylim(0,210)
    ax[1,i].set_xlim(-1.1,1.1)

    ax[0,i].set_title(year, fontweight='bold', fontsize=14)



ax[0,0].set_ylabel('Bias', fontweight='bold', fontsize=14)    
ax[1,0].set_ylabel('ECRPS', fontweight='bold', fontsize=14)

for i in range(0,4):
    ax[0,i].set_xticks(offsets)
    ax[0,i].tick_params(labelbottom=False)
    ax[0,i].grid(True)
    ax[1,i].set_xticks(offsets)
    ax[1,i].set_xticklabels(lead_labels)

for i in range(0,4):
    if i > 0:
        ax[0,i].tick_params(labelleft=False)
    ax[0,i].grid(True, alpha=0.5)
    
    if i > 0:
        ax[1,i].tick_params(labelleft=False)
    ax[1,i].grid(True, alpha=0.5)

hefs_legend = plt.Line2D([0], [0], color='black', marker='*', markersize = 10, lw=0, label='HEFS')
syn_legend = mpatches.Patch(edgecolor='blue', facecolor='none', label='Synthetic')

ax[0,0].legend(handles=[hefs_legend, syn_legend], edgecolor='black')

#fig.legend(handles=[hefs_legend, syn_legend], loc='upper center', bbox_to_anchor=(0.52, -0.01), ncol=2)
#fig.suptitle('Forecast Bias and ECRPS for HEFS and 100 Synthetic Samples', fontweight='bold', fontsize=16)    
plt.tight_layout()
plt.savefig('/Users/williamtaylor/Documents/Github/Robust-FIRO/figures/Figure_7.pdf', format='pdf', bbox_inches='tight', transparent=False)
plt.show()

#%% plot the st dev between ensembles at each lead time

fig, ax = plt.subplots(1,4, figsize=(12,4))
leads = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
for i,year in enumerate(years):
    opt_forecast = 'hefs'
    gen_path = 'hefs-86' if year == 1986 else 'r-gen'
    syn_samp = 1 # placeholder
    Q,Qf_hefs,dowy,df_idx = model.extract(years[year]['start'], 
                                          years[year]['end'], 
                                          forecast_type=opt_forecast, 
                                          gen_path=gen_path, 
                                          syn_sample=syn_samp)
        
    Qf_hefs = Qf_hefs[:,:,:14]
    cv_per_lead = np.std(Qf_hefs, axis=1)/np.mean(Qf_hefs, axis=1)
    mean_cv_per_lead = np.mean(cv_per_lead, axis=0)
    
    ax[i].scatter(leads, mean_cv_per_lead, marker='*', c='green', s=100, zorder=2, edgecolor='black')
    
    ds_syn = xr.open_dataset('data/Qf-syn_pcnt=0.99_ORDC1_5fold-test.nc') # replaced with newest synthetic dataset
    da_syn = ds_syn['syn'].sel(site=2, lead=slice(0,13), date=slice(years[year]['start'], years[year]['end']))
    da_syn = da_syn.transpose('ensemble','date','trace','lead')
    
    for j in range(0,100):
        Qf_syn = da_syn.sel(ensemble=j)*kcfs_to_tafd

        cv_per_lead = np.std(Qf_syn, axis=1)/np.mean(Qf_syn, axis=1)
        mean_cv_per_lead = np.mean(cv_per_lead, axis=0)
        
        ax[i].scatter(leads, mean_cv_per_lead, c='red', alpha=0.1)

    ax[i].set_title(year, fontweight='bold', fontsize=12)
    ax[i].set_xticks(leads)
    ax[i].set_xticklabels(leads)
    if i > 0:
        ax[i].tick_params(axis='y', left=False, labelleft=False)

ax[0].set_ylabel('Coefficient of Var')

hefs_legend = plt.Line2D([0], [0], color='green', marker='*', lw=0, label='HEFS')
syn_legend = plt.Line2D([0],[0], color='red', marker = 'o', lw=0, label='Synthetic')

fig.legend(handles=[hefs_legend, syn_legend], loc='upper center', bbox_to_anchor=(0.52, -0.01), ncol=2)
fig.suptitle('Ensemble Coefficient of Variation by Lead Day', fontweight='bold', fontsize=16)    

plt.tight_layout()
plt.show()
#%% first differences

fig, ax = plt.subplots(1,4, figsize=(12,4))
leads = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
for i,year in enumerate(years):
    opt_forecast = 'hefs'
    gen_path = 'hefs-86' if year == 1986 else 'r-gen'
    syn_samp = 1 # placeholder
    Q,Qf_hefs,dowy,df_idx = model.extract(years[year]['start'], 
                                          years[year]['end'], 
                                          forecast_type=opt_forecast, 
                                          gen_path=gen_path, 
                                          syn_sample=syn_samp)
       
    # HEFS
    for lead in leads:
        first_diff = np.diff(Qf_hefs[:,:,lead], axis=0)
        first_diff_std = np.std(first_diff, axis=0)
        mean_std_ensembles = np.mean(first_diff_std, axis=0)
        ax[i].scatter(lead, mean_std_ensembles, c='green', marker='*', zorder=2, s=100, edgecolor='black')
    
    
    ds_syn = xr.open_dataset('data/Qf-syn_pcnt=0.99_ORDC1_5fold-test.nc') # replaced with newest synthetic dataset
    da_syn = ds_syn['syn'].sel(site=2, lead=slice(0,13), date=slice(years[year]['start'], years[year]['end']))
    da_syn = da_syn.transpose('ensemble','date','trace','lead')
    
    for j in range(0,100):
    # select a synthetic forecast  
        Qf_syn = da_syn.sel(ensemble=j)*kcfs_to_tafd
        for lead in leads:
            first_diff = np.diff(Qf_syn[:,:,lead], axis=0)
            first_diff_std = np.std(first_diff, axis=0)
            mean_std_ensembles = np.mean(first_diff_std, axis=0)
            ax[i].scatter(lead, mean_std_ensembles, c='red', alpha=0.15)

    ax[i].set_xticks(leads)
    ax[i].set_xticklabels(leads)
    ax[i].set_title(year, fontweight='bold')
    ax[i].set_ylim(10,120)

for i in range(0,4):
    if i > 0:
        ax[i].tick_params(labelleft=False)
    #ax[0,i].grid(True)
    
ax[0].set_ylabel('First Diff St Dev')

hefs_legend = plt.Line2D([0], [0], color='green', marker='*', lw=0, label='HEFS')
syn_legend = plt.Line2D([0],[0], color='red', marker = 'o', lw=0, label='Synthetic')

fig.legend(handles=[hefs_legend, syn_legend], loc='upper center', bbox_to_anchor=(0.52, -0.01), ncol=2)
fig.suptitle('Standard Deviation of First Differences by Lead Day', fontweight='bold', fontsize=16)    

plt.tight_layout()
plt.show()


