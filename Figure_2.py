#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 14:37:40 2025

@author: williamtaylor
"""
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
plt.style.use('default')
import xarray as xr
import model
import matplotlib.dates as mdates
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
kcfs_to_tafd = 2.29568411*10**-5 * 86400

opt_forecast = 'hefs'
gen_path = 'r-gen'
syn_samp = 1 # placeholder
sd = '1996-12-28'
ed = '1997-01-10'
Q,Qf_hefs,dowy,df_idx = model.extract(sd,ed,forecast_type=opt_forecast,gen_path=gen_path,syn_sample=syn_samp)

fig, axes = plt.subplots(1, 2, figsize=(12,4))
axes = axes.flatten()

axes[0].plot(Q[1:], c='black', alpha=0.75, label='Observed Inflow')
axes[0].plot(Qf_hefs[0,0,:14], c='gray',alpha=0.25, label='Forecast Inflow')
axes[0].plot(Qf_hefs[0,:,:14].T, c='gray',alpha=0.25)

lead_positions = np.arange(0,14,1)
lead_labels = np.arange(1,15,1)
axes[0].set_xticks(lead_positions, lead_labels)


axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

axes[0].axvline(4, c='orange', linestyle='--')
axes[0].annotate('1997-01-05', (5, 550), c='orange')
axes[0].set_xlabel('Forecast Lead Day', fontweight='bold')
axes[0].set_ylabel('Inflow (TAF)', fontweight='bold')
axes[0].set_title('(a) Hydrologic Ensemble Forecasts', fontweight='bold', loc='left', fontsize=16)
axes[0].set_ylim(0,700)
axes[0].legend(edgecolor='black')

# Compute percentiles for shading
ensemble=15
ds_syn = xr.open_dataset('data/Qf-syn_pcnt=0.99_ORDC1_5fold-test.nc')
da_syn = ds_syn['syn'].sel(site=2, lead=slice(0,14-1), date=slice(sd, ed))
Qf_syn = da_syn.sel(ensemble=ensemble).values * kcfs_to_tafd

q10 = np.percentile(Qf_syn[0,:,:14], 5, axis=0)
q90 = np.percentile(Qf_syn[0,:,:14], 95, axis=0)
q25 = np.percentile(Qf_syn[0,:,:14], 25, axis=0)
q75 = np.percentile(Qf_syn[0,:,:14], 75, axis=0)
qmean = np.mean(Qf_syn[0,:,:14], axis=0)

lead_positions = np.arange(0,14,1)
lead_labels = np.arange(1,15,1)

axes[1].plot(Q[1:], c='black', alpha=0.75, label='Observed Inflow')

# Plot median forecast
axes[1].plot(qmean, c='blue', alpha=1, label='Ensemble 1')

# Plot shaded area for forecast uncertainty
axes[1].fill_between(lead_positions, q10, q90, color='blue', alpha=0.15)
axes[1].fill_between(lead_positions, q25, q75, color='blue', alpha=0.3)


axes[1].set_xticks(lead_positions, lead_labels)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

# Add vertical line and annotation
axes[1].axvline(4, c='orange', linestyle='--')
axes[1].annotate('1997-01-05', (5, 550), c='orange')


ensemble=4
ds_syn = xr.open_dataset('data/Qf-syn_pcnt=0.99_ORDC1_5fold-test.nc')
da_syn = ds_syn['syn'].sel(site=2, lead=slice(0,14-1), date=slice(sd, ed))
Qf_syn = da_syn.sel(ensemble=ensemble).values * kcfs_to_tafd

q10 = np.percentile(Qf_syn[0,:,:14], 5, axis=0)
q90 = np.percentile(Qf_syn[0,:,:14], 95, axis=0)
q25 = np.percentile(Qf_syn[0,:,:14], 25, axis=0)
q75 = np.percentile(Qf_syn[0,:,:14], 75, axis=0)
qmean = np.mean(Qf_syn[0,:,:14], axis=0)

lead_positions = np.arange(0,14,1)
lead_labels = np.arange(1,15,1)

# Plot median forecast
axes[1].plot(qmean, c='red', alpha=1, label='Ensemble 2')

# Plot shaded area for forecast uncertainty
axes[1].fill_between(lead_positions, q10, q90, color='red', alpha=0.15)
axes[1].fill_between(lead_positions, q25, q75, color='red', alpha=0.3)

axes[1].set_xticks(lead_positions, lead_labels)
axes[1].set_title('(b) Synthetic Forecasts', fontweight='bold', loc='left', fontsize=16)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

axes[1].set_xlabel('Forecast Lead Day', fontweight='bold')
axes[1].set_ylabel('Inflow (TAF)', fontweight='bold')
axes[1].set_ylim(0,700)
axes[1].legend(edgecolor='black')
plt.savefig('/Users/williamtaylor/Documents/Github/Robust-FIRO/figures/figure_2.pdf', format='pdf', bbox_inches='tight', transparent=False)
plt.tight_layout()
plt.show()
