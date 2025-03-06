#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 11:08:04 2024

@author: williamtaylor
"""

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('default')
import pandas as pd
import numpy as np
import model
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

#%% 
colors = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02','#a6761d','#666666']
weight = 0.0001
event = '200_year'
years = {
     1986: {'start': '1986-01-29', 'end': '1986-03-10', 'scale': 1.91, 'plot_end': '1986-02-25', 'bias_start':'1986-02-02', 'bias_end':'1986-02-16'}, 
     1997: {'start': '1996-12-15', 'end': '1997-01-30', 'scale': 1.3374, 'plot_end': '1997-01-12', 'bias_start':'1996-12-19', 'bias_end':'1997-01-02'}, 
     2006: {'start': '2005-12-15', 'end': '2006-01-30', 'scale': 2.59, 'plot_end': '2006-01-15', 'bias_start':'2005-12-18', 'bias_end':'2006-01-01'}, 
     2017: {'start': '2017-01-20', 'end': '2017-03-10', 'scale': 2.33, 'plot_end': '2017-02-27', 'bias_start':'2017-01-27', 'bias_end':'2017-02-10'}, 
     }


fig, ax = plt.subplots(2,4, figsize=(12,6))

# load observed data
for i,year in enumerate(years):
    
    # load model results
    filename = f"results/storage_estimates_{year}/res_df_mpc_{event}_weight{weight}.csv"
    mpc_dfs = pd.read_csv(filename, index_col=0)
    mpc_dfs.index = pd.to_datetime(mpc_dfs.index)
    
    # HEFS EFO
    filename = f"results/storage_estimates_{year}/res_df_hefs_{event}_weight{weight}.csv"
    hefs_dfs = pd.read_csv(filename, index_col=0)
    
    # Syn EFO
    filename = f"results/storage_estimates_{year}/res_df_syn_train_{event}_weight{weight}.csv"
    syn_dfs = pd.read_csv(filename, index_col=0)

    # Cumulative
    filename = f"results/storage_estimates_{year}/res_df_cumulative_{event}.csv"
    cumulative_dfs = pd.read_csv(filename, index_col=0)
    
    # begin plotting
    # inflow
    #ax[1,i].plot(mpc_dfs.index, mpc_dfs['Q'], c='blue')
    
    # plot baseline
    #ax[0,i].plot(mpc_dfs.index, mpc_dfs['S_baseline_mpc'], c=colors[0])
    #ax[1,i].plot(mpc_dfs.index, mpc_dfs['S_baseline_mpc'], c=colors[0])
    
    # plot perfect
    #ax[0,i].plot(mpc_dfs.index, mpc_dfs['S_perfect_mpc'], c=colors[1])
    #ax[1,i].plot(mpc_dfs.index, mpc_dfs['R_perfect_mpc'], c=colors[1])
    
    # plot MPC HEFS
    ax[1,i].plot(mpc_dfs.index, mpc_dfs['S_hefs_mpc'], c=colors[2])
    
    # plot MPC median synthetic ensemble
    # find the median synthetic ensemble
    pools = []
    for j in range(0,100):
        pool = mpc_dfs['S_syn_'+str(j)].max()
        pools.append(pool)
    med = np.median(pools)
    med_index = np.argmin(np.abs(np.array(pools) - med))
    print('MPC: ' +str(med_index))

    ax[0,i].plot(mpc_dfs.index, mpc_dfs['S_syn_'+str(med_index)], c=colors[2])
    
    # plot EFO HEFS
    ax[1,i].plot(mpc_dfs.index, hefs_dfs['S_hefs_EFO'], c=colors[3])
    
    # plot HEFS EFO median synthetic ensemble
    pools = []
    for j in range(0,100):
        pool = hefs_dfs['S_syn_EFO_'+str(j)].max()
        pools.append(pool)
    med = np.median(pools)
    med_index = np.argmin(np.abs(np.array(pools) - med))
    print('HEFS EFO: '+str(med_index))

    ax[0,i].plot(mpc_dfs.index, hefs_dfs['S_syn_EFO_'+str(med_index)], c=colors[3])
    
    # plot EFO Syn
    #ax[0,i].plot(mpc_dfs.index, syn_dfs['S_hefs_EFO'], c=colors[6])
    ax[1,i].plot(mpc_dfs.index, syn_dfs['S_hefs_EFO'], c=colors[5])

    # plot Syn EFO median synthetic ensemble
    pools = []
    for j in range(0,100):
        pool = syn_dfs['S_syn_EFO_'+str(j)].max()
        pools.append(pool)
    med = np.median(pools)
    med_index = np.argmin(np.abs(np.array(pools) - med))
    print('Syn EFO: '+str(med_index))

    #ax[0,i].plot(mpc_dfs.index, syn_dfs['S_syn_EFO_'+str(med_index)], c=colors[5])
    ax[0,i].plot(mpc_dfs.index, syn_dfs['S_syn_EFO_'+str(med_index)], c=colors[5])    

    # plot cumulative method
    #ax[0,i].plot(mpc_dfs.index, cumulative_dfs['S_cumulative'], c=colors[8])
    ax[1,i].plot(mpc_dfs.index, cumulative_dfs['S_cumulative'], c=colors[7])

    # plot cumulative median synthetic ensemble
    pools = []
    for j in range(0,100):
        pool = cumulative_dfs['S_syn_cumulative_'+str(j)].max()
        pools.append(pool)
    med = np.median(pools)
    med_index = np.argmin(np.abs(np.array(pools) - med))
    ax[0,i].plot(mpc_dfs.index, cumulative_dfs['S_syn_cumulative_'+str(med_index)], c=colors[7])

    # formatting
    ax[0,i].xaxis.set_major_locator(mdates.DayLocator(interval=7))  # Adjust the interval (e.g., every 4 days)
    ax[0,i].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))  # Format to show month and day
    ax[0,i].xaxis.set_ticklabels([])
    ax[0,i].grid(True)
    ax[1,i].xaxis.set_major_locator(mdates.DayLocator(interval=7))  # Adjust the interval (e.g., every 4 days)
    ax[1,i].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))  # Format to show month and day
    
    # axes adjustments    
    ax[0,i].set_ylim(-1600,1400)
    ax[1,i].set_ylim(-1600,1400)

    ax[0,i].set_xlim(pd.to_datetime(years[year]['start']), pd.to_datetime(years[year]['plot_end']))
    ax[1,i].set_xlim(pd.to_datetime(years[year]['start']), pd.to_datetime(years[year]['plot_end']))
    ax[1,i].set_xlabel('Date')
    ax[0,i].set_title(year, fontweight='bold', fontsize=16)
    
    ax[0,i].axhline(0, linestyle='--', c='black', zorder=0)
    ax[1,i].axhline(0, linestyle='--', c='black', zorder=0)

# remove the y axis from all but the leftmost panels
for row in ax:
    for i in range(1,4):
        row[i].yaxis.set_ticklabels([])
        row[i].grid(True)
ax[1,0].grid(True)
ax[0,0].set_ylabel(r'$\Delta$ Storage (TAF)', fontweight='bold', fontsize=12)
ax[1,0].set_ylabel(r'$\Delta$ Storage (TAF)', fontweight='bold', fontsize=12)

legend_baseline = plt.Line2D([0], [0], color=colors[0], lw=2, label='MPC - Naive (Baseline)')
legend_perf = plt.Line2D([0], [0], color=colors[1], lw=2, label='MPC - Perfect (Upper Bound)')
legend_MPC = plt.Line2D([0], [0], color=colors[2], lw=2, label='MPC - Ensemble Forecasts')
legend_EFO = plt.Line2D([0], [0], color=colors[3], lw=2, label='EFO - HEFS')
legend_EFO_held = plt.Line2D([0], [0], color=colors[4], lw=2, label='EFO - HEFS without 1997')
legend_syn = plt.Line2D([0], [0], color=colors[5], lw=2, label='EFO - Synthetic')
legend_syn_held = plt.Line2D([0], [0], color=colors[6], lw=2, label='EFO - Synthetic without 1997')
legend_cumulative = plt.Line2D([0], [0], color=colors[7], lw=2, label='Cumulative Method')

fig.legend(handles=[legend_MPC, legend_EFO, legend_syn, legend_cumulative], loc='upper center', bbox_to_anchor=(0.52, -0.01), ncol=4, edgecolor='black')
fig.text(0.05, 0.75, 'Simulated with\nSynthetic Forecasts', ha='center', va='center', fontweight='bold', fontsize=16, rotation=90)
fig.text(0.05, 0.30, 'Simulated with\nHEFS Forecasts', ha='center', va='center', fontweight='bold', fontsize=16, rotation=90)
#fig.suptitle(f'Simulated Hydrographs - Event: {event}, Weight:{weight}', fontweight='bold', fontsize=16)
plt.tight_layout(rect=[0.08,0,1,1])
plt.savefig('/Users/williamtaylor/Documents/Github/Robust-FIRO/figures/Figure_6.pdf', format='pdf', bbox_inches='tight', transparent=False)
plt.show()

#%% coefficient of variation by policy

colors = sns.color_palette("tab10")
years = {
    1986: {'start': '1986-01-29', 'end': '1986-03-10', 'scale': 1.91, 'plot_end': '1986-02-25', 'bias_start':'1986-02-02', 'bias_end':'1986-02-16'}, 
    1997: {'start': '1996-12-15', 'end': '1997-01-30', 'scale': 1.3374, 'plot_end': '1997-01-12', 'bias_start':'1996-12-19', 'bias_end':'1997-01-02'}, 
    2006: {'start': '2005-12-15', 'end': '2006-01-30', 'scale': 2.59, 'plot_end': '2006-01-15', 'bias_start':'2005-12-18', 'bias_end':'2006-01-01'}, 
    2017: {'start': '2017-01-20', 'end': '2017-03-10', 'scale': 2.33, 'plot_end': '2017-02-27', 'bias_start':'2017-01-27', 'bias_end':'2017-02-10'}, 
    }

handles = ['Baseline','MPC - Perfect','MPC - Forecasts','EFO - Fully Trained','EFO - 1997 Held Out','EFO - Syn Trained', 'EFO - Syn Trained, 1997 Held Out','Cumulative Method']
weight = 0.0001
positions = np.arange(0,8,1)
fig, ax = plt.subplots(1,4, figsize=(12,4))
ax = ax.flatten()
# load observed data
for i,year in enumerate(years):
    
    # load model results
    filename = f"results/storage_estimates_{year}/res_df_mpc_200_year_weight{weight}.csv"
    mpc_dfs = pd.read_csv(filename, index_col=0)
    mpc_dfs.index = pd.to_datetime(mpc_dfs.index)
    
    # HEFS EFO
    filename = f"results/storage_estimates_{year}/res_df_hefs_200_year_weight{weight}.csv"
    hefs_dfs = pd.read_csv(filename, index_col=0)
    
    # Syn EFO
    filename = f"results/storage_estimates_{year}/res_df_syn_train_200_year_weight{weight}.csv"
    syn_dfs = pd.read_csv(filename, index_col=0)

    # Cumulative
    filename = f"results/storage_estimates_{year}/res_df_cumulative_200_year.csv"
    cumulative_dfs = pd.read_csv(filename, index_col=0)
    
    
    # plot baseline
    ax[i].bar(positions[0], mpc_dfs['R_baseline_mpc'].std() / mpc_dfs['R_baseline_mpc'].mean(), color=colors[0], edgecolor='black')
    
    # plot perfect
    ax[i].bar(positions[1], mpc_dfs['R_perfect_mpc'].std() / mpc_dfs['R_perfect_mpc'].mean(), color=colors[1], edgecolor='black')
    

    
    # plot MPC median synthetic ensemble
    # find the median synthetic ensemble
    pools = []
    for j in range(0,100):
        pool = mpc_dfs['S_syn_'+str(j)].max()
        pools.append(pool)
    med = np.median(pools)
    med_index = np.argmin(np.abs(np.array(pools) - med))
    print('MPC: ' +str(med_index))

    ax[i].bar(positions[2], mpc_dfs['R_syn_'+str(med_index)].std() / mpc_dfs['R_syn_'+str(med_index)].mean(), color=colors[2], edgecolor='black')

    # plot HEFS EFO median synthetic ensemble
    pools = []
    for j in range(0,100):
        pool = hefs_dfs['S_syn_EFO_'+str(j)].max()
        pools.append(pool)
    med = np.median(pools)
    med_index = np.argmin(np.abs(np.array(pools) - med))
    print('HEFS EFO: '+str(med_index))

    ax[i].bar(positions[3], hefs_dfs['R_syn_EFO_'+str(med_index)].std() / hefs_dfs['R_syn_EFO_'+str(med_index)].mean(), color=colors[3], edgecolor='black')

    # plot Syn EFO median synthetic ensemble
    pools = []
    for j in range(0,100):
        pool = syn_dfs['S_syn_EFO_'+str(j)].max()
        pools.append(pool)
    med = np.median(pools)
    med_index = np.argmin(np.abs(np.array(pools) - med))
    print('Syn EFO: '+str(med_index))

    ax[i].bar(positions[4], syn_dfs['R_syn_EFO_'+str(med_index)].std() / syn_dfs['R_syn_EFO_'+str(med_index)].mean(), color=colors[4], edgecolor='black')

    # plot cumulative median synthetic ensemble
    pools = []
    for j in range(0,100):
        pool = cumulative_dfs['S_syn_cumulative_'+str(j)].max()
        pools.append(pool)
    med = np.median(pools)
    med_index = np.argmin(np.abs(np.array(pools) - med))
    ax[i].bar(positions[5], cumulative_dfs['R_syn_cumulative_'+str(med_index)].std() / cumulative_dfs['R_syn_cumulative_'+str(med_index)].mean(), color=colors[9], edgecolor='black')

  
    # axes adjustments    
   
    ax[i].set_title(year, fontweight='bold')
    ax[i].set_xticklabels([])
    ax[i].set_ylim(0,1.5)

ax[0].set_ylabel('Release CV', fontweight='bold', fontsize=14)

for i in range(1,4):
    ax[i].yaxis.set_ticklabels([])
    ax[i].grid(False)

legend_baseline = plt.Line2D([0], [0], color=colors[0], lw=2, label='MPC - Naive (Baseline)')
legend_perf = plt.Line2D([0], [0], color=colors[1], lw=2, label='MPC - Perfect (Upper Bound)')
legend_MPC = plt.Line2D([0], [0], color=colors[2], lw=2, label='MPC - Ensemble Forecasts')
legend_EFO = plt.Line2D([0], [0], color=colors[3], lw=2, label='EFO - HEFS')
legend_EFO_held = plt.Line2D([0], [0], color=colors[4], lw=2, label='EFO - HEFS without 1997')
legend_syn = plt.Line2D([0], [0], color=colors[5], lw=2, label='EFO - Synthetic')
legend_syn_held = plt.Line2D([0], [0], color=colors[6], lw=2, label='EFO - Synthetic without 1997')
legend_cumulative = plt.Line2D([0], [0], color=colors[9], lw=2, label='Cumulative Method')
legend_scalar = plt.Line2D([0], [0], color='gray', marker='o', lw=0, markerfacecolor='black', label='HEFS Simulation')
legend_boxplot = mpatches.Patch(edgecolor='black', facecolor='none', label='Synthetic Simulation')

fig.legend(handles=[legend_baseline, legend_perf, legend_MPC, legend_EFO, legend_syn, legend_cumulative], loc='upper center', bbox_to_anchor=(0.52, -0.01), ncol=3)
plt.show()
