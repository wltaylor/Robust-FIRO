#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 11:23:09 2024
@author: williamtaylor
"""
# grouped scatter plot, showing median storage under synthetic forecasts for each method
# columns are inconsistently named so I need to treat each method a little differently
# start with just one event, read in data, extract median flood pool

import matplotlib.pyplot as plt
plt.style.use('default')
import pandas as pd
import numpy as np
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

offsets = np.linspace(-0.3,0.3,8)
#[-0.3,-0.2,-0.1,0,0.1,0.2,0.3]
#colors = sns.color_palette("tab10")
colors = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02','#a6761d','#666666']
handles = ['Baseline','MPC - Perfect','MPC - Forecasts','EFO - Fully Trained','EFO - 1997 Held Out','EFO - Syn Trained', 'EFO - Syn Trained, 1997 Held Out','Cumulative Method']
weights = [0.0001,0.001,0.01,0.1,0.5]
weights = [0.5,0.1,0.01,0.001,0.0001]
years = [1986, 1997, 2006, 2017]
methods = ['res_df_mpc','res_df_hefs','res_df_syn_train','res_df_syn_train_no1997','res_df_cumulative']
markers = ['o','s','P','*','d']
fig, ax = plt.subplots(figsize=(10,5))
# baseline
for i,year in enumerate(years):
    filename = f"results/storage_estimates_{year}/res_df_mpc_200_year_weight0.0001.csv"
    df = pd.read_csv(filename, index_col=0)
    ax.scatter(years.index(year)+offsets[0], df['S_baseline_mpc'].max(), color=colors[0])


# perfect
for i, year in enumerate(years):
        
    # read in MPC data
    for k,weight in enumerate(weights):
        filename = f"results/storage_estimates_{year}/res_df_mpc_200_year_weight{weight}.csv"
        df = pd.read_csv(filename, index_col=0)
        ax.scatter(years.index(year)+offsets[1], df['S_perfect_mpc'].max(), color=colors[1], marker=markers[k])


# MPC
for i, year in enumerate(years):
        
    # read in MPC data
    for k,weight in enumerate(weights):
        filename = f"results/storage_estimates_{year}/res_df_mpc_200_year_weight{weight}.csv"
        df = pd.read_csv(filename, index_col=0)
        pools = []
        for j in range(0,100):
            pool = df['S_syn_'+str(j)].max()
            pools.append(pool)
        med = np.median(pools)
        ax.scatter(years.index(year)+offsets[2], med, color=colors[2], marker=markers[k])

# EFO-HEFS trained
for i, year in enumerate(years):
        
    # read in MPC data
    for k,weight in enumerate(weights):
        filename = f"results/storage_estimates_{year}/res_df_hefs_200_year_weight{weight}.csv"
        df = pd.read_csv(filename, index_col=0)
        pools = []
        for j in range(0,100):
            pool = df['S_syn_EFO_'+str(j)].max()
            pools.append(pool)
        med = np.median(pools)
        ax.scatter(years.index(year)+offsets[3], med, color=colors[3], marker=markers[k])

# EFO-HEFS trained, no 1997
for i, year in enumerate(years):
        
    # read in MPC data
    for k,weight in enumerate(weights):
        filename = f"results/storage_estimates_{year}/res_df_hefs_200_year_weight{weight}.csv"
        df = pd.read_csv(filename, index_col=0)
        pools = []
        for j in range(0,100):
            pool = df['S_syn_EFO_no_1997_'+str(j)].max()
            pools.append(pool)
        med = np.median(pools)
        ax.scatter(years.index(year)+offsets[4], med, color=colors[4], marker=markers[k])

# EFO-syn trained
for i, year in enumerate(years):
        
    # read in MPC data
    for k,weight in enumerate(weights):
        filename = f"results/storage_estimates_{year}/res_df_syn_train_200_year_weight{weight}.csv"
        df = pd.read_csv(filename, index_col=0)
        pools = []
        for j in range(0,100):
            pool = df['S_syn_EFO_'+str(j)].max()
            pools.append(pool)
        med = np.median(pools)
        ax.scatter(years.index(year)+offsets[5], med, color=colors[5], marker=markers[k])

# EFO-syn trained, no 1997
for i, year in enumerate(years):
        
    # read in MPC data
    for k,weight in enumerate(weights):
        filename = f"results/storage_estimates_{year}/res_df_syn_train_no1997_200_year_weight{weight}.csv"
        df = pd.read_csv(filename, index_col=0)
        pools = []
        for j in range(0,100):
            pool = df['S_syn_EFO_'+str(j)].max()
            pools.append(pool)
        med = np.median(pools)
        ax.scatter(years.index(year)+offsets[6], med, color=colors[6], marker=markers[k])

        # if (year == 1986) and (weight != 0.5):
        #     ax.annotate(weight, (years.index(1986)+offsets[6]+0.05, med-20), fontsize=10)
        # if (year == 1986) and (weight == 0.5):
        #     ax.annotate(weight, (years.index(1986)+offsets[6]+0.05, med+15), fontsize=10)



# cumulative
for i, year in enumerate(years):
    filename = f"results/storage_estimates_{year}/res_df_cumulative_200_year.csv"
    df = pd.read_csv(filename, index_col=0)
    pools = []
    for j in range(0,100):
        pool = df['S_syn_cumulative_'+str(j)].max()
        pools.append(pool)
    med = np.median(pools)
    ax.scatter(years.index(year)+offsets[7], med, color=colors[7])



# formatting
ax.set_xlabel('Flood Event (Scaled to 200-year)', fontweight='bold', fontsize=14)
ax.set_ylabel('Required Flood Pool (TAF)', fontweight='bold', fontsize=14)
ax.set_xticks(range(len(years)))
ax.set_xticklabels(years, fontweight='bold', fontsize=12)
ax.set_ylim(0,2100)
ax.grid(True, axis='y')
ax.axvline(0.5, linestyle='--', c='black')
ax.axvline(1.5, linestyle='--', c='black')
ax.axvline(2.5, linestyle='--', c='black')

# legend for the markers
marker_handles = [plt.Line2D([0],[0], marker=m, lw=0, markersize=8, c='black', label=f'weight: {w}') for m,w in zip(markers, weights)]
ax.legend(handles = marker_handles, loc='upper right', title='Weight', edgecolor='black')

#fig.suptitle('Required Flood Pool Size by Event and Drawdown Weight', fontweight='bold', fontsize=18)

# legend_baseline = plt.Line2D([0], [0], color=colors[0], lw=0, label='Baseline', marker='o')
# legend_perf = plt.Line2D([0], [0], color=colors[1], lw=0, label='Perfect', marker='o')
# legend_MPC = plt.Line2D([0], [0], color=colors[2], lw=0, label='MPC - HEFS', marker='o')
# legend_EFO = plt.Line2D([0], [0], color=colors[3], lw=0, label='EFO - Fully Trained', marker='o')
# legend_EFO_held = plt.Line2D([0], [0], color=colors[4], lw=0, label='EFO - 1997 Held Out', marker='o')
# legend_syn = plt.Line2D([0], [0], color=colors[5], lw=0, label='Synthetic - Fully Trained', marker='o')
# legend_syn_held = plt.Line2D([0], [0], color=colors[6], lw=0, label='Synthetic - 1997 Held Out', marker='o')
# legend_cum = plt.Line2D([0], [0], color=colors[7], lw=0, label='Cumulative', marker='o')

legend_baseline = plt.Line2D([0], [0], color=colors[0], lw=0, label='MPC - Baseline (Lower Bound)', marker='o')
legend_perf = plt.Line2D([0], [0], color=colors[1], lw=0, label='MPC - Perfect (Upper Bound)', marker='o')
legend_MPC = plt.Line2D([0], [0], color=colors[2], lw=0, label='MPC - Ensemble Forecasts', marker='o')
legend_EFO = plt.Line2D([0], [0], color=colors[3], lw=0, label='EFO - HEFS', marker='o')
legend_EFO_held = plt.Line2D([0], [0], color=colors[4], lw=0, label='EFO - HEFS without 1997', marker='o')
legend_syn = plt.Line2D([0], [0], color=colors[5], lw=0, label='EFO - Synthetic', marker='o')
legend_syn_held = plt.Line2D([0], [0], color=colors[6], lw=0, label='EFO - Synthetic without 1997', marker='o')
legend_cumulative = plt.Line2D([0], [0], color=colors[7], lw=0, label='Cumulative Method', marker='o')

fig.legend(handles=[legend_baseline, legend_perf, legend_MPC, legend_EFO, legend_EFO_held, legend_syn, legend_syn_held, legend_cumulative], loc='upper center', bbox_to_anchor=(0.52, -0.01), ncol=4, edgecolor='black')
plt.tight_layout()
plt.savefig('/Users/williamtaylor/Documents/Github/Robust-FIRO/figures/Figure_5.pdf', format='pdf', bbox_inches='tight', transparent=False)
plt.show()

#%%
offsets = np.linspace(-0.3,0.3,8)
#[-0.3,-0.2,-0.1,0,0.1,0.2,0.3]
colors = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02','#a6761d','#666666']
handles = ['Baseline','MPC - Perfect','MPC - Forecasts','EFO - Fully Trained','EFO - 1997 Held Out','EFO - Syn Trained', 'EFO - Syn Trained, 1997 Held Out','Cumulative Method']
weights = [0.0001,0.001,0.01,0.1,0.5]
years = [1986, 1997, 2006, 2017]
methods = ['res_df_mpc','res_df_hefs','res_df_syn_train','res_df_syn_train_no1997','res_df_cumulative']
markers = ['o','s','P','*','d']
fig, ax = plt.subplots(figsize=(12,6))
# baseline
for i,year in enumerate(years):
    filename = f"results/storage_estimates_{year}/res_df_mpc_200_year_weight0.0001.csv"
    df = pd.read_csv(filename, index_col=0)
    ax.scatter(years.index(year)+offsets[0], df['S_baseline_mpc'].min(), color=colors[0])


# perfect
for i, year in enumerate(years):
        
    # read in MPC data
    for k,weight in enumerate(weights):
        filename = f"results/storage_estimates_{year}/res_df_mpc_200_year_weight{weight}.csv"
        df = pd.read_csv(filename, index_col=0)
        ax.scatter(years.index(year)+offsets[1], df['S_perfect_mpc'].min(), color=colors[1], marker=markers[k])


# MPC
for i, year in enumerate(years):
        
    # read in MPC data
    for k,weight in enumerate(weights):
        filename = f"results/storage_estimates_{year}/res_df_mpc_200_year_weight{weight}.csv"
        df = pd.read_csv(filename, index_col=0)
        pools = []
        for j in range(0,100):
            pool = df['S_syn_'+str(j)].min()
            pools.append(pool)
        med = np.median(pools)
        ax.scatter(years.index(year)+offsets[2], med, color=colors[2], marker=markers[k])

# EFO-HEFS trained
for i, year in enumerate(years):
        
    # read in MPC data
    for k,weight in enumerate(weights):
        filename = f"results/storage_estimates_{year}/res_df_hefs_200_year_weight{weight}.csv"
        df = pd.read_csv(filename, index_col=0)
        pools = []
        for j in range(0,100):
            pool = df['S_syn_EFO_'+str(j)].min()
            pools.append(pool)
        med = np.median(pools)
        ax.scatter(years.index(year)+offsets[3], med, color=colors[3], marker=markers[k])

# EFO-HEFS trained, no 1997
for i, year in enumerate(years):
        
    # read in MPC data
    for k,weight in enumerate(weights):
        filename = f"results/storage_estimates_{year}/res_df_hefs_200_year_weight{weight}.csv"
        df = pd.read_csv(filename, index_col=0)
        pools = []
        for j in range(0,100):
            pool = df['S_syn_EFO_no_1997_'+str(j)].min()
            pools.append(pool)
        med = np.median(pools)
        ax.scatter(years.index(year)+offsets[4], med, color=colors[4], marker=markers[k])

# EFO-syn trained
for i, year in enumerate(years):
        
    # read in MPC data
    for k,weight in enumerate(weights):
        filename = f"results/storage_estimates_{year}/res_df_syn_train_200_year_weight{weight}.csv"
        df = pd.read_csv(filename, index_col=0)
        pools = []
        for j in range(0,100):
            pool = df['S_syn_EFO_'+str(j)].min()
            pools.append(pool)
        med = np.median(pools)
        ax.scatter(years.index(year)+offsets[5], med, color=colors[5], marker=markers[k])

# EFO-syn trained, no 1997
for i, year in enumerate(years):
        
    # read in MPC data
    for k,weight in enumerate(weights):
        filename = f"results/storage_estimates_{year}/res_df_syn_train_no1997_200_year_weight{weight}.csv"
        df = pd.read_csv(filename, index_col=0)
        pools = []
        for j in range(0,100):
            pool = df['S_syn_EFO_'+str(j)].min()
            pools.append(pool)
        med = np.median(pools)
        ax.scatter(years.index(year)+offsets[6], med, color=colors[6], marker=markers[k])

        # if (year == 1986) and (weight != 0.5):
        #     ax.annotate(weight, (years.index(1986)+offsets[6]+0.05, med-20), fontsize=10)
        # if (year == 1986) and (weight == 0.5):
        #     ax.annotate(weight, (years.index(1986)+offsets[6]+0.05, med+15), fontsize=10)



# cumulative
for i, year in enumerate(years):
    filename = f"results/storage_estimates_{year}/res_df_cumulative_200_year.csv"
    df = pd.read_csv(filename, index_col=0)
    pools = []
    for j in range(0,100):
        pool = df['S_syn_cumulative_'+str(j)].min()
        pools.append(pool)
    med = np.median(pools)
    ax.scatter(years.index(year)+offsets[7], med, color=colors[7])



# formatting
ax.set_xlabel('Flood Event (Scaled to 200-year)', fontweight='bold', fontsize=14)
ax.set_ylabel('Minimum Storage (TAF)', fontweight='bold', fontsize=14)
ax.set_xticks(range(len(years)))
ax.set_xticklabels(years, fontweight='bold', fontsize=12)
ax.set_ylim(-1600,100)
ax.grid(True, axis='y')
ax.axvline(0.5, linestyle='--', c='black')
ax.axvline(1.5, linestyle='--', c='black')
ax.axvline(2.5, linestyle='--', c='black')

# legend for the markers
marker_handles = [plt.Line2D([0],[0], marker=m, lw=0, markersize=8, c='black', label=f'weight: {w}') for m,w in zip(markers, weights)]
ax.legend(handles = marker_handles, loc='upper right', title='Weight', edgecolor='black', bbox_to_anchor=(1.20,1))

fig.suptitle('Maximum Drawdown by Model and Drawdown Weight', fontweight='bold', fontsize=18)

# legend_baseline = plt.Line2D([0], [0], color=colors[0], lw=0, label='Baseline', marker='o')
# legend_perf = plt.Line2D([0], [0], color=colors[1], lw=0, label='Perfect', marker='o')
# legend_MPC = plt.Line2D([0], [0], color=colors[2], lw=0, label='MPC - HEFS', marker='o')
# legend_EFO = plt.Line2D([0], [0], color=colors[3], lw=0, label='EFO - Fully Trained', marker='o')
# legend_EFO_held = plt.Line2D([0], [0], color=colors[4], lw=0, label='EFO - 1997 Held Out', marker='o')
# legend_syn = plt.Line2D([0], [0], color=colors[5], lw=0, label='Synthetic - Fully Trained', marker='o')
# legend_syn_held = plt.Line2D([0], [0], color=colors[6], lw=0, label='Synthetic - 1997 Held Out', marker='o')
# legend_cum = plt.Line2D([0], [0], color=colors[7], lw=0, label='Cumulative', marker='o')

legend_baseline = plt.Line2D([0], [0], color=colors[0], lw=0, label='MPC - Baseline (Lower Bound)', marker='o')
legend_perf = plt.Line2D([0], [0], color=colors[1], lw=0, label='MPC - Perfect (Upper Bound)', marker='o')
legend_MPC = plt.Line2D([0], [0], color=colors[2], lw=0, label='MPC - Ensemble Forecasts', marker='o')
legend_EFO = plt.Line2D([0], [0], color=colors[3], lw=0, label='EFO - HEFS', marker='o')
legend_EFO_held = plt.Line2D([0], [0], color=colors[4], lw=0, label='EFO - HEFS without 1997', marker='o')
legend_syn = plt.Line2D([0], [0], color=colors[5], lw=0, label='EFO - Synthetic', marker='o')
legend_syn_held = plt.Line2D([0], [0], color=colors[6], lw=0, label='EFO - Synthetic without 1997', marker='o')
legend_cumulative = plt.Line2D([0], [0], color=colors[7], lw=0, label='Cumulative Method', marker='o')



fig.legend(handles=[legend_baseline, legend_perf, legend_MPC, legend_EFO, legend_EFO_held, legend_syn, legend_syn_held, legend_cumulative], loc='upper center', bbox_to_anchor=(0.52, -0.01), ncol=4, edgecolor='black')
plt.tight_layout()
plt.show()

#%% pareto front of required flood pool vs maximum drawdown

fig, axes = plt.subplots(1,4, figsize=(12,4))
axes = axes.flatten()

for i,year in enumerate(years):
    filename = f"results/storage_estimates_{year}/res_df_mpc_200_year_weight0.0001.csv"
    df = pd.read_csv(filename, index_col=0)
    axes[i].scatter(df['S_baseline_mpc'].max(), df['S_baseline_mpc'].min(), color=colors[0])        

# perfect
for i, year in enumerate(years):
        
    # read in MPC data
    for k,weight in enumerate(weights):
        filename = f"results/storage_estimates_{year}/res_df_mpc_200_year_weight{weight}.csv"
        df = pd.read_csv(filename, index_col=0)
        axes[i].scatter(df['S_perfect_mpc'].max(), df['S_perfect_mpc'].min(), color=colors[1], marker=markers[k])


# MPC
for i, year in enumerate(years):
        
    # read in MPC data
    for k,weight in enumerate(weights):
        filename = f"results/storage_estimates_{year}/res_df_mpc_200_year_weight{weight}.csv"
        df = pd.read_csv(filename, index_col=0)
        pools = []
        for j in range(0,100):
            pool = df['S_syn_'+str(j)].min()
            pools.append(pool)
        draw = np.median(pools)
        pools = []
        for j in range(0,100):
            pool = df['S_syn_'+str(j)].max()
            pools.append(pool)
        flood = np.median(pools)
        axes[i].scatter(flood, draw, color=colors[2], marker=markers[k])

# EFO-HEFS trained
for i, year in enumerate(years):
        
    # read in MPC data
    for k,weight in enumerate(weights):
        filename = f"results/storage_estimates_{year}/res_df_hefs_200_year_weight{weight}.csv"
        df = pd.read_csv(filename, index_col=0)
        pools = []
        for j in range(0,100):
            pool = df['S_syn_EFO_'+str(j)].min()
            pools.append(pool)
        draw = np.median(pools)
        pools = []
        for j in range(0,100):
            pool = df['S_syn_EFO_'+str(j)].max()
            pools.append(pool)
        flood = np.median(pools)
        
        axes[i].scatter(flood, draw, color=colors[3], marker=markers[k])

# EFO-HEFS trained, no 1997
for i, year in enumerate(years):
        
    # read in MPC data
    for k,weight in enumerate(weights):
        filename = f"results/storage_estimates_{year}/res_df_hefs_200_year_weight{weight}.csv"
        df = pd.read_csv(filename, index_col=0)
        pools = []
        for j in range(0,100):
            pool = df['S_syn_EFO_no_1997_'+str(j)].min()
            pools.append(pool)
        draw = np.median(pools)
        pools = []
        for j in range(0,100):
            pool = df['S_syn_EFO_no_1997_'+str(j)].max()
            pools.append(pool)
        flood = np.median(pools)
        axes[i].scatter(flood, draw, color=colors[4], marker=markers[k])

# EFO-syn trained
for i, year in enumerate(years):
        
    # read in MPC data
    for k,weight in enumerate(weights):
        filename = f"results/storage_estimates_{year}/res_df_syn_train_200_year_weight{weight}.csv"
        df = pd.read_csv(filename, index_col=0)
        pools = []
        for j in range(0,100):
            pool = df['S_syn_EFO_'+str(j)].min()
            pools.append(pool)
        draw = np.median(pools)
        pools = []
        for j in range(0,100):
            pool = df['S_syn_EFO_'+str(j)].max()
            pools.append(pool)
        flood = np.median(pools)
        
        axes[i].scatter(flood, draw, color=colors[5], marker=markers[k])

# EFO-syn trained, no 1997
for i, year in enumerate(years):
        
    # read in MPC data
    for k,weight in enumerate(weights):
        filename = f"results/storage_estimates_{year}/res_df_syn_train_no1997_200_year_weight{weight}.csv"
        df = pd.read_csv(filename, index_col=0)
        pools = []
        for j in range(0,100):
            pool = df['S_syn_EFO_'+str(j)].min()
            pools.append(pool)
        draw = np.median(pools)
        pools = []
        for j in range(0,100):
            pool = df['S_syn_EFO_'+str(j)].max()
            pools.append(pool)
        flood = np.median(pools)
        
        axes[i].scatter(flood, draw, color=colors[6], marker=markers[k])

# cumulative
for i, year in enumerate(years):
    filename = f"results/storage_estimates_{year}/res_df_cumulative_200_year.csv"
    df = pd.read_csv(filename, index_col=0)
    pools = []
    for j in range(0,100):
        pool = df['S_syn_cumulative_'+str(j)].min()
        pools.append(pool)
    draw = np.median(pools)
    pools = []
    for j in range(0,100):
        pool = df['S_syn_cumulative_'+str(j)].max()
        pools.append(pool)
    flood = np.median(pools)
    axes[i].scatter(flood, draw, color=colors[7])


    axes[i].set_title(year, fontweight='bold', fontsize=14)
    axes[i].set_xlabel('Required Flood Pool (TAF)', fontweight='bold', fontsize=12)

axes[0].set_ylim(-1600,100)
axes[0].set_xlim(0,1900)

for i in range(1,4):
    axes[i].set_ylim(-1600,100)
    axes[i].set_xlim(0,1900)
    axes[i].yaxis.set_ticklabels([])
    #axes[i].grid(True)

axes[0].set_ylabel('Maximum Drawdown (TAF)', fontweight='bold', fontsize=12)
marker_handles = [plt.Line2D([0],[0], marker=m, lw=0, markersize=8, c='black', label=f'weight: {w}') for m,w in zip(markers, weights)]
axes[3].legend(handles = marker_handles, loc='lower right', title='Weight', edgecolor='black')

legend_baseline = plt.Line2D([0], [0], color=colors[0], lw=0, label='MPC - Baseline (Lower Bound)', marker='o')
legend_perf = plt.Line2D([0], [0], color=colors[1], lw=0, label='MPC - Perfect (Upper Bound)', marker='o')
legend_MPC = plt.Line2D([0], [0], color=colors[2], lw=0, label='MPC - Ensemble Forecasts', marker='o')
legend_EFO = plt.Line2D([0], [0], color=colors[3], lw=0, label='EFO - HEFS', marker='o')
legend_EFO_held = plt.Line2D([0], [0], color=colors[4], lw=0, label='EFO - HEFS without 1997', marker='o')
legend_syn = plt.Line2D([0], [0], color=colors[5], lw=0, label='EFO - Synthetic', marker='o')
legend_syn_held = plt.Line2D([0], [0], color=colors[6], lw=0, label='EFO - Synthetic without 1997', marker='o')
legend_cumulative = plt.Line2D([0], [0], color=colors[7], lw=0, label='Cumulative Method', marker='o')

fig.legend(handles=[legend_baseline, legend_perf, legend_MPC, legend_EFO, legend_EFO_held, legend_syn, legend_syn_held, legend_cumulative], loc='upper center', bbox_to_anchor=(0.52, -0.01), ncol=4, edgecolor='black')
plt.tight_layout()
plt.show()
        
