#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 10:52:58 2024

@author: williamtaylor
"""

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('default')
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

#%% version 2 (with weight input)
def plot_boxplots(ax, data, event_idx, offset, color):
    for i in range(100):
        data[i] = data[i].max()
    boxplot_position = event_idx + offset
    ax.boxplot(data, positions=[boxplot_position], vert=True, widths=0.05,
               boxprops=dict(color=color), capprops=dict(color=color), 
               whiskerprops=dict(color=color), flierprops=dict(markeredgecolor=color),
               medianprops=dict(color='black'))  # You can adjust median color separately if needed

offsets = [-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4]
#colors = sns.color_palette("tab10")
colors = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02','#a6761d','#666666']
events = ['observed','100_year','200_year','300_year']
handles = ['Baseline','MPC - Perfect','MPC - Forecasts','EFO - Fully Trained','EFO - 1997 Held Out','EFO - Syn Trained', 'EFO - Syn Trained, 1997 Held Out','Cumulative Method']
weight = 0.5
# load all dataframes
# model predictive control results
mpc_dfs = {}
for event in events:
    filename = f"results/storage_estimates_1986/res_df_mpc_{event}_weight{weight}.csv"
    mpc_dfs[event] = pd.read_csv(filename, index_col=0)

# hefs trained, fully and held out
hefs_dfs = {}
for event in events:
    filename = f"results/storage_estimates_1986/res_df_hefs_{event}_weight{weight}.csv"
    hefs_dfs[event] = pd.read_csv(filename, index_col=0)

# syn fully trained
syn_hefs_dfs = {}
for event in events:
    filename = f"results/storage_estimates_1986/res_df_syn_train_{event}_weight{weight}.csv"
    syn_hefs_dfs[event] = pd.read_csv(filename, index_col=0)

# syn trained held out
syn_held_dfs = {}
for event in events:
    filename = f"results/storage_estimates_1986/res_df_syn_train_no1997_{event}_weight{weight}.csv"
    
    syn_held_dfs[event] = pd.read_csv(filename, index_col=0)

# cumulative method
cumulative_dfs = {}
for event in events:
    filename = f"results/storage_estimates_1986/res_df_cumulative_{event}.csv"
    cumulative_dfs[event] = pd.read_csv(filename, index_col=0)

fig, ax = plt.subplots(figsize=(10,5))
for i, event in enumerate(events):
    df = mpc_dfs[event]
    ax.scatter(events.index(event)+offsets[0], df['S_baseline_mpc'].max(), label = 'MPC - Baseline', c=colors[0], marker='*', s=150, edgecolor='black', zorder=3)
    ax.scatter(events.index(event)+offsets[1], df['S_perfect_mpc'].max(), label = 'MPC - Perfect', c=colors[1], marker='*', s=150, edgecolor='black', zorder=3)
    ax.scatter(events.index(event)+offsets[2], df['S_hefs_mpc'].max(), label = 'MPC - HEFS', c=colors[2], marker='*', s=150, edgecolor='black', zorder=3)
    syn_mpc = [df['S_syn_'+str(i)] for i in range(100)]
    plot_boxplots(ax, syn_mpc, events.index(event), offsets[2], colors[2])
    
    df = hefs_dfs[event]
    ax.scatter(events.index(event)+offsets[3], df['S_hefs_EFO'].max(), label = 'EFO - Fully Trained', c=colors[3], marker='*', s=150, edgecolor='black', zorder=3)
    ax.scatter(events.index(event)+offsets[4], df['S_hefs_EFO_no_1997'].max(), label = 'EFO - 1997 Held Out', c=colors[4], marker='*', s=150, edgecolor='black', zorder=3)
    hefs_syn = [df['S_syn_EFO_'+str(i)] for i in range(100)]
    plot_boxplots(ax, hefs_syn, events.index(event), offsets[3], colors[3])
    hefs_syn = [df['S_syn_EFO_no_1997_'+str(i)] for i in range(100)]
    plot_boxplots(ax, hefs_syn, events.index(event), offsets[4], colors[4])

    df = syn_hefs_dfs[event]
    ax.scatter(events.index(event)+offsets[5], df['S_hefs_EFO'].max(), label = 'EFO - Syn Fully Trained', c=colors[5], marker='*', s=150, edgecolor='black', zorder=3)
    syn_hefs = [df['S_syn_EFO_'+str(i)] for i in range(100)]
    plot_boxplots(ax, syn_hefs, events.index(event), offsets[5], colors[5])

    df = syn_held_dfs[event]
    ax.scatter(events.index(event)+offsets[6], df['S_hefs_EFO'].max(), label = 'EFO - Syn Held Out', c=colors[6], marker='*', s=150, edgecolor='black', zorder=3)
    syn_held = [df['S_syn_EFO_'+str(i)] for i in range(100)]
    plot_boxplots(ax, syn_held, events.index(event), offsets[6], colors[6])

    df = cumulative_dfs[event]
    ax.scatter(events.index(event)+offsets[7], df['S_cumulative'].max(), c=colors[7], marker='*', s=150, edgecolor='black', zorder=3)
    syn_cumulative = [df['S_syn_cumulative_'+str(i)] for i in range(100)]
    plot_boxplots(ax, syn_cumulative, events.index(event), offsets[7], colors[7])


ax.set_ylabel('Required Flood Pool Size (TAF)', fontweight='bold')
ax.set_xlabel('Simulated Flood Event', fontweight='bold')
ax.set_xticks(range(len(events)))
ax.set_xticklabels(['1986 Flood','100 Year Scaled','200 Year Scaled','300 Year Scaled'])
ax.set_ylim(-5,2500)
ax.set_xlim(-0.5,3.5)
ax.annotate(text=('Scaled Events'), xy=(2,2300), ha='center', fontweight='bold', fontsize=12)
ax.annotate(text=('Observed Event'), xy=(0,2300), ha='center', fontweight='bold', fontsize=12)
ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1)

for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1)

fig.suptitle(f'Required Flood Pool by Policy - 1986 Flood, Weight: {weight}', fontweight='bold', fontsize=18)

legend_baseline = plt.Line2D([0], [0], color=colors[0], lw=2, label='MPC - Baseline (Lower Bound)')
legend_perf = plt.Line2D([0], [0], color=colors[1], lw=2, label='MPC - Perfect (Upper Bound)')
legend_MPC = plt.Line2D([0], [0], color=colors[2], lw=2, label='MPC - Ensemble Forecasts')
legend_EFO = plt.Line2D([0], [0], color=colors[3], lw=2, label='EFO - HEFS')
legend_EFO_held = plt.Line2D([0], [0], color=colors[4], lw=2, label='EFO - HEFS without 1997')
legend_syn = plt.Line2D([0], [0], color=colors[5], lw=2, label='EFO - Synthetic')
legend_syn_held = plt.Line2D([0], [0], color=colors[6], lw=2, label='EFO - Synthetic without 1997')
legend_cumulative = plt.Line2D([0], [0], color=colors[7], lw=2, label='Cumulative Method')
legend_scalar = plt.Line2D([0], [0], color='gray', marker='*', markersize=12, lw=0, markerfacecolor='black', label='HEFS Simulation')
legend_boxplot = mpatches.Patch(edgecolor='black', facecolor='none', label='Synthetic Simulation')

fig.legend(handles=[legend_baseline, legend_perf, legend_MPC, legend_EFO, legend_EFO_held, legend_syn, legend_syn_held, legend_cumulative, legend_scalar, legend_boxplot], loc='upper center', bbox_to_anchor=(0.52, -0.01), ncol=4)

plt.tight_layout()
plt.show()


