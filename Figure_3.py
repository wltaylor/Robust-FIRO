#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 09:48:41 2025

@author: williamtaylor
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

year = 1986
weight = 0.0001
fig, ax = plt.subplots(figsize=(8,6))
filename = f"results/storage_estimates_{year}/res_df_mpc_200_year_weight{weight}.csv"
mpc_dfs = pd.read_csv(filename, index_col=0)
mpc_dfs.index = pd.to_datetime(mpc_dfs.index)

pools = []
for j in range(0,100):
    pool = mpc_dfs['S_syn_'+str(j)].max()
    pools.append(pool)
med = np.median(pools)
med_index = np.argmin(np.abs(np.array(pools) - med))
print('MPC: ' +str(med_index))

ax.plot(mpc_dfs.index, mpc_dfs['S_syn_'+str(med_index)], c='blue')
ax.axhline(0, linestyle='--', c='black')
ax.fill_between(mpc_dfs.index, mpc_dfs['S_syn_'+str(med_index)], 0, color='blue', alpha=0.25)

#ax.annotate('Penalty for TOCS Exceedance', (mpc_dfs.index[10], 500), fontweight='bold', fontsize=12)
#ax.annotate('Penalty for Drawdown', (mpc_dfs.index[20], -500), fontweight='bold', fontsize=12)

peak_day = datetime.datetime(1986,2,21)
peak_start = 0
peak_end = 750

ax.annotate("",
             xy = (peak_day, peak_end), xytext=(peak_day, peak_start),
             arrowprops=dict(facecolor='black', edgecolor='black', linewidth=2, arrowstyle='<->'))
ax.annotate('Size of required flood pool', xy=(mpc_dfs.index[12], 700), fontsize=12, fontweight='bold')


ax.set_xlim(pd.to_datetime('1986-02-05'), pd.to_datetime('1986-02-28'))
num_ticks = (mpc_dfs.index[30] - mpc_dfs.index[7]).days
ticks = np.linspace(mpc_dfs.index[7].value, mpc_dfs.index[30].value, num=num_ticks)
ticks = pd.to_datetime(ticks)
ax.set_xticks(ticks)
labels = np.arange(1,len(ticks)+1,1)
ax.set_xticklabels(range(1, len(ticks)+1))  # Label as 1, 2, ..., 40

ax.set_ylim(-1000,1000)
ax.set_ylabel(r'$\Delta$ Storage (TAF)', fontweight='bold')
ax.set_xlabel('Simulation Day', fontweight='bold')

for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1)

plt.savefig('/Users/williamtaylor/Documents/Github/Robust-FIRO/figures/figure_3.pdf', format='pdf', bbox_inches='tight', transparent=False)
plt.show()

