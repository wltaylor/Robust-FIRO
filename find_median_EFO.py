#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 12:53:20 2025

@author: williamtaylor
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('default')
import pandas as pd
import xarray as xr
import model
from time import localtime, strftime
import seaborn as sns

weight = 0.0001
#%%
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
# What do the fully trained and held out policies look like?
# visualize the median policy between random seeds
plt.figure(figsize=(8,6))
df = pd.DataFrame()

# fully trained
for i in range(0,10):
    risk_curve = np.loadtxt(f'results/EFO_policies/EFO_risk_thresholds_weight{weight}_seed{i}.csv')  
    temp_df = pd.DataFrame(risk_curve, columns=[f'Policy_{i}'])
    df = pd.concat([df, temp_df], axis=1)

for i in range(1,10):
    plt.plot(model.create_risk_curve(df.iloc[:,i].values)*100, c='grey', alpha=0.5)
median_policy = df.median(axis=1)
plt.plot(model.create_risk_curve(median_policy.values)*100, c='red', label='Median Fully Trained Policy')


# held out 1997
df_held = pd.DataFrame()

for i in range(0,10):
    risk_curve = np.loadtxt(f'results/EFO_policies/EFO_risk_thresholds_no1997_weight{weight}_seed{i}.csv')
    temp_df = pd.DataFrame(risk_curve, columns=[f'Policy_{i}'])
    df_held = pd.concat([df_held, temp_df], axis=1)

plt.plot(model.create_risk_curve(df_held.iloc[:,0].values)*100, c='grey', alpha=0.5, label='Held Out Random Seed Policies', linestyle='--')
for i in range(1,10):
    plt.plot(model.create_risk_curve(df_held.iloc[:,i].values)*100, c='grey', alpha=0.5, linestyle='--')
median_policy_held = df_held.median(axis=1)
plt.plot(model.create_risk_curve(median_policy_held.values)*100, c='blue', label='Median Held Out Policy')
plt.xlabel('Lead days')
plt.ylabel('Risk percent')
plt.legend()
plt.title('HEFS EFO Policies')
plt.show()

# # save the median policies
np.savetxt(f'results/EFO_policies/EFO_risk_thresholds_weight{weight}.csv', median_policy)
np.savetxt(f'results/EFO_policies/EFO_risk_thresholds_no1997_weight{weight}.csv', median_policy_held)

#%%
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
# what about all the synthetic policies?
plt.figure(figsize=(8,6))
df = pd.DataFrame()
for i in range(0,100):
    risk_curve = np.loadtxt(f'results/EFO_policies/synthetic_risk_thresholds_weight{weight}_{i}.csv')  
    temp_df = pd.DataFrame(risk_curve, columns=[f'Policy_{i}'])
    df = pd.concat([df, temp_df], axis=1)

for i in range(0,100):
    plt.plot(model.create_risk_curve(df.iloc[:,i].values)*100, c='grey', alpha=0.5)
median_policy = df.median(axis=1)
plt.plot(model.create_risk_curve(median_policy.values)*100, c='red', label='Median Fully Trained Policy')
plt.xlabel('Lead days')
plt.ylabel('Risk percent')
plt.legend()
plt.title('Synthetic EFO Policies')
plt.show()
rng = np.random.default_rng(0)

plt.figure(figsize=(8,6))
df = pd.DataFrame()
for i in range(0,100):
    risk_curve = np.loadtxt(f'results/EFO_policies/synthetic_risk_thresholds_no1997_weight{weight}_{i}.csv')  
    temp_df = pd.DataFrame(risk_curve, columns=[f'Policy_{i}'])
    df = pd.concat([df, temp_df], axis=1)

for i in range(0,100):
    curve = model.create_risk_curve(df.iloc[:,i].values)*100
    plt.plot(curve, c='grey', alpha=0.5)
    #plt.annotate(str(i), (2+rng.uniform(-1,1)*2,curve[3]+rng.uniform(-1,1)*4))

median_held_policy = df.median(axis=1)
plt.plot(model.create_risk_curve(median_held_policy.values)*100, c='red', label='Median Fully Trained Policy')
plt.xlabel('Lead days')
plt.ylabel('Risk percent')
plt.legend()
plt.title('Synthetic EFO Policies (with 1997 held out)')
plt.show()

np.savetxt(f'results/EFO_policies/synthetic_risk_thresholds_weight{weight}.csv', median_policy)
np.savetxt(f'results/EFO_policies/synthetic_risk_thresholds_no1997_weight{weight}.csv', median_held_policy)