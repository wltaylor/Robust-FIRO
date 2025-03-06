import numpy as np
import matplotlib.pyplot as plt
plt.style.use('default')
import pandas as pd
import model
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42


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

plt.plot(model.create_risk_curve(df.iloc[:,0].values)*100, c='grey', alpha=0.5, label='Full Trained Random Seed Policies')
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
#%% all EFO policies on one figure
weight = weight
leads = np.arange(1,15,1)
#colors = sns.color_palette('tab10')
colors = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02','#a6761d','#666666']

fig, ax = plt.subplots(1,3,figsize=(12,4))

# ax0
df = pd.DataFrame()
# fully trained
for i in range(0,10):
    risk_curve = np.loadtxt(f'results/EFO_policies/EFO_risk_thresholds_weight{weight}_seed{i}.csv')  
    temp_df = pd.DataFrame(risk_curve, columns=[f'Policy_{i}'])
    df = pd.concat([df, temp_df], axis=1)

ax[0].plot(leads, model.create_risk_curve(df.iloc[:,0].values)*100, c='grey', alpha=0.25, label='Fully Trained Random Seed Policies')
for i in range(1,10):
    ax[0].plot(leads, model.create_risk_curve(df.iloc[:,i].values)*100, c='grey', alpha=0.25)
median_policy = df.median(axis=1)
ax[0].plot(leads, model.create_risk_curve(median_policy.values)*100, c=colors[3], label='Median Fully Trained Policy', linewidth=3, zorder=4)

# held out 1997
df_held = pd.DataFrame()

for i in range(0,10):
    risk_curve = np.loadtxt(f'results/EFO_policies/EFO_risk_thresholds_no1997_weight{weight}_seed{i}.csv')
    temp_df = pd.DataFrame(risk_curve, columns=[f'Policy_{i}'])
    df_held = pd.concat([df_held, temp_df], axis=1)

ax[0].plot(leads, model.create_risk_curve(df_held.iloc[:,0].values)*100, c='grey', alpha=0.25, label='Held Out Random Seed Policies', linestyle='--')
for i in range(1,10):
    ax[0].plot(leads, model.create_risk_curve(df_held.iloc[:,i].values)*100, c='grey', alpha=0.25)
median_policy_held = df_held.median(axis=1)
ax[0].plot(leads, model.create_risk_curve(median_policy_held.values)*100, c=colors[4], label='Median Held Out Policy', linewidth=3, zorder=4)
ax[0].set_xticks(leads)
ax[0].set_xticklabels(leads)
ax[0].set_xlabel('Lead days')
ax[0].set_ylabel('Risk percent')
#ax[0].legend(bbox_to_anchor=(0.5, -0.13), loc='upper center')
ax[0].set_title('(a) HEFS Policies', fontweight='bold', loc='left', fontsize=12)

# ax1
df = pd.DataFrame()
for i in range(0,100):
    risk_curve = np.loadtxt(f'results/EFO_policies/synthetic_risk_thresholds_weight{weight}_{i}.csv')  
    temp_df = pd.DataFrame(risk_curve, columns=[f'Policy_{i}'])
    df = pd.concat([df, temp_df], axis=1)

ax[1].plot(leads, model.create_risk_curve(df.iloc[:,0].values)*100, c='grey', alpha=0.25, label='Synthetic Policies')
for i in range(1,100):
    ax[1].plot(leads, model.create_risk_curve(df.iloc[:,i].values)*100, c='grey', alpha=0.25)
median_policy = np.loadtxt(f'results/EFO_policies/synthetic_risk_thresholds_weight{weight}.csv')

ax[1].plot(leads, model.create_risk_curve(median_policy)*100, c=colors[5], lw=3, label='Median Fully Trained Policy', zorder=4)
ax[1].set_xlabel('Lead days')
ax[1].set_ylabel('Risk percent')
ax[1].set_xticks(leads)
ax[1].set_xticklabels(leads)
#ax[1].legend(bbox_to_anchor=(0.5, -0.13), loc='upper center')
ax[1].set_title('(b) Synthetic Policies', fontweight='bold', loc='left', fontsize=12)

# ax2
df = pd.DataFrame()
for i in range(0,100):
    risk_curve = np.loadtxt(f'results/EFO_policies/synthetic_risk_thresholds_no1997_weight{weight}_{i}.csv')  
    temp_df = pd.DataFrame(risk_curve, columns=[f'Policy_{i}'])
    df = pd.concat([df, temp_df], axis=1)

ax[2].plot(leads, model.create_risk_curve(df.iloc[:,0].values)*100, c='grey', alpha=0.25, label='Synthetic Policies')
for i in range(1,100):
    ax[2].plot(leads, model.create_risk_curve(df.iloc[:,i].values)*100, c='grey', alpha=0.25)
median_policy = np.loadtxt(f'results/EFO_policies/synthetic_risk_thresholds_no1997_weight{weight}.csv')
ax[2].plot(leads, model.create_risk_curve(median_policy)*100, c=colors[6], lw=3, zorder=5, label='Median Held Out Policy')
ax[2].set_xlabel('Lead days')
ax[2].set_ylabel('Risk percent')
ax[2].set_xticks(leads)
ax[2].set_xticklabels(leads)
#ax[2].legend(bbox_to_anchor=(0.5, -0.13), loc='upper center')
ax[2].set_title('(c) Synthetic Policies without 1997', fontweight='bold', loc='left', fontsize=12)

legend_seeds = plt.Line2D([0],[0], c='grey', lw=3, alpha=0.25, label='EFO Policies')
legend_hefs = plt.Line2D([0],[0], c=colors[3], lw=3, label='Median HEFS')
legend_hefs_held = plt.Line2D([0],[0], c=colors[4], lw=3, label='Median HEFS without 1997')

legend_syn = plt.Line2D([0],[0], c=colors[5], lw=3, label='Median Synthetic')
legend_syn_held = plt.Line2D([0],[0], c=colors[6], lw=3, label='Median Synthetic without 1997')


fig.legend(handles=[legend_hefs, legend_hefs_held, legend_syn, legend_syn_held, legend_seeds], loc='upper center', bbox_to_anchor=(0.52, -0.01), ncol=5, edgecolor='black')

#fig.suptitle(f'EFO Policy Comparison - Weight: {weight}', fontweight='bold', fontsize=20, y=1.02)
plt.tight_layout()
plt.savefig('/Users/williamtaylor/Documents/Github/Robust-FIRO/figures/Figure_8.pdf', format='pdf', bbox_inches='tight', transparent=False)
plt.show()
#%%
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
# HEFS policies trained with each year individually
kcfs_to_tafd = 2.29568411*10**-5 * 86400
sd = '1990-10-01' 
ed = '2019-08-31'
df = pd.read_csv('data/observed_flows.csv', parse_dates=True)
df['ORDC1'] = df['ORDC1']*kcfs_to_tafd
df['date'] = pd.to_datetime(df['Date'])
df.sort_values(by='date', inplace=True)
df = df.loc[(df['date'] >= sd) & (df['date'] <= ed)]
df['water_year'] = df['date'].apply(lambda x: x.year + 1 if x.month >= 10 else x.year)
unique_water_years = df['water_year'].unique()
water_years = df['water_year'].values
unique_water_years = np.unique(water_years)

def create_risk_curve(x):
  # convert 0-1 values to non-decreasing risk curve
    x_copy = np.copy(x)
    for i in range(1, len(x_copy)):
        x_copy[i] = x_copy[i-1] + (1 - x_copy[i-1]) * x_copy[i]
    return x_copy



#%% grid version by seed

fig, axes = plt.subplots(5,6, figsize=(12,12))

axes = axes.flatten()

total_plots = len(axes)

for i, year in enumerate(unique_water_years):
    ax = axes[i]
    df = pd.read_csv(f"results/EFO_policies/EFO_risk_thresholds_{year}.csv")
    for j in range(0,9):
        ax.plot(create_risk_curve(df.iloc[j,:])*100, c='black', alpha=0.5)
    ax.set_title(f'WY {year}')
    ax.set_xlabel('Lead day')
    ax.set_ylabel('Risk (%)')

    if i % 6 != 0:
        ax.set_ylabel('')
    if i < (total_plots - 6):
        ax.set_xlabel('')

for i in range(len(unique_water_years), total_plots):
    fig.delaxes(axes[i])

fig.suptitle('HEFS EFO Policies, trained one water year at a time', fontweight='bold', fontsize=20)
plt.tight_layout()
plt.show()





#%%
scales = [100,200,300]
leads = np.arange(1,15,1)
fig, axes = plt.subplots(1,3, figsize=(12,4))
axes = axes.flatten()
for j,scale in enumerate(scales):
    for i in range(0,10):
        risk_curve = np.loadtxt(f'results/EFO_policies/EFO_risk_thresholds_scale{scale}_weight{weight}_seed{i}.csv')  
        axes[j].plot(leads, model.create_risk_curve(risk_curve)*100, c='red', alpha=0.5)

    axes[j].set_xlabel('Lead days')
    axes[j].set_title(f'Scaled event size: {scale} year')
    axes[j].set_xticks(leads)
    axes[j].set_xticklabels(leads)
    if j > 0:
        axes[j].set_yticklabels('')

axes[0].set_ylabel('Risk Percent')

fig.suptitle('Scaled EFO Policies - Trained with HEFS', fontweight='bold', fontsize=16)
plt.tight_layout()
plt.show()


