import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from util import water_day
from numba import njit
import calendar

kcfs_to_tafd = 2.29568411*10**-5 * 86400
K = 3524 # TAF
Rmax = 150 * kcfs_to_tafd # estimate - from MBK, this is correct

# ramping rate from ORO WCM is 10,000 cfs every two hours for an increase and 5,000 cfs every two hours for a decrease
ramping_rate = 120 * kcfs_to_tafd

def extract_obs(sd,ed,gen_path='r-gen'):
    
    if gen_path == 'py-gen':
        df = pd.read_csv('data/Qobs.csv', index_col=0, parse_dates=True)[sd:ed]
    
    if gen_path == 'r-gen':
        df = pd.read_csv('%s/data/observed_flows.csv', index_col=0, parse_dates=True)[sd:ed]

    df = df * kcfs_to_tafd
    Q = df['ORDC1'].values 
    #Q_MSG = df['MSGC1L'].values

    dowy = np.array([water_day(d, calendar.isleap(d.year)) for d in df.index])
    
    return Q,dowy
    
def extract(sd,ed,forecast_type,gen_path,syn_sample):
    
    #df_tocs = pd.read_csv('data/precip/precip-based_tocs.csv', index_col=0, parse_dates=True)[sd:ed]
    #tocs = df_tocs['tocs'].values
    
    if gen_path == 'py-gen':
        #path = 'data/Qf-%s.nc' % (forecast_type)
        path = 'data/Qf-hefs.nc'
        da = xr.open_dataset(path)['Qf'] 

        df = pd.read_csv('data/observed_flows.csv', index_col=0, parse_dates=True)[sd:ed]
    
    if gen_path == 'r-gen':
        path = 'data/Qf-%s.nc' % (forecast_type)
        da = xr.open_dataset(path)[forecast_type]
        df = pd.read_csv('data/observed_flows.csv', index_col=0, parse_dates=True)[sd:ed]
    
    if gen_path == 'hefs-86':
        da = xr.open_dataset('data/Qf-hefs86.nc')
        df = pd.read_csv('data/observed_flows.csv', index_col=0, parse_dates=True)[sd:ed]

    df = df * kcfs_to_tafd
    Q = df['ORDC1'].values 
    #Q_MSG = df['MSGC1L'].values

    dowy = np.array([water_day(d, calendar.isleap(d.year)) for d in df.index])
    #dowy -=1
    # if the start date is in a leap year adjust by 1
    
    # (ensemble: 4, site: 2, date: 15326, trace: 42, lead: 15)
    if forecast_type == 'hefs':
        if gen_path == 'py-gen':
            Qf = da.sel(ensemble='HEFS', site='ORDC1', date=df.index).values * kcfs_to_tafd # np.array (time, trace, lead)
            #Qf_MSG = da.sel(ensemble='HEFS', site='MSGC1L', date=df.index).values * kcfs_to_tafd # np.array (time, trace, lead)
        if gen_path == 'r-gen':
            Qf = da.sel(ensemble=0, site=2, date=df.index).values * kcfs_to_tafd # np.array (time, trace, lead)

        if gen_path == 'hefs-86':
            Qf = da['hefs'].sel(site=2, lead=slice(0,13), ensemble=0, date=slice(sd, ed)) * kcfs_to_tafd
    if forecast_type == 'syn':
        if gen_path == 'py-gen':
            Qf = da.sel(ensemble=syn_sample, site='ORDC1', date=df.index).values * kcfs_to_tafd # np.array (time, trace, lead)
            #Qf_MSG = da.sel(ensemble=syn_sample, site='MSGC1L', date=df.index).values * kcfs_to_tafd # np.array (time, trace, lead)
        if gen_path == 'r-gen':
            #Qf = da.sel(ensemble=int(syn_sample[5:])-1, site=2, date=df.index).values * kcfs_to_tafd # np.array (time, trace, lead)
            Qf = da.sel(ensemble=syn_sample, site=2, date=df.index).values * kcfs_to_tafd # np.array (time, trace, lead)
    
    
    df_idx = df.index
    
    #return Q,Q_MSG,Qf,Qf_MSG,dowy,tocs,df_idx
    return Q, Qf, dowy, df_idx

# helper functions
@njit
def get_tocs(d):
    tp = np.array([0, 7, 140, 340, 366], dtype=np.float64)
    sp = np.array([K, 0.5*K, 0.5*K, K, K], dtype=np.float64)
    return np.interp(d, tp, sp)

@njit
def firo_curve(d, firo_pool):
  tp = np.array([0, 7, 140, 340, 366], dtype=np.float64)
  sp = np.array([K, (0.5+firo_pool)*K, (0.5+firo_pool)*K, K, K], dtype=np.float64)
  return np.interp(d, tp, sp)

@njit
def create_risk_curve(x):
  # convert 0-1 values to non-decreasing risk curve
    x_copy = np.copy(x)
    for i in range(1, len(x_copy)):
        x_copy[i] = x_copy[i-1] + (1 - x_copy[i-1]) * x_copy[i]
    return x_copy

@njit
def clip(x, l, u):
    return max(min(x, u), l)


@njit
def daily_opt(S0, fmax, Q, Qf_summed_sorted, ix):
    ne,nl = Qf_summed_sorted.shape
    rel_ld = 0
    Qf_quantiles = np.zeros(nl)
    # iterate over lead times
    for l in range(nl):
        Qf_quantiles[l] = Qf_summed_sorted[ix[l],l]
    
    # vectorized computation of future storage
    Sf_q = S0 + Qf_quantiles
    # vectorized computation to find where future storage exceeds fmax and calculate the required release
    releases = np.maximum(Sf_q - fmax,0) / (np.arange(nl) + 1)
    # find the max release and the corresponding lead time
    R = np.max(releases)
    #print("Sf_q:", Sf_q)
    #print("Releases:",releases)
    #print("R:",R)

    return R

def exploratory_daily_opt(S0, fmax, Q, Qf_summed_sorted, ix):
    ne,nl = Qf_summed_sorted.shape
    rel_ld = 0
    Qf_quantiles = np.zeros(nl)
    # iterate over lead times
    for l in range(nl):
        Qf_quantiles[l] = Qf_summed_sorted[ix[l],l]
    
    # vectorized computation of future storage
    Sf_q = S0 + Qf_quantiles
    # vectorized computation to find where future storage exceeds fmax and calculate the required release
    releases = np.maximum(Sf_q - fmax,0) / (np.arange(nl) + 1)
    # find the max release and the corresponding lead time
    R = np.max(releases)
    
    df = pd.DataFrame({
        'Qf_quantiles':Qf_quantiles,
        'Predicted_storage':Sf_q,
        'Required_releases':releases
        })

    return df

@njit
def simulate_baseline(Q, Qf, tocs, Q_avg):
  T = len(Q)
  S = np.full(T+1, np.nan)
  R = np.full(T, np.nan)
  spill = np.zeros(T)
  S[0] = 0 # start with no storage
  
  for t in range(T):
      R[t] = Q_avg[t] # release the median inflow by day of water year
      if S[t] > 0:
          R[t] = S[t] # if the storage is above 0 (TOCS) release enough to get back down to 0
      if R[t] > Rmax:
          R[t] = Rmax
      if np.abs(R[t] - R[t-1]) > ramping_rate:
          R[t] = R[t-1] + np.sign((R[t] - R[t-1])) * ramping_rate
      if S[t] + Q[t] - R[t] > K:
          spill[t] = S[t] + Q[t] - R[t] - K
        
      S[t+1] = S[t] + Q[t] - R[t] - spill[t]

  return S[:T], R, spill



@njit 
def simulate_EFO(firo_pool, ix, Q, Qf, dowy, tocs, S_start):
    T = len(Q)
    S = np.full(T+1, np.nan)
    R = np.full(T, np.nan)
    spill = np.zeros(T)
    firo = np.ones(T) * firo_pool # change to np.ones if running with respect to K
    S[0] = S_start
    
    for t in range(T):
        R[t] = daily_opt(S[t], firo[t], Q[t], Qf[t-1,:,:], ix)
        if R[t] > Rmax:
            R[t] = Rmax
        if np.abs(R[t] - R[t-1]) > ramping_rate:
            R[t] = R[t-1] + np.sign((R[t] - R[t-1])) * ramping_rate
        if S[t] + Q[t] - R[t] > K:
            spill[t] = S[t] + Q[t] - R[t] - K
          
        S[t+1] = S[t] + Q[t] - R[t] - spill[t]
    return S[:T], R, spill

#@njit 
def simulate_EFO_annual(firo_pool, ix, Q, Qf, dowy, tocs, S_start, R_med):
    '''
    Parameters
    ----------
    firo_pool : int
        Allowable FIRO pool limit.
    ix : array
        Risk thresholds, translated to indices of the 40-member ensemble.
    Q : array
        Inflow values.
    Qf : array [day, ensemble, lead]
        Forecasted inflow values.
    dowy : array
        Day of water year.
    tocs : array
        Top of conservation storage limit, based on DOWY.
    S_start : float
        Observed historical storage level on October 1 of a specific water year.
    R_avg : array
        Median release for this day of the year based on historical demand factor

    Returns
    -------
    S : array
        Daily simulated storage values.
    R : array
        Daily simulated release values.
    spill : array
        Daily calculated spill (usually 0).

    '''
    T = len(Q)
    S = np.full(T+1, np.nan)
    R = np.full(T, np.nan)
    spill = np.zeros(T)
    firo = np.ones(T) * firo_pool # start with this as a flat value through the year, may transition to a TOCS-style value
    S[0] = S_start
    
    for t in range(T):
        R[t] = R_med[t]
        R_firo = daily_opt(S[t], firo[t], Q[t], Qf[t-1,:,:], ix)
        #print('FIRO R: ' + str(R_firo))
        #print('Median R: '+str(R_med[t]))
        if R_firo > R[t]:
            R[t] = R_firo
        if R[t] > Rmax:
            R[t] = Rmax
        if np.abs(R[t] - R[t-1]) > ramping_rate:
            R[t] = R[t-1] + np.sign((R[t] - R[t-1])) * ramping_rate
        if S[t] + Q[t] - R[t] > K:
            spill[t] = S[t] + Q[t] - R[t] - K
          
        S_target = S[t] + Q[t] - R[t] - spill[t]
        
        if S_target < K * 0.05: # deadpool limit
            R[t] -= (K * 0.05 - S_target)
        S[t+1] = S[t] + Q[t] - R[t] - spill[t]
    return S[:T], R, spill

def simulate_cumulative(K, S_start, Q, Qf, firo_pool):
    """
    Simulate reservoir operations with the cumulative inflow method

    Args:
        S_start (float): initial storage value (acre-feet)
        Q (array): array of observed inflow values (acre-feet)
        Qf (array): array of cumulative forecasted inflow values (acre-feet)
        firo_pool (scalar): size of reserved FIRO pool (acre-feet)
    """
    nl = Qf.shape[1]
    T = len(Q)
    S = np.full(T+1, np.nan)
    R = np.full(T, np.nan)
    tocs = np.ones(T)*firo_pool
    spill = np.zeros(T)
    S[0] = S_start
    for t in range(T):
        S_placeholder = np.zeros(nl)
        # calculate forecasted storage
        for l in range(0,nl):
            S_placeholder[l] = S[t] + Qf[t-1,l]
        releases = np.maximum(S_placeholder - tocs[t],0) / (np.arange(nl) + 1)
        # choose largest
        R_firo = np.max(releases)
        
        # simulate reservoir timestep
        R[t] = R_firo
        if R_firo > Rmax:
            R[t] = Rmax
        if np.abs(R[t] - R[t-1]) > ramping_rate:
            R[t] = R[t-1] + np.sign((R[t] - R[t-1])) * ramping_rate
        if S[t] + Q[t] - R[t] > K:
            spill[t] = S[t] + Q[t] - R[t] - K
        
        S[t+1] = S[t] + Q[t] - R[t] - spill[t]
    
    return S[:T], R, spill

@njit
def objective(S,R,Rmax,spill):
  obj = 0
  #obj = S.mean()
  obj += np.sum(S < 0) * 0.01 # small penalty for going below TOCS
  obj += np.sum(S > 0) * 1 # penalize for going above TOCS
  obj += np.sum(R > Rmax) * 100 # large penalty for releases above downstream max
  obj += np.sum(spill > 0) * 100 # large penalty for any spill at all

  return obj

@njit
def magnitude_objective(S,R,Rmax,spill, weight):
  obj = 0
  T = len(S)
  obj += np.sum(np.abs(S[S < 0])) * weight # small penalty for going below TOCS
  obj += np.sum(np.abs(S[S > 0])) * (1-weight) # penalize for going above TOCS
  obj += np.sum(R[R > Rmax]) * 10000000 # large penalty for releases above downstream max
  obj += np.sum(spill[spill > 0]) * 10000000 # large penalty for any spill at all

  return obj

# @njit
# def objective(S,R,Rmax,spill):
#   obj = 0
#   obj = np.sum(np.abs(S)**2)
#   obj += np.sum(R > Rmax) * 10 # large penalty for releases above downstream max
#   obj += np.sum(spill > 0) * 10 # large penalty for any spill at all

#   return obj

# @njit
# def objective(S,R,Rmax,spill):
#   obj = 0
#   obj = np.std(S)
#   obj += np.sum(R > Rmax) * 10 # large penalty for releases above downstream max
#   obj += np.sum(spill > 0) * 10 # large penalty for any spill at all

#   return obj


# def plot_results(Q,S,R,tocs,firo,spill,Q_cp,df_idx,title):
#     plt.figure(figsize=(10,8))
#     plt.subplot(3,1,1)
#     plt.plot(df_idx, tocs, c='gray')
#     plt.plot(df_idx, firo, c = 'green')
#     plt.plot(df_idx, S, c='blue')
#     plt.axhline(K, color='red')
#     plt.ylabel('TAF')
#     plt.ylim([0, K+50])
#     plt.gcf().autofmt_xdate()
#     plt.legend(['TOCS','FIRO Pool','Storage'], loc = 'lower right')

#     plt.subplot(3,1,2)
#     plt.plot(df_idx, Q / kcfs_to_tafd)
#     # plt.plot(df.index, (R+spill) / cfs_to_tafd)
#     #plt.plot(df_idx, Q_cp / kcfs_to_tafd)
#     plt.axhline(Rmax / kcfs_to_tafd, color='red')
#     plt.ylabel('kcfs')
#     plt.gcf().autofmt_xdate()
#     plt.legend(['Inflow', 'Q_cp', 'Max safe release'], loc = 'lower right')
    
#     plt.subplot(3,1,3)
#     plt.plot(df_idx, spill)
#     plt.legend(['Spill'], loc = 'lower right')
#     plt.suptitle(title)
#     plt.show()

def plot_results(Q,S,R,tocs,spill,df_idx,title):
    plt.figure(figsize=(10,8))
    plt.subplot(3,1,1)
    plt.plot(df_idx, tocs, c='gray', label='TOCS')
    plt.plot(df_idx, S, c='green', label='FIRO')
    plt.axhline(K, color='red', linestyle='--')
    plt.ylabel('TAF')
    #plt.ylim([0, K+50])
    plt.gcf().autofmt_xdate()
    #plt.legend(loc = 'lower right')

    plt.subplot(3,1,2)
    plt.plot(df_idx, Q, c='blue', alpha=0.5, label='Inflow')
    plt.plot(df_idx, R, c='green', label='FIRO')
    plt.axhline(Rmax, color='red', linestyle='--')
    plt.ylabel('Release (TAF)')
    plt.gcf().autofmt_xdate()
    plt.legend(loc = 'lower right')
    
    plt.subplot(3,1,3)
    plt.plot(df_idx, spill)
    #plt.legend(loc = 'lower right')
    plt.suptitle(title)
    plt.show()