import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from os.path import splitext
import calendar
kcfs_to_tafd = 2.29568411*10**-5 * 86400


# def water_day(d):
#   return d - 274 if d >= 274 else d + 91

def calc_obs_medians(df):
  is_leap_years = np.array([calendar.isleap(d.year) for d in df.index])
  
  df['dowy'] = np.array([water_day(d, is_leap_year) for d, is_leap_year in zip(df.index, is_leap_years)])

  df_median = pd.DataFrame(index=df.index, columns=df.columns)
  df_median['dowy'] = df['dowy']

  for dowy in range(0,365):
    for k in df.columns:
      if k == 'dowy': continue
      m = np.median(df.loc[df.dowy==dowy, k])
      df_median.loc[df_median.dowy == dowy, k] = m

  # leap years
  for k in df.columns:
    if k == 'dowy': continue
    m = np.median(df.loc[df.dowy.isin([364,365]), k])
    df_median.loc[df_median.dowy == 365, k] = m

  return df_median

# forecasts must be (date, trace, lead)
# helper functions for baseline/perfect cases

def get_baseline_forecast(Q, Q_median, NL):
  T = len(Q)
  Qf = np.zeros((T, 1, NL))
  for t in range(T - NL):
    Qf[t,0,:] = Q_median[(t+1) : (t+1+NL)]
  return Qf


def get_perfect_forecast(Q, NL):
  T = len(Q)
  Qf = np.zeros((T, 1, NL))
  for t in range(T - NL):
    Qf[t,0,:] = Q[(t+1) : (t+1+NL)]
  return Qf

def water_day(d, is_leap_year):
    # Convert the date to day of the year
    day_of_year = d.timetuple().tm_yday
    
    # For leap years, adjust the day_of_year for dates after Feb 28
    if is_leap_year and day_of_year > 59:
        day_of_year -= 1  # Correcting the logic by subtracting 1 instead of adding
    
    # Calculate water day
    if day_of_year >= 274:
        # Dates on or after October 1
        dowy = day_of_year - 274
    else:
        # Dates before October 1
        dowy = day_of_year + 91  # Adjusting to ensure correct offset
    
    return dowy

