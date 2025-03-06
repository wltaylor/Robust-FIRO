# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 13:10:44 2024

@author: zpb4
"""
import numpy as np
import multiprocessing as mp

#input is a n x n_ens x n_leads HEFS 'forward-looking' forecast
#rearranges forecast to align each lead with target observation
def onesamp_forecast_rearrange(forc_ensemble):
    n,ne,nl = np.shape(forc_ensemble)
    forc_out = np.zeros_like(forc_ensemble)
    for i in range(nl):
        forc_out[(i+1):,:,i] = forc_ensemble[:(n-(i+1)),:,i]
    return forc_out

#input is a n_samp x n x n_ens x n_leads syn-HEFS 'forward-looking' forecast
#rearranges syn-forecasts to align each lead with target observation
def multisamp_forecast_rearrange(forc_ensemble):
    nsamp,n,ne,nl = np.shape(forc_ensemble)
    forc_out = np.zeros_like(forc_ensemble)
    for i in range(nsamp):
        forc_out[i,:,:,:] = onesamp_forecast_rearrange(forc_ensemble[i,:,:,:])
    return forc_out

#baseline eCRPS function
#'ensemble' is the ensemble prediction at timestep t
#'tgt' is the target observation

def ensemble_crps(ensemble,tgt):
    ne = len(ensemble)
    term1 = (1/ne) * np.sum(np.abs(ensemble - tgt))
    #elementwise matrix calculation for term2
    mat1 = np.reshape(np.repeat(ensemble,ne),(ne,ne))
    mat2 = np.reshape(np.tile(ensemble,ne),(ne,ne))
    idx = np.tril_indices(ne)
    term2 = np.abs(mat1[idx] - mat2[idx])
    
    term2_result = (1 / (ne * (ne-1))) * np.sum(term2)
    out = term1 - term2_result
    return out

#apply ensemble_crps function to an ensemble and target timeseries (e.g. one forecast lead time)
#'ensemble' is a n x n_ens array if 'forc=True' or a n_ens x n array if 'forc=False'
#'tgt' is a length n vector of target observations
#returns eCRPS value for each ensemble/tgt pair
def ensemble_crps_sample(ensemble,tgt,forc=False):
    if forc == False:
        ne,n = np.shape(ensemble)
        inp_ens = np.copy(ensemble)
    else:
        n,ne = np.shape(ensemble)
        ens = np.copy(ensemble)
        inp_ens = np.transpose(ens)
    ecrps_out = np.empty(n)
    for i in range(n):
        ecrps_out[i] = ensemble_crps(inp_ens[:,i],tgt[i])
    ecrps_out[ecrps_out < 0] = 0
    return ecrps_out

#implementation of ensemble_crps_sample with a percentile specification
#'pcntile' should be a tuple with upper and lower value, e.g. (0,1) is the entire dataset, (0.5,1) would be the upper 50th percentile, etc
#unless wanting to sort by forecast value, leave 'forc_sort=False'
#returns the timeseries of ecrps values in index 0 and the mean of those values in index 1
def onesamp_ecrps(ensemble,tgt,pcntile,forc_sort=False):
    n,ne = np.shape(ensemble)
    lwr_idx = int(pcntile[0]*n)
    upr_idx = int(pcntile[1]*n-1)
    if forc_sort == False:
        obs_srt = np.sort(tgt)
        rtn_idx = np.where((tgt >= obs_srt[lwr_idx]) & (tgt <= obs_srt[upr_idx]))[0]
    elif forc_sort == True:
        ensmean = ensemble[:,:].mean(axis=1)
        forc_srt = np.sort(ensmean)
        rtn_idx = np.where((ensmean >= forc_srt[lwr_idx]) & (ensmean <= forc_srt[upr_idx]))[0]
        
    ecrps_out = ensemble_crps_sample(ensemble[rtn_idx,:],tgt[rtn_idx],forc=True)
    ecrps_mean = np.mean(ecrps_out)
    
    return ecrps_out,ecrps_mean

#as above for multisample application to syn-HEFS
#'par=True' means running with parallel processing
#this function can be quite slow, parallel processing can help on an HPC
#returns a distribution of size n_samp of the eCRPS means for the syn-HEFS samples
def multisamp_ecrps(ensemble,tgt,pcntile,par=True,forc_sort=False):
    nsamp,n,ne = np.shape(ensemble)
    lwr_idx = int(pcntile[0]*n)
    upr_idx = int(pcntile[1]*n-1)
    rtn_idx = []
    if forc_sort == False:
        obs_srt = np.sort(tgt)
        for k in range(nsamp):
            rtn_idx.append(np.where((tgt >= obs_srt[lwr_idx]) & (tgt <= obs_srt[upr_idx]))[0])
    elif forc_sort == True:
        for k in range(nsamp):
            ensmean = ensemble[k,:,:].mean(axis=1)
            forc_srt = np.sort(ensmean)
            rtn_idx.append(np.where((ensmean >= forc_srt[lwr_idx]) & (ensmean <= forc_srt[upr_idx]))[0])
    if par==False:
        ecrps_out = np.arange(nsamp)
        for i in range(nsamp):
            ecrps = ensemble_crps_sample(ensemble[i,rtn_idx[i],:],tgt[rtn_idx[i]],forc=True)
            ecrps_out[i] = np.mean(ecrps)
            
    elif par==True:
        global ecrps_par_fun
        def ecrps_par_fun(i):
            ecrps = ensemble_crps_sample(ensemble[i,rtn_idx[i],:],tgt[rtn_idx[i]],forc=True)
            ecrps_mean = np.mean(ecrps)
            return ecrps_mean
    
        pool = mp.Pool(mp.cpu_count()-2)
        ecrps_out = pool.map_async(ecrps_par_fun,np.arange(nsamp)).get()
        pool.close()
    return ecrps_out