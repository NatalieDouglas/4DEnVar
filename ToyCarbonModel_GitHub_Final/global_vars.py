## Define globally used variables, tools, libraries and functions

import numpy as np
import matplotlib.pyplot as plt
import functools
import importlib
import seaborn as sns
import sys
sys.path.insert(1, '/Users/nataliedouglas/Documents/Research/Reading University Work/ToyCarbonModel_GitHub_Final')
sys.path.insert(1, '/Users/nataliedouglas/Documents/Research/Reading University Work/ToyCarbonModel_GitHub_Final/Classes')

## Import function classes
import fourDEnVar_engine
importlib.reload(fourDEnVar_engine)
import modelruns_class
importlib.reload(modelruns_class)
import genensemble_class
importlib.reload(genensemble_class)
import genDAfeatures_class
importlib.reload(genDAfeatures_class)
import gensurfaces_class
importlib.reload(gensurfaces_class)

## Define variables for model runs
g_x0 = [1,1] # True initial conditions
g_t0=0 # Start time
g_tf = 1000 # Final time
g_discard = 200
g_deltat = 1 # Time step
g_t1 = np.arange(g_t0, g_tf + g_discard + g_deltat / 2, g_deltat) 
g_t2 = np.arange(g_t0, g_tf + g_deltat / 2, g_deltat) 
g_forcing_long = np.load('forcing.npy')
g_forcing = g_forcing_long[g_discard:g_discard+g_tf+1]


## Truth runs and parameter settings for each of the parameter sets
## this line can only be executed after initial model runs
filepath = '/Users/nataliedouglas/Documents/Research/Reading University Work/ToyCarbonModel_GitHub_Final'
g_xtrue_noise = np.load(filepath+'/Model Runs/xtrue_noisy_paramsref.npy') 
g_paramtrue = [1.0, 1.0, 0.2, 0.1] # reference parameters
## use x_ens=np.load('ideal_ensemble_50.npy')

#g_xtrue_noise = np.load(filepath+'/Model Runs/xtrue_noisy_paramsA1-A11.npy')     
#g_paramtrue = [1.04,1.35,0.23,0.08] # A1-A11 experiment
## use x_ens=np.load('ideal_ensemble_50.npy')

#g_xtrue_noise = np.load(filepath+'/Model Runs/xtrue_noisy_paramsB1-B6.npy')     
#g_paramtrue = [2.44,2.45,0.11,0.031] # B1-B6 experiment
## use x_ens=np.load('ideal_ensemble_50_B.npy')

#g_xtrue_noise = np.load(filepath+'/Model Runs/xtrue_noisy_paramsC1.npy')  
#g_paramtrue = [2.44,2.45,0.11,0.011] # C1 experiment
## use x_ens=np.load('ideal_ensemble_50.npy')

#g_xtrue_noise = np.load(filepath+'/Model Runs/xtrue_noisy_paramsC2.npy')
#g_paramtrue = [0.77,2.73,0.033,0.025] # C2 experiment
## use x_ens=np.load('ideal_ensemble_200_C2.npy')

g_xlabels=[r'$p_1$',r'$p_2$',r'$k_1$',r'$k_2$']


## Observations
g_obs_err=0.1    
g_initstate=g_xtrue_noise[:,0]
g_y=np.concatenate(g_xtrue_noise.T,axis=0)
g_R=g_obs_err*np.diag(g_y)

## Prior Information
g_nens=50
g_prior = {
        'c_p1': [2.75, (0.5, 5)],
        'd_p2': [2.75, (0.5, 5)],
        'e_k1': [0.465, (0.03, 0.9)],
        'f_k2': [0.065, (0.01, 0.12)],
               }
g_prior_err=0.25
g_xkeys = list(g_prior.keys())
g_xb = []
g_val_bnds = []
for xkey in g_xkeys:
    g_xb.append(g_prior[xkey][0])
    g_val_bnds.append(g_prior[xkey][1])
    
g_xb = np.array(g_xb)
g_xb_sd = g_xb * g_prior_err
g_B_mat = np.eye(len(g_xb))*((g_xb_sd)**2)

## Plot settings
g_palette=sns.color_palette("colorblind", 11)

## Define RMSE function used throughout
def RMSE(array1, array2):
    rmsediff=(array1-array2)**2.0
    rmsemean=np.mean(rmsediff)
    rmse=rmsemean**0.5
        
    return rmse
