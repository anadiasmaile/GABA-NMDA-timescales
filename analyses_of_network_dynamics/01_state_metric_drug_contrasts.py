#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 13:51:23 2025

@author: okohl

Calculate Lorazepam vs. Placebo condition contrasts.
Contrasts are calculated with within GLMs testing differences between the conditions against 0.
Significance is assessed with maximum-tstatistic permutation tests pooling across states, which corrects for multiple comparisons across states.
"""

import numpy as np
import glmtools as glm 
from scipy.stats import percentileofscore
import pandas as pd

# run glms with max tstatistic/copes permutation tests
def condition_contrast_glms(data, metric='tstats',pooled_dims=(1),n_jobs=4):

    # Define Dataset for GLM    
    data = glm.data.TrialGLMData(data=data)
                                 
    # Specify regressors and Contrasts in GLM Model 
    DC = glm.design.DesignConfig()
    DC.add_regressor(name='Constant',rtype='Constant')
    
    DC.add_contrast(name="On > Off", values=[1])
    
    # Create design martix and fit model and grab tstats 
    des = DC.design_from_datainfo(data.info)
    model = glm.fit.OLSModel(des,data)
     
    # Permutation Test Pooling Across States 
    perm = glm.permutations.MaxStatPermutation(des, data, 0, nperms=10000,
                                            metric=metric, nprocesses=n_jobs,
                                            pooled_dims=pooled_dims)
       
    thresh = perm.get_thresh([95])
     
    # Get p-values and significance mask
    if pooled_dims:
        if metric == "tstats":
            metrics = model.tstats[0]
            percentiles = percentileofscore(perm.nulls, abs(metrics))            
            mask = abs(metrics) > thresh[0]
        elif metric == "copes":
            metrics = model.copes[0]
            percentiles = percentileofscore(perm.nulls, abs(metrics))
            mask = abs(metrics) > thresh[0]
        pvalues = 1 - percentiles / 100 
    else:
        if metric == "tstats":
            metrics = model.tstats[0]
            percentiles = [percentileofscore(perm.nulls[:,i], abs(metrics[i])) for i in range(perm.nulls.shape[1])] 
            percentiles = np.array(percentiles)    
            mask = abs(metrics) > thresh[0]
        elif metric == "copes":
            metrics = model.copes[0]
            percentiles = [percentileofscore(perm.nulls[:,i], abs(metrics[i])) for i in range(perm.nulls.shape[1])] 
            percentiles = np.array(percentiles)  
            mask = abs(metrics) > thresh[0]
        pvalues = 1 - percentiles / 100
    
    return metrics, pvalues, model.dof_model, thresh, mask

n_states = 8
irun = 8

# Set dirs
hmm_dir = f".../data/inf/{n_states:02d}_states/run{irun:02d}"
stats_dir = f'.../analysis/hmm/state_metrics_contrasts/{n_states:02d}_states'
demo_dir = ".../datasets/lorasick/rawdata"

# Prepare Session information
df = pd.read_csv( f'{demo_dir}/sessions.tsv',sep='\t',header=0)
exclude = (
    (df["participant_id"] == 14)
    | (df["participant_id"] == 44)
    | ((df["participant_id"] == 28) & (df["session_id"] == 1))
)

# Stats
excluded_from_stats = df["participant_id"][~exclude] == 28
condition = df['drug_id'][~exclude][~excluded_from_stats]

#%% Loar vs Placebo
for label in ['fo','lt','intv','sr','cr']:
    
    # Load State MNetrics
    metric = np.load(f"{hmm_dir}/{label}.npy" )[~excluded_from_stats]
    
    # Calculate difference between COnditions
    diff = metric[condition == 0] - metric[condition == 1]
    
    # Calculate 1 Sample Ttest
    t, p, dof , _, _ = condition_contrast_glms(diff,pooled_dims=(1))
    
    np.save(f'{stats_dir}/{label}_lora_pvalues.npy',p)
    
    print(f"Lora vs Placebo {label}:")
    print(t)
    print(p)
    
 
