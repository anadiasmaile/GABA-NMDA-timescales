#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 14:12:11 2025

@author: okohl

Contrast network specific timescales against time-averaged timescale.

This analysis is performed for only the placebo condition.
Calculate difference between network-specific timescale from time-averaged timescale
and check whether it is different from 0.
Apply FDR correction to control for multiple comparisons.

Plot percentage of change from time-averaged time constant of parcels that cross
significance threshold.
"""

import numpy as np
import glmtools as glm 
from osl_dynamics.analysis import power
from scipy.stats import percentileofscore
import pandas as pd
import matplotlib.pyplot as plt  
import seaborn as sns 
from matplotlib.lines import Line2D
from mne.stats import fdr_correction, bonferroni_correction

import sys
sys.path.append('/.../scripts/helper')
from parc_reindexer import get_osl_reindexer

# -------------------------------

def compute_percentiles(perm_nulls, metrics):
    """
    Vectorized percentile computation.
    
    Works for:
      - perm_nulls: (n_perm, n_metrics), metrics: (n_metrics,)
      - perm_nulls: (n_perm, n_rows, n_cols), metrics: (n_rows, n_cols)
    """
    # Take absolute values of metrics
    metrics_abs = np.abs(metrics)
    
    # Compare permuted nulls against metric values with broadcasting
    # Shape: (n_perm, ...) vs (...)
    counts = (perm_nulls <= metrics_abs).sum(axis=0)
    
    # Convert to percentiles
    percentiles = 100 * counts / perm_nulls.shape[0]
    return percentiles

# run glms with max tstatistic/copes permutation tests
def within_glms(data, metric='tstats',pooled_dims=(1),n_jobs=4):

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
        metrics = model.tstats[0]
        percentiles = percentileofscore(perm.nulls, abs(metrics))            
        mask = abs(metrics) > thresh[0]
        pvalues = 1 - percentiles / 100 
    else:
        metrics = model.tstats[0]
        percentiles = compute_percentiles(perm.nulls, metrics)   
        mask = abs(metrics) > thresh[0]
        pvalues = 1 - percentiles / 100
    
    return metrics, pvalues, model.dof_model, thresh, mask


def pivot_df(df):

    pivot_df = df.pivot_table(index=["sub", "label"],columns="state",values="time_constant")
    
    # Convert to numpy
    arr = pivot_df.to_numpy()
    
    # Reshape to (sub, label, state)
    # Get sizes
    n_sub = df["sub"].nunique()
    n_label = df["label"].nunique()
    n_state = df["state"].nunique()
    
    return arr.reshape(n_sub, n_label, n_state)

# ---------------------------------

n_states = 8
irun = 8

# Set dirs
hmm_dir = f".../data/inf/{n_states:02d}_states/run{irun:02d}"
output_dir = '.../results/hmm/08_states/fooof/per_vertex_hmm/dynamic_contrasts_percentage/'
demo_dir = "/.../datasets/lorasick/rawdata"

# Prepare Session information
df = pd.read_csv( f'{demo_dir}/sessions.tsv',sep='\t',header=0)
exclude = (
    (df["participant_id"] == 14)
    | (df["participant_id"] == 44)
    | ((df["participant_id"] == 28) & (df["session_id"] == 1))
)

excluded_from_stats = df["participant_id"][~exclude] == 28
condition = df['drug_id'][~exclude][~excluded_from_stats]

# Source reconstruction files
mask_file = "MNI152_T1_8mm_brain.nii.gz"
parcellation_file = ".../osl/osl/source_recon/parcellation/files/dk_cortical.nii.gz"
reindexer, osl_labels = get_osl_reindexer() # Get vector sorting from freesurfer order into osl order

# Load Fractional Occupancies
fo = np.load(f"{hmm_dir}/fo.npy" )[~excluded_from_stats]

#%% Time Constants

# --- Load Data ---

# Load Exponents
tc = pd.read_csv('.../analysis/hmm/state_time_scales/fooof_fits_pervertex_allexp_hmm_all.csv')
tc = tc.groupby(['sub', 'drug', 'label','state'], as_index=False)['time_constant'].median()
tc = tc[~(tc['sub'] == 28)]

tc_plac = tc[tc['drug']=='plac']
tc_plac = pivot_df(tc_plac)
tc_plac = tc_plac[:,reindexer]

# --- Calculate mean time constant across all states and calculate diff and percentage change ---

metric = np.transpose(tc_plac, (0, 2, 1))

# Subtract mean across State
metric_change = np.empty(metric.shape)
metric_diff = np.empty(metric.shape)
mean_metric = np.empty([metric.shape[0],metric.shape[2]])
for iSub in range(metric.shape[0]):
    mean_metric[iSub] = np.average(metric[iSub], axis=0, weights=fo[iSub])
    metric_diff[iSub] = (metric[iSub] - mean_metric[iSub])
    metric_change[iSub] = (metric[iSub] - mean_metric[iSub]) / mean_metric[iSub] * 100 # only used for plotting
    
# --- Run within GLMs contrasting difference against 0 ---

# Calculate 1 Sample Ttest
t, p, dof , _, _ = within_glms(metric_diff,pooled_dims=[])

# FDR Correction 
fdr_corrected_pmask, lora_fdr_corrected_pvalues = fdr_correction(p)  

#%% --- Plotting ---

# --- Get one figure per state ---
for iK in range(8):
       
    maps = np.median(metric_change, axis=0)[iK] # median percentage of change
    maps[~fdr_corrected_pmask[iK]] = 0 # Set non-significant parcels to 0
    
    # Plot
    power.save(
        maps,
        parcellation_file=parcellation_file,
        mask_file=mask_file,
        component=0,
        subtract_mean=True,
        plot_kwargs={"cmap": "RdBu_r", "bg_on_data": True, 'views':['lateral']},
        filename=f"{output_dir}/state{iK}_change_fdr_thresh_percentage_of_change_.svg",
    )
    
    plt.close()
    
    

# --- Make overview Figure with all states/brain in one large figure ---
# Plotted Range (color intensities) is the same for all brains

all_maps = []
for iK in range(8):           
        maps = np.median(metric_change, axis=0)[iK]
        maps[~fdr_corrected_pmask[iK]] = 0
        
        all_maps.append(maps)

# Limits for common scaling of all brain surface plots
ymax = np.max(abs(np.array(all_maps))) 

power.save(
    all_maps,
    parcellation_file=parcellation_file,
    mask_file=mask_file,
    component=0,
    subtract_mean=False,
    plot_kwargs={"cmap": "RdBu_r", "bg_on_data": True, 'views':['lateral'], 'vmin':-ymax, 'vmax':ymax},
    combined=True,
    #titles=[f'State {iK}' for iK in range(1,9)],
    n_rows=2,
    filename=f"{output_dir}/combined_change_fdr_thresh_percentage_of_change_same_cbar_range_.png",
)
    
