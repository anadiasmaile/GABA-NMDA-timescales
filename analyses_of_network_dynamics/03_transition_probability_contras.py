#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 18:26:24 2025

@author: okohl

Calculate Transition Probabilities for all participants and contrast Lorazepam vs. Placebo.
Significance assessed with within GLMs and maximum tstatistic permutation tests pooling across
transitions, accounting for multiple comparisons.
Transition probability contrast is visualised by plotting t-statitics of contrasts and highlighting significant changes that survive multiple comparison correction with **.

This plot is shown in the lower pannerl of Figure 5.
"""
import pickle
import numpy as np
import pandas as pd
import glmtools as glm 
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore
from osl_dynamics.analysis.modes import calc_trans_prob_matrix

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


# Best Fit
n_states = 8
irun = 8

# Data Dir
demo_dir = ".../datasets/lorasick/rawdata"
preproc_dir = '.../data/hmm_in'
hmm_dir = f'.../data/inf/{n_states:02d}_states/run{irun:02d}'
output_dir = f'.../results/hmm/08_states/fooof/per_vertex_hmm/transition_probability_contrast'

# --- Load Group Info ---
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

#%% Load state probability time courses and calculate transition probabilities ---

# Get Alphas
alp = pickle.load(open(f"{hmm_dir}/alp.pkl", "rb"))

# Calculate Transition Probabilities
trans = [ calc_trans_prob_matrix(a, n_states=n_states) for a in alp]
trans = np.stack(trans, axis=0)
trans = trans[~excluded_from_stats]


#%% Contrast tranistion probabilities between Lorazepam and Placebo condition 

# flatten transition probabilities
metric = trans
metric = metric.reshape(metric.shape[0], -1) # flatten
    
# 1 Sample Test with maxTstat pooling across states
diff = metric[condition == 0] - metric[condition == 1]
t, p, _ , _, _ = condition_contrast_glms(diff,pooled_dims=(1))

# Reshape t and p
t = t.reshape([8,8])
p = p.reshape([8,8,])


#%% Plot transition probability contrast -> plot t-statistics

# Preallocate annot matrix
annot_matrix = np.full(p.shape, "", dtype=object)

# Assign significance stars
annot_matrix[p < 0.05] = "*"
annot_matrix[p < 0.01] = "**"
annot_matrix[p < 0.001] = "***"

# Find Max and min
vmax = np.nanmax(abs(t))

# Make Heatmap
plt.figure(figsize=(5,4), dpi=600)

ax = sns.heatmap(
    t,
    annot=annot_matrix,
    center=0,
    vmin=-vmax,
    vmax=vmax,
    fmt="",
    cmap="RdBu_r",
    cbar=True,
    linewidths=1,  # this enables cell borders
    linecolor='grey',  # base color for borders
    annot_kws={
        "size": 20,
        "color": "white",
        "ha": "center",
        "va": "center",
        "fontfamily": "monospace"
    },

)

# Set dashed borders for each cell
for _, spine in ax.spines.items():
    spine.set_linestyle((0, (5, 5)))  # 5pt dash, 5pt space
    spine.set_color('grey')

# Symmetric colorbar with 5 ticks
cbar = ax.collections[0].colorbar
ticks = np.linspace(-vmax, vmax, 5)
cbar.set_ticks(ticks)

# Round tick labels and set fontsize
cbar.set_ticklabels([f"{tick:.0f}" for tick in ticks], fontsize=12)

# Set colorbar label
cbar.set_label('T-Statistic', fontsize=12, rotation=270, labelpad=15)

# Set x and y axis labels
states = [f"State {i}" for i in range(1, t.shape[0]+1)]
ax.set_xticklabels(states, rotation=45, ha="right", fontsize=12)
ax.set_yticklabels(states, rotation=0, fontsize=12)

plt.tight_layout()

#Save Figure
plt.savefig(f"{output_dir}/t_star_heatmap.svg",
            transparent = False,bbox_inches="tight",format="svg")


