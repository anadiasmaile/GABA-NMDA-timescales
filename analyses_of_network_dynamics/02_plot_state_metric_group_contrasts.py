#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 14:00:49 2025

@author: okohl

Make upper half of Figure 5.
Plot GLMs contrasting state metrics between Lorazepam and placebo condition.

Stats are loaded from outputs of previous script.
"""

import numpy as np
import glmtools as glm 
from scipy.stats import percentileofscore
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns 
from matplotlib.lines import Line2D

n_states = 8
irun = 8

# Set dirs
hmm_dir = f".../data/inf/{n_states:02d}_states/run{irun:02d}"
stats_dir = f'.../analysis/hmm/state_metrics_contrasts/{n_states:02d}_states'
out_dir = f'.../results/hmm/{n_states:02d}_states/state_metrics'
demo_dir = ".../datasets/lorasick/rawdata"

# Prepare Session information
df = pd.read_csv( f'{demo_dir}/sessions.tsv',sep='\t',header=0)
exclude = (
    (df["participant_id"] == 14)
    | (df["participant_id"] == 44)
    | ((df["participant_id"] == 28) & (df["session_id"] == 1))
)

excluded_from_stats = df["participant_id"][~exclude] == 28
condition = df['drug_id'][~exclude][~excluded_from_stats]

# Keep only lora and plac
condition_mask = condition < 2
condition = condition[condition_mask]

#%%  Set up plot of Lora vs Placebo State Metric Contrast
cols = ["#66C2A5","#B3B3B3","#FC8D62"] 
xticks = ['State\n1','State\n2','State\n3','State\n4',
          'State\n5','State\n6','State\n7','State\n8',
          'State\n9','State\n10','State\n11','State\n12']
ylabels = ['Fractional Occupancy','Lifetimes (ms)','Interval Times (sec)','State Rates']

# Initialise figure
fig = plt.figure(figsize=(8,6), dpi=600)

gs1 = GridSpec(1, 1, left=0.05, right=0.95, top=.98, bottom=.55 ,hspace=0.05)
ax1 = fig.add_subplot(gs1[0, 0])

gs2 = GridSpec(1, 3, left=0.05, right=0.95, top=.40, bottom=.02 , wspace=0.45)
ax2 = fig.add_subplot(gs2[0, 0])
ax3 = fig.add_subplot(gs2[0, 1])
ax4 = fig.add_subplot(gs2[0, 2])

#%% --- Figure 1: FO for all States ----

# --- Looad metric and p valiues ---
metric = np.load(f"{hmm_dir}/fo.npy")[~excluded_from_stats][condition_mask]
ps_lora = np.load(f'{stats_dir}/fo_lora_pvalues.npy')

# Bring Data into longformat for plotting
df = []
for iState in range(metric.shape[1]):
    
    # Get Vars vor Columns
    m = metric[:,iState]
    c = condition
    s = np.ones(len(metric)) * (iState + 1) 
    
    # Stack Horizontally and put into longformat
    df.append(np.vstack([m,c,s]).T)

df = pd.DataFrame(np.vstack(df),columns=["metric","condition","state"])

# --- Make Boxplot with 1 dot per participant --- 
bplot = sns.boxplot(x="state", y='metric', hue='condition', data=df, 
                    palette=['white','white'], width=.7, legend=False, 
                    showfliers=False,ax = ax1)
points = sns.stripplot(x="state", y='metric',hue='condition',data=df, 
              palette=cols, size=3.5, dodge=True, 
              legend=False,ax = ax1) 
 
# Make Axis pretty
ax1.xaxis.labelpad = 10
ax1.set_xticklabels(xticks, fontsize=14)
ax1.set_xlabel('', fontsize=16)
ax1.locator_params(axis="y", nbins=5)
ax1.set_ylabel(ylabels[0], fontsize = 16, labelpad = 5)

# Legend
custom_lines = [Line2D([0], [0], color=cols[0], lw=0, marker="o", markersize=10),
                Line2D([0], [0], color=cols[1], lw=0, marker="o",markersize=10),]

legend = ax1.legend(custom_lines,['Lorazepam','Placebo'],
             bbox_to_anchor=(1, 1.02),
             frameon=False,
             facecolor='white',
             framealpha=1,
             prop={"size": 14},
             labelspacing=0.1,
             handletextpad=.6,
             handlelength=1,
             #title="Group",
             #title_fontsize=12,
             )
  
# Remove Box Around Subplot
sns.despine(ax=ax1 , top=True, right=True, left=False,
        bottom=False, offset=None, trim=False)


# Significance of Condition Contrast
for iState in range(metric.shape[1]):
    # Get Significance stars 
    if ps_lora[iState] < .01:      
        p = '**' 
        x = (iState) - .185    
        
        y = np.max(metric[:,iState]) * 1.07 
        ax1.text(x=x , y=y , s=p , zorder=10, size=20)
        ax1.plot([iState - .18, iState + .15], [y, y], 'k-', lw=1.5) 
    elif ps_lora[iState] < .05:
        p = "*" 
        x = (iState) - .085  
        
        y = np.max(metric[:,iState]) * 1.07
        ax1.text(x=x , y=y , s=p , zorder=10, size=20)
        ax1.plot([iState - .18, iState + .15], [y, y], 'k-', lw=1.5) 
    else:
        p = ""
         
    
#%% --- Make Boxplots with 1 dot per participant for other state metrics ---
ylabels = ['Lifetimes (ms)','Interval Times (sec)','State Rates']
xticks = ['State\n1','State\n3']

for ind, (lab, ax) in enumerate(zip(['lt','intv','sr'], [ax2, ax3, ax4])):
    
    # Looad metric and p valiues
    metric = np.load(f"{hmm_dir}/{lab}.npy")[~excluded_from_stats][condition_mask]
    ps_lora = np.load(f'{stats_dir}/{lab}_lora_pvalues.npy')
    
    # Bring Data into longformat for plotting
    df = []
    for iState in [0,2]:#range(metric.shape[1]):
        
        # Get Vars vor Columns
        m = metric[:,iState]
        c = condition
        s = np.ones(len(metric)) * (iState + 1) 
        
        # Stack Horizontally and put into longformat
        df.append(np.vstack([m,c,s]).T)
    
    df = pd.DataFrame(np.vstack(df),columns=["metric","condition","state"])
    
    # Make PLott 
    
    # --- Add Burst x State Overlap Figure ----
    bplot = sns.boxplot(x="state", y='metric', hue='condition', data=df, 
                        palette=['white','white'], width=.7, legend=False, 
                        showfliers=False,ax = ax)
    points = sns.stripplot(x="state", y='metric',hue='condition',data=df, 
                  palette=cols, size=3, dodge=True, 
                  legend=False,ax = ax) 
     
    # Make Axis pretty
    ax.xaxis.labelpad = 10
    ax.set_xticklabels(xticks, fontsize=14)
    ax.set_xlabel('', fontsize=16)
    ax.locator_params(axis="y", nbins=5)
    ax.set_ylabel(ylabels[ind], fontsize = 18, labelpad = 5)
    
  
    # Remove Box Around Subplot
    sns.despine(ax=ax , top=True, right=True, left=False,
            bottom=False, offset=None, trim=False)
    
    
    ps_in_lora = ps_lora[[0,2]]
    metric_in = metric[:,[0,2]]
    
    # Significance of Condition Contrast
    for iState in range(2):
        # Get Significance stars 
        if ps_in_lora[iState] < .01:      
            p = '**' 
            x = (iState) - .185    
            
            y = np.max(metric_in[:,iState]) * 1.07 
            ax.text(x=x , y=y , s=p , zorder=10, size=20)
            ax.plot([iState - .18, iState + .11], [y, y], 'k-', lw=1.5) 
        elif ps_in_lora[iState] < .05:
            p = "*" 
            x = (iState) - .085 
            
            y = np.max(metric_in[:,iState]) * 1.07
            ax.text(x=x , y=y , s=p , zorder=10, size=20)
            ax.plot([iState - .18, iState + .15], [y, y], 'k-', lw=1.5) 
        else:
            p = ""
   
#Save Figure
plt.savefig( f"{out_dir}/state_metric_contrast_overview.svg",
            transparent = True,bbox_inches="tight",format="svg")  
