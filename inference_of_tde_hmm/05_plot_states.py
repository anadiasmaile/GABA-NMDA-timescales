#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 10:46:59 2025

@author: okohl

Lorasick Dataset
Get Network Descriptions for extracted TDE-HMM states.
These figures are presented in figure 3 in the manuscript.
1) State-specific whole brain-average power spectrum vs time-averaged whole-brain average power spectrum.
2) State-specific wideband power change (relative to time-averaged wideband power).
3) State-specific coherence. Top 2% of stongest connections are plotted.
"""

import os
import numpy as np
from osl_dynamics.analysis import power, connectivity
from osl_dynamics.utils import plotting
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd
  
import sys
sys.path.append('.../scripts/helper')
from parc_reindexer import get_osl_reindexer

# Define Plotting Function

def tsplot(ax, data, mean_data,time, color_data = 'blue', color_mean = 'k', linestyle = 'solid', linewidth = 2):
    x = time
    
    # Data Line
    est = np.mean(data, axis=0)
    sd = np.std(data, axis=0)
    se = sd/np.sqrt(len(data))
    cis = (est - se, est + se)
    ax.fill_between(x,cis[0],cis[1],alpha = 0.2, facecolor = color_data)
    ax.plot(x,est,color = color_data, linestyle = linestyle, linewidth = 2.5)
    ax.margins(x=0)
    
    # Mean Data Line
    est = np.mean(mean_data, axis=0)
    sd = np.std(mean_data, axis=0)
    se = sd/np.sqrt(len(mean_data))
    cis = (est - se, est + se)
    ax.fill_between(x,cis[0],cis[1],alpha = 0.2, facecolor = color_mean)
    ax.plot(x,est,color = color_mean, linestyle = linestyle , linewidth = 1.5)
    ax.margins(x=0)
    
    # Make Axes pretty
    ax.set_xlim([2, 30])
    ax.set_ylim([0,0.12]) # 6 = .041, 8= .063, 10 = .06
    ax.set_xlabel('Frequency (Hz)',fontsize=20, labelpad=12)
    ax.set_ylabel('Power (a.u.)', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16) 
    ax.ticklabel_format(scilimits=(-1,1))

    # Set x-Ticks
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_major_formatter('{x:.0f}')    
    ax.xaxis.set_minor_locator(MultipleLocator(5))

    # Set y-ticks
    ax.yaxis.set_major_locator(MultipleLocator(.03))
    #ax.yaxis.set_major_formatter('{x:.0f}')    
    ax.yaxis.set_minor_locator(MultipleLocator(.015))

    # Despine
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)


n_states = 8
irun = 8

# Set Dirs
indir = f".../data/inf/{n_states:02d}_states/run{irun:02d}"
output_dir = '.../results/hmm/08_states/state_description/plac'

# Make output directory
os.makedirs(output_dir, exist_ok=True)

# Source reconstruction files
mask_file = "MNI152_T1_8mm_brain.nii.gz"
parcellation_file = ".../osl/osl/source_recon/parcellation/files/dk_cortical.nii.gz"
reindexer, _ = get_osl_reindexer() # Get vector sorting from freesurfer order into osl order

# Get indices of Placebo Condition
df = pd.read_csv( '.../datasets/lorasick/rawdata/sessions.tsv',sep='\t',header=0)
exclude = (
    (df["participant_id"] == 14)
    | (df["participant_id"] == 44)
    | ((df["participant_id"] == 28) & (df["session_id"] == 1))
)
plac_mask = df["drug_id"][~exclude] == 1

# Load spectra
f = np.load(indir + "/f.npy")
psd = np.load(indir + "/psd.npy")[plac_mask][:,:,reindexer]
coh = np.load(indir + "/coh.npy")[plac_mask][:,:,reindexer][:,:,:,reindexer]
w = np.load(indir + "/w.npy")[plac_mask]
fo = np.load(indir + '/fo.npy')[plac_mask]

# Subtract mean across State
mean_psd = np.empty([psd.shape[0],psd.shape[2],psd.shape[3]])
for iSub in range(psd.shape[0]):
    mean_psd[iSub] = np.average(psd[iSub], axis=0, weights=fo[iSub])
    psd[iSub] = psd[iSub] - mean_psd[iSub]
         

#%% --- Plot State-specific power spectra ---

# Calculate the group average power spectrum for each state
gpsd = psd
m_gpsd = mean_psd

# Pick Frequency Range
f_in = np.logical_and(f>=3,f<=30)
gpsd = gpsd[:,:,:,f_in]
m_gpsd = m_gpsd[:,:,f_in]
freqs = f[f_in]

# Combine mean and state specific changes
comb_gpsd = gpsd + m_gpsd[:,np.newaxis]

# --- Plot Grand Average ---
col =  [ '#727272','#3d3d3d']

for i in range(gpsd.shape[1]):
    
    # Make Plot
    fig, ax = plt.subplots()
    tsplot(ax, comb_gpsd[:,i].mean(axis=1), m_gpsd.mean(axis=1), time=freqs, color_data=col[1], color_mean='grey')
    
    #Save Figure
    plt.savefig(output_dir +  f"/psd_{i}.svg",
                transparent = True,bbox_inches="tight",format="svg")


#%% --- Plot state-specific power maps ---
# Percentage of avg power power

# Load spectra
psd = np.load(indir + "/psd.npy")[plac_mask][:,:,reindexer]

# Calculate state specific percentage of change from average power (weighted mean across all states)
mean_psd = np.empty([psd.shape[0],psd.shape[2],psd.shape[3]])
for iSub in range(psd.shape[0]):
    mean_psd[iSub] = np.average(psd[iSub], axis=0, weights=fo[iSub])
    psd[iSub] = (psd[iSub] - mean_psd[iSub]) / mean_psd[iSub] * 100 

# Calculate the group average power spectrum for each state
gpsd = np.average(psd, axis=0, weights=w)

# Calculate the power map by integrating the power spectra over a frequency range
p = power.variance_from_spectra(f, gpsd, frequency_range = [5,30])

# Plot
power.save(
    p,
    parcellation_file=parcellation_file,
    mask_file=mask_file,
    component=0,
    subtract_mean=True,
    plot_kwargs={"cmap": "RdBu_r", "bg_on_data": True, 'views': ['lateral','medial']},
    filename=f"{output_dir}/pow_percentage_.svg",
)

plt.close()
    

#%% --- Plot coherence networks ---

# Calculate the group average
gcoh = np.average(coh, axis=0, weights=w)

# Calculate the coherence network by averaging over a frequency range
c = connectivity.mean_coherence_from_spectra(f, gcoh, frequency_range = [2,30])

# Threshold the top 2% of connections
c = connectivity.threshold(c, percentile=99, subtract_mean=True)

# Plot
connectivity.save(
    c,
    parcellation_file=parcellation_file,
    component=0,
    plot_kwargs={"edge_cmap": "Reds"},
    filename=output_dir + "/coh_.svg",
)

