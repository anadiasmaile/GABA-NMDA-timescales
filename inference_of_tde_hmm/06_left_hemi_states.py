#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 10:46:59 2025

@author: okohl

Lorasick Dataset

Make Surface Polots of state-specific 2-30Hz wideband power 
of left hemisphere for only placebo condition.
Plots used in Figure 5f.

Best run:
    8K -> run5
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


#%% Prepare Plotting

n_states = 8
irun = 8

# Set Dirs
indir = f".../data/inf/{n_states:02d}_states/run{irun:02d}"
output_dir = '.../results/hmm/08_states/state_description/plac/left'

# Make output directory
os.makedirs(output_dir, exist_ok=True)

# Get Parcellation and Surface Innformation
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

# Subtract mean across State and calculate percentage of change from average across states
mean_psd = np.empty([psd.shape[0],psd.shape[2],psd.shape[3]])
for iSub in range(psd.shape[0]):
    mean_psd[iSub] = np.average(psd[iSub], axis=0, weights=fo[iSub])
    psd[iSub] = (psd[iSub] - mean_psd[iSub]) / mean_psd[iSub] * 100 

#%% --- Plot power maps ---

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
    plot_kwargs={"cmap": "RdBu_r", "bg_on_data": True, 'hemispheres': ['left'],'views': ['lateral']},
    filename=f"{output_dir}/left_hem_pow_percentage_.svg",
)

plt.close()
    
