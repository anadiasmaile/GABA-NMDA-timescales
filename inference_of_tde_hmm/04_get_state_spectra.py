#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 10:44:05 2025

@author: okohl

Lorasick
Calculate Post-hoc metrics: state-specific psd and coh spectra.
Lowest FE Runs:
    8K = run6
    10K = run5
    12K = run10

"""

import pickle
import numpy as np
from glob import glob
from osl_dynamics.data import Data
from osl_dynamics.analysis import spectral

# Best Run
n_states = 12
irun = 5

# Data Dir
preproc_dir = '.../data/hmm_in'
output_dir = f'.../data/inf/{n_states:02d}_states/run{irun:02d}'

# Load state probability time courses
alp = pickle.load(open(f"{output_dir}/alp.pkl", "rb"))

# --- Calculate Multitaper ---
print("Calculating Multitaper")

# Load input data, filter and trim to match inferred alps
data = Data(preproc_dir, picks="misc", reject_by_annotation="omit", sampling_frequency=250, n_jobs=8)
data = data.standardize()
x = data.trim_time_series(n_embeddings=15, sequence_length=2000)

# Calculate multitaper spectra
f, psd, coh, w = spectral.multitaper_spectra(
    data=x,
    alpha=alp,
    sampling_frequency=250,
    time_half_bandwidth=4,
    n_tapers=7,
    frequency_range=[1, 45],
    standardize=True,
    return_weights=True,
    n_jobs=8,
)

np.save(f"{output_dir}/f.npy", f)
np.save(f"{output_dir}/psd.npy", psd)
np.save(f"{output_dir}/coh.npy", coh)
np.save(f"{output_dir}/w.npy", w)

# Calculate non-negative matrix factorisation on the stacked coherences
# nnmf = spectral.decompose_spectra(coh, n_components=2)
# np.save(f"{output_dir}/nnmf_2.npy", nnmf)
