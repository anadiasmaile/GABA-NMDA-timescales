#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 18:03:17 2025

@author: okohl

Prepare Data for Lorasick HMMs.
"""

import numpy as np
from osl_dynamics.data import Data

# Load and Save Data 
data = Data('.../lorasick_source', reject_by_annotation="omit", sampling_frequency=250,n_jobs=15)

# Sign Flip Data
data = data.align_channel_signs(n_embeddings=15)

# Apply HMM preparation and save to Hard Drive. Used for HMM inference
data = data.prepare({
    #"align_channel_signs": {n_embeddings:15},
    "filter": {"low_freq": 1, "high_freq": 45},
    "tde_pca": {"n_embeddings": 15, "n_pca_components": 80},
    "standardize": {},
})

# Save a TFRecord dataset
data.save("data/training_dataset")

# Delete TMP folder
data.delete_dir()
