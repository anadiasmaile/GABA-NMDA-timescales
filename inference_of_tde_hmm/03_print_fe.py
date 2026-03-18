#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 10:35:49 2025

@author: okohl

Lorasick Analysis
Print free energy.
"""

n_states = 8
output_dir = f".../data/inf/{n_states:02d}_states/"

import pickle
import numpy as np

def get_best_run(min_, max_):
    best_fe = np.inf
    for run in range(min_, max_ + 1):
        try:
            history = pickle.load(open(f"{output_dir}/run{run:02d}/history.pkl", "rb"))
            fe = history["free_energy"]
            print(f"run {run}: {fe}")
            if fe < best_fe:
                best_run = run
                best_fe = fe
        except:
            print(f"run {run} missing")
            pass
    return best_run

print("Best run:", get_best_run(1, 10))
