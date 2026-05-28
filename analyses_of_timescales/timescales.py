"""
plot timescales across cortex and analyse drug effects

@author: Ana Antonia Dias Maile
"""

import pandas as pd
import numpy as np
import osl_dynamics.analysis.power as power
import os.path as op
from mne.stats import permutation_t_test, fdr_correction
import seaborn as sns
import matplotlib.pyplot as plt

""" OVERHEAD """
# prepare paths
root = "."
plot_path = op.join(root,"plots")
# load fits
fits = pd.read_csv(op.join(root, "spectral_fits_per_vertex.csv"))
# load settings
drugs = np.unique(fits["drug"])
parcellation_file=('/Users/antoniadiasmaile/miniconda3/envs/osle/lib/python3.12/'
                  'site-packages/osl/source_recon/parcellation/files/dk_cortical.nii.gz')
cols = ['#66C2A5', '#B3B3B3', '#FC8D62']
# freesurfer has a different label order than osl
lh_order = [
    'bankssts-lh', 'caudalanteriorcingulate-lh', 'caudalmiddlefrontal-lh',
    'cuneus-lh', 'entorhinal-lh', 'fusiform-lh', 'inferiorparietal-lh',
    'inferiortemporal-lh', 'isthmuscingulate-lh', 'lateraloccipital-lh',
    'lateralorbitofrontal-lh', 'lingual-lh', 'medialorbitofrontal-lh',
    'middletemporal-lh', 'parahippocampal-lh', 'paracentral-lh',
    'parsopercularis-lh', 'parsorbitalis-lh', 'parstriangularis-lh',
    'pericalcarine-lh', 'postcentral-lh', 'posteriorcingulate-lh',
    'precentral-lh', 'precuneus-lh', 'rostralanteriorcingulate-lh',
    'rostralmiddlefrontal-lh', 'superiorfrontal-lh', 'superiorparietal-lh',
    'superiortemporal-lh', 'supramarginal-lh', 'frontalpole-lh',
    'temporalpole-lh', 'transversetemporal-lh', 'insula-lh'
]
rh_order = [label.replace('-lh', '-rh') for label in lh_order]
full_label_order = lh_order + rh_order

""" TIMESCALES """
time_constant = fits.groupby(['sub', 'drug', 'label'], as_index=False)
                            ['time_constant'].median()
time_constant['label'] = pd.Categorical(time_constant['label'],
                                        categories=full_label_order,
                                        ordered=True)
time_constant.sort_values(['label','sub'], inplace = True, ignore_index = True)
time_constant = time_constant.pivot_table(index=['sub', 'label'], columns='drug',
                                          values='time_constant',
                                          observed = True).reset_index()
time_constant['lora-plac'] = time_constant['lora'] - time_constant['plac']
time_constant['cyc-plac'] = time_constant['cyc'] - time_constant['plac']
m_time_constant = time_constant.groupby(['label'], observed = True,
                                        as_index=False)[['cyc', 'lora', 'plac',
                                                         'lora-plac',
                                                         'cyc-plac']].median()
# test drug effect lorazepam
results_time_constant_lora = []
diff_matrix = time_constant.pivot(
    index='sub',
    columns='label',
    values='lora-plac')
for label in full_label_order:
    x = diff_matrix[label].dropna().to_numpy()[:, np.newaxis]
    t_val, p_val, H0 = permutation_t_test(x, n_permutations=10000, tail=0)
    results_time_constant_lora.append({'label': label, 't':t_val[0], 'p':p_val[0]})
results_time_constant_lora = pd.DataFrame(results_time_constant_lora)
# correct p-values for multiple comparisons
sig_time_constant_lora = fdr_correction(results_time_constant_lora["p"])[0]
results_time_constant_lora["p"] = fdr_correction(results_time_constant_lora["p"])[1]
# get significant labels
sig_labels = pd.Series(sig_time_constant_lora, index=results_time_constant_lora.index)
# percentage of change per label
change_lora_sub = time_constant[["sub", "label"]]
change_lora_sub["change"] = time_constant["lora-plac"] / time_constant["plac"] * 100
change_lora = change_lora_sub.groupby("label", as_index=False)["change"].median()
# test drug effect d-cycloserine
results_time_constant_cyc = []
diff_matrix = time_constant.pivot(
    index='sub',
    columns='label',
    values='cyc-plac')
for label in full_label_order:
    x = diff_matrix[label].dropna().to_numpy()[:, np.newaxis]
    t_val, p_val, H0 = permutation_t_test(x, n_permutations=10000, tail=0)
    results_time_constant_cyc.append({'label': label, 't':t_val[0], 'p':p_val[0]})
results_time_constant_cyc = pd.DataFrame(results_time_constant_cyc)
# correct p-values for multiple comparisons
sig_time_constant_cyc = fdr_correction(results_time_constant_cyc["p"])[0]
results_time_constant_cyc["p"] = fdr_correction(results_time_constant_cyc["p"])[1]
# get significant labels
results_time_constant_cyc.loc[sig_time_constant_cyc,]
# plotting
# plot each condition
for drug in drugs:
    power.save(power_map=1000*m_time_constant[drug],
               mask_file="MNI152_T1_8mm_brain.nii.gz",
               parcellation_file=parcellation_file,
               filename=op.join(plot_path,"time_constant_" + str(drug) + ".png"),
               subtract_mean=True,
               plot_kwargs={"cmap" : "Reds",
               "bg_on_data" : 1,
               "views": ["lateral", "medial"],
               "vmin": 0,
               "vmax": 176,
               "symmetric_cbar": False,
               "cbar_tick_format": "%d",
               "cbar_label": "Timescale(ms)",
               "bg_on_data": True})
# plot tmap lorazepam
power.save(power_map=results_time_constant_lora["t"],
           mask_file="MNI152_T1_8mm_brain.nii.gz",
           parcellation_file=parcellation_file,
           filename=op.join(plot_path,"time_constant_lora-plac_t.png"),
           subtract_mean=True,
           plot_kwargs={"cmap" : "RdBu_r",
           "bg_on_data" : 1,
           "views": ["lateral", "medial"],
           "vmin": -2,
           "vmax": 5,
           "symmetric_cbar": True,
           "cbar_tick_format": "%d",
           "cbar_label": "t-value",
           "bg_on_data": True})
# plot tmap for significant parcels lorazepam
results_time_constant_lora.loc[~sig_labels,"t"] = 0
power.save(power_map=results_time_constant_lora["t"],
           mask_file="MNI152_T1_8mm_brain.nii.gz",
           parcellation_file=parcellation_file,
           filename=op.join(plot_path,"time_constant_lora-plac_t_sig.png"),
           subtract_mean=True,
           plot_kwargs={"cmap" : "RdBu_r",
           "bg_on_data" : 1,
           "views": ["lateral", "medial"],
           "vmin": -2,
           "vmax": 5,
           "symmetric_cbar": True,
           "cbar_tick_format": "%d",
           "cbar_label": "t-value",
           "bg_on_data": True})
# plot percentage of change lorazepam
power.save(power_map=change_lora["change"],
           mask_file="MNI152_T1_8mm_brain.nii.gz",
           parcellation_file=parcellation_file,
           filename=op.join(plot_path,"time_constant_lora-plac_change.png"),
           subtract_mean=True,
           plot_kwargs={"cmap" : "RdBu_r",
           "bg_on_data" : 1,
           "views": ["lateral", "medial"],
           "symmetric_cbar": True,
           "cbar_tick_format": "%d",
           "cbar_label": "Timescale Change (%)",
           "bg_on_data": True})
# plot tmap d-cycloserine
power.save(power_map=results_time_constant_cyc["t"],
           mask_file="MNI152_T1_8mm_brain.nii.gz",
           parcellation_file=parcellation_file,
           filename=op.join(plot_path,"time_constant_cyc-plac_t.png"),
           subtract_mean=True,
           plot_kwargs={"cmap" : "RdBu_r",
           "bg_on_data" : 1,
           "views": ["lateral", "medial"],
           "vmin": -2,
           "vmax": 5,
           "symmetric_cbar": True,
           "cbar_tick_format": "%d",
           "cbar_label": "t-value",
           "bg_on_data": True})
