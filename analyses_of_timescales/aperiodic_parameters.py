""""
plot aperiodic parameters across cortex and analyse drug effects

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
fits = pd.read_csv(op.join(root, "fooof_fits_pervertex_allexp.csv"))
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

""" KNEE FREQUENCY """
knee_freq = fits.groupby(['sub', 'drug', 'label'], as_index=False)
                         ['knee_freq'].median()
knee_freq['label'] = pd.Categorical(knee_freq['label'],
                                    categories=full_label_order,
                                    ordered=True)
knee_freq.sort_values(['label','sub'], inplace = True, ignore_index = True)
knee_freq = knee_freq.pivot_table(index=['sub', 'label'], columns='drug',
                                   values='knee_freq', observed = True).reset_index()
knee_freq['lora-plac'] = knee_freq['lora'] - knee_freq['plac']
knee_freq['cyc-plac'] = knee_freq['cyc'] - knee_freq['plac']
m_knee_freq = knee_freq.groupby(['label'], observed = True, as_index=False)
                                [['cyc', 'lora', 'plac',
                                'lora-plac', 'cyc-plac']].median()
# test drug effect lorazepam
results_knee_freq_lora = []
diff_matrix = knee_freq.pivot(
    index='sub',
    columns='label',
    values='lora-plac')
for label in full_label_order:
    x = diff_matrix[label].dropna().to_numpy()[:, np.newaxis]
    t_val, p_val, H0 = permutation_t_test(x, n_permutations=10000, tail=0)
    results_knee_freq_lora.append({'label': label, 't':t_val[0], 'p':p_val[0]})
results_knee_freq_lora = pd.DataFrame(results_knee_freq_lora)
# correct p-values for multiple comparisons
sig_knee_freq_lora = fdr_correction(results_knee_freq_lora["p"])[0]
results_knee_freq_lora["p"] = fdr_correction(results_knee_freq_lora["p"])[1]
# get significant labels
results_knee_freq_lora.loc[sig_knee_freq_lora,]
# test drug effect d-cycloserine
results_knee_freq_cyc = []
diff_matrix = knee_freq.pivot(
    index='sub',
    columns='label',
    values='cyc-plac')
for label in full_label_order:
    x = diff_matrix[label].dropna().to_numpy()[:, np.newaxis]
    t_val, p_val, H0 = permutation_t_test(x, n_permutations=10000, tail=0)
    results_knee_freq_cyc.append({'label': label, 't':t_val[0], 'p':p_val[0]})
results_knee_freq_cyc = pd.DataFrame(results_knee_freq_cyc)
# correct p-values for multiple comparisons
sig_knee_freq_cyc = fdr_correction(results_knee_freq_cyc["p"])[0]
results_knee_freq_cyc["p"] = fdr_correction(results_knee_freq_cyc["p"])[1]
# get significant labels
results_knee_freq_cyc.loc[sig_knee_freq_cyc,]
# plotting
# plot under each drug condition
for drug in drugs:
    power.save(power_map=m_knee_freq[drug],
               mask_file="MNI152_T1_8mm_brain.nii.gz",
               parcellation_file=parcellation_file,
               filename=op.join(plot_path,"knee_freq_" + str(drug) + ".png"),
               subtract_mean=True,
               plot_kwargs={"cmap" : "RdBu_r",
                            "bg_on_data" : 1,
                            "views": ["lateral", "medial"],
                            "vmin"})
# plot tmap lorazepam
power.save(power_map=results_knee_freq_lora["t"],
           mask_file="MNI152_T1_8mm_brain.nii.gz",
           parcellation_file=parcellation_file,
           filename=op.join(plot_path,"knee_freq_lora-plac_t.png"),
           subtract_mean=True,
           plot_kwargs={"cmap" : "RdBu_r",
                        "bg_on_data" : 1,
                        "views": ["lateral", "medial"],
                        "vmin": -2.87,
                        "cbar_tick_format": "%d",
                        "cbar_label": "t-value",
                        "bg_on_data": True})

# plot tmap d-cycloserine
power.save(power_map=results_knee_freq_cyc["t"],
           mask_file="MNI152_T1_8mm_brain.nii.gz",
           parcellation_file=parcellation_file,
           filename=op.join(plot_path,"knee_freq_cyc-plac_t.png"),
           subtract_mean=True,
           plot_kwargs={"cmap" : "RdBu_r",
                        "bg_on_data" : 1,
                        "views": ["lateral", "medial"],
                        "cbar_tick_format": "%d",
                        "cbar_label": "t-value",
                        "bg_on_data": True})

""" EXPONENT """
exponent = fits.groupby(['sub', 'drug', 'label'], as_index=False)
                        ['exponent'].median()
exponent['label'] = pd.Categorical(exponent['label'],
                                   categories=full_label_order, ordered=True)
exponent.sort_values(['label','sub'], inplace = True, ignore_index = True)
exponent = exponent.pivot_table(index=['sub', 'label'], columns='drug',
                                values='exponent', observed = True).reset_index()
exponent['lora-plac'] = exponent['lora'] - exponent['plac']
exponent['cyc-plac'] = exponent['cyc'] - exponent['plac']
m_exponent = exponent.groupby(['label'], observed = True, as_index=False)
                                [['cyc', 'lora', 'plac',
                                'lora-plac', 'cyc-plac']].median()
# test drug effect lorazepam
results_exponent_lora = []
diff_matrix = exponent.pivot(
    index='sub',
    columns='label',
    values='lora-plac')
for label in full_label_order:
    x = diff_matrix[label].dropna().to_numpy()[:, np.newaxis]
    t_val, p_val, H0 = permutation_t_test(x, n_permutations=10000, tail=0)
    results_exponent_lora.append({'label': label, 't':t_val[0], 'p':p_val[0]})
results_exponent_lora = pd.DataFrame(results_exponent_lora)
# correct p-values for multiple comparisons
sig_exponent_lora = fdr_correction(results_exponent_lora["p"])[0]
results_exponent_lora["p"] = fdr_correction(results_exponent_lora["p"])[1]
# get significant labels
results_exponent_lora.loc[sig_exponent_lora,]
# test drug effect d-cycloserine
results_exponent_cyc = []
diff_matrix = exponent.pivot(
    index='sub',
    columns='label',
    values='cyc-plac')
for label in full_label_order:
    x = diff_matrix[label].dropna().to_numpy()[:, np.newaxis]
    t_val, p_val, H0 = permutation_t_test(x, n_permutations=10000, tail=0)
    results_exponent_cyc.append({'label': label, 't':t_val[0], 'p':p_val[0]})
results_exponent_cyc = pd.DataFrame(results_exponent_cyc)
# correct p-values for multiple comparisons
sig_exponent_cyc = fdr_correction(results_exponent_cyc["p"])[0]
results_exponent_cyc["p"] = fdr_correction(results_exponent_cyc["p"])[1]
# get significant labels
results_exponent_cyc.loc[sig_exponent_cyc,]
# plotting
# plot for each drug
for drug in drugs:
    power.save(power_map=m_exponent[drug],
               mask_file="MNI152_T1_8mm_brain.nii.gz",
               parcellation_file=parcellation_file,
               filename=op.join(plot_path,"exponent_" + str(drug) + ".png"),
               subtract_mean=True,
               plot_kwargs={"cmap" : "RdBu_r",
                            "bg_on_data" : 1,
                            "views": ["lateral", "medial"]})

# plot tmap lorazepam
power.save(power_map=results_exponent_lora["t"],
           mask_file="MNI152_T1_8mm_brain.nii.gz",
           parcellation_file=parcellation_file,
           filename=op.join(plot_path,"exponent_lora-plac_t.png"),
           subtract_mean=True,
           plot_kwargs={"cmap" : "RdBu_r",
                        "bg_on_data" : 1,
                        "views": ["lateral", "medial"]})

# plot tmap d-cycloserine
power.save(power_map=results_exponent_cyc["t"],
           mask_file="MNI152_T1_8mm_brain.nii.gz",
           parcellation_file=parcellation_file,
           filename=op.join(plot_path,"exponent_cyc-plac_t.png"),
           subtract_mean=True,
           plot_kwargs={"cmap" : "RdBu_r",
                        "bg_on_data" : 1,
                        "views": ["lateral", "medial"]})

""" OFFSET """
offset = fits.groupby(['sub', 'drug', 'label'], as_index=False)['offset'].median()
offset['label'] = pd.Categorical(offset['label'], categories=full_label_order, ordered=True)
offset.sort_values(['label','sub'], inplace = True, ignore_index = True)
offset = offset.pivot_table(index=['sub', 'label'], columns='drug',
                            values='offset', observed = True).reset_index()
offset['lora-plac'] = offset['lora'] - offset['plac']
offset['cyc-plac'] = offset['cyc'] - offset['plac']
m_offset = offset.groupby(['label'], observed = True, as_index=False)
                          [['cyc', 'lora', 'plac',
                          'lora-plac', 'cyc-plac']].median()
# test drug effects lorazepam
results_offset_lora = []
diff_matrix = offset.pivot(
    index='sub',
    columns='label',
    values='lora-plac')
for label in full_label_order:
    x = diff_matrix[label].dropna().to_numpy()[:, np.newaxis]
    t_val, p_val, H0 = permutation_t_test(x, n_permutations=10000, tail=0)
    results_offset_lora.append({'label': label, 't':t_val[0], 'p':p_val[0]})
results_offset_lora = pd.DataFrame(results_offset_lora)
# correct p-values for multiple comparisons
sig_offset_lora = fdr_correction(results_offset_lora["p"])[0]
results_offset_lora["p"] = fdr_correction(results_offset_lora["p"])[1]
# get significant labels
results_offset_lora.loc[sig_offset_lora,]
# test drug effects d-cycloserine
results_offset_cyc = []
diff_matrix = offset.pivot(
    index='sub',
    columns='label',
    values='cyc-plac')
for label in full_label_order:
    x = diff_matrix[label].dropna().to_numpy()[:, np.newaxis]
    t_val, p_val, H0 = permutation_t_test(x, n_permutations=10000, tail=0)
    results_offset_cyc.append({'label': label, 't':t_val[0], 'p':p_val[0]})
results_offset_cyc = pd.DataFrame(results_offset_cyc)
# correct p-values for multiple comparisons
sig_offset_cyc = fdr_correction(results_offset_cyc["p"])[0]
results_offset_cyc["p"] = fdr_correction(results_offset_cyc["p"])[1]
# get significant labels
results_offset_cyc.loc[sig_offset_cyc,]
# plotting
# plot for each drug
for drug in drugs:
    power.save(power_map=m_offset[drug],
               mask_file="MNI152_T1_8mm_brain.nii.gz",
               parcellation_file=parcellation_file,
               filename=op.join(plot_path,"offset_" + str(drug) + ".png"),
               subtract_mean=True,
               plot_kwargs={"cmap" : "RdBu_r",
                            "bg_on_data" : 1,
                            "views": ["lateral", "medial"]})
# plot tmap lorazepam
power.save(power_map=results_offset_lora["t"],
           mask_file="MNI152_T1_8mm_brain.nii.gz",
           parcellation_file=parcellation_file,
           filename=op.join(plot_path,"offset_lora-plac_t.png"),
           subtract_mean=True,
           plot_kwargs={"cmap" : "RdBu_r",
                        "bg_on_data" : 1,
                        "views": ["lateral", "medial"]})
# plot tmap d-cycloserine
power.save(power_map=results_offset_cyc["t"],
           mask_file="MNI152_T1_8mm_brain.nii.gz",
           parcellation_file=parcellation_file,
           filename=op.join(plot_path,"offset_cyc-plac_t.png"),
           subtract_mean=True,
           plot_kwargs={"cmap" : "RdBu_r",
                        "bg_on_data" : 1,
                        "views": ["lateral", "medial"]})
