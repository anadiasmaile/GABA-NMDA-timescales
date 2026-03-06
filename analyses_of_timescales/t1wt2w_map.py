"""
compare computed timescales to
publicly available t1w/t1w map using
spatial null significance testing

@author: Ana Antonia Dias Maile
"""

from neuromaps.parcellate import Parcellater
from neuromaps.datasets import fetch_annotation
from neuromaps import plotting, nulls, stats
import pandas as pd
import numpy as np
from netneurotools import datasets
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os.path as op
import seaborn as sns
from scipy.stats import linregress, ttest_rel, zscore
from matplotlib.lines import Line2D

""" OVERHEAD """
# load atlas
annot = datasets.fetch_cammoun2012('fslr32k')['scale033']
annot_lh, annot_rh = annot
# Load LH GIFTI file
gifti_lh = nib.load(annot_lh)
gifti_rh = nib.load(annot_rh)
# Extract label table (index -> name)
labels_fsLR = gifti_lh.labeltable.get_labels_as_dict()
# load map
map = fetch_annotation(source='hcps1200', desc='myelinmap', space='fsLR', den='32k')
map_lh = map[0]
map_rh = map[1]
# load timescales
root = "."
fits = pd.read_csv(op.join(root, "fooof_fits_pervertex_allexp.csv"))
time_constant = fits.groupby(['sub', 'drug', 'label'], as_index=False)
                             ['time_constant'].median()[["sub","drug",
                                                         "label","time_constant"]]
# order
label_order = np.array(list(labels_fsLR.values()))
label_order = np.delete(label_order,[0,4]) # delete ??? and corpuscallosum
label_order_lh = np.char.add(label_order, '-lh') # add -lh
label_order_rh = np.char.add(label_order, '-rh') # add -lh
label_order_fsLR = np.concatenate([label_order_lh, label_order_rh])
time_constant['label'] = pd.Categorical(time_constant['label'],
                                        categories=label_order_fsLR, ordered=True)
time_constant = time_constant.sort_values(['label', 'sub']).reset_index(drop=True)
m_time_constant = time_constant.groupby(['drug', 'label'], observed = True)
                                        ['time_constant'].median().reset_index()
m_time_constant_plac = m_time_constant.loc[m_time_constant['drug'] == "plac",
                                          ["label","time_constant"]].reset_index(drop=True)
m_time_constant_lora = m_time_constant.loc[m_time_constant['drug'] == "lora",
                                          ["label","time_constant"]].reset_index(drop=True)
m_time_constant_cyc = m_time_constant.loc[m_time_constant['drug'] == "cyc",
                                         ["label","time_constant"]].reset_index(drop=True)
""" PARCELLATE MAP """
# background data = ??? and corpuscallosum
# ??? automatically ignored + corpuscallosum has background value and is turned to nan
parc_lh = Parcellater(annot_lh, 'fsLR', hemi = 'L')
parc_rh = Parcellater(annot_rh, 'fsLR', hemi = 'R')
map_parc_lh = parc_lh.fit_transform(data = map_lh, space = 'fsLR',
                                    background_value = 0,
                                    ignore_background_data = True, hemi = 'L')
map_parc_rh = parc_rh.fit_transform(data = map_rh, space = 'fsLR',
                                    background_value = 0,
                                    ignore_background_data = True, hemi = 'R')
map_parc = np.concatenate([map_parc_lh, map_parc_rh])
map_parc = map_parc[~np.isnan(map_parc)] # exclude ??? and corpuscallosum

""" SPATIAL NULL SIGNIFICANCE TESTING """
# create null maps
rotated = nulls.alexander_bloch(map_parc, atlas='fsLR', density='32k',
                                n_perm=1000, seed=69, parcellation=annot)
# compute correlation
corr = {}
pval = {}
corr['lora'], pval['lora'] = stats.compare_images(map_parc,
                                                  np.array(m_time_constant_lora["time_constant"]),
                                                  nulls=rotated, metric = "spearmanr")
corr['pla'], pval['pla'] = stats.compare_images(map_parc,
                                                np.array(m_time_constant_plac["time_constant"]),
                                                nulls=rotated, metric = "spearmanr")
corr['cyc'], pval['cyc'] = stats.compare_images(map_parc,
                                                np.array(m_time_constant_cyc["time_constant"]),
                                                nulls=rotated, metric = "spearmanr")

""" PLOTTING """
# prepare plotting
cols = ['#66C2A5', '#B3B3B3', '#FC8D62']
p_corr = {k: round(v, 2) for k, v in corr.items()}
p_pval = {k: round(v, 3) for k, v in pval.items()}
x_pos = 0
y_pos = 160
xlabel = 'T1w/T2w (σ)'
ylabel = 'Timescale (ms)'
xticks = [-2,-1,0,1,2]
yticks = [40,80,120,160]
fig = plt.figure(figsize=(14,4.5), dpi=600)
gs1 = GridSpec(1, 3, left=0.08, right=0.95, top=0.9, bottom=0.15, wspace=0.35)
ax1 = fig.add_subplot(gs1[0, 0])
ax2 = fig.add_subplot(gs1[0, 1])
ax3 = fig.add_subplot(gs1[0, 2])
# prepare data
correlation_pla = pd.DataFrame(zip(zscore(map_parc),
                                   1000*np.array(m_time_constant_plac["time_constant"])),
                                   columns = ["t1wt2w", "time_constant"])
correlation_lora = pd.DataFrame(zip(zscore(map_parc),
                                    1000*np.array(m_time_constant_lora["time_constant"])),
                                    columns = ["t1wt2w", "time_constant"])
correlation_cyc = pd.DataFrame(zip(zscore(map_parc),
                                   1000*np.array(m_time_constant_cyc["time_constant"])),
                                   columns = ["t1wt2w", "time_constant"])
# plot
lora_cplot = sns.regplot(x="t1wt2w",y="time_constant",data=correlation_lora,truncate=True,
            line_kws={"color": cols[0]}, scatter_kws={"color": cols[0]}, ax=ax1)
pla_cplot = sns.regplot(x="t1wt2w",y="time_constant",data=correlation_pla,truncate=True,
            line_kws={"color": cols[1]}, scatter_kws={"color": cols[1]}, ax=ax2)
cyc_cplot = sns.regplot(x="t1wt2w",y="time_constant",data=correlation_cyc,truncate=True,
            line_kws={"color": cols[2]}, scatter_kws={"color": cols[2]}, ax=ax3)
# add statistics
ax1.text(x_pos, y_pos, f"$r_s$ = {corr['lora']:.2f}, $p$ = {pval['lora']:.3f}", fontsize=12)
ax2.text(x_pos, y_pos, f"$r_s$ = {corr['pla']:.2f}, $p$ = {pval['pla']:.3f}", fontsize=12)
ax3.text(x_pos, y_pos, f"$r_s$ = {corr['cyc']:.2f}, $p$ = {pval['cyc']:.3f}", fontsize=12)
# make axis pretty and get rid of suplot boxes
axes = [ax1, ax2, ax3]
for ax in axes:
    ax.xaxis.labelpad = 10
    ax.set_ylim(0,180)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.locator_params(axis="y", nbins=5)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, fontsize=14)
    ax.set_ylabel(ylabel, fontsize = 16, labelpad = 5)
    sns.despine(ax=ax , top=True, right=True, left=False,
            bottom=False, offset=None, trim=False)
# legend
custom_lines = [Line2D([0], [0], color=cols[0], lw=0, marker="o", markersize=10),
                Line2D([0], [0], color=cols[1], lw=0, marker="o",markersize=10),
                Line2D([0], [0], color=cols[2], lw=0, marker="o",markersize=10)]
labels = ['Lorazepam','Placebo', 'D-cycloserine']
for ax, line, label in zip([ax1, ax2, ax3], custom_lines, labels):
    ax.legend(
        [line], [label],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        frameon=False,
        prop={"size": 14},
        handletextpad=.6,
        handlelength=1,
    )
plt.savefig('./plots/manuscript/t1wt2w.png')
