"""
validation check: compare computed timescales to
publicly available intrinsic timescale map using
spatial null significance testing

@author: Ana Antonia Dias Maile
"""
from neuromaps.parcellate import Parcellater
from neuromaps.datasets import fetch_annotation
from neuromaps import plotting, nulls, stats, transforms
import pandas as pd
import numpy as np
from netneurotools import datasets
import nibabel as nib
import matplotlib.pyplot as plt
import os.path as op
import seaborn as sns
from scipy.stats import linregress, ttest_rel

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
map = fetch_annotation(source='hcps1200', desc='megtimescale', space='fsLR', den='4k')
# transform to 32k
map = transforms.fslr_to_fslr(data = map, target_density = "32k")
map_lh = map[0]
map_rh = map[1]
# load timescales
root = "."
fits = pd.read_csv(op.join(root, "spectral_fits_per_vertex.csv"))
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
corr['pla'], pval['pla'] = stats.compare_images(map_parc,
                                                m_time_constant_plac["time_constant"],
                                                nulls=rotated, metric = "spearmanr")
corr['lora'], pval['lora'] = stats.compare_images(map_parc,
                                                  m_time_constant_lora["time_constant"],
                                                  nulls=rotated, metric = "spearmanr")
corr['cyc'], pval['cyc'] = stats.compare_images(map_parc,
                                                m_time_constant_cyc["time_constant"],
                                                nulls=rotated, metric = "spearmanr")
