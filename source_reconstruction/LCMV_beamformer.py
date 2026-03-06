"""
Source reconstruction using LCMV beamformer

Compute covariance matrices and apply beamformer per participant,
session and task. Uses forward solution and preprocessed data
(and freesurfer for plotting).

@author: Ana Antonia Dias Maile
"""
import json
import sys
import os.path as op
from os import (makedirs, getcwd)
from glob import glob
import datalad.api as dl
import mne
from mne import (set_log_file, setup_source_space, write_source_spaces)
from mne.beamformer import  apply_lcmv_raw, make_lcmv
import numpy as np
from datetime import datetime
import logging as log
from eduTools import meg
from eduTools import io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from osl_ephys.source_recon import parcellation, beamforming
from osl.source_recon import parcellation
sys.path.append("./code/")
from ep_preproc_emptyroom import emptyroom_preproc
from merging import merge_epochs_to_raw, check_merging
import pandas as pd

""" OVERHEAD """
root = op.abspath(sys.argv[1])
# bids fields
sub = int(sys.argv[2])
ses = int(sys.argv[3])
task = sys.argv[4]
# general settings
# save beamformed data?
save_src = True
date = str(datetime.now()).replace(' ', '-').replace(':', '-')
# subject paths
meg_dir = op.join(root, 'input_clean', f'sub-{sub:02d}', f'ses-{ses:02d}',
                  'meg')
forward_dir = op.join(root, 'input_forward', f'sub-{sub:02d}', f'ses-{ses:02d}')
subjects_dir = op.join(root, 'input_anat', 'freesurfer') # freesurfer
deriv_dir = op.join(root, f'sub-{sub:02d}', f'ses-{ses:02d}')
log_dir = op.join(deriv_dir, 'log')
qc_dir = op.join(deriv_dir, 'qc')
makedirs(log_dir, exist_ok=True)
makedirs(qc_dir, exist_ok=True)
# subject stubs
ep_stub = f'sub-{sub:02d}_ses-{ses:02d}_task-{task}_%s'
# define file names
log_file = op.join(log_dir, ep_stub % f'LCMV_beamformer_{date}.log')
# set up logging
log.basicConfig(filename=log_file, level=log.INFO,
                filemode = "a", format='%(asctime)s %(message)s')
set_log_file(fname=log_file, overwrite=False)
log.info('Logfile initialized.')

""" READ EPOCHs AND FORWARD SOLUTION """
log.info("Retrieving files from remote.")
# read epochs
megname = sorted(glob(op.join(meg_dir, ep_stub % 'long_clean-ep*.fif')))
dl.get(megname, dataset=f'{root}', source='origin')
epochs = mne.read_epochs(megname[-1])
dl.drop(megname, dataset=f'{root}')
# merge epochs to continuous data again and check if it worked
raw = merge_epochs_to_raw(epochs, overlap=0, bad_id=99)
check_merging(raw,epochs,log,op.join(qc_dir, ep_stub % 'raw_psd.png'))
# exclude bad channels
raw.pick_types(meg='grad', exclude = []) # only gradiometers
# get sampling frequency
sfreq = raw.info['sfreq']
# read forward solution
fwd_name = op.join(forward_dir, ep_stub % 'fwd.fif')
dl.get(fwd_name, dataset=f'{root}', source='origin')
fwd = mne.read_forward_solution(fwd_name)
dl.drop(fwd_name, dataset=f'{root}')
log.info("Files retrieved.")

""" COMPUTE COVARIANCE MATRIX """
log.info(f"Compute data and noise covariance matrix.")
# bad annotations are rejected by default
data_cov_raw = mne.compute_raw_covariance(raw, method="empirical",rank='info')
# calculate noise covariance from empty room
emptyroom = emptyroom_preproc(root, sub, ses, task) # preprocess emptyroom
emptyroom.pick_types(meg='grad', exclude=[]) # only pick gradiometers
noise_cov = mne.compute_raw_covariance(emptyroom, method = "empirical", rank="info")

""" DEFINE AND APPLY BEAMFORMER """
log.info(f"Set up and apply 3D LCMV beamformer.")
filters = make_lcmv(
    raw.info,
    fwd,
    data_cov_raw,
    # reguarize covariance matrix to counteract slight rank-deficiency
    reg=0.05,
    noise_cov=noise_cov,
    # following mne tutorial and Britta Westner
    pick_ori="max-power",
    # normalisation to counteract center-of-head-bias
    weight_norm="unit-noise-gain-invariant",
    rank="info",
    # recommended for MEG data since not sensitive to radial sources
    reduce_rank=True
)
# save a bit of memory
src = fwd["src"] # source space
del fwd
# apply beamformer --> use osl function to reject bad annotations
stc = beamforming.apply_lcmv_raw(raw, filters, reject_by_annotation = 'omit')
# save source estimate
if save_src == True:
    stc.save(op.join(deriv_dir, ep_stub % 'raw_source_estimate'), overwrite=True)
log.info(f"Source reconstruction done for sub {sub}, ses {ses} and task {task}.")
del filters

""" WHICH VERTEX BELONG TO WHICH PARCEL """
labels = mne.read_labels_from_annot(f'sub-{sub:02d}', parc="aparc",
                                    subjects_dir=subjects_dir, hemi = "both")
n_vertices_lh = len(stc.vertices[0])
n_vertices_rh = len(stc.vertices[1])
vertex_to_label = np.empty(n_vertices_lh + n_vertices_rh, dtype=object)
for label in labels:
    if label.hemi == 'lh':
        # Convert from full-surface vertex numbers to indices in stc.vertices[0]
        idx_in_stc = np.searchsorted(stc.vertices[0], label.vertices)
        idx_in_stc = idx_in_stc[np.isin(label.vertices, stc.vertices[0])]
        vertex_to_label[idx_in_stc] = label.name
    else:
        # Same for right hemisphere, but offset by LH length
        idx_in_stc = np.searchsorted(stc.vertices[1], label.vertices)
        idx_in_stc = idx_in_stc[np.isin(label.vertices, stc.vertices[1])]
        vertex_to_label[idx_in_stc + n_vertices_lh] = label.name
vertex_label_map = pd.DataFrame({
    'vertex': np.arange(len(vertex_to_label)),
    'parcel': vertex_to_label
})
# check if all labels present
missing = set(label.name for label in labels) -
          set(lbl for lbl in vertex_to_label if lbl is not None)
if missing:
    raise ValueError(f"Missing parcels ({len(missing)}): {missing}")
# save for later averaging over vertex using atlas
vertex_label_map.to_csv(op.join(deriv_dir, ep_stub % 'vertex_to_label_map.csv'),
                        index=False)
log.info("Done!")
