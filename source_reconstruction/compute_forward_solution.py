"""
Generate a forward solution. Combines coregistration with bem setup.

@author: Eduard Ort
"""

import sys
import os.path as op
from os import (makedirs, getcwd)
from glob import glob
import datalad.api as dl
from mne import (set_log_file, read_trans, write_forward_solution,
                 read_bem_solution, read_source_spaces, make_forward_solution)
from mne_bids import(BIDSPath, read_raw_bids)
import numpy as np
from datetime import datetime
import logging as log

""" OVERHEAD """
root = op.abspath(sys.argv[1])
raw_root = op.join(root, 'input_raw')
anat_root = op.join(root, 'input_anat')

# bids fields
sub = int(sys.argv[2])
ses = int(sys.argv[3])
task = sys.argv[4]
try:
    acq = int(sys.argv[5])
except:
    acq = None
subject = f'sub-{sub:02d}'
session = f'ses-{ses:02d}'

# general settings
date = str(datetime.now()).replace(' ', '-').replace(':', '-')

# define paths
bids_path = BIDSPath(root=raw_root, subject=f'{sub:02d}',
                     session=f'{ses:02d}', task=task, datatype='meg')
if acq is not None:
    bids_path = bids_path.update(acquisition=f'{acq:02d}')

subjects_dir = op.join(anat_root, 'freesurfer')
sub_dir = op.join(root, subject)
ses_dir = op.join(sub_dir, session)
log_dir = op.join(ses_dir, 'log', task)

makedirs(log_dir, exist_ok=True)

# define file names
raw_stub = f'{subject}_{session}_task-{task}_%s'
bem_stub = f'{subject}_%s'
if acq is not None:
    raw_stub = f'{subject}_{session}_task-{task}_acq-{acq:02d}_%s'
raw_fname = glob(op.join(bids_path.directory, raw_stub % '*meg.fif'))
anat_fname = glob(op.join(subjects_dir, subject))
log_file = op.join(log_dir, raw_stub % f'coreg_{date}.log')
fwd_file = op.join(ses_dir, raw_stub % f'fwd.fif')
trans_file = op.join(ses_dir, raw_stub % f'trans.fif')
src_file = op.join(sub_dir, bem_stub % f'oct6-src.fif')
bem_file = op.join(sub_dir, bem_stub % f'bem-sol.fif')

# set up logging
log.basicConfig(filename=log_file, level=log.INFO,
                format='%(asctime)s %(message)s')
set_log_file(fname=log_file, overwrite=True)
log.info('Logfile initialized.')

""" READ RAW AND ANAT """
log.info("Retrieving files from remote.")

log.info("Get meg data")
dl.get(raw_fname, source='data-source', dataset=raw_root)

log.info("Get anatomical data")
dl.get(anat_fname, source='data-source', dataset=anat_root)

log.info("Files retrieved.")

log.info("Load the required files...")
raw = read_raw_bids(bids_path)
trans = read_trans(trans_file)
src = read_source_spaces(src_file)
bem_sol = read_bem_solution(bem_file)

# compute forward solution
fwd = make_forward_solution(raw.info, trans=trans, src=src, bem=bem_sol,
                            meg=True, eeg=False, mindist=5)
log.info("Leadfield size : %d sensors x %d dipoles" % fwd['sol']['data'].shape)
write_forward_solution(fwd_file, fwd, overwrite=True)

log.info(f"Forward solution set up for sub {sub} and ses {ses} and {task}.")

# drop files again
dl.save(fwd_file, dataset=root,
        message=f'save BEM model for {subject}_{session}_{task}')
dl.drop(raw_fname, dataset=raw_root)
dl.drop(anat_fname, dataset=anat_root)
