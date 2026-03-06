"""
Runs coregistration of meg to mr space

Because we don't have anatomical fiducials (in the mr scan) we
need to estimate nasion, lpa, rpa manually. To do so, we need
to call the coregistration GUI after generating the bem surfaces
That prevents, a completely automatic execution of this script.
However, this only needs to be done once per subject. For reruns
the fiducalials will be read from the bem directory (provided) that
the users saved the manual points at the first run.

@author: Eduard Ort
"""
import sys
import os.path as op
from os import (makedirs, getcwd)
from glob import glob
import datalad.api as dl
from mne import (set_log_file, write_trans)
from mne.io import read_raw
from mne.coreg import Coregistration
from mne_bids import BIDSPath
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
deriv_dir = op.join(root, subject, session)
log_dir = op.join(deriv_dir, 'log')

makedirs(log_dir, exist_ok=True)

# define file names
raw_stub = f'{subject}_{session}_task-{task}_%s'
if acq is not None:
    raw_stub = f'{subject}_{session}_task-{task}_acq-{acq:02d}_%s'
raw_fname = glob(op.join(bids_path.directory, raw_stub % '*meg.fif'))
anat_fname = glob(op.join(subjects_dir, subject))
fid_file = op.join(subjects_dir, subject, 'bem', f'{subject}-fiducials.fif')
log_file = op.join(log_dir, raw_stub % f'coreg_{date}.log')
trans_file = op.join(deriv_dir, raw_stub % f'trans.fif')

# set up logging
log.basicConfig(filename=log_file, level=log.INFO,
                format='%(asctime)s %(message)s')
set_log_file(fname=log_file, overwrite=True)
log.info('Logfile initialized.')

""" READ RAW AND ANAT """
log.info("Retrieving files from remote.")

log.info("Get meg data")
dl.get(raw_fname, source='data-source', dataset=f'{raw_root}')

log.info("Get anatomical data")
dl.get(anat_fname, source='data-source', dataset=f'{anat_root}')


log.info("Files retrieved.")

log.info("Reading raw...")
raw = read_raw(bids_path.fpath)

""" COREGISTRATION """
log.info("Start coregistration.")
if not op.exists(fid_file):
    log.error("The fiducials do not exist! Run make_fiducials.py first!")
    sys.exit(1)

log.info("set up coregistration model and fix bem surfaces if needed")
coreg = Coregistration(raw.info, subject, subjects_dir)

log.info("initial fit with fiducials")
coreg.set_fid_match('matched')
coreg.fit_fiducials(verbose=True)

log.info("refining with icp")
coreg.fit_icp(n_iterations=20, nasion_weight=20, verbose=True)

log.info("omitting bad points (> 5 mm)")
coreg.omit_head_shape_points(distance=0.005)

log.info("final coregistration fit")
coreg.fit_icp(n_iterations=30, nasion_weight=20, hsp_weight=10, verbose=True)

dists = coreg.compute_dig_mri_distances() * 1e3
log.info(
    f"Distance between HSP and MRI (mean/min/max):\n{dists.mean():.2f} mm "
    f"/ {dists.min():.2f} mm / {dists.max():.2f} mm"
)

# save data
write_trans(trans_file, coreg.trans, overwrite=True)
log.info(f"Coregistration of sub {sub} and ses {ses} and {task} completed.")

# drop files again
dl.save(trans_file, dataset=root,
         message=f'save trans file for {subject}')
dl.drop(raw_fname, dataset=raw_root)
dl.drop(anat_fname, dataset=anat_root)
