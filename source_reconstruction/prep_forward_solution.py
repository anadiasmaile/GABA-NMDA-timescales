"""
Starting point for source reconstruction. Compute BEM surfaces and
source space per participant.

The script generates new data in input_anat (the bem directory), so
after it is run for the first time, the new files have to be saved with
datalad before they can be dropped.

@author: Eduard Ort
"""
import sys
import os.path as op
from os import (makedirs, getcwd)
from glob import glob
import datalad.api as dl
from mne import (set_log_file, setup_source_space, write_source_spaces)
from mne.bem import (make_scalp_surfaces, make_watershed_bem,
                     make_bem_model, make_bem_solution,
                     write_bem_surfaces, write_bem_solution,
                     read_bem_surfaces)
from datetime import datetime
import logging as log
from eduTools import meg

""" OVERHEAD """
root = op.abspath(sys.argv[1])
anat_root = op.join(root, 'input_anat')

# bids fields
sub = int(sys.argv[2])

# general settings
date = str(datetime.now()).replace(' ', '-').replace(':', '-')

subject = f'sub-{sub:02d}'
subjects_dir = op.join(anat_root, 'freesurfer')
deriv_dir = op.join(root, subject)
log_dir = op.join(deriv_dir, 'log')

makedirs(log_dir, exist_ok=True)

# define file names
out_stub = op.join(deriv_dir, f'{subject}_%s')
anat_fname = glob(op.join(subjects_dir, subject))
log_file = op.join(log_dir, f'{subject}_prep_bem_{date}.log')

# set up logging
log.basicConfig(filename=log_file, level=log.INFO,
                format='%(asctime)s %(message)s')
set_log_file(fname=log_file, overwrite=True)
log.info('Logfile initialized.')

""" READ RAW AND ANAT """
log.info("Retrieving files from remote.")

# get anatomical data
dl.get(anat_fname, source='data-source', dataset=anat_root)
log.info("Files retrieved.")
dl.unlock(anat_fname, dataset=anat_root)
dl.unlock(['sub-36/sub-36_bem-sol.fif', 'sub-36/sub-36_oct6-src.fif'],
          dataset='.')

log.info(f"Make surfaces for the layers")
make_watershed_bem(subject, subjects_dir=subjects_dir, overwrite=True)

log.info(f"Generate scalp surfaces for the coregistration")
make_scalp_surfaces(subject, subjects_dir=subjects_dir, overwrite=True)

try:
    read_bem_surfaces(op.join(subjects_dir, subject, 'bem',
                              f'{subject}-head-dense.fif'))
except ValueError as e:
    log.info("Topological defects detected that need fixing!")
    print("Topological defects detected that need fixing!")
    dl.unlock(dataset=anat_root, path=anat_fname)
    meg.fix_bem_surfaces(subjects_dir, subject)
    make_scalp_surfaces(subject, subjects_dir=subjects_dir, overwrite=True)


log.info(f"Setup BEM model, standard single-shell layer")
model = make_bem_model(subject=subject, ico=4, conductivity=[0.3],
                       subjects_dir=subjects_dir)

log.info(f"make BEM solution")
bem_sol = make_bem_solution(model)
write_bem_solution(out_stub % 'bem-sol.fif', bem_sol, overwrite=True)

#from IPython import embed as shell;shell()
log.info(f"Setup source space, surface based with oct6 spacing")
src = setup_source_space(subject=subject, subjects_dir=subjects_dir,
                         spacing='oct6', verbose=10, n_jobs=6, add_dist='patch')
write_source_spaces(out_stub % 'oct6-src.fif', src, overwrite=True)

log.info(f"Done.")

dl.save(op.join(subjects_dir, subject), dataset=anat_root,
         message=f'Save bem stuff for {subject}')
dl.save(deriv_dir, dataset=root,
        message=f'Save source space and bem model for {subject}')
# drop files again
dl.drop(anat_fname, dataset=anat_root)
