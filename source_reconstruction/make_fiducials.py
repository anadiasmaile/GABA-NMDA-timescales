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
from glob import glob
import datalad.api as dl
from mne.gui import coregistration

""" OVERHEAD """
root = op.abspath(sys.argv[1])
anat_root = op.join(root, 'input_anat')

# bids fields
sub = int(sys.argv[2])
subject = f'sub-{sub:02d}'
subjects_dir = op.join(anat_root, 'freesurfer')

anat_fname = glob(op.join(subjects_dir, subject))
fid_file = op.join(subjects_dir, subject, 'bem', f'{subject}-fiducials.fif')

""" READ RAW AND ANAT """
# Get anatomical data
dl.get(anat_fname, source='data-source', dataset=f'{anat_root}')

if not op.exists(op.dirname(fid_file)):
    print("Abort! First the bem needs to be set up!")
    sys.exit(1)

# Run the gui to set the mr fiducials")
coregistration(subject=subject, subjects_dir=subjects_dir, verbose=True,
               block=True)
dl.save(fid_file, dataset=root, message=f'save MR fiducials for {subject}')
dl.drop(anat_fname, dataset=anat_root)
