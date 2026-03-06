"""
@author: Ana Antonia Dias Maile
"""
import os.path as op
import sys
from mne.viz import set_browser_backend
import mne
from mne import read_epochs
from eduTools import io
import datalad.api as dl
from glob import glob

set_browser_backend('matplotlib')

# read config file
root = op.abspath(sys.argv[1])
raw_root = op.join(root, 'input')

# bids fields
sub = int(sys.argv[2])
ses = int(sys.argv[3])
task = sys.argv[4]
try:
   acq = int(sys.argv[5])
except:
   acq = None
bids_fields = [sub, ses, task, acq]

# define paths
deriv_dir = op.join(root, f'sub-{sub:02d}', f'ses-{ses:02d}')
meg_dir = op.join(deriv_dir, 'meg')

# define stubs
ep_stub = f'sub-{sub:02d}_ses-{ses:02d}_task-{task}_long_%s'
if acq is not None:
    ep_stub = f'sub-{sub:02d}_ses-{ses:02d}_task-{task}_acq-{acq:02d}_%s'
deriv_stub = op.join(meg_dir, ep_stub)

# read the epochs
infiles = sorted(glob(op.join(meg_dir, ep_stub % 'clean-ep*.fif')))
dl.get(infiles, dataset=f'{root}')
ep_clean = read_epochs(infiles[-1])

# unlock files from previous runs
dl.unlock(dataset='.', path=deriv_dir)

# information about bad epochs
bads_info = op.join(deriv_dir, ep_stub % 'bad_info.json')
bad_epochs = io.read_json(bads_info)['bad_epochs']['noise']

# mark bad epochs
ep_clean.events[ep_clean.events == 99] = 1
ep_clean.events[bad_epochs, 2] = 99
ep_clean.event_id['bad_noise'] = 99

# save data
ep_clean.save(deriv_stub % 'clean-epo.fif', overwrite=True)
