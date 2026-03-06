"""
@author: Eduard Ort
"""

import os.path as op
import sys
from mne.viz import set_browser_backend
import mne
from mne.preprocessing import read_ica
import logging as log
from numpy import where
from datetime import datetime
from eduTools import meg as mt
from eduTools import io
from mne_bids import BIDSPath
import datalad.api as dl
from glob import glob
from autoreject import read_auto_reject
from idiosyncracies import add_metadata
from pandas import DataFrame
from IPython import embed as shell

set_browser_backend('matplotlib')

""" OVERHEAD """
# read config file
root = op.abspath(sys.argv[1])
cfg = io.read_json(sys.argv[2])
raw_root = op.join(root, 'input')

# bids fields
sub = int(sys.argv[3])
ses = int(sys.argv[4])
task = cfg['task']
bids_fields = [sub, ses, task]

# general settings
n_jobs = cfg['n_jobs']
seed = cfg['seed']

# preproc settings
ep_info = cfg['ep_info']
ch_picks = ['meg', 'eog', 'ecg']

# define paths
deriv_dir = op.join(root, f'sub-{sub:02d}', f'ses-{ses:02d}')
log_dir = op.join(deriv_dir, 'log', task)
qc_dir = op.join(deriv_dir, 'qc', task)
meg_dir = op.join(deriv_dir, 'meg')

# define bids-related paths
bids_path = BIDSPath(root=raw_root, subject=f'{sub:02d}',
                     session=f'{ses:02d}', task=task, datatype='meg')
raw_path = bids_path.copy().update(root=root)

# define stubs
file_stub = f'sub-{sub:02d}_ses-{ses:02d}_task-{task}_%s'
deriv_stub = op.join(meg_dir, file_stub)
qc_stub = op.join(qc_dir, file_stub)

raw_files = sorted(glob(str(raw_path.directory /
                            f'{file_stub}*raw.fif') % ''))
dl.get(raw_files, source=cfg['file_src'], dataset=f'{root}')

# unlock files from previous runs
dl.unlock(dataset='.', path=deriv_dir)

# information about excluded channels or epochs
bads_info = op.join(deriv_dir, file_stub % 'bad_info.json')

# set up logging
date = str(datetime.now()).replace(' ', '-').replace(':', '-')
log_file = op.join(log_dir, file_stub % f'preproc_{date}.log')
log.basicConfig(filename=log_file, level=log.DEBUG,
                format='%(asctime)s %(message)s')
mne.set_log_file(fname=log_file, overwrite=True)
log.info('Logfile initialized.')
for k, v in cfg.items():
    log.info(f'Parameter {k} set to {v} in config file')
log.getLogger('matplotlib').setLevel(log.WARNING)

""" READ RAW """
raw = mne.io.read_raw_fif(raw_files[0])
megs = mne.pick_types(raw.info, meg=True, eeg=False)

# update the bad channels with manually created bad channels
raw.info['bads'] = mt.set_bad_channels(op.join(raw_path.directory,
                                               file_stub % 'channels.tsv'))
if sub == 39 and ses == 1 and task == 'restingstate':
    raw.crop(tmin = 34)
elif sub == 3 and ses == 1 and task == 'restingstate':
    raw.crop(tmin = 18)
else:
    log.info("No cropping needed.")

""" Update Bad Annotations """
if cfg['update_annot']:
    log.info("Processing step: Update annotations")
    annots = mt.shift_onset_annotations(raw.annotations, annot_info['event'],
                                        annot_info['time_shift'])
    raw = raw.set_annotations(annots)
else:
    log.info("Step skipped: Don't update annotations")

# make annoying warnings disappear
raw.info.normalize_proj()

""" CREATE EPOCHS """
log.info("Define event dicts and several types of epochs.")

# create epochs for autoreject and ica
ep_art = mne.make_fixed_length_epochs(raw,
                                      duration=ep_info['epoch_length_art'])

# and the real epochs
ep_clean = mne.make_fixed_length_epochs(raw,
                                        duration=ep_info['fixed_epoch_length'],
                                        overlap=ep_info['overlap'])

# one more set of events that has potential alpha activity removed
# this one is used to compute 2nd autoreject on before applying it
# to the real epochs
raw_notch = raw.copy().load_data().notch_filter(10,
                                                notch_widths=3)
ep_notch = mne.make_fixed_length_epochs(raw_notch,
                                        duration=ep_info['fixed_epoch_length'],
                                        overlap=ep_info['overlap'])

""" ICA """
log.info("Processing step: Run ICA")
if cfg['compute_ica'] == 'run':
    AR_settings = dict(apply_ar=cfg['preICA_clean'],
                       ar_path=deriv_stub % 'preICA_autorej.hdf5')
    ica = mt.compute_ICA(ep_art, AR_settings, l_freq=1.2, n_components=60,
                         random_state=seed, n_jobs=n_jobs, qc_stub=qc_stub)
    ica.save(deriv_stub % 'epo-ica.fif', overwrite=True)
elif cfg['compute_ica'] == 'load':
    log.info("Skip ICA and run solution from file")
    ica = read_ica(deriv_stub % 'epo-ica.fif')
elif cfg['compute_ica'] == 'skip':
    log.info("Step skipped: No ICA present. Next step also needs to go")

# load epochs
ep_notch = ep_notch.load_data()
ep_clean = ep_clean.load_data()

log.info("Link ICA components to artifacts + diagnostics")
if cfg['inspect_ica'] == 'run':
    # store explain variance to the bads file (for lack of better places)
    io.write_json(bads_info,
                  {'ica_variance':ica.get_explained_variance_ratio(ep_clean)})
    veog_idx = mt.classify_components(ica, raw, ep_notch, 'veog', deriv_stub,
                                      qc_stub)
    ecg_idx = mt.classify_components(ica, raw, ep_notch, 'ecg', deriv_stub,
                                     qc_stub)
    # plot artifact
    mt.plot_artifact(ica, raw, 'veog', veog_idx, False, deriv_stub, qc_stub)
    mt.plot_artifact(ica, raw, 'ecg', ecg_idx, False, deriv_stub, qc_stub)

    # save components
    io.write_json(bads_info, {'ica_veog': [int(i) for i in veog_idx]})
    io.write_json(bads_info, {'ica_ecg': [int(i) for i in ecg_idx]})

elif cfg['inspect_ica'] == 'load':
    log.info("Load artifact-linked components from file.")
    veog_idx = io.read_json(bads_info)['ica_veog']
    ecg_idx = io.read_json(bads_info)['ica_ecg']

    # plot artifact
    mt.plot_artifact(ica, raw, 'veog', veog_idx, True,  deriv_stub, qc_stub)
    mt.plot_artifact(ica, raw, 'ecg', ecg_idx, True, deriv_stub, qc_stub)

elif cfg['inspect_ica'] == 'skip':
    log.info("Step skipped. No ICA diagnostics computed")

# apply solution
if cfg['inspect_ica'] != 'skip':
    ica.exclude = veog_idx + ecg_idx
    if len(ica.exclude) == 0:
        io.write_json(bads_info,
                     {'ica_noise_variance': 0})
    else:
        io.write_json(bads_info,
                  {'ica_noise_variance':
                   ica.get_explained_variance_ratio(ep_clean,
                                                    components=ica.exclude)})
    ica.apply(ep_notch)
    ica.apply(ep_clean)

# center epoch at zero
ep_notch.apply_baseline((None, None))

""" AUTOMATIC ARTIFACT FIXING PART 2 """
log.info("Processing step: Automatic noise detection with autorej. Part 2")
if cfg['postICA_clean'] == 'run':
    # once for grads
    meg_rej, rej_log = mt.run_autorej(ep_notch.load_data(), megs, n_jobs, seed,
                                      qc_stub % 'qc_autorej.svg')
    meg_rej.save(deriv_stub % 'autorej.hdf5', overwrite=True)

    # extract the rejection log file and set the bad epochs to empty to avoid
    # them to be dropped
    bad_meg = rej_log.bad_epochs
    rej_log.bad_epochs = []

    # apply interpolation part of autoreject
    ep_clean = meg_rej.transform(ep_clean, reject_log=rej_log)

    # get indices of bad epochs for log purposes
    bad_epochs = where(bad_meg)[0].tolist()

    # write meta data to bads file
    AR = dict(n_interpolate_grad=int(meg_rej.n_interpolate_['grad']),
              consensus_grad=meg_rej.consensus_['grad'],
              n_interpolate_mag=int(meg_rej.n_interpolate_['mag']),
              consensus_mag=meg_rej.consensus_['mag'])

    io.write_json(bads_info, {'autorej':AR})
    io.write_json(bads_info, {'bad_epochs':{'noise':bad_epochs}})

elif cfg['postICA_clean'] == 'load':
    log.info("Skip running autoreject, and read bad epoch info from file.")
    # load cleaned epochs (load only mags, as grads are included here)

    meg_rej = read_auto_reject(deriv_stub % 'autorej.hdf5')
    rej_log = meg_rej.get_reject_log(ep_clean)
    bad_meg = rej_log.bad_epochs
    rej_log.bad_epochs = []

    # apply autoreject
    ep_clean = meg_rej.transform(ep_clean, reject_log=rej_log)
    bad_epochs = io.read_json(bads_info)['bad_epochs']['noise']

elif cfg['postICA_clean'] == 'skip':
    bad_epochs = []
    log.info("Step skipped: No noisy epochs marked")

# mark bad epochs
ep_clean.events[bad_epochs, 2] = 99
ep_clean.event_id['bad_noise'] = 99

""" SAVE FULLY PREPROCESSED """
log.info("Preprocessing finished. Store clean epochs to file.")
ep_clean.save(deriv_stub % 'clean-epo.fif', overwrite=True)
