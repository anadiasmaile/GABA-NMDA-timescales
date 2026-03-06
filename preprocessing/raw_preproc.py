"""
@author: Eduard Ort
"""
from os import (makedirs, getcwd)
import os.path as op
import sys
import mne
from mne import pick_types
import logging as log
from datetime import datetime
from eduTools import meg as mt
from eduTools import io
from mne_bids import (BIDSPath, read_raw_bids)
import datalad.api as dl
from glob import glob

""" OVERHEAD """
# read config file
root = op.abspath(sys.argv[1])
raw_root = op.join(root, 'input')
cfg = io.read_json(sys.argv[2])

# bids fields
sub = int(sys.argv[3])
ses = int(sys.argv[4])
task = cfg['task']

# general settings
seed = cfg['seed']
n_jobs = cfg['n_jobs']

# preproc settings
sss_settings = cfg['sss_settings']
l_freq = cfg['lp_freq']
date = str(datetime.now()).replace(' ', '-').replace(':', '-')

# define paths
bids_path = BIDSPath(root=raw_root, subject=f'{sub:02d}',
                     session=f'{ses:02d}', task=task, datatype='meg')

deriv_dir = op.join(root, f'sub-{sub:02d}', f'ses-{ses:02d}')
log_dir = op.join(deriv_dir, 'log', task)
qc_dir = op.join(deriv_dir, 'qc', task)
meg_dir = op.join(deriv_dir, 'meg')

for directory in [log_dir, meg_dir, qc_dir]:
    makedirs(directory, exist_ok=True)

# define stubs
raw_dir = bids_path.directory
raw_stub = f'sub-{sub:02d}_ses-{ses:02d}_task-{task}_%s'
deriv_stub = op.join(meg_dir, raw_stub)
qc_stub = op.join(qc_dir, raw_stub)
stub = "filt_raw"

# preprocessing files
finecalibration = bids_path.meg_calibration_fpath
crosstalk = bids_path.meg_crosstalk_fpath
log_file = op.join(log_dir, raw_stub % f'preproc_{date}.log')

# information about excluded channels or epochs
bads_info = deriv_stub % 'bad_info.json'

# set up logging
log.basicConfig(filename=log_file, level=log.DEBUG,
                format='%(asctime)s %(message)s')
mne.set_log_file(fname=log_file, overwrite=True)
log.getLogger('matplotlib').setLevel(log.WARNING)
log.info('Logfile initialized.')
for k, v in cfg.items():
    log.info(f'Parameter {k} set to {v} in config file')

""" READ RAW """
log.info("Retrieving files from remote.")
infiles = glob(op.join(raw_dir, raw_stub % '*meg.fif'))
dl.get(infiles, source=cfg['file_src'], dataset=f'{raw_root}')
log.info("Files retrieved")

log.info("Reading raw...")
raw = read_raw_bids(bids_path)

megs = pick_types(raw.info, meg=True, eeg=False)

# drop stim channel to save memory
raw.drop_channels([raw.ch_names[i] for i in pick_types(raw.info, stim=True)])

"""EXCLUDE BAD PERIODS"""
# get rid of breaks in the experiment as they only blow up
# size of files without contributing meaningful data
log.info("Processing step: Experimental break annotation")
if cfg['bad_periods'] == 'run':
    log.info("Reading events...")
    events, _ = mt.read_events(raw, bids_path.update(suffix='events'))
    bad_periods = mt.annotate_breaks(raw, events, cfg['ep_info'])
    raw.set_annotations(raw.annotations + bad_periods)
    raw.annotations.save(deriv_stub % 'bad_annot.csv', overwrite=True)
    log.info("Bad episodes successfully annotated.")

elif cfg['bad_periods'] == 'load':
    log.info("Annotationg pauses skipped. Read annotations from file.")
    bad_annots = mne.read_annotations(deriv_stub % 'bad_annot.csv')
    raw.set_annotations(bad_annots)

elif cfg['bad_periods'] == 'skip':
    log.info("Step skipped: No breaks annotated")

""" BAD CHANNELS """
log.info("Processing step: Mark bad channels with SSS")
if cfg['find_bads'] == 'run':
    bads = mt.find_bad_channels(raw, crosstalk, finecalibration,
                                min_count=sss_settings['min_count'],
                                duration=sss_settings['duration'],
                                limit=sss_settings['limit'],
                                outpath=qc_stub % 'qc_bads.png')
    raw.info['bads'] = list(set(bads + raw.info['bads']))

    # save bad channels
    io.write_json(bads_info, {'bad_channels':raw.info['bads']})

elif cfg['find_bads'] == 'load':
    log.info("Load SSS-based bads from file.")
    bad_channels = io.read_json(bads_source)['bad_channels']
    raw.info['bads'] = bad_channels

elif cfg['find_bads'] == 'skip':
    log.info("Step skipped: No SSS-based bad channel detection")

""" ENVIRONMENT NOISE (uncorrelated over channels)- OTP """
log.info("Processing step: Oversampled temporal projection")
if cfg['otp']:
    raw = mt.oversampled_temporal_projection_wrapper(raw,
                                                     qc_stub % 'qc_otp.png')
else:
    log.info("Step skipped: No OTP cleaned data")

""" ENVIRONMENT NOISE - (correlated over channels) """
log.info("Processing step: SSS to clean data")
if cfg['sss']:
    raw = mt.maxwell_filter_wrapper(raw, calibration=finecalibration,
                                    cross_talk=crosstalk,
                                    duration=cfg['st_duration'],
                                    correlation=cfg['st_correlation'],
                                    outpath=qc_stub % 'qc_sss.png')
    stub = "raw_sss"
else:
    log.info("Step skipped: No SSS cleaned data")

"""  LINE NOISE REMOVAL  """
log.info("Processing step: Notch filter (Zapline)")
if cfg['zapline']:
    raw = raw.load_data()
    log.info("Notch filtering with ZAPline.")
    raw = mt.zapline(raw, nremove=cfg['zap_comp'],
                     outpath=qc_stub % 'qc_zapline.png')
else:
    log.info("Step skipped: No Notch filtering!")

""" HIGHPASS FILTERING """
log.info("Processing step: Highpass filter")
if cfg['hp_filt']:
    raw = raw.load_data()
    log.info("Apply a highpass filter with a frequency of %fHz." % l_freq)
    raw = mt.filtering(raw, l_freq=l_freq, fmin=0.01, picks=megs,
                       outpath=qc_stub % 'qc_hpFiltering.png')
else:
    log.info("Step skipped: No highpass filtering!")

# save data
raw.save(deriv_stub % f'{stub}.fif', overwrite=True,
                  split_naming='bids')
log.info(f"Processing of sub {sub} and ses {ses} completed.")

# drop files again
dl.drop(infiles, dataset=f'{raw_root}')
