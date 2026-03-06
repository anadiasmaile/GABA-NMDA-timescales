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
from pandas import DataFrame

def emptyroom_preproc(root, sub, ses, task):
    """ FUNCTION TO PREPROC EMPTY ROOM BASED ON RESTING STATE PREPROC """
    ##############
    ## OVERHEAD ##
    ##############
    raw_root = op.join(root, 'input_raw')
    preproc_root = op.join(root, 'input_clean')
    # bids fields
    bids_fields = [sub, ses, task]
    # general settings
    n_jobs = 7
    seed = 10
    # preproc settings
    ch_picks = ['meg']
    # define paths to get restingstate preproc
    preproc_dir = op.join(preproc_root, f'sub-{sub:02d}', f'ses-{ses:02d}')
    short_stub = f'sub-{sub:02d}_ses-{ses:02d}_task-{task}_%s'
    # define bids-related paths
    bids_path = BIDSPath(root=raw_root, subject=f'{sub:02d}',
                         session=f'{ses:02d}', task=task, datatype='meg')
    bids_path_preproc = BIDSPath(root=preproc_root, subject=f'{sub:02d}',
                         session=f'{ses:02d}', task=task, datatype='meg')
    preproc_path = bids_path_preproc.copy()
    raw_files = bids_path.find_empty_room()
    dl.get(raw_files, source="origin", dataset=f'{raw_root}')
    # information about extra bad channels
    bad_info = op.join(preproc_dir, short_stub % 'long_bad_info.json')
    cfg = io.read_json(bad_info)
    # information about ica
    ica_info = op.join(preproc_dir, short_stub % 'bad_info.json')

    ##############
    ## READ RAW ##
    ##############
    raw = mne.io.read_raw_fif(raw_files)
    megs = mne.pick_types(raw.info, meg=True, eeg=False)
    # update the bad channels with manually created bad channels
    raw.info['bads'] = mt.set_bad_channels(op.join(preproc_path.directory,
                                                   short_stub % 'channels.tsv'))
    # make annoying warnings disappear
    raw.info.normalize_proj()
    # crop chunks with open door noise
    if raw.info["meas_date"].year == 2020 and raw.info["meas_date"].month == 11:
        if raw.info["meas_date"].day == 20:
            raw_first = raw.copy().crop(tmax=122)
            raw_second = raw.copy().crop(tmin=126)
            raw_first.append([raw_second])
            raw = raw_first
        elif raw.info["meas_date"].day == 24:
            raw_first = raw.copy().crop(tmax=56)
            raw_second = raw.copy().crop(tmin=58)
            raw_first.append([raw_second])
            raw = raw_first
        else:
            pass
    elif raw.info["meas_date"].year == 2021:
        if raw.info["meas_date"].month == 3 and raw.info["meas_date"].day == 23:
            raw_first = raw.copy().crop(tmax=110)
            raw_second = raw.copy().crop(tmin=118)
            raw_first.append([raw_second])
            raw = raw_first
        elif raw.info["meas_date"].month == 4:
            if raw.info["meas_date"].day == 6:
                raw_first = raw.copy().crop(tmax=115)
                raw_second = raw.copy().crop(tmin=117)
                raw_first.append([raw_second])
                raw = raw_first
            elif raw.info["meas_date"].day == 27:
                raw.crop(tmin=18)
            elif raw.info["meas_date"].day == 28:
                raw_first = raw.copy().crop(tmax=88)
                raw_second = raw.copy().crop(tmin=94)
                raw_first.append([raw_second])
                raw = raw_first
            else:
                pass
        elif raw.info["meas_date"].month == 5 and raw.info["meas_date"].day == 11:
            raw_first = raw.copy().crop(tmax=615)
            raw_second = raw.copy().crop(tmin=628)
            raw_first.append([raw_second])
            raw = raw_first
        elif raw.info["meas_date"].month == 6:
            if raw.info["meas_date"].day == 1:
                raw_first = raw.copy().crop(tmax=71)
                raw_second = raw.copy().crop(tmin=72)
                raw_first.append([raw_second])
                raw = raw_first
            elif raw.info["meas_date"].day == 9:
                raw_first = raw.copy().crop(tmax=136)
                raw_second = raw.copy().crop(tmin=142)
                raw_first.append([raw_second])
                raw = raw_first
            else:
                pass
        elif raw.info["meas_date"].month == 7 and raw.info["meas_date"].day == 14:
            raw_first = raw.copy().crop(tmax=176)
            raw = raw_first
        elif raw.info["meas_date"].month == 9 and raw.info["meas_date"].day == 7:
            raw_first = raw.copy().crop(tmax=40)
            raw_second = raw.copy().crop(tmin=50, tmax = 184)
            raw_first.append([raw_second])
            raw = raw_first
        elif raw.info["meas_date"].month == 11 and raw.info["meas_date"].day == 2:
            raw_first = raw.copy().crop(tmax=311)
            raw = raw_first
        else:
            pass
    elif raw.info["meas_date"].year == 2022:
        if raw.info["meas_date"].month == 1 and raw.info["meas_date"].day == 11:
            raw.crop(tmin=3)
        elif raw.info["meas_date"].month == 2 and raw.info["meas_date"].day == 16:
            raw_first = raw.copy().crop(tmax=164)
            raw = raw_first
        elif raw.info["meas_date"].month == 3 and raw.info["meas_date"].day == 23:
            raw_first = raw.copy().crop(tmax=188)
            raw = raw_first
        elif raw.info["meas_date"].month == 4:
            if raw.info["meas_date"].day == 5:
                raw.crop(tmin=6)
            elif raw.info["meas_date"].day == 26:
                raw.crop(tmin=34)
            else:
                pass
        elif raw.info["meas_date"].month == 5 and raw.info["meas_date"].day == 11:
            raw_first = raw.copy().crop(tmax=138)
            raw_second = raw.copy().crop(tmin=142)
            raw_first.append([raw_second])
            raw = raw_first
        elif raw.info["meas_date"].month == 7:
            if raw.info["meas_date"].day == 5:
                raw_first = raw.copy().crop(tmax=70)
                raw_second = raw.copy().crop(tmin=84)
                raw_first.append([raw_second])
                raw = raw_first
            elif raw.info["meas_date"].day == 21:
                raw_first = raw.copy().crop(tmax=178)
                raw_second = raw.copy().crop(tmin=188)
                raw_first.append([raw_second])
                raw = raw_first
            else:
                pass
        else:
            pass
    else:
        pass
    #######################################
    ## RAW PREPROC (FILTERS AND ZAPLINE) ##
    #######################################
    l_freq = 0.1
    emptyroom = raw.load_data()
    if sub == 50 and ses == 3:
        emptyroom.pick_types(meg=True, stim=True, eog=True, ecg=True)
    else:
        pass
    # line noise removal
    emptyroom = mt.zapline(emptyroom, nremove=10)
    # highpass filtering
    emptyroom = mt.filtering(emptyroom, l_freq=l_freq, fmin=0.01, picks=megs)

    #########
    ## ICA ##
    #########
    dl.get(op.join(preproc_dir, "meg", short_stub % 'epo-ica.fif'), source='data-source',
           dataset=f'{preproc_root}')
    ica = read_ica(op.join(preproc_dir, "meg", short_stub % 'epo-ica.fif'))
    # Load artifact-linked components from file
    veog_idx = io.read_json(ica_info)['ica_veog']
    ecg_idx = io.read_json(ica_info)['ica_ecg']
    ica.apply(emptyroom)

    ########################
    ## EXTRA BAD CHANNELS ##
    ########################
    # add extra bad channels from visual inspection of emptyroom
    if 'extra_bad_channels_emptyroom' in cfg:
        bads = io.read_json(bad_info)['extra_bad_channels_emptyroom']
        raw.info['bads'] = list(set(raw.info['bads'] + bads))

    ###############################
    ## RETURN FULLY PREPROCESSED ##
    ###############################
    return emptyroom
