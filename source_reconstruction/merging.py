#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 14:47:57 2025

@author: Oliver Kohl
"""
import mne
import numpy as np
import copy
from collections import Counter
import matplotlib.pyplot as plt
import sys

def merge_epochs_to_raw(epochs, duration=10, overlap=0.5, bad_id=99):
    """
    Merge MNE epochs into a continuous MNE raw object while removing overlap
    and marking bad epochs in annotations.

    Parameters
    ----------
    epochs : mne.Epochs
        The MNE Epochs object created with `mne.make_fixed_length_epochs`.
    duration : float, optional
        The duration of each epoch in seconds. Defaults to 10 seconds.
    overlap : float, optional
        The fraction of overlap between consecutive epochs (0 to 1).
        Defaults to 0.5 (50% overlap).
    bad_id : int, optional
        The event ID used to mark bad epochs in the events array.
        Defaults to 99.

    Returns
    -------
    raw_merged : mne.io.RawArray
        A continuous MNE Raw object with overlap removed and bad epochs marked
        as annotations.
    """
    # Sampling frequency
    sfreq = epochs.info['sfreq']
    # Step size to avoid overlap
    step_size = duration * (1 - overlap) if overlap < 1 else duration
    # Extract all epochs, removing overlap
    data_list = []

    for i in range(len(epochs)):
        epoch_data = epochs[i].get_data()

        if i == 0 or overlap == 0:
            data_list.append(epoch_data)  # Keep full epoch when no overlap
        else:
            samples_to_keep = int(step_size * sfreq)  # Keep only second half
            data_list.append(epoch_data[:, :, samples_to_keep:])

    # Merge data
    merged_data = np.concatenate(data_list, axis=2)  # Shape: (n_channels, n_times)

    # Preserve as much of the info object as possible
    info = copy.deepcopy(epochs.info)  # Deep copy to retain all metadata
    # This only works because we know that first epoch starts at beginning of
    # continuous tc.
    first_samp = epochs.events[0][0]

    # Create a new Raw object
    raw_merged = mne.io.RawArray(merged_data.squeeze(), info, first_samp=first_samp)

    # Handle existing annotations
    if epochs.annotations:
        merged_annotations = epochs.annotations.copy()
    else:
        merged_annotations = mne.Annotations(
            onset=[], duration=[], description=[],
            orig_time=epochs.info['meas_date']
        )

    # Add annotations for bad epochs marked in events (ID 99)
    bad_epoch_annotations = get_bad_epoch_annotations(epochs, duration=duration,
                                                      bad_id=bad_id)

    print(f'Time of first sample in raw: {first_samp/sfreq}')
    for ann in bad_epoch_annotations:
       print(f"'BAD' Annotation at {ann['onset']} (Duration: {ann['duration']})")

    # Merge annotations properly
    if bad_epoch_annotations:
        merged_annotations += bad_epoch_annotations

    # Assign annotations to the new raw object
    raw_merged.set_annotations(merged_annotations)

    return raw_merged


def get_bad_epoch_annotations(epochs, duration=10, bad_id=99):
    """
    Create annotations for bad epochs based on event markers
    (e.g., ID 99 for bad epochs).

    Parameters
    ----------
    epochs : mne.Epochs
        The MNE Epochs object containing events where bad epochs are marked.
    duration : float, optional
        The duration of each epoch in seconds. Defaults to 10 seconds.
    bad_id : int, optional
        The event ID used to mark bad epochs in the events array. Defaults to 99.

    Returns
    -------
    bad_annotations : mne.Annotations
        Annotations marking bad epochs, or an empty Annotations object if none exist.
    """
    sfreq = epochs.info['sfreq']
    bad_events = epochs.events[epochs.events[:, 2] == bad_id]

    if len(bad_events) == 0:
        return mne.Annotations(onset=[], duration=[], description=[],
                               orig_time=epochs.info['meas_date'])

    # Convert sample indices to absolute time
    bad_times = bad_events[:, 0] / sfreq

    return mne.Annotations(
        onset=bad_times,
        duration=[duration] * len(bad_times),
        description=["BAD_epoch"] * len(bad_times),
        orig_time=epochs.info['meas_date']
    )


def check_merging(raw, epochs, log, psd_path, plot_psd=True):
    # check if sample sizes make sense
    n_epochs, _, len_epochs = epochs.get_data().shape
    # Add half epoch because for first epoch we need whole epoch
    n_expected = n_epochs * len_epochs
    n_raw = raw.get_data().shape[1]
    if n_expected == n_raw:
        print(f'Sample Sizes in merged_raw (n={n_raw}) match expected sample size.')
    else:
        sys.exit(f'Sample size of merged_raw (n={n_raw})
                 f'does not match to expected sample size (n={n_expected}).')
    # check if any values are repeated in time courses
    is_dublication = any(count > 2 for count in Counter(raw.get_data().mean(axis=0)).values())
    if is_dublication:
        sys.exit('Some samples show identical values. Carefully, check whether'
                 'redundant samples occure because of imperfect alignement.')
    else:
        print('No redundant samples.')
    # check number of epochs and annotations
    n_bads_raw = sum([description == 'BAD_epoch' for description in raw.annotations.description])
    n_bads_epochs = len(epochs['bad_noise'])
    if n_bads_raw == n_bads_epochs:
        print(f'Number of bad epochs (n={n_bads_raw}) is the same for raw and epochs object.')
    else:
        sys.exit(f'Number of bad epochs in raw object (n={n_bads_raw}) does not'
                 f'match bad epochs (n={n_bads_epochs}) in epoch object.')

    # quick psd plot ----
    if plot_psd == True:
        sfreq = raw.info['sfreq']
        raw.compute_psd(n_per_seg= int(sfreq*2),
                        n_fft=int(sfreq*2),
                        fmax=30).plot(show = False, spatial_colors = True)
        plt.savefig(f'{psd_path}')
        plt.close('all')
