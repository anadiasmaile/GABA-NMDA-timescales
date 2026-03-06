"""
run fooof analysis on continuous source reconstructed data.
PSD and fooof is computed per vertex and then averaged over parcels
of the Desikan-Killiany atlas

@author: Ana Antonia Dias Maile
"""

import mne
from mne import read_source_estimate
from mne.time_frequency import psd_array_welch
from os import makedirs
import os.path as op
import datalad.api as dl
import matplotlib.pyplot as plt
import numpy as np
from fooof import FOOOF
from fooof.plts.annotate import plot_annotated_model
from fooof.utils.params import compute_time_constant, compute_knee_frequency
import pandas as pd
import numpy as np

""" OVERHEAD """
root = "."
raw_dir = op.join(root, '..', 'rawdata')
drugs = pd.read_csv(op.join(raw_dir, 'sessions.tsv'), sep='\t', header=0)
subs = 60
sess = 3
task = 'restingstate'
fooof_sets_knee = {
'peak_width_limits': [1, 6],
'max_n_peaks': 6,
'min_peak_height': 0.1,
'peak_threshold': 2,
'aperiodic_mode': 'knee'
}
fooof_sets_noknee = {
'peak_width_limits': [1, 6],
'max_n_peaks': 6,
'min_peak_height': 0.1,
'peak_threshold': 2,
'aperiodic_mode': 'fixed'
}
freq_range = [0.5, 40]
fits = []
save_model = False
quality_checks = False

""" FITTING """
for sub in range(51,subs+1):
    if sub == 14 or sub == 44:
        continue
    for ses in range(1,sess+1):
        if sub == 28 and ses == 1:
            continue
        # paths
        meg_dir = op.join(root, 'input_source', f'sub-{sub:02d}', f'ses-{ses:02d}')
        model_dir = op.join(root, f'sub-{sub:02d}', f'ses-{ses:02d}', 'meg')
        qc_dir = op.join(root, f'sub-{sub:02d}', f'ses-{ses:02d}', 'qc')
        for directory in [model_dir, qc_dir]:
            makedirs(directory, exist_ok=True)
        #subject stubs
        stub = f'sub-{sub:02d}_ses-{ses:02d}_task-{task}_%s'
        # unlock subject directory
        dl.unlock(op.join(root,f'sub-{sub:02d}', f'ses-{ses:02d}'))
        # read drug
        drug = drugs.loc[(drugs['participant_id']==sub)&(drugs['session_id']==ses),
                                'drug_label'].values[0]
        vertex_label_map = pd.read_csv(op.join(root, 'input_source',
                                               f'sub-{sub:02d}',
                                               f'ses-{ses:02d}',
                                               f'sub-{sub:02d}_ses-{ses:02d}_'
                                               f'task-{task}_vertex_to_label_map.csv'))
        """ READ STC """
        megname_lh = op.join(meg_dir, stub % 'raw_source_estimate-lh.stc')
        megname_rh = op.join(meg_dir, stub % 'raw_source_estimate-rh.stc')
        dl.get(megname_lh, dataset=f'{root}', source='origin')
        dl.get(megname_rh, dataset=f'{root}', source='origin')
        stc = read_source_estimate(megname_lh)
        data = stc.data
        sfreq = stc.sfreq
        psds, freqs = psd_array_welch(data, sfreq = sfreq, fmin = freq_range[0],
                                      fmax = freq_range[1], n_fft = 10000,
                                      n_per_seg = 10000, n_overlap = 5000,
                                      average = 'median', window = 'hamming',
                                      n_jobs = -1)
        for vertex in range(psds.shape[0]):
            power = psds[vertex,:]
            """ FOOOF """
            model = FOOOF(**fooof_sets_knee)
            model.fit(freqs, power, freq_range)
            # check if reasonable knee
            knee_freq = compute_knee_frequency(model.aperiodic_params_[1],
                                               exponent = model.aperiodic_params_[2])
            if knee_freq >= freq_range[0] and knee_freq <= freq_range[1]:
                pass
            else:
                model = FOOOF(**fooof_sets_noknee)
                model.fit(freqs, power, freq_range)
            # save model
            if save_model:
                model.save(op.join(model_dir, stub + str(label) + '_result'),
                save_results = True, save_settings = True, save_data = True)
            # quality checks
            if quality_checks:
                if len(model.peak_params_) == 0:
                    model.plot(save_fig = True, file_name =
                               op.join(qc_dir, stub + str(label) +
                               '_basic_plot.png'))
                else:
                    plot_annotated_model(model, plt_log = True)
                    plt.savefig(op.join(qc_dir, stub + str(label) +
                                        '_annotated_plot.png'))
            # save fitting results in table
            if model.get_settings()[4] == 'knee':
                fits.append({
                    'sub': sub,
                    'ses': ses,
                    'drug': drug,
                    'vertex': vertex,
                    'label': vertex_label_map.loc[vertex_label_map.vertex==vertex,
                                                  "parcel"].values[0],
                    'n_peaks':model.n_peaks_,
                    'offset': model.aperiodic_params_[0],
                    'knee': model.aperiodic_params_[1],
                    'knee_freq': compute_knee_frequency(model.aperiodic_params_[1],
                                                        exponent = model.aperiodic_params_[2]),
                    'time_constant': compute_time_constant(knee =
                                                           compute_knee_frequency(
                                                           model.aperiodic_params_[1],
                                                           exponent = model.aperiodic_params_[2])),
                    'exponent': model.aperiodic_params_[2],
                    'R^2': round(model.r_squared_,2)
                })
            else:
                fits.append({
                    'sub': sub,
                    'ses': ses,
                    'drug': drug,
                    'vertex': vertex,
                    'label': vertex_label_map.loc[vertex_label_map.vertex==vertex,
                                                  "parcel"].values[0],
                    'n_peaks':model.n_peaks_,
                    'offset': model.aperiodic_params_[0],
                    'knee': np.nan,
                    'knee_freq': np.nan,
                    'time_constant': np.nan,
                    'exponent': model.aperiodic_params_[1],
                    'R^2': round(model.r_squared_,2)
            })

fits = pd.DataFrame(fits)
fits.to_csv(op.join(root, 'fooof_fits_pervertex_allexp.csv'))
