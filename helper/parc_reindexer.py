#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 14:06:14 2025
@author: okohl
Function to reorder osl and fressurfer parcel orders.
"""

import numpy as np

# Label Definitions:
# OSL 68 parcels first lh then rh
def load_osl_labels():
    lh_order = [
        'bankssts-lh', 'caudalanteriorcingulate-lh', 'caudalmiddlefrontal-lh',
        'cuneus-lh', 'entorhinal-lh', 'fusiform-lh', 'inferiorparietal-lh',
        'inferiortemporal-lh', 'isthmuscingulate-lh', 'lateraloccipital-lh',
        'lateralorbitofrontal-lh', 'lingual-lh', 'medialorbitofrontal-lh',
        'middletemporal-lh', 'parahippocampal-lh', 'paracentral-lh',
        'parsopercularis-lh', 'parsorbitalis-lh', 'parstriangularis-lh',
        'pericalcarine-lh', 'postcentral-lh', 'posteriorcingulate-lh',
        'precentral-lh', 'precuneus-lh', 'rostralanteriorcingulate-lh',
        'rostralmiddlefrontal-lh', 'superiorfrontal-lh', 'superiorparietal-lh',
        'superiortemporal-lh', 'supramarginal-lh', 'frontalpole-lh',
        'temporalpole-lh', 'transversetemporal-lh', 'insula-lh'
    ]
    rh_order = [lbl.replace('-lh', '-rh') for lbl in lh_order]
    return lh_order + rh_order

# Freesurfer 68 labels, interleaved L/R
def load_freesurfer_labels():
    return [  
        'bankssts-lh','bankssts-rh','caudalanteriorcingulate-lh','caudalanteriorcingulate-rh',
        'caudalmiddlefrontal-lh','caudalmiddlefrontal-rh','cuneus-lh','cuneus-rh',
        'entorhinal-lh','entorhinal-rh','frontalpole-lh','frontalpole-rh',
        'fusiform-lh','fusiform-rh','inferiorparietal-lh','inferiorparietal-rh',
        'inferiortemporal-lh','inferiortemporal-rh','insula-lh','insula-rh',
        'isthmuscingulate-lh','isthmuscingulate-rh','lateraloccipital-lh','lateraloccipital-rh',
        'lateralorbitofrontal-lh','lateralorbitofrontal-rh','lingual-lh','lingual-rh',
        'medialorbitofrontal-lh','medialorbitofrontal-rh','middletemporal-lh','middletemporal-rh',
        'paracentral-lh','paracentral-rh','parahippocampal-lh','parahippocampal-rh',
        'parsopercularis-lh','parsopercularis-rh','parsorbitalis-lh','parsorbitalis-rh',
        'parstriangularis-lh','parstriangularis-rh','pericalcarine-lh','pericalcarine-rh',
        'postcentral-lh','postcentral-rh','posteriorcingulate-lh','posteriorcingulate-rh',
        'precentral-lh','precentral-rh','precuneus-lh','precuneus-rh',
        'rostralanteriorcingulate-lh','rostralanteriorcingulate-rh',
        'rostralmiddlefrontal-lh','rostralmiddlefrontal-rh',
        'superiorfrontal-lh','superiorfrontal-rh','superiorparietal-lh','superiorparietal-rh',
        'superiortemporal-lh','superiortemporal-rh','supramarginal-lh','supramarginal-rh',
        'temporalpole-lh','temporalpole-rh','transversetemporal-lh','transversetemporal-rh'
    ]


def get_osl_reindexer():
    """
    Return
    ------
    reindexer : np.ndarray (int64, shape=(68,))
        Fancy‑indexing vector that maps a FreeSurfer‑ordered
        1‑D/2‑D array to OSL parcel order.
    osl_labels : list[str] (length 68)
        The OSL/Desikan label list corresponding to `reindexer`.
    """

    # --- build reindexer -----------------------------------------------
    freesurfer_labels = load_freesurfer_labels()
    osl_labels        = load_osl_labels()

    lut = {lbl: i for i, lbl in enumerate(freesurfer_labels)}
    reindexer = np.array([lut[lbl] for lbl in osl_labels], dtype=np.int64)

    return reindexer, osl_labels
