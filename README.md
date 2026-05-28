# GABA-NMDA-timescales

> Analysis code and scripts used for the manuscript:
> **Role of GABA- and NMDA-receptors in shaping cortical timescales and network dynamics**.

---

## Overview

This repository contains the full analysis pipeline used to generate the main results and figures for the manuscript.

It includes workflows for:

- Data preprocessing  
- Spectral analysis and timescale estimation  
- Inference of dynamic cortical networks  
- Network-specific timescale analysis  
- Statistical testing  
- Figure generation  

The pipeline integrates signal processing, computational modeling, and statistical analysis to investigate how GABA- and NMDA-mediated receptor activity shapes cortical dynamics.

📄 **Manuscript / preprint:** (https://www.biorxiv.org/content/10.64898/2026.05.11.723797v1)

---

## Repository structure
- preprocessing (filtering, artefact removal)
- source_reconstruction (source reconstruction, atlas parcellating)
- computation_of_timescales (calculation of power spectra and timescales across the cortex, calculation of network-specific timescales across cortex)
- analysis_of_timescales (associations with publicly avaible T1wT2w map and timescale map, condition contrasts, plotting)
- inference_of_tde_hmm (preparation & running of TDE-HMM, calculation of network descriptions & metrics)
- analyses_of_network_specific_timescales (dynamic & condition contrasts for network-specific timescales)
- analyses_of_network_dynamics (condition contrasts for state metrics and tranistion probabilities)
- helper (sorting of parcels in volume versions according to order in surface version of DK-Atlas )

---

## Methods Summary

The analysis pipeline consists of the following major steps:

1. **Preprocessing**  
   Filtering, artefact rejection, source reconstruction, and cortical parcellation.

2. **Timescale Estimation**  
   - Power spectral density (PSD) estimation  
   - Extraction of timescales using spectral parameterization (fooof)

3. **Dynamic Network Inference**  
   - Time-delay embedded Hidden Markov Models (TDE-HMM)  
   - Identification of recurring brain states and their temporal dynamics  

4. **Network Analysis**  
   - State-specific metrics  
   - Transition probabilities  
   - Network-specific timescales  

5. **Statistical Analysis & Visualization**  
   - Condition contrasts (e.g., Lorazepam vs. Placebo)  
   - Associations with structural maps (e.g., T1/T2)  
   - Figure generation for publication  

---

## Environment & Dependencies

### Core environments

- **Preprocessing**
   - MNE-Python
   - eduTools
     
- **Timescale analysis**
  - FOOOF
  - neuromaps 

- **Dynamic network analysis**
  - OSL-Dynamics  

### Installation

- MNE: __[https://mne.tools/stable/install/index.html](https://mne.tools/stable/install/index.html)__
- eduTools: __[https://github.com/eort/eduTools](https://github.com/eort/eduTools)__
- FOOOF: __[https://fooof-tools.github.io/fooof/](https://fooof-tools.github.io/fooof/)__
- neuromaps: __[https://github.com/netneurolab/neuromaps](https://github.com/netneurolab/neuromaps)__
- OSL-Dynamics: __[https://osl-dynamics.readthedocs.io/en/latest/](https://osl-dynamics.readthedocs.io/en/latest/)__ 

We recommend using separate virtual environments for:
- Signal processing (MNE + eduTools + FOOOF + neuromaps)
- Network modeling (OSL-Dynamics)
