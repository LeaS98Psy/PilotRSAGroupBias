# RSA Memory Project

## Purpose

This project investigates how political group membership (ingroup vs. outgroup) affects memory representations. Using Representational Similarity Analysis (RSA), it compares the representational geometry of item recognition (remembering a statement) and source memory (remembering who said it) across three conditions: baseline (non-political), ingroup (politically aligned speaker), and outgroup (politically opposing speaker).

The project serves two main purposes:

1. **Demonstrate RSA on real behavioural data**: showing how Representational Dissimilarity Matrices (RDMs) can reveal the similarity structure of memory representations.

2. **Simulate realistic fMRI data**: providing a complete pipeline that extends RSA from behaviour to neuroimaging, including synthetic data generation, noise modelling, and statistical validation (bootstrap, cross-validation).

## What You Can Do With These Scripts

### Script 1: `01_behavioural_RSA.py` Behavioural RSA

- Load real behavioural data (item recognition and source memory accuracies from 50 participants)
- Compute Representational Dissimilarity Matrices (RDMs) for both memory domains
- Normalise the RDMs to [0,1] for interpretability
- Compare observed RDMs with four theoretical models (null, ingroup/outgroup similar, ingroup advantage, categorical)
- Generate and save heatmaps of the RDMs as PNG files

**Use this when:** You have behavioural memory data with three conditions and want to understand their representational geometry.

### Script 2: `02_simulated_fmri_data.py` Generate Synthetic fMRI Data

- Create realistic ROI-based fMRI activation patterns for 50 participants
- Control data quality with presets: good (high signal, low noise), medium (balanced), bad (low signal, high noise), or custom parameters
- Define the true correlation structure (ingroup and outgroup similar, baseline different)
- Add voxel-level noise to simulate real measurement error
- Save the generated data as a CSV file ready for RSA

**Use this when:** You need synthetic fMRI data to test RSA pipelines or to simulate power analyses.

### Script 3: `03_fmri_RSA.py` fMRI RSA Analysis

- Load synthetic (or real) fMRI data from a CSV file
- Compute RDMs for each participant individually
- Average across participants to obtain a group-level RDM
- Correlate the group fMRI RDM with behavioural RDMs (item and source memory)
- Perform bootstrap resampling to estimate 95% confidence intervals
- Run leave-one-out cross-validation to assess individual-level reliability
- Generate and save:
  - Heatmap of the group-level fMRI RDM
  - Bootstrap distribution plots for both memory domains

**Use this when:** You have fMRI data (real or simulated) and want to compare neural representational geometry with behavioural patterns.

## For Other Users

Clone the repo, install dependencies with `pip install -r requirements.txt`, place your behavioural CSV in `data/raw/`, then run the three scripts in `src/` in numerical order. All outputs go to the `outputs/` folder.