#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fMRI Representational Similarity Analysis (RSA)

This script reads fMRI data (mean activation per condition and participant)
from a CSV file, computes Representational Dissimilarity Matrices (RDMs) for
each participant, averages them across participants, and compares the group-
level RDM with fixed behavioural RDMs (source memory and item recognition).

It also performs bootstrap resampling to estimate confidence intervals and
optionally a leave-one-out cross-validation. All figures are saved in the
'outputs/' folder.

Usage: python src/03_fmri_RSA.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist, squareform
import os

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
CSV_FILE = "data/processed/synthetic_fmri_data.csv"   # Path to fMRI data
OUTPUT_DIR = "outputs"                                 # Where to save figures
N_BOOTSTRAP = 1000                                     # Number of bootstrap samples

# Create output folder if needed
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# 2. BEHAVIOURAL RDMs (fixed, from your study)
# ============================================================================
# These RDMs come from the behavioural analysis (01_behavioural_RSA.py)
# They represent the representational geometry of memory for political statements

# Item recognition RDM: remembering the statement itself
behav_item_rdm = np.array([[0.000, 1.000, 0.889],
                           [1.000, 0.000, 0.635],
                           [0.889, 0.635, 0.000]])

# Source memory RDM: remembering which speaker said the statement
behav_source_rdm = np.array([[0.000, 1.000, 0.949],
                             [1.000, 0.000, 0.339],
                             [0.949, 0.339, 0.000]])

conditions = ['baseline', 'ingroup', 'outgroup']

# ============================================================================
# 3. HELPER FUNCTIONS
# ============================================================================
def rdm_from_activations(activations):
    """
    Compute a 3x3 RDM from a 3-element activation vector.
    
    Uses Euclidean distance between conditions and normalises to [0,1].
    This creates a dissimilarity matrix where 0 = identical patterns,
    1 = maximally dissimilar patterns.
    
    Parameters
    ----------
    activations : numpy.ndarray
        Array of length 3 containing mean activation values for
        baseline, ingroup, and outgroup conditions.
    
    Returns
    -------
    rdm : numpy.ndarray
        3x3 dissimilarity matrix normalised to [0,1] with zero diagonal.
    """
    rdm = squareform(pdist(activations.reshape(-1, 1), metric='euclidean'))
    if rdm.max() > rdm.min():
        rdm = (rdm - rdm.min()) / (rdm.max() - rdm.min())
    else:
        rdm = np.zeros_like(rdm)
    np.fill_diagonal(rdm, 0)
    return rdm

def rdm_to_vector(rdm):
    """
    Extract the three unique upper-triangular values from a 3x3 RDM.
    
    Order of extracted values: (baseline-ingroup, baseline-outgroup, ingroup-outgroup)
    This vectorisation is necessary for correlation analysis.
    
    Parameters
    ----------
    rdm : numpy.ndarray
        3x3 dissimilarity matrix.
    
    Returns
    -------
    numpy.ndarray
        Vector of length 3 containing the three unique dissimilarity values.
    """
    return rdm[np.triu_indices_from(rdm, k=1)]

def load_and_analyze_fmri_csv(csv_file, output_folder=OUTPUT_DIR,
                              n_bootstrap=N_BOOTSTRAP):
    """
    Main analysis pipeline for fMRI RSA.
    
    This function performs the complete RSA analysis on synthetic or real fMRI data:
    1. Load CSV, create pivot table (participants x conditions)
    2. Compute RDM for each participant
    3. Average across participants -> group RDM
    4. Correlate group RDM with behavioural RDMs (item and source)
    5. Bootstrap to obtain 95% confidence intervals
    6. Leave-one-out cross-validation to assess individual-level reliability
    7. Save heatmap and bootstrap distribution plots
    
    Parameters
    ----------
    csv_file : str
        Path to the CSV file containing fMRI data.
    output_folder : str
        Directory where figures will be saved.
    n_bootstrap : int
        Number of bootstrap resampling iterations.
    
    Returns
    -------
    tuple
        (avg_rdm, corr_source, corr_item) - group RDM and correlations
    """
    # Load data
    df = pd.read_csv(csv_file)
    print(f"Loaded file: '{csv_file}'")
    print(f"Number of participants: {df['participant_id'].nunique()}")
    print(f"Conditions found: {df['condition'].unique()}")

    # Pivot table: participants x conditions
    # This reorganises the long-format data into a matrix
    pivot = df.pivot_table(index='participant_id', columns='condition',
                           values='mean_activation')
    # Ensure correct order of conditions
    pivot = pivot[conditions]

    # Compute RDM for each participant individually
    participant_rdms = []
    for _, row in pivot.iterrows():
        rdm = rdm_from_activations(row.values)
        participant_rdms.append(rdm)

    # Group average RDM (mean across participants)
    # This represents the typical representational geometry in the sample
    avg_rdm = np.mean(participant_rdms, axis=0)
    vec_avg = rdm_to_vector(avg_rdm)

    # Behavioural vectors for comparison
    vec_behav_source = rdm_to_vector(behav_source_rdm)
    vec_behav_item = rdm_to_vector(behav_item_rdm)

    # Correlations between fMRI group RDM and behavioural RDMs
    # High correlation indicates that the neural representational geometry
    # mirrors the behavioural memory structure
    corr_source = pearsonr(vec_avg, vec_behav_source)[0]
    corr_item = pearsonr(vec_avg, vec_behav_item)[0]

    print("\nGroup average fMRI RDM:")
    print(pd.DataFrame(avg_rdm, index=conditions, columns=conditions).round(3))
    print(f"\nCorrelation with behavioural source memory: r = {corr_source:.3f}")
    print(f"Correlation with behavioural item recognition: r = {corr_item:.3f}")

    # Bootstrap resampling to estimate confidence intervals
    # This shows how stable the correlations are given the sample size
    boot_source = []
    boot_item = []
    n_part = len(participant_rdms)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n_part, size=n_part, replace=True)
        boot_rdms = [participant_rdms[i] for i in idx]
        boot_avg = np.mean(boot_rdms, axis=0)
        boot_vec = rdm_to_vector(boot_avg)
        r_src = pearsonr(boot_vec, vec_behav_source)[0]
        r_itm = pearsonr(boot_vec, vec_behav_item)[0]
        boot_source.append(r_src if not np.isnan(r_src) else 0.0)
        boot_item.append(r_itm if not np.isnan(r_itm) else 0.0)

    ci_source = np.percentile(boot_source, [2.5, 97.5])
    ci_item = np.percentile(boot_item, [2.5, 97.5])
    print(f"\nBootstrap 95% CI (fMRI vs. source): [{ci_source[0]:.3f}, {ci_source[1]:.3f}]")
    print(f"Bootstrap 95% CI (fMRI vs. item): [{ci_item[0]:.3f}, {ci_item[1]:.3f}]")

    # Leave-one-out cross-validation (single subject vs. behaviour)
    # This tests whether individual participants' representational geometries
    # resemble the behavioural pattern
    cv_source = []
    cv_item = []
    for leave_out in range(n_part):
        train_rdms = [participant_rdms[i] for i in range(n_part) if i != leave_out]
        train_avg = np.mean(train_rdms, axis=0)
        test_rdm = participant_rdms[leave_out]
        test_vec = rdm_to_vector(test_rdm)
        cv_source.append(pearsonr(test_vec, vec_behav_source)[0])
        cv_item.append(pearsonr(test_vec, vec_behav_item)[0])
    
    print(f"\nLeave-one-out CV (single subject vs. source): mean r = {np.nanmean(cv_source):.3f}, SD = {np.nanstd(cv_source):.3f}")
    print(f"Leave-one-out CV (single subject vs. item): mean r = {np.nanmean(cv_item):.3f}, SD = {np.nanstd(cv_item):.3f}")

    # ----- Heatmap of group RDM -----
    plt.figure(figsize=(6,5))
    sns.heatmap(avg_rdm, xticklabels=conditions, yticklabels=conditions,
                square=True, annot=True, fmt='.3f', cmap='viridis',
                vmin=0, vmax=1, cbar_kws={'label': 'Dissimilarity'})
    plt.title('Group-average fMRI RDM')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'group_fmri_rdm.png'), dpi=150)
    plt.close()

    # ----- Bootstrap distribution plots -----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
    ax1.hist(boot_source, bins=30, color='skyblue', edgecolor='black')
    ax1.axvline(corr_source, color='red', linestyle='--', label='Observed')
    ax1.set_xlabel('Correlation (fMRI vs. source)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Bootstrap distribution (source)')
    ax1.legend()

    ax2.hist(boot_item, bins=30, color='lightgreen', edgecolor='black')
    ax2.axvline(corr_item, color='red', linestyle='--', label='Observed')
    ax2.set_xlabel('Correlation (fMRI vs. item)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Bootstrap distribution (item)')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'bootstrap_correlations.png'), dpi=150)
    plt.close()

    print(f"\nAll outputs saved in '{output_folder}/'")
    return avg_rdm, corr_source, corr_item

# ============================================================================
# 4. MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    avg_rdm, corr_source, corr_item = load_and_analyze_fmri_csv(CSV_FILE)
    
    # ========================================================================
    # 5. INTERPRETATION OF POSSIBLE OUTCOMES
    # ========================================================================
    print("\n" + "="*70)
    print("INTERPRETATION OF fMRI RSA RESULTS")
    print("="*70)
    
    # Extract the three unique dissimilarity values from the fMRI group RDM
    d_bi = avg_rdm[0, 1]  # baseline vs ingroup
    d_bo = avg_rdm[0, 2]  # baseline vs outgroup
    d_io = avg_rdm[1, 2]  # ingroup vs outgroup
    
    print("\n" + "-"*70)
    print("1. Representational Geometry in fMRI Data")
    print("-"*70)
    print(f"\n   Dissimilarity baseline vs ingroup:  {d_bi:.3f}")
    print(f"   Dissimilarity baseline vs outgroup: {d_bo:.3f}")
    print(f"   Dissimilarity ingroup vs outgroup:  {d_io:.3f}")
    
    # Determine which pair is most similar (smallest dissimilarity)
    min_val = min(d_bi, d_bo, d_io)
    if min_val == d_io:
        min_pair = "ingroup vs outgroup"
        pattern = "POLITICAL_CLUSTER"
    elif min_val == d_bi:
        min_pair = "baseline vs ingroup"
        pattern = "INGROUP_ADVANTAGE"
    else:
        min_pair = "baseline vs outgroup"
        pattern = "OUTGROUP_EFFECT"
    
    print(f"\n   Most similar pair: {min_pair} (dissimilarity = {min_val:.3f})")
    
    print("\n" + "-"*70)
    print("2. Correlation with Behavioural Memory Patterns")
    print("-"*70)
    print(f"\n   Correlation with source memory RDM: r = {corr_source:.3f}")
    print(f"   Correlation with item recognition RDM: r = {corr_item:.3f}")
    
    print("\n" + "-"*70)
    print("3. Possible Outcome Patterns and Their Interpretations")
    print("-"*70)
    
    print("\n   --- Pattern A: POLITICAL CLUSTER (ingroup and outgroup similar) ---")
    print("   Observed when: d_io is the smallest value (ingroup vs outgroup most similar)")
    print("   Interpretation:")
    print("   - The brain represents both political groups in a similar manner")
    print("   - This mirrors the behavioural finding that ingroup and outgroup")
    print("     are encoded similarly in memory")
    print("   - Suggests that political categorization (political vs. non-political)")
    print("     is a primary organising principle in neural representational space")
    print("   - High correlation with behavioural RDMs (r > 0.7) indicates that")
    print("     neural patterns track the behavioural memory structure")
    
    print("\n   --- Pattern B: INGROUP ADVANTAGE (ingroup distinct from outgroup) ---")
    print("   Observed when: d_bi is the smallest value")
    print("   Interpretation:")
    print("   - The brain represents ingroup as most similar to itself")
    print("   - This would suggest a neural bias favouring one's own political group")
    print("   - Would contrast with behavioural findings (which showed no ingroup advantage)")
    print("   - Could indicate that neural representations are more sensitive to")
    print("     social identity than behavioural measures")
    
    print("\n   --- Pattern C: CATEGORICAL DISTINCTION (all conditions distinct) ---")
    print("   Observed when: All d_bi, d_bo, d_io are similarly large (> 0.6)")
    print("   Interpretation:")
    print("   - The brain treats all three conditions as equally distinct")
    print("   - Would suggest that political and non-political information are")
    print("     not organised along a single dimension")
    print("   - Low correlation with behavioural RDMs (r < 0.3) would indicate")
    print("     that neural representational geometry does not mirror behaviour")
    
    print("\n   --- Pattern D: BASELINE DISTINCTION (baseline different from both) ---")
    print("   Observed when: d_bi and d_bo are large, d_io is small")
    print("   Interpretation:")
    print("   - The brain distinguishes political from non-political information")
    print("   - Ingroup and outgroup are clustered together in representational space")
    print("   - This is the pattern predicted by the behavioural results")
    print("   - High correlation with source memory (r > 0.8) suggests that")
    print("     speaker identity is encoded in a 'political speaker' category")
    
    print("\n" + "-"*70)
    print("4. Bootstrap Confidence Intervals")
    print("-"*70)
    print("\n   Narrow CI (e.g., width < 0.2): High confidence in the correlation estimate")
    print("   Wide CI (e.g., width > 0.5): High variability, small sample size, or high noise")
    print("   CI crossing zero: Correlation not reliably different from zero")
    
    print("\n" + "-"*70)
    print("5. Leave-One-Out Cross-Validation (LOOCV)")
    print("-"*70)
    print("\n   High mean r (> 0.5): Individual participants reliably show the pattern")
    print("   Low mean r (< 0.2): Pattern is only visible at the group level")
    print("   High SD (> 0.4): Large individual differences in representational geometry")
    
    print("\n" + "-"*70)
    print("6. Summary Interpretation for the Current Results")
    print("-"*70)
    
    # Provide specific interpretation based on the actual values
    if d_io < d_bi and d_io < d_bo:
        print("\n   The fMRI data show the POLITICAL CLUSTER pattern:")
        print("   Ingroup and outgroup are most similar in neural representational space.")
        print(f"   This aligns with the behavioural finding (ingroup-outgroup distance = {d_io:.3f}).")
    elif d_bi < d_io and d_bi < d_bo:
        print("\n   The fMRI data show an INGROUP ADVANTAGE pattern:")
        print("   Baseline and ingroup are most similar, suggesting a neural bias.")
    else:
        print("\n   The fMRI data show a CATEGORICAL or BASELINE DISTINCTION pattern.")
    
    if corr_source > 0.7:
        print(f"\n   High correlation with source memory (r = {corr_source:.3f}) indicates")
        print("   that neural patterns strongly track the memory for speaker identity.")
    elif corr_source > 0.3:
        print(f"\n   Moderate correlation with source memory (r = {corr_source:.3f}) suggests")
        print("   partial alignment between neural and behavioural representations.")
    else:
        print(f"\n   Low correlation with source memory (r = {corr_source:.3f}) suggests")
        print("   that neural representational geometry does not mirror behaviour.")
    
    print("\n" + "="*70)
    print("INTERPRETATION COMPLETE")
    print("="*70)