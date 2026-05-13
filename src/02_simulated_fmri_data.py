#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate synthetic fMRI data (ROI-based mean activations) for three conditions:
baseline, ingroup, outgroup. The data can be used for Representational Similarity Analysis (RSA).

The script supports different data quality presets (good, medium, bad, custom) and
saves the output as a CSV file in 'data/processed/synthetic_fmri_data.csv'.

Usage:
    python src/02_simulated_fmri_data.py          # interactive mode
    Or modify the parameters in __main__ to run non-interactively.
"""

import numpy as np
import pandas as pd
import os

# ============================================================================
# CORE GENERATION FUNCTION
# ============================================================================
def generate_synthetic_fmri_csv(filename='data/processed/synthetic_fmri_data.csv',
                                n_participants=50,
                                n_voxels=50,
                                signal_strength=0.5,
                                noise_sd=0.6,
                                random_seed=42):
    """
    Generate synthetic fMRI data and save as CSV.

    This function simulates fMRI activation patterns for a group of participants
    across three experimental conditions. The data follows either a true
    correlation structure (ingroup-outgroup similar, baseline different) or a
    random structure, depending on the signal_strength parameter.

    Parameters
    ----------
    filename : str
        Output CSV file path (relative to current working directory).
    n_participants : int
        Number of simulated participants.
    n_voxels : int
        Number of voxels in the ROI (higher values increase stability).
    signal_strength : float (0 to 1)
        Proportion of participants whose activation pattern follows the true structure
        (ingroup-outgroup similar, baseline different). The rest get random correlations.
    noise_sd : float
        Standard deviation of Gaussian noise added to each voxel (higher = more noise).
    random_seed : int or None
        Seed for reproducibility. Use None for truly random data.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with columns: participant_id, condition, mean_activation
    """
    # Create output directory if needed
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    if random_seed is not None:
        np.random.seed(random_seed)

    condition_names = ['baseline', 'ingroup', 'outgroup']
    n_conditions = len(condition_names)

    # True correlation matrix representing the expected pattern:
    # - Ingroup and outgroup are positively correlated (0.6)
    # - Baseline is weakly correlated with both political groups (0.2)
    true_corr = np.array([[1.0, 0.2, 0.2],
                          [0.2, 1.0, 0.6],
                          [0.2, 0.6, 1.0]])

    all_rows = []

    for subj_id in range(1, n_participants + 1):
        # Decide correlation structure for this participant
        # Higher signal_strength = more participants show the true pattern
        if np.random.rand() < signal_strength:
            corr = true_corr
        else:
            # Generate a random positive semi-definite correlation matrix
            # This creates a random but valid correlation structure
            random_corr = np.random.randn(n_conditions, n_conditions)
            random_corr = random_corr @ random_corr.T
            d = np.sqrt(np.diag(random_corr))
            corr = random_corr / np.outer(d, d)

        # Cholesky decomposition to create correlated patterns
        # This transforms uncorrelated random variables into correlated ones
        L = np.linalg.cholesky(corr)
        z = np.random.normal(0, 1, size=(n_conditions, n_voxels))
        patterns = L @ z
        
        # Add independent voxel noise to simulate measurement error
        noise = np.random.normal(0, noise_sd, size=patterns.shape)
        patterns = patterns + noise

        # Average across voxels -> one activation value per condition
        # This simulates extracting mean activation from a functional ROI
        mean_activations = np.mean(patterns, axis=1)

        for cond, act in zip(condition_names, mean_activations):
            all_rows.append({
                'participant_id': subj_id,
                'condition': cond,
                'mean_activation': act
            })

    df = pd.DataFrame(all_rows)
    df.to_csv(filename, index=False)
    print(f"Synthetic fMRI data saved to '{filename}'")
    print(f"  Participants: {n_participants}")
    print(f"  Voxels: {n_voxels}")
    print(f"  Signal strength (true pattern proportion): {signal_strength}")
    print(f"  Noise SD: {noise_sd}")
    return df


# ============================================================================
# INTERACTIVE CONFIGURATION (only used when script is run directly)
# ============================================================================
def ask_yes_no(question, default='y'):
    """
    Ask a yes/no question and return a boolean.
    
    Parameters
    ----------
    question : str
        The question to ask the user.
    default : str, either 'y' or 'n'
        Default answer if user presses Enter.
    
    Returns
    -------
    bool
        True for yes, False for no.
    """
    valid = {'y': True, 'n': False}
    prompt = f"{question} (Y/n): " if default == 'y' else f"{question} (y/N): "
    while True:
        answer = input(prompt).strip().lower()
        if answer == '':
            return valid[default]
        if answer in valid:
            return valid[answer]
        print("Please answer 'y' or 'n'.")

def ask_int(question, default=None):
    """
    Ask for an integer input with validation.
    
    Parameters
    ----------
    question : str
        The question to ask the user.
    default : int or None
        Default value if user presses Enter.
    
    Returns
    -------
    int
        The user's answer as an integer.
    """
    while True:
        answer = input(question).strip()
        if answer == '' and default is not None:
            return default
        try:
            return int(answer)
        except ValueError:
            print("Please enter a valid integer.")

def ask_float(question, default=None):
    """
    Ask for a float input with validation.
    
    Parameters
    ----------
    question : str
        The question to ask the user.
    default : float or None
        Default value if user presses Enter.
    
    Returns
    -------
    float
        The user's answer as a float.
    """
    while True:
        answer = input(question).strip()
        if answer == '' and default is not None:
            return default
        try:
            return float(answer)
        except ValueError:
            print("Please enter a valid number.")

def configure_and_generate():
    """
    Run interactive configuration and generate synthetic fMRI data.
    
    This function presents the user with data quality presets (good, medium, bad, custom)
    and allows adjustment of all generation parameters before creating the CSV file.
    
    Returns
    -------
    df : pandas.DataFrame or None
        The generated DataFrame, or None if generation was cancelled.
    """
    print("="*60)
    print("SYNTHETIC FMRI DATA GENERATOR")
    print("="*60)
    print("This script creates a CSV file with simulated fMRI activation values.")
    print("You can choose the data quality: good, medium, or bad.\n")
    print("Data quality presets:")
    print("  good   - low noise, high signal (80% true pattern, noise SD = 0.3)")
    print("  medium - moderate noise, moderate signal (50% true pattern, noise SD = 0.6)")
    print("  bad    - high noise, low signal (20% true pattern, noise SD = 1.2)")
    print("  custom - set parameters manually\n")

    preset = input("Choose preset (good/medium/bad/custom): ").strip().lower()

    if preset == 'good':
        signal_strength = 0.8
        noise_sd = 0.3
        print("\nSelected GOOD: high signal (80%), low noise (0.3)")
    elif preset == 'medium':
        signal_strength = 0.5
        noise_sd = 0.6
        print("\nSelected MEDIUM: moderate signal (50%), moderate noise (0.6)")
    elif preset == 'bad':
        signal_strength = 0.2
        noise_sd = 1.2
        print("\nSelected BAD: low signal (20%), high noise (1.2)")
    elif preset == 'custom':
        print("\nCustom settings:")
        signal_strength = ask_float("Proportion with true pattern (0-1) [0.5]: ", default=0.5)
        noise_sd = ask_float("Voxel noise standard deviation [0.8]: ", default=0.8)
    else:
        print("Invalid preset. Using medium as default.")
        signal_strength = 0.5
        noise_sd = 0.6

    print("\nNow you can change other parameters (press Enter to keep default):")
    n_participants = ask_int("Number of participants [50]: ", default=50)
    n_voxels = ask_int("Number of voxels in ROI [50]: ", default=50)
    random_seed = ask_int("Random seed (0 for random) [42]: ", default=42)
    if random_seed == 0:
        random_seed = None

    filename = input("Output CSV filename [data/processed/synthetic_fmri_data.csv]: ").strip()
    if filename == '':
        filename = 'data/processed/synthetic_fmri_data.csv'

    print("\n" + "-"*40)
    print("Summary of settings:")
    print(f"  Participants: {n_participants}")
    print(f"  Voxels: {n_voxels}")
    print(f"  Signal strength: {signal_strength}")
    print(f"  Noise SD: {noise_sd}")
    print(f"  Random seed: {random_seed if random_seed is not None else 'random'}")
    print(f"  Output file: {filename}")
    print("-"*40)

    proceed = ask_yes_no("\nGenerate data with these settings?", default='y')
    if not proceed:
        print("Generation cancelled.")
        return None

    df = generate_synthetic_fmri_csv(
        filename=filename,
        n_participants=n_participants,
        n_voxels=n_voxels,
        signal_strength=signal_strength,
        noise_sd=noise_sd,
        random_seed=random_seed
    )
    print("\nDone! You can now run the analysis script (e.g., 03_fmri_RSA.py).")
    return df


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    # For interactive use, call configure_and_generate()
    configure_and_generate()

    # If you prefer non-interactive default generation, uncomment the next line
    # and comment the line above.
    # generate_synthetic_fmri_csv()