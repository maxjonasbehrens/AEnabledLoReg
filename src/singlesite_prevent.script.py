
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Seed Composite Autoencoder Analysis Script

This script performs a multi-seed analysis to evaluate the stability and
interpretability of patient subgroups identified from clinical data. It compares
a novel Composite Autoencoder (CompositeAE) against a standard Vanilla Autoencoder
(VanillaAE) and Principal Component Analysis (PCA).

The core objectives are:
1.  Train dimension reduction models (AEs, PCA) on patient data across multiple random seeds.
2.  Identify patient subgroups based on deviations of local model coefficients from global trends.
3.  Analyze the stability of these subgroups and their defining characteristics across seeds.
4.  Generate plots and summary statistics for interpretation and publication.

The script is structured as follows:
- Imports and setup of global parameters.
- Data loading and preprocessing functions.
- Core analysis functions for model training, stability analysis, and plotting.
- A main execution block that orchestrates the multi-seed runs.
- Post-analysis cells for detailed stability reporting and result exploration.

To run this script, ensure all required packages are installed and the dataset
'prevent_direct_train_data.csv' is in the 'data/processed/' directory.

Version: 0.13.4
"""

__generated_with = "0.13.4"

# %%
# =============================================================================
# 0. IMPORTS AND SETUP
# =============================================================================

# Standard library and data analysis imports
import os
import sys
import itertools
from collections import Counter, defaultdict
import copy

# Core scientific computing and machine learning imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# Removed TensorBoard import for reproducibility - not needed for core functionality
from tqdm import tqdm

# Scikit-learn and statsmodels for modeling and evaluation
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import ttest_ind, rankdata
import statsmodels.api as sm

# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Custom helper modules (ensure they are in the same directory or Python path)
# These modules contain the core model architecture and training logic.
from methods.single_site_ae_loreg_module import SingleSiteDataset, LikelihoodLoss, AutoencoderWithRegression, create_data_loader, initialize_training, train_model
from utilities import weights
from utilities import loregs

# %%
# =============================================================================
# 1. GLOBAL PARAMETERS AND CONFIGURATION
# =============================================================================

# --- File Paths ---
DATA_PATH = 'data/processed/prevent_direct_train_data.csv'
OUTPUT_DIR = 'results/figures/'

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- Reproducibility ---
BASE_SEED = 42  # Initial seed to generate other seeds from
N_SEEDS = 15    # Number of different random seeds to run for stability analysis

# --- Data splitting constants ---
TEST_SIZE_RATIO = 0.2  # Proportion of data to use for testing
MIN_PATIENTS_FOR_STD = 2  # Minimum number of patients required to calculate standard deviation
NUMERICAL_TOLERANCE = 1e-6  # Tolerance for numerical comparisons

# --- Model Training Hyperparameters ---
# These parameters control the architecture and training process of the autoencoders.
TRAIN_HYPERPARAMETERS = {
    # Architecture
    'latent_size': 4,       # Number of dimensions in the bottleneck layer
    'hidden_factor_1': 3,   # Exponent for first hidden layer size (latent_size^factor)
    'hidden_factor_2': 2,   # Exponent for second hidden layer size (latent_size^factor)
    # Training Loop
    'num_epochs': 300,
    'learning_rate': 1e-4,
    'weight_decay': 1e-3,   # L2 regularization
    'batch_size_train': 1,  # Using a batch size of 1 for stochastic updates in local regression loss
    'switch_shuffle_epoch': 1, # Epoch to switch from unshuffled to shuffled data loader
    'log_interval': 10,     # How often to log training progress
    # Loss Function Weights (for CompositeAE)
    'alpha_recon_loss': 1.0,            # Weight for reconstruction loss
    'theta_null_loss_composite': 0.06,  # Weight for the prediction likelihood loss
    'gamma_global_loss_composite': 0.3, # Weight for the global orthogonality loss
    # Local Regression Parameters (for loss calculation)
    'sigma_weights': 1.0,         # Sigma for Gaussian kernel in local regression weighting
    'k_nearest_weights': 0.1,     # K-nearest neighbors fraction for local regression weighting
    'kernel_type_weights': 'gaussian', # Kernel type for weighting
    'num_batches_weights': 1,
    # Data Preprocessing
    'outcome_var': 'Y',                 # Name of the outcome variable
}

# --- Post-Hoc Analysis Hyperparameters ---
# These parameters control the analysis and plotting after model training.
ANALYSIS_HYPERPARAMETERS = {
    # Local Regression for Beta Interpretation
    'sigma_local_regression': TRAIN_HYPERPARAMETERS['sigma_weights'],
    'k_nearest_local_regression': TRAIN_HYPERPARAMETERS['k_nearest_weights'],
    'kernel_local_regression': TRAIN_HYPERPARAMETERS['kernel_type_weights'],
    # Feature Interpretation
    'ttest_mi_top_n': 10,  # Number of top variables to show from t-tests and mutual information
    'top_n_vars_forest_plot': 5, # Number of variables for interaction forest plots
    # Subgroup Definition
    'beta_coefficient_plot_type': "Difference to Global Coefficients", # Plot local betas or their diff from global
    'beta_diff_threshold_subgroup': 0.01, # Threshold for defining subgroups based on beta differences
    # Plotting
    'z_profile_range': [-1, 1],  # Range for Z-score profile plots
    'ci_alpha': 0.1 #0.05,            # Alpha for confidence intervals (1 - confidence level)
}

# %%
# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def set_seed(seed_value=42):
    """
    Set seed for reproducibility across NumPy, PyTorch, and CUDA.

    Args:
        seed_value (int): The seed value to use. Must be a non-negative integer.
        
    Raises:
        ValueError: If seed_value is not a valid integer
    """
    if not isinstance(seed_value, int) or seed_value < 0:
        raise ValueError(f"Seed value must be a non-negative integer, got {seed_value}")
        
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed_value} for reproducibility.")

# %%
from collections import defaultdict # Make sure this is imported

def calculate_benchmark_inter_patient_variability(
    all_seeds_results_by_method,
    manual_ld_alignment_map,
    component_col_names_by_method,
    unique_patient_ids
):
    """
    Calculates:
    1. Per-seed inter-patient standard deviations (scale factors) for signed and absolute deviations.
    2. Average inter-patient standard deviations (benchmarks) for signed and absolute deviations,
       averaged across all seeds for each conceptual latent dimension.

    Args:
        all_seeds_results_by_method (dict): Results from the multi-seed run.
        manual_ld_alignment_map (dict): Defines how original LD names map to conceptual LD names,
                                        optionally including sign correction.
        component_col_names_by_method (dict): Maps method_name to ordered list of component columns for OLS.
        unique_patient_ids (list or pd.Index): List of all unique patient IDs.

    Returns:
        tuple: (final_averaged_benchmarks, per_seed_scale_factors)
            final_averaged_benchmarks (dict):
                Keys: (method_name, conceptual_ld_name)
                Values: {'benchmark_std_signed_deviation': avg_std,
                         'benchmark_std_abs_deviation': avg_abs_std,
                         'num_seeds_in_benchmark_avg': count}
            per_seed_scale_factors (dict):
                Keys: (method_name, conceptual_ld_name, seed_val)
                Values: {'scale_factor_signed': std_of_signed_devs_this_seed,
                         'scale_factor_abs': std_of_abs_devs_this_seed}
    """
    print("\n--- Calculating Benchmark: Average & Per-Seed Inter-Patient Deviation Variability ---")

    # To store lists of per-seed SDs before averaging for the final benchmark
    # Key: (method_name, conceptual_ld_name)
    # Value: {'all_per_seed_signed_dev_stds': [], 'all_per_seed_abs_dev_stds': []}
    aggregated_per_seed_stds_for_averaging = defaultdict(lambda: {'all_per_seed_signed_dev_stds': [], 'all_per_seed_abs_dev_stds': []})

    # To store the scale factor for each specific seed/cld
    # Key: (method_name, conceptual_ld_name, seed_val)
    # Value: {'scale_factor_signed': float, 'scale_factor_abs': float}
    per_seed_scale_factors = {}

    for method_name, seed_results_list in all_seeds_results_by_method.items():
        if method_name not in manual_ld_alignment_map or method_name not in component_col_names_by_method:
            print(f"  Skipping method '{method_name}' for benchmark: Missing in alignment map or component_col_names.")
            continue

        print(f"  Processing Method: '{method_name}' for benchmark calculation")
        method_alignment_map = manual_ld_alignment_map[method_name]
        ordered_component_cols_for_ols = component_col_names_by_method[method_name]

        all_conceptual_lds_for_method = set()
        for seed_val_map in method_alignment_map.values():
            for conceptual_name_or_detail in seed_val_map.values():
                if isinstance(conceptual_name_or_detail, dict):
                    all_conceptual_lds_for_method.add(conceptual_name_or_detail['conceptual_name'])
                else:
                    all_conceptual_lds_for_method.add(conceptual_name_or_detail)

        if not all_conceptual_lds_for_method:
            print(f"    No conceptual LDs found in the alignment map for method '{method_name}'.")
            continue

        for conceptual_ld_name in sorted(list(all_conceptual_lds_for_method)):
            for seed_result in seed_results_list:
                seed_val = seed_result.get('seed')
                if seed_val is None or seed_val not in method_alignment_map:
                    continue

                seed_specific_ld_alignment = method_alignment_map[seed_val]
                original_ld_name_in_seed = None
                sign_multiplier = 1

                for orig_ld, conceptual_info in seed_specific_ld_alignment.items():
                    current_conceptual_name_from_map = None
                    current_sign_from_map = 1
                    if isinstance(conceptual_info, dict):
                        current_conceptual_name_from_map = conceptual_info.get('conceptual_name')
                        current_sign_from_map = conceptual_info.get('sign', 1)
                    elif isinstance(conceptual_info, str):
                        current_conceptual_name_from_map = conceptual_info

                    if current_conceptual_name_from_map == conceptual_ld_name:
                        original_ld_name_in_seed = orig_ld
                        sign_multiplier = current_sign_from_map
                        break

                if not original_ld_name_in_seed:
                    continue

                analysis_data = seed_result.get('train_analysis', {})
                local_betas_df_seed = analysis_data.get('local_betas_df')
                global_coeffs_df_seed = analysis_data.get('global_ols_coeffs_df')

                if local_betas_df_seed is None or global_coeffs_df_seed is None:
                    continue

                all_patient_deviations_this_seed_signed = []
                all_patient_deviations_this_seed_abs = []

                for patient_id in unique_patient_ids:
                    if patient_id not in local_betas_df_seed.index or \
                       original_ld_name_in_seed not in local_betas_df_seed.columns:
                        continue

                    local_coeff_raw = local_betas_df_seed.loc[patient_id, original_ld_name_in_seed]
                    if pd.isna(local_coeff_raw):
                        continue
                    local_coeff_corrected = local_coeff_raw * sign_multiplier

                    global_coeff_raw = np.nan
                    try:
                        idx_in_ols_components = ordered_component_cols_for_ols.index(original_ld_name_in_seed)
                        potential_ols_coeff_name = None
                        if global_coeffs_df_seed.index[0].lower() == 'const' and (idx_in_ols_components + 1) < len(global_coeffs_df_seed.index):
                            potential_ols_coeff_name = global_coeffs_df_seed.index[idx_in_ols_components + 1]
                        elif original_ld_name_in_seed in global_coeffs_df_seed.index:
                            potential_ols_coeff_name = original_ld_name_in_seed

                        if potential_ols_coeff_name and potential_ols_coeff_name in global_coeffs_df_seed.index:
                            global_coeff_raw = global_coeffs_df_seed.loc[potential_ols_coeff_name, 'Coefficient']
                    except (ValueError, IndexError):
                        pass

                    if pd.isna(global_coeff_raw):
                        continue
                    global_coeff_corrected = global_coeff_raw * sign_multiplier

                    signed_deviation = local_coeff_corrected - global_coeff_corrected
                    all_patient_deviations_this_seed_signed.append(signed_deviation)
                    all_patient_deviations_this_seed_abs.append(abs(signed_deviation))

                # Calculate SD across patients for this seed and conceptual_ld
                scale_factor_signed_this_seed = np.nan
                scale_factor_abs_this_seed = np.nan

                if len(all_patient_deviations_this_seed_signed) > MIN_PATIENTS_FOR_STD:
                    scale_factor_signed_this_seed = np.std(all_patient_deviations_this_seed_signed)
                    aggregated_per_seed_stds_for_averaging[(method_name, conceptual_ld_name)]['all_per_seed_signed_dev_stds'].append(scale_factor_signed_this_seed)

                if len(all_patient_deviations_this_seed_abs) > MIN_PATIENTS_FOR_STD:
                    scale_factor_abs_this_seed = np.std(all_patient_deviations_this_seed_abs)
                    aggregated_per_seed_stds_for_averaging[(method_name, conceptual_ld_name)]['all_per_seed_abs_dev_stds'].append(scale_factor_abs_this_seed)

                # Store the per-seed scale factor regardless of whether it was NaN (if <2 patients)
                per_seed_scale_factors[(method_name, conceptual_ld_name, seed_val)] = {
                    'scale_factor_signed': scale_factor_signed_this_seed,
                    'scale_factor_abs': scale_factor_abs_this_seed
                }

    final_averaged_benchmarks = {}
    for key, data in aggregated_per_seed_stds_for_averaging.items():
        method_name_key, conceptual_ld_name_key = key
        avg_std_signed = np.nanmean(data['all_per_seed_signed_dev_stds']) if data['all_per_seed_signed_dev_stds'] else np.nan
        avg_std_abs = np.nanmean(data['all_per_seed_abs_dev_stds']) if data['all_per_seed_abs_dev_stds'] else np.nan

        num_seeds_avg_signed = len([s for s in data['all_per_seed_signed_dev_stds'] if pd.notna(s)])
        num_seeds_avg_abs = len([s for s in data['all_per_seed_abs_dev_stds'] if pd.notna(s)])

        final_averaged_benchmarks[key] = {
            'benchmark_std_signed_deviation': avg_std_signed,
            'benchmark_std_abs_deviation': avg_std_abs,
            'num_seeds_in_benchmark_avg_signed': num_seeds_avg_signed,
            'num_seeds_in_benchmark_avg_abs': num_seeds_avg_abs
        }
        print(f"\n    Benchmark for Method '{method_name_key}', Conceptual LD '{conceptual_ld_name_key}':")
        print(f"      Avg Inter-Patient STD (Signed Dev): {avg_std_signed:.4f} (from {num_seeds_avg_signed} seeds with valid inter-patient SD)")
        print(f"      Avg Inter-Patient STD (Abs Dev):    {avg_std_abs:.4f} (from {num_seeds_avg_abs} seeds with valid inter-patient SD)")

    return final_averaged_benchmarks, per_seed_scale_factors


def analyze_patient_coefficient_stability(
    all_seeds_results_by_method,
    manual_ld_alignment_map,
    component_col_names_by_method,
    unique_patient_ids,
    per_seed_contextual_sds # Output from calculate_benchmark_inter_patient_variability
):
    """
    Analyzes the stability of individual patient local coefficient deviations
    (raw signed, raw absolute, normalized signed, normalized absolute)
    from global coefficients across seeds, using a manual alignment of latent dimensions
    and per-seed contextual scale factors for normalization.

    Args:
        all_seeds_results_by_method (dict): Results from the multi-seed run.
        manual_ld_alignment_map (dict): Defines how original LD names map to conceptual LD names.
        component_col_names_by_method (dict): Maps method_name to ordered list of component columns for OLS.
        unique_patient_ids (list or pd.Index): List of all unique patient IDs.
        per_seed_contextual_sds (dict): Per-seed scale factors.
            Keys: (method_name, conceptual_ld_name, seed_val)
            Values: {'scale_factor_signed': std_of_signed_devs_this_seed,
                     'scale_factor_abs': std_of_abs_devs_this_seed}

    Returns:
        pd.DataFrame: DataFrame with columns including raw and normalized stability stats.
    """
    print("\n--- Analyzing Stability of Patient-Level Local Coefficient Deviations (Raw & Normalized by Seed-Context) ---")
    stability_records = []

    for method_name, seed_results_list in all_seeds_results_by_method.items():
        if method_name not in manual_ld_alignment_map or method_name not in component_col_names_by_method:
            print(f"  Skipping method '{method_name}': No alignment map or component_col_names.")
            continue

        print(f"\n  Processing Method: '{method_name}' for patient stability")
        method_alignment_map = manual_ld_alignment_map[method_name]
        ordered_component_cols_for_ols = component_col_names_by_method[method_name]

        all_conceptual_lds_for_method = set()
        for seed_val_map in method_alignment_map.values():
            for conceptual_name_or_detail in seed_val_map.values():
                if isinstance(conceptual_name_or_detail, dict):
                    all_conceptual_lds_for_method.add(conceptual_name_or_detail['conceptual_name'])
                else:
                    all_conceptual_lds_for_method.add(conceptual_name_or_detail)

        if not all_conceptual_lds_for_method:
            print(f"    No conceptual LDs found in the alignment map for method '{method_name}'.")
            continue

        for patient_id in unique_patient_ids:
            for conceptual_ld_name in sorted(list(all_conceptual_lds_for_method)):
                raw_deviation_scores_signed = []
                raw_deviation_scores_abs = []
                normalized_deviation_scores_signed = []
                normalized_deviation_scores_abs = []
                num_valid_seeds_for_this_comparison = 0

                for seed_result in seed_results_list:
                    seed_val = seed_result.get('seed')
                    if seed_val is None or seed_val not in method_alignment_map:
                        continue

                    seed_specific_ld_alignment = method_alignment_map[seed_val]
                    original_ld_name_in_seed = None
                    sign_multiplier = 1

                    for orig_ld, conceptual_info in seed_specific_ld_alignment.items():
                        current_conceptual_name_from_map = None
                        current_sign_from_map = 1
                        if isinstance(conceptual_info, dict):
                            current_conceptual_name_from_map = conceptual_info.get('conceptual_name')
                            current_sign_from_map = conceptual_info.get('sign', 1)
                        elif isinstance(conceptual_info, str):
                            current_conceptual_name_from_map = conceptual_info

                        if current_conceptual_name_from_map == conceptual_ld_name:
                            original_ld_name_in_seed = orig_ld
                            sign_multiplier = current_sign_from_map
                            break

                    if not original_ld_name_in_seed:
                        continue

                    analysis_data = seed_result.get('train_analysis', {})
                    local_betas_df_seed = analysis_data.get('local_betas_df')
                    global_coeffs_df_seed = analysis_data.get('global_ols_coeffs_df')

                    if local_betas_df_seed is None or global_coeffs_df_seed is None:
                        continue

                    if patient_id not in local_betas_df_seed.index or \
                       original_ld_name_in_seed not in local_betas_df_seed.columns:
                        continue

                    local_coeff_raw = local_betas_df_seed.loc[patient_id, original_ld_name_in_seed]
                    if pd.isna(local_coeff_raw):
                        continue
                    local_coeff_corrected = local_coeff_raw * sign_multiplier

                    global_coeff_raw = np.nan
                    try:
                        idx_in_ols_components = ordered_component_cols_for_ols.index(original_ld_name_in_seed)
                        potential_ols_coeff_name = None
                        if global_coeffs_df_seed.index[0].lower() == 'const' and (idx_in_ols_components + 1) < len(global_coeffs_df_seed.index):
                            potential_ols_coeff_name = global_coeffs_df_seed.index[idx_in_ols_components + 1]
                        elif original_ld_name_in_seed in global_coeffs_df_seed.index:
                            potential_ols_coeff_name = original_ld_name_in_seed

                        if potential_ols_coeff_name and potential_ols_coeff_name in global_coeffs_df_seed.index:
                            global_coeff_raw = global_coeffs_df_seed.loc[potential_ols_coeff_name, 'Coefficient']
                    except (ValueError, IndexError):
                        pass

                    if pd.isna(global_coeff_raw):
                        continue
                    global_coeff_corrected = global_coeff_raw * sign_multiplier

                    raw_signed_deviation_this_seed = local_coeff_corrected - global_coeff_corrected
                    raw_deviation_scores_signed.append(raw_signed_deviation_this_seed)
                    raw_deviation_scores_abs.append(abs(raw_signed_deviation_this_seed))

                    # --- Normalization ---
                    context_sds = per_seed_contextual_sds.get((method_name, conceptual_ld_name, seed_val))
                    if context_sds:
                        scale_factor_signed = context_sds.get('scale_factor_signed')
                        scale_factor_abs = context_sds.get('scale_factor_abs') # This is SD of abs_devs across patients

                        if pd.notna(scale_factor_signed) and abs(scale_factor_signed) > NUMERICAL_TOLERANCE:
                            normalized_deviation_scores_signed.append(raw_signed_deviation_this_seed / scale_factor_signed)
                        # else:
                            # normalized_deviation_scores_signed.append(np.nan) # Or skip if scale factor is problematic

                        if pd.notna(scale_factor_abs) and abs(scale_factor_abs) > NUMERICAL_TOLERANCE:
                            normalized_deviation_scores_abs.append(abs(raw_signed_deviation_this_seed) / scale_factor_abs)
                        # else:
                            # normalized_deviation_scores_abs.append(np.nan)
                    # else: # Contextual SDs not found for this seed/cld combo
                        # normalized_deviation_scores_signed.append(np.nan)
                        # normalized_deviation_scores_abs.append(np.nan)

                    num_valid_seeds_for_this_comparison += 1

                if num_valid_seeds_for_this_comparison > 0:
                    # Stats for raw deviations
                    mean_raw_dev_signed = np.nanmean(raw_deviation_scores_signed) if raw_deviation_scores_signed else np.nan
                    std_raw_dev_signed = np.nanstd(raw_deviation_scores_signed) if len(raw_deviation_scores_signed) > MIN_PATIENTS_FOR_STD else np.nan
                    mean_raw_dev_abs = np.nanmean(raw_deviation_scores_abs) if raw_deviation_scores_abs else np.nan
                    std_raw_dev_abs = np.nanstd(raw_deviation_scores_abs) if len(raw_deviation_scores_abs) > MIN_PATIENTS_FOR_STD else np.nan

                    # Stats for normalized deviations
                    mean_norm_dev_signed = np.nanmean(normalized_deviation_scores_signed) if normalized_deviation_scores_signed else np.nan
                    std_norm_dev_signed = np.nanstd(normalized_deviation_scores_signed) if len(normalized_deviation_scores_signed) > MIN_PATIENTS_FOR_STD else np.nan
                    mean_norm_dev_abs = np.nanmean(normalized_deviation_scores_abs) if normalized_deviation_scores_abs else np.nan
                    std_norm_dev_abs = np.nanstd(normalized_deviation_scores_abs) if len(normalized_deviation_scores_abs) > MIN_PATIENTS_FOR_STD else np.nan

                    stability_records.append({
                        'patient_id': patient_id,
                        'method': method_name,
                        'conceptual_ld': conceptual_ld_name,
                        'mean_deviation': mean_raw_dev_signed,
                        'std_deviation': std_raw_dev_signed,
                        'mean_abs_deviation': mean_raw_dev_abs,
                        'std_abs_deviation': std_raw_dev_abs,
                        'mean_norm_dev_signed': mean_norm_dev_signed,
                        'std_norm_dev_signed': std_norm_dev_signed,
                        'mean_norm_dev_abs': mean_norm_dev_abs,
                        'std_norm_dev_abs': std_norm_dev_abs,
                        'num_seeds_compared': num_valid_seeds_for_this_comparison,
                        'all_signed_deviation_scores': list(raw_deviation_scores_signed),
                        'all_abs_deviation_scores': list(raw_deviation_scores_abs),
                        'all_norm_signed_deviation_scores': list(normalized_deviation_scores_signed),
                        'all_norm_abs_deviation_scores': list(normalized_deviation_scores_abs)
                    })

    if not stability_records:
        print("No patient stability records generated. Check alignment maps, data availability, and patient IDs.")
        return pd.DataFrame()

    results_df = pd.DataFrame(stability_records)
    return results_df

# %%

from scipy.stats import rankdata # For ranking

def analyze_patient_deviation_rank_stability(
    all_seeds_results_by_method,
    manual_ld_alignment_map,
    component_col_names_by_method,
    unique_patient_ids
):
    """
    Analyzes the stability of individual patient's rank of deviation
    (based on absolute magnitude of local_coeff - global_coeff) across seeds,
    using a manual alignment of latent dimensions.

    Args:
        all_seeds_results_by_method (dict): Results from the multi-seed run.
        manual_ld_alignment_map (dict): Defines how original LD names map to conceptual LD names,
                                        optionally including sign correction for LD orientation.
        component_col_names_by_method (dict): Maps method_name to ordered list of component columns for OLS.
        unique_patient_ids (list or pd.Index): List of all unique patient IDs.

    Returns:
        pd.DataFrame: DataFrame with columns:
            ['patient_id', 'method', 'conceptual_ld', 'mean_rank',
             'std_rank', 'num_seeds_ranked_in', 'all_ranks']
    """
    print("\n--- Analyzing Stability of Patient-Level Deviation Ranks ---")
    rank_stability_records = []

    for method_name, seed_results_list in all_seeds_results_by_method.items():
        if method_name not in manual_ld_alignment_map or method_name not in component_col_names_by_method:
            print(f"  Skipping method '{method_name}': No alignment map or component_col_names.")
            continue

        print(f"\n  Processing Method: '{method_name}' for rank stability")
        method_alignment_map = manual_ld_alignment_map[method_name]
        ordered_component_cols_for_ols = component_col_names_by_method[method_name]

        all_conceptual_lds_for_method = set()
        for seed_val_map in method_alignment_map.values():
            for conceptual_name_or_detail in seed_val_map.values():
                if isinstance(conceptual_name_or_detail, dict):
                    all_conceptual_lds_for_method.add(conceptual_name_or_detail['conceptual_name'])
                else:
                    all_conceptual_lds_for_method.add(conceptual_name_or_detail)

        if not all_conceptual_lds_for_method:
            print(f"    No conceptual LDs found in the alignment map for method '{method_name}'.")
            continue

        # Step 1 & 2: Calculate deviations and then ranks per seed/cLD
        # Store ranks: ranks_per_cld_seed[(cld, seed_val)] = {patient_id: rank}
        all_ranks_for_method = defaultdict(dict)

        for conceptual_ld_name in sorted(list(all_conceptual_lds_for_method)):
            for seed_result in seed_results_list:
                seed_val = seed_result.get('seed')
                if seed_val is None or seed_val not in method_alignment_map:
                    continue

                seed_specific_ld_alignment = method_alignment_map[seed_val]
                original_ld_name_in_seed = None
                sign_multiplier = 1 # For LD orientation

                for orig_ld, conceptual_info in seed_specific_ld_alignment.items():
                    current_conceptual_name_from_map = None
                    current_sign_from_map = 1
                    if isinstance(conceptual_info, dict):
                        current_conceptual_name_from_map = conceptual_info.get('conceptual_name')
                        current_sign_from_map = conceptual_info.get('sign', 1)
                    elif isinstance(conceptual_info, str):
                        current_conceptual_name_from_map = conceptual_info

                    if current_conceptual_name_from_map == conceptual_ld_name:
                        original_ld_name_in_seed = orig_ld
                        sign_multiplier = current_sign_from_map
                        break

                if not original_ld_name_in_seed:
                    continue

                analysis_data = seed_result.get('train_analysis', {})
                local_betas_df_seed = analysis_data.get('local_betas_df')
                global_coeffs_df_seed = analysis_data.get('global_ols_coeffs_df')

                if local_betas_df_seed is None or global_coeffs_df_seed is None:
                    continue

                patient_abs_deviations_this_seed = {} # Store {patient_id: abs_deviation}

                for patient_id_iter in local_betas_df_seed.index: # Iterate over patients present in this seed's local betas
                    if patient_id_iter not in unique_patient_ids: # Optional: ensure we only process target patients
                        # This check might be redundant if unique_patient_ids is derived from all local_betas_df
                        continue

                    if original_ld_name_in_seed not in local_betas_df_seed.columns:
                        continue

                    local_coeff_raw = local_betas_df_seed.loc[patient_id_iter, original_ld_name_in_seed]
                    if pd.isna(local_coeff_raw):
                        continue
                    local_coeff_corrected = local_coeff_raw * sign_multiplier

                    global_coeff_raw = np.nan
                    try:
                        idx_in_ols_components = ordered_component_cols_for_ols.index(original_ld_name_in_seed)
                        potential_ols_coeff_name = None
                        if global_coeffs_df_seed.index[0].lower() == 'const' and (idx_in_ols_components + 1) < len(global_coeffs_df_seed.index):
                            potential_ols_coeff_name = global_coeffs_df_seed.index[idx_in_ols_components + 1]
                        elif original_ld_name_in_seed in global_coeffs_df_seed.index:
                            potential_ols_coeff_name = original_ld_name_in_seed

                        if potential_ols_coeff_name and potential_ols_coeff_name in global_coeffs_df_seed.index:
                            global_coeff_raw = global_coeffs_df_seed.loc[potential_ols_coeff_name, 'Coefficient']
                    except (ValueError, IndexError):
                        pass

                    if pd.isna(global_coeff_raw):
                        continue
                    global_coeff_corrected = global_coeff_raw * sign_multiplier

                    signed_deviation = local_coeff_corrected - global_coeff_corrected
                    patient_abs_deviations_this_seed[patient_id_iter] = abs(signed_deviation)

                if patient_abs_deviations_this_seed:
                    pids_for_ranking = list(patient_abs_deviations_this_seed.keys())
                    abs_devs_for_ranking = list(patient_abs_deviations_this_seed.values())

                    # Rank patients by absolute deviation (higher absolute deviation = higher rank)
                    ranks_for_pids = rankdata(abs_devs_for_ranking, method='average')

                    all_ranks_for_method[(conceptual_ld_name, seed_val)] = dict(zip(pids_for_ranking, ranks_for_pids))

        # Step 3 & 4: Collect ranks per patient and calculate stability
        for patient_id in unique_patient_ids:
            for conceptual_ld_name in sorted(list(all_conceptual_lds_for_method)):
                patient_specific_ranks_across_seeds = []
                num_seeds_this_patient_ranked_in = 0

                for seed_result in seed_results_list: # Iterate again to ensure order if needed, or use seed_vals from map
                    seed_val = seed_result.get('seed')
                    if seed_val is None:
                        continue

                    if (conceptual_ld_name, seed_val) in all_ranks_for_method:
                        patient_rank_in_seed = all_ranks_for_method[(conceptual_ld_name, seed_val)].get(patient_id)
                        if patient_rank_in_seed is not None: # Patient was successfully ranked in this seed/cLD
                            patient_specific_ranks_across_seeds.append(patient_rank_in_seed)
                            num_seeds_this_patient_ranked_in += 1

                if num_seeds_this_patient_ranked_in > MIN_PATIENTS_FOR_STD: # Need at least 2 ranks to calculate std
                    mean_rank = np.mean(patient_specific_ranks_across_seeds)
                    std_rank = np.std(patient_specific_ranks_across_seeds)

                    rank_stability_records.append({
                        'patient_id': patient_id,
                        'method': method_name,
                        'conceptual_ld': conceptual_ld_name,
                        'mean_rank': mean_rank,
                        'std_rank': std_rank,
                        'num_seeds_ranked_in': num_seeds_this_patient_ranked_in,
                        'all_ranks': list(patient_specific_ranks_across_seeds)
                    })

    if not rank_stability_records:
        print("No rank stability records generated. Check alignment maps, data availability, and patient IDs.")
        return pd.DataFrame()

    results_df = pd.DataFrame(rank_stability_records)
    print(f"\nGenerated {len(results_df)} patient rank stability records.")
    return results_df

# %%
def analyze_rank_stability_for_cohort_by_cld(
    all_seeds_results_by_method,
    manual_ld_alignment_map,
    component_col_names_by_method
):
    """
    Identifies cohorts of patients who appeared at least once in a specific
    type of conceptual subgroup. Then, for each patient in such a cohort,
    tracks their rank (based on deviations for the conceptual LD that defined
    their cohort membership) across ALL seeds, irrespective of their subgroup
    status in each of those seeds. Finally, calculates rank stability for these
    cohort members.

    Args:
        all_seeds_results_by_method (dict): Results from the multi-seed run.
        manual_ld_alignment_map (dict): Defines how original LD names map to conceptual LD names.
        component_col_names_by_method (dict): Maps method_name to ordered list of original component columns.

    Returns:
        pd.DataFrame: DataFrame with stability metrics.
    """
    print("\n--- Analyzing Rank Stability for Patient Cohorts (defined by initial subgroup) across ALL Seeds ---")
    stability_records = []

    # Step 1: Identify all unique cohort definitions and their members
    # cohort_patients_map[(method, cld_that_defined_sg, sg_type)] = {patient_id1, ...}
    cohort_patients_map = defaultdict(set)

    for method_name, seed_results_list_for_method in all_seeds_results_by_method.items():
        seed_results_list = [r for r in seed_results_list_for_method if 'error' not in r]
        if not seed_results_list: continue
        if method_name not in manual_ld_alignment_map: continue

        method_alignment_map_for_method = manual_ld_alignment_map.get(method_name, {})

        for seed_result in seed_results_list:
            seed_val = seed_result.get('seed')
            if seed_val is None or seed_val not in method_alignment_map_for_method: continue

            analysis_data = seed_result.get('train_analysis', {})
            if not analysis_data: continue
            subgroup_patient_ids_dict_seed = analysis_data.get('subgroup_patient_ids_dict')
            if subgroup_patient_ids_dict_seed is None: continue

            seed_specific_ld_alignment_map = method_alignment_map_for_method.get(seed_val, {})

            for sg_key_from_dict, patient_list_in_sg_this_seed in subgroup_patient_ids_dict_seed.items():
                if not patient_list_in_sg_this_seed: continue
                parts = sg_key_from_dict.split('_')
                if len(parts) >= 3 and parts[0] == "Dynamic":
                    orig_ld_name_from_sg_key = parts[1]
                    sg_type_str_from_sg_key = "_".join(parts[2:])

                    # Map original LD of subgroup key back to its conceptual LD for this seed
                    conceptual_ld_for_this_sg_key = None
                    for orig_ld_map, c_info_map in seed_specific_ld_alignment_map.items():
                        c_name_map = c_info_map.get('conceptual_name') if isinstance(c_info_map, dict) else c_info_map
                        if orig_ld_map == orig_ld_name_from_sg_key:
                            conceptual_ld_for_this_sg_key = c_name_map
                            break

                    if conceptual_ld_for_this_sg_key:
                        cohort_key = (method_name, conceptual_ld_for_this_sg_key, sg_type_str_from_sg_key)
                        cohort_patients_map[cohort_key].update(patient_list_in_sg_this_seed)

    if not cohort_patients_map:
        print("No patient cohorts identified from any subgroup appearances.")
        return pd.DataFrame()

    # Step 2: For each defined cohort, collect ranks for its members across ALL seeds
    #         based on the cLD that defined the cohort.
    # collected_ranks_for_cohort_members[(cohort_key)][patient_id] = [rank_seed1, rank_seed2, ...]
    collected_ranks_for_cohort_members = defaultdict(lambda: defaultdict(list))
    num_successful_seeds_per_method = {}

    for method_name, seed_results_list_for_method in all_seeds_results_by_method.items():
        seed_results_list = [r for r in seed_results_list_for_method if 'error' not in r]
        num_successful_seeds_per_method[method_name] = len(seed_results_list)
        if not seed_results_list: continue
        if method_name not in manual_ld_alignment_map or method_name not in component_col_names_by_method: continue

        method_alignment_map_for_method = manual_ld_alignment_map.get(method_name, {})
        ordered_component_cols_for_ols_method = component_col_names_by_method.get(method_name, [])

        for seed_result in seed_results_list:
            seed_val = seed_result.get('seed')
            if seed_val is None or seed_val not in method_alignment_map_for_method: continue

            analysis_data = seed_result.get('train_analysis', {})
            if not analysis_data: continue
            local_betas_df_seed = analysis_data.get('local_betas_df')
            global_coeffs_df_seed = analysis_data.get('global_ols_coeffs_df')
            if local_betas_df_seed is None or global_coeffs_df_seed is None: continue

            seed_specific_ld_alignment_map = method_alignment_map_for_method.get(seed_val, {})

            # Iterate through all *conceptual dimensions* that could be used for ranking
            all_clds_in_seed_map = { (c_info.get('conceptual_name') if isinstance(c_info, dict) else c_info) : o_ld 
                                     for o_ld, c_info in seed_specific_ld_alignment_map.items() }

            for cld_being_ranked_in_this_seed, original_ld_for_ranking_in_seed in all_clds_in_seed_map.items():
                if cld_being_ranked_in_this_seed is None: continue

                sign_multiplier_for_ranking = 1
                c_info_detail = seed_specific_ld_alignment_map.get(original_ld_for_ranking_in_seed)
                if isinstance(c_info_detail, dict):
                    sign_multiplier_for_ranking = c_info_detail.get('sign', 1)

                # Calculate deviations & ranks for ALL patients for this cld_being_ranked_in_this_seed
                patient_abs_deviations_this_seed_cld = {}
                for patient_id_iter in local_betas_df_seed.index:
                    if original_ld_for_ranking_in_seed not in local_betas_df_seed.columns: continue
                    local_coeff_raw = local_betas_df_seed.loc[patient_id_iter, original_ld_for_ranking_in_seed]
                    if pd.isna(local_coeff_raw): continue
                    local_coeff_corrected = local_coeff_raw * sign_multiplier_for_ranking

                    global_coeff_raw = np.nan
                    try:
                        idx_in_ols_components = ordered_component_cols_for_ols_method.index(original_ld_for_ranking_in_seed)
                        potential_ols_coeff_name = None
                        if global_coeffs_df_seed.index[0].lower() == 'const' and \
                           (idx_in_ols_components + 1) < len(global_coeffs_df_seed.index):
                            potential_ols_coeff_name = global_coeffs_df_seed.index[idx_in_ols_components + 1]
                        elif original_ld_for_ranking_in_seed in global_coeffs_df_seed.index:
                            potential_ols_coeff_name = original_ld_for_ranking_in_seed
                        if potential_ols_coeff_name and potential_ols_coeff_name in global_coeffs_df_seed.index:
                            global_coeff_raw = global_coeffs_df_seed.loc[potential_ols_coeff_name, 'Coefficient']
                    except (ValueError, IndexError): pass
                    if pd.isna(global_coeff_raw): continue
                    global_coeff_corrected = global_coeff_raw * sign_multiplier_for_ranking
                    patient_abs_deviations_this_seed_cld[patient_id_iter] = abs(local_coeff_corrected - global_coeff_corrected)

                if not patient_abs_deviations_this_seed_cld: continue
                pids_for_ranking = list(patient_abs_deviations_this_seed_cld.keys())
                abs_devs_for_ranking = list(patient_abs_deviations_this_seed_cld.values())
                ranks_values = rankdata(abs_devs_for_ranking, method='average')
                ranks_all_patients_for_this_cld_this_seed = dict(zip(pids_for_ranking, ranks_values))

                # Now, for any cohort defined by this cld_being_ranked_in_this_seed, add ranks for its members
                for (coh_method, coh_cld_def, coh_sg_type), patient_set in cohort_patients_map.items():
                    if coh_method == method_name and coh_cld_def == cld_being_ranked_in_this_seed:
                        for patient_id_in_cohort in patient_set:
                            rank_to_add = ranks_all_patients_for_this_cld_this_seed.get(patient_id_in_cohort, np.nan)
                            collected_ranks_for_cohort_members[(coh_method, coh_cld_def, coh_sg_type)][patient_id_in_cohort].append(rank_to_add)


    # Step 3: Aggregate Stability Metrics
    for (method, cld_defined_cohort, sg_type_of_cohort), patient_to_ranks_map in collected_ranks_for_cohort_members.items():
        for patient_id, ranks_list_possibly_with_nan in patient_to_ranks_map.items():
            valid_ranks = [r for r in ranks_list_possibly_with_nan if pd.notna(r)]
            num_seeds_actually_ranked = len(valid_ranks)
            # num_seeds_attempted = num_successful_seeds_per_method.get(method, 0) # Total successful seeds for the method

            if num_seeds_actually_ranked > MIN_PATIENTS_FOR_STD:
                mean_r = np.mean(valid_ranks)
                std_r = np.std(valid_ranks)
                stability_records.append({
                    'method': method,
                    'patient_id': patient_id,
                    'cohort_defined_by_cld': cld_defined_cohort,
                    'cohort_defined_by_sg_type': sg_type_of_cohort,
                    'mean_rank_all_seeds': mean_r,
                    'std_rank_all_seeds': std_r,
                    'num_seeds_ranked_for_cld': num_seeds_actually_ranked,
                    # 'total_successful_seeds_for_method': num_seeds_attempted,
                    'all_ranks_collected': list(valid_ranks)
                })

    if not stability_records:
        print("No rank stability records generated for any cohorts.")
        return pd.DataFrame()

    results_df = pd.DataFrame(stability_records)
    print(f"\nGenerated {len(results_df)} rank stability records for patient cohorts (ranks from all seeds).")
    return results_df

# %%
def analyze_subgroup_member_rank_stability(
    all_seeds_results_by_method,
    manual_ld_alignment_map,
    component_col_names_by_method,
):
    """
    Analyzes the stability of individual patient's rank of deviation
    (based on absolute magnitude of local_coeff - global_coeff)
    for patients within dynamically defined subgroups, using conceptual alignment.

    The rank is calculated w.r.t. the (aligned) latent dimension that
    defined the subgroup in a given seed. Stability is assessed across seeds
    for patients consistently part of a conceptually similar subgroup.

    Args:
        all_seeds_results_by_method (dict): Results from the multi-seed run.
        manual_ld_alignment_map (dict): Defines how original LD names map to conceptual LD names.
        component_col_names_by_method (dict): Maps method_name to ordered list of original component columns.

    Returns:
        pd.DataFrame: DataFrame with columns:
            ['method', 'patient_id', 'conceptual_ld_subgroup_defined_by',
             'subgroup_definition_type', 'mean_rank_in_subgroup_context',
             'std_rank_in_subgroup_context', 'num_seeds_ranked_in_subgroup',
             'all_ranks_in_subgroup_context']
    """
    print("\n--- Analyzing Rank Stability for Subgroup Members based on Aligned Latent Dimensions ---")
    stability_records = []

    # Storage: patient_ranks_in_csg[(method, patient_id, cld_def, sg_type_def)] = [rank_seedA, rank_seedB, ...]
    #   cld_def: Conceptual LD that DEFINED the subgroup.
    #   sg_type_def: How the subgroup was defined (e.g., 'belowLCI', 'aboveUCI').
    #   The rank itself is calculated for the original LD corresponding to cld_def in that seed.
    patient_ranks_in_conceptual_subgroups = defaultdict(list)

    for method_name, seed_results_list_for_method in all_seeds_results_by_method.items():
        # Filter out error results
        seed_results_list = [r for r in seed_results_list_for_method if 'error' not in r]

        if not seed_results_list:
            print(f"  Skipping method '{method_name}': No valid seed results.")
            continue

        if method_name not in manual_ld_alignment_map or method_name not in component_col_names_by_method:
            print(f"  Skipping method '{method_name}': Missing in alignment map or component_col_names.")
            continue

        print(f"\n  Processing Method: '{method_name}' for subgroup rank stability")
        method_alignment_map_for_method = manual_ld_alignment_map.get(method_name, {})
        ordered_component_cols_for_ols_method = component_col_names_by_method.get(method_name, [])

        all_conceptual_lds_for_method = set()
        for seed_val_map_inner in method_alignment_map_for_method.values():
            for conceptual_name_or_detail_inner in seed_val_map_inner.values():
                if isinstance(conceptual_name_or_detail_inner, dict):
                    all_conceptual_lds_for_method.add(conceptual_name_or_detail_inner['conceptual_name'])
                else:
                    all_conceptual_lds_for_method.add(conceptual_name_or_detail_inner)

        if not all_conceptual_lds_for_method:
            print(f"    No conceptual LDs found for method '{method_name}' in its alignment map.")
            continue

        for seed_result in seed_results_list:
            seed_val = seed_result.get('seed')
            if seed_val is None or seed_val not in method_alignment_map_for_method:
                continue

            analysis_data = seed_result.get('train_analysis', {})
            if not analysis_data:
                continue

            local_betas_df_seed = analysis_data.get('local_betas_df')
            global_coeffs_df_seed = analysis_data.get('global_ols_coeffs_df')
            subgroup_patient_ids_dict_seed = analysis_data.get('subgroup_patient_ids_dict')

            if local_betas_df_seed is None or global_coeffs_df_seed is None or subgroup_patient_ids_dict_seed is None:
                continue

            seed_specific_ld_alignment_map = method_alignment_map_for_method.get(seed_val, {})

            for cld_def_name in all_conceptual_lds_for_method:
                original_ld_name_in_seed_for_cld_def = None
                sign_multiplier_for_cld_def = 1

                for orig_ld, conceptual_info in seed_specific_ld_alignment_map.items():
                    current_conceptual_name_from_map = None
                    current_sign_from_map = 1
                    if isinstance(conceptual_info, dict):
                        current_conceptual_name_from_map = conceptual_info.get('conceptual_name')
                        current_sign_from_map = conceptual_info.get('sign', 1)
                    elif isinstance(conceptual_info, str):
                        current_conceptual_name_from_map = conceptual_info

                    if current_conceptual_name_from_map == cld_def_name:
                        original_ld_name_in_seed_for_cld_def = orig_ld
                        sign_multiplier_for_cld_def = current_sign_from_map
                        break

                if not original_ld_name_in_seed_for_cld_def:
                    continue # This conceptual LD isn't mapped in this seed

                patient_abs_deviations_this_seed_orig_ld = {}
                for patient_id_iter in local_betas_df_seed.index:
                    if original_ld_name_in_seed_for_cld_def not in local_betas_df_seed.columns:
                        continue
                    local_coeff_raw = local_betas_df_seed.loc[patient_id_iter, original_ld_name_in_seed_for_cld_def]
                    if pd.isna(local_coeff_raw):
                        continue
                    local_coeff_corrected = local_coeff_raw * sign_multiplier_for_cld_def

                    global_coeff_raw = np.nan
                    try:
                        idx_in_ols_components = ordered_component_cols_for_ols_method.index(original_ld_name_in_seed_for_cld_def) #
                        potential_ols_coeff_name = None
                        # Check if OLS model used 'const' and generic names like 'x1', 'x2'
                        if global_coeffs_df_seed.index[0].lower() == 'const' and \
                           (idx_in_ols_components + 1) < len(global_coeffs_df_seed.index):
                            potential_ols_coeff_name = global_coeffs_df_seed.index[idx_in_ols_components + 1] #
                        # Fallback if OLS used original component names or other structures
                        elif original_ld_name_in_seed_for_cld_def in global_coeffs_df_seed.index:
                            potential_ols_coeff_name = original_ld_name_in_seed_for_cld_def #

                        if potential_ols_coeff_name and potential_ols_coeff_name in global_coeffs_df_seed.index:
                            global_coeff_raw = global_coeffs_df_seed.loc[potential_ols_coeff_name, 'Coefficient'] #
                    except (ValueError, IndexError):
                        pass

                    if pd.isna(global_coeff_raw):
                        continue
                    global_coeff_corrected = global_coeff_raw * sign_multiplier_for_cld_def

                    signed_deviation = local_coeff_corrected - global_coeff_corrected
                    patient_abs_deviations_this_seed_orig_ld[patient_id_iter] = abs(signed_deviation)

                if not patient_abs_deviations_this_seed_orig_ld:
                    continue

                pids_for_ranking = list(patient_abs_deviations_this_seed_orig_ld.keys())
                abs_devs_for_ranking = list(patient_abs_deviations_this_seed_orig_ld.values())
                ranks_for_pids_values = rankdata(abs_devs_for_ranking, method='average') #
                ranks_all_patients_this_seed_orig_ld = dict(zip(pids_for_ranking, ranks_for_pids_values))

                for sg_key_from_dict, patient_list_in_sg_this_seed in subgroup_patient_ids_dict_seed.items(): #
                    if not patient_list_in_sg_this_seed:
                        continue

                    parts = sg_key_from_dict.split('_') # Example "Dynamic_Latent0_belowLCI"
                    if len(parts) >= 3 and parts[0] == "Dynamic": #
                        orig_ld_name_from_sg_key = parts[1] # e.g., "Latent0"
                        sg_type_str_from_sg_key = "_".join(parts[2:]) # e.g., "belowLCI" or "aboveUCI"

                        if orig_ld_name_from_sg_key == original_ld_name_in_seed_for_cld_def:
                            for patient_id_in_sg in patient_list_in_sg_this_seed:
                                if patient_id_in_sg in ranks_all_patients_this_seed_orig_ld:
                                    rank_of_patient = ranks_all_patients_this_seed_orig_ld[patient_id_in_sg]
                                    storage_key = (method_name, patient_id_in_sg, cld_def_name, sg_type_str_from_sg_key)
                                    patient_ranks_in_conceptual_subgroups[storage_key].append(rank_of_patient)

    if not patient_ranks_in_conceptual_subgroups:
        print("No patient ranks collected in any conceptual subgroups.")
        return pd.DataFrame()

    for (method, patient_id, cld_definition, sg_type_definition), ranks_list in patient_ranks_in_conceptual_subgroups.items():
        if len(ranks_list) >= MIN_PATIENTS_FOR_STD: # Need at least 2 ranks for std deviation
            mean_r = np.mean(ranks_list)
            std_r = np.std(ranks_list)
            stability_records.append({
                'method': method,
                'patient_id': patient_id,
                'conceptual_ld_subgroup_defined_by': cld_definition,
                'subgroup_definition_type': sg_type_definition,
                'mean_rank_in_subgroup_context': mean_r,
                'std_rank_in_subgroup_context': std_r,
                'num_seeds_ranked_in_subgroup': len(ranks_list),
                'all_ranks_in_subgroup_context': list(ranks_list)
            })

    if not stability_records:
        print("No stability records generated for subgroup member ranks.")
        return pd.DataFrame()

    results_df = pd.DataFrame(stability_records)
    print(f"\nGenerated {len(results_df)} subgroup member rank stability records.")
    return results_df

# %%
# Cluster names for Z-score plot (from streamlit_singlesite.py)
CLUSTER_NAMES_DICT = {
        'alter_W': 'Age',
        'BORG_W': 'Dyspnea',
        'BPDIA_W': 'Blood Pressure',
        'BPSYS_W': 'Blood Pressure',
        'BREATH_W': 'Respiratory Rate',
        'COPDDIA_W': 'COPD and Symptoms Duration',
        'COPDSYM_W': 'COPD and Symptoms Duration',
        'DIST_W': 'Exercise Capacity',
        'DLCOMP_W': 'Diffusion Capacity',
        'DLCOPP_W': 'Diffusion Capacity',
        'DLCOVAMP_W': 'Diffusion Capacity',
        'DLCOVAPP_W': 'Diffusion Capacity',
        'ERVL_W': 'Expiratory Reserve',
        'ERVLP_W': 'Expiratory Reserve',
        'ERVP_W': 'Expiratory Reserve',
        'ERVPP_W': 'Expiratory Reserve',
        'FENO': 'Inflammatory Markers',
        'FEVL_W': 'Diffusion Capacity',
        'FEVLP_W': 'Diffusion Capacity',
        'FEVP_W': 'Diffusion Capacity',
        'FEVPP_W': 'Diffusion Capacity',
        'FEVVCP_W': 'Diffusion Capacity',
        'FEVVCPP_W': 'Diffusion Capacity',
        'FVCL_W': 'Lung Volume Parameters',
        'FVCLP_W': 'Lung Volume Parameters',
        'FVCP_W': 'Lung Capacity Percentage',
        'FVCPP_W': 'Lung Capacity Percentage',
        'HEIGHT_W': 'Height',
        'HHR_W': 'Exercise Heart Rate',
        'HR_W': 'Exercise Heart Rate',
        'HRREST_W': 'Exercise Heart Rate',
        'ITGVL_W': 'Lung Intrathoracic Gas',
        'ITGVLP_W': 'Lung Intrathoracic Gas',
        'ITGVP_W': 'Lung Intrathoracic Gas',
        'ITGVPP_W': 'Lung Intrathoracic Gas',
        'LOXAT_W': 'Exercise Oxygen Desaturation',
        'MEDDIS_W': 'Medication Count',
        'MEF25P_W': 'Lung Flow Measurements',
        'MEF25PP_W': 'Lung Flow Measurements',
        'MEF50LS_W': 'Diffusion Capacity',
        'MEF50LSP_W': 'Diffusion Capacity',
        'MEF50P_W': 'Diffusion Capacity',
        'MEF50PP_W': 'Diffusion Capacity',
        'MEF75LS_W': 'Diffusion Capacity',
        'MEF75LSP_W': 'Diffusion Capacity',
        'MEF75P_W': 'Diffusion Capacity',
        'MEF75PP_W': 'Diffusion Capacity',
        'NO1_W': 'Inflammatory Markers',
        'PEFLS_W': 'Diffusion Capacity',
        'PEFLSP_W': 'Diffusion Capacity',
        'PEFP_W': 'Diffusion Capacity',
        'PEFPP_W': 'Diffusion Capacity',
        'POXSAT_W': 'Baseline Oxygen Levels',
        'POXSAT_WDT6_W': 'Baseline Oxygen Levels',
        'PY_W': 'Smoking History',
        'RVL_W': 'Lung Intrathoracic Gas',
        'RVLP_W': 'Lung Intrathoracic Gas',
        'RVP_W': 'Lung Intrathoracic Gas',
        'RVPP_W': 'Lung Intrathoracic Gas',
        'RVTLCP_W': 'Lung Intrathoracic Gas',
        'RVTLCPP_W': 'Lung Intrathoracic Gas',
        'TLCL_W': 'Lung Intrathoracic Gas',
        'TLCLP_W': 'Lung Intrathoracic Gas',
        'TLCP_W': 'Lung Intrathoracic Gas',
        'TLCPP_W': 'Lung Intrathoracic Gas',
        'VALP_W': 'Ventilation Efficiency',
        'VAPP_W': 'Ventilation Efficiency',
        'VCL_W': 'Lung Volume Parameters',
        'VCLP_W': 'Lung Volume Parameters',
        'VCMAXL_W': 'Lung Volume Parameters',
        'VCMAXLP_W': 'Lung Volume Parameters',
        'VCMAXP_W': 'Lung Capacity Percentage',
        'VCMAXPP_W': 'Lung Capacity Percentage',
        'VCP_W': 'Lung Capacity Percentage',
        'VCPP_W': 'Lung Capacity Percentage',
        'WEIGHT_W': 'Body Weight'
        }

# %%
def load_and_process_data(data_path, train_params, current_seed):
    """
    Load and preprocess data for analysis.
    
    Args:
        data_path (str): Path to the data file
        train_params (dict): Training parameters dictionary
        current_seed (int): Random seed for reproducible train/test split
    
    Returns:
        tuple: (train_data, test_data, base_feature_cols, train_data_copy, test_data_copy)
               Returns (None, None, None, None, None) if data loading fails
    """
    try:
        data = pd.read_csv(data_path)
        print(f"Successfully loaded data from {data_path}. Shape: {data.shape}")
    except FileNotFoundError:
        print(f"ERROR: Data file not found at {data_path}.")
        return None, None, None, None, None
    except Exception as e:
        print(f"ERROR: Failed to load data from {data_path}. Error: {str(e)}")
        return None, None, None, None, None

    # Drop latent variables (if they exist)
    data = data.drop(columns=['Latent0', 'Latent1', 'Latent2','Latent3'], errors='ignore')
    outcome_var = train_params["outcome_var"]

    # Filter out non-numeric columns (identifiers like CENTER, PID, VISIT, etc.)
    # Keep only numeric columns and the outcome variable
    numeric_columns = data.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns.tolist()
    non_numeric_columns = data.select_dtypes(exclude=['float64', 'int64', 'float32', 'int32']).columns.tolist()
    
    # Ensure outcome variable is included if it exists
    if outcome_var in data.columns:
        if outcome_var not in numeric_columns:
            # Try to convert outcome variable to numeric if possible
            try:
                data[outcome_var] = pd.to_numeric(data[outcome_var])
                numeric_columns.append(outcome_var)
            except (ValueError, TypeError):
                print(f"Warning: Outcome variable '{outcome_var}' is not numeric and cannot be converted.")
    
    # Print info about filtered columns
    if non_numeric_columns:
        print(f"Filtered out non-numeric columns: {non_numeric_columns}")
    
    # Keep only numeric columns
    data_numeric = data[numeric_columns]

    # Check if 'train' column exists for predefined split
    if 'train' in data_numeric.columns:
        # Split train and test based on train column
        train_data = data_numeric[data_numeric['train']==1]
        test_data = data_numeric[data_numeric['train']==0]
        # Drop train column
        train_data = train_data.drop(columns=['train'])
        test_data = test_data.drop(columns=['train'])
    else:
        # Create train/test split programmatically
        print(f"No 'train' column found. Creating {int((1-TEST_SIZE_RATIO)*100)}/{int(TEST_SIZE_RATIO*100)} train/test split with seed {current_seed}...")
        train_data, test_data = train_test_split(
            data_numeric, 
            test_size=TEST_SIZE_RATIO, 
            random_state=current_seed,
            stratify=None  # Set to None since we don't know if Y is categorical
        )

    base_feature_cols = [
        col
        for col in data_numeric.columns
        if col not in [outcome_var, "Y", "train", "index"]
        and not col.startswith("Latent")
        and not col.startswith("PC")
    ]

    # Add outcome variable statistics
    if outcome_var in train_data.columns:
        outcome_values = train_data[outcome_var].dropna()
        #if not outcome_values.empty:
            #print(f"\n[1] Outcome Variable Statistics (Training Data):")
            #print(f"  {outcome_var}: Mean = {outcome_values.mean():.4f}, SD = {outcome_values.std():.4f}, N = {len(outcome_values)}")

    return (
        train_data,
        test_data,
        base_feature_cols,
        train_data,
        test_data,
    )

# %%
def train_ae_model(df_train_input, df_test_input, train_params, device_in, current_seed_val, is_vanilla=False):
    """
    Train an autoencoder model (either Vanilla or Composite).
    
    Args:
        df_train_input (pd.DataFrame): Training data including outcome variable
        df_test_input (pd.DataFrame): Test data including outcome variable  
        train_params (dict): Training hyperparameters
        device_in (torch.device): Device to run training on
        current_seed_val (int): Random seed for this training run
        is_vanilla (bool): If True, train VanillaAE; if False, train CompositeAE
    
    Returns:
        tuple: (model, final_recon_loss, final_null_loss, final_global_loss, 
                eval_recon_loss_train, eval_recon_loss_test)
               Returns (None, nan, nan, nan, nan, nan) if training fails
    """
    outcome_var = train_params['outcome_var']
    if outcome_var not in df_train_input.columns:
        print(f"ERROR: Outcome variable '{outcome_var}' not in df_train_input for AE training.")
        return None, np.nan, np.nan, np.nan, np.nan, np.nan

    input_size_train = df_train_input.drop(columns=[outcome_var], errors='ignore').shape[1]
    latent_size = train_params['latent_size']
    hidden_size_1 = latent_size ** train_params['hidden_factor_1']
    hidden_size_2 = latent_size ** train_params['hidden_factor_2']

    model, optimizer, scheduler, _ = initialize_training(
        input_size_train, hidden_size_1, hidden_size_2, latent_size, device_in,
        train_params['learning_rate'], train_params['weight_decay']
    )

    # TensorBoard logging removed for reproducibility
    print(f"Training {'CompositeAE' if not is_vanilla else 'VanillaAE'} for seed {current_seed_val}...")

    current_theta = 0.0 if is_vanilla else train_params['theta_null_loss_composite']
    current_gamma = 0.0 if is_vanilla else train_params['gamma_global_loss_composite']

    # This function returns a dictionary of loss histories
    training_losses_history = train_model(
        input_data=df_train_input.copy(), outcome_var=outcome_var,
        batch_size=train_params['batch_size_train'],
        switch_shuffle=train_params['switch_shuffle_epoch'],
        num_batches=train_params['num_batches_weights'],
        model=model, optimizer=optimizer, scheduler=scheduler,
        num_epochs=train_params['num_epochs'],
        log_interval=train_params['log_interval'],
        alpha=train_params['alpha_recon_loss'],
        theta=current_theta, gamma=current_gamma,
        device=device_in, sigma=train_params['sigma_weights'],
        k_nearest=train_params['k_nearest_weights'],
        kernel=train_params['kernel_type_weights']
        #progress_bar_class=tqdm
    )

    # Extract final logged epoch average losses from training history
    # Handle cases where history might be empty if num_epochs < log_interval
    final_epoch_recon_loss_train = training_losses_history['reconstruction_loss'][-1] if training_losses_history['reconstruction_loss'] else np.nan
    final_epoch_null_loss_train = training_losses_history['null_loss'][-1] if training_losses_history['null_loss'] else np.nan
    final_epoch_global_loss_train = training_losses_history['global_loss'][-1] if training_losses_history['global_loss'] else np.nan

    # Evaluate reconstruction loss on the full training set post-training
    model.eval()
    train_features_for_recon = df_train_input.drop(columns=[outcome_var], errors='ignore')
    train_loader_recon = create_data_loader(df_train_input.copy(), outcome_var, train_params['batch_size_train'], train_params['num_batches_weights'], shuffle=False)
    with torch.no_grad():
        data_x_to_device = train_loader_recon.dataset.data_x.to(device_in)
        latent_train_tensor_recon = model.encoder(data_x_to_device)
        reconstruction_train_np = model.decoder(latent_train_tensor_recon).cpu().numpy()
    eval_recon_loss_train = np.mean((train_features_for_recon.values - reconstruction_train_np)**2)

    # Evaluate reconstruction loss on the full test set post-training
    eval_recon_loss_test = np.nan
    if not df_test_input.empty:
        test_features_for_recon = df_test_input.drop(columns=[outcome_var], errors='ignore')
        test_loader_recon = create_data_loader(df_test_input.copy(), outcome_var, train_params['batch_size_train'], train_params['num_batches_weights'], shuffle=False)
        with torch.no_grad():
            data_x_test_to_device = test_loader_recon.dataset.data_x.to(device_in)
            latent_test_tensor_recon = model.encoder(data_x_test_to_device)
            reconstruction_test_np = model.decoder(latent_test_tensor_recon).cpu().numpy()
        eval_recon_loss_test = np.mean((test_features_for_recon.values - reconstruction_test_np)**2)

    return model, final_epoch_recon_loss_train, final_epoch_null_loss_train, final_epoch_global_loss_train, eval_recon_loss_train, eval_recon_loss_test

# %%
def extract_ae_latent_variables(model_trained, df_data, outcome_var_orig, train_params_local, device_local, is_train_set):
    """
    Extract latent variables from a trained autoencoder model.
    
    Args:
        model_trained: Trained autoencoder model
        df_data (pd.DataFrame): Input data to extract latent variables from
        outcome_var_orig (str): Name of the outcome variable
        train_params_local (dict): Training parameters
        device_local (torch.device): Device to run inference on
        is_train_set (bool): Whether this is training data (affects 'train' column)
    
    Returns:
        pd.DataFrame: Original data with latent variables and train flag added
    """
    model_trained.eval()
    latent_size_local = train_params_local['latent_size']
    df_temp_for_loader = df_data.copy()
    
    # Handle outcome variable naming consistency
    if outcome_var_orig not in df_temp_for_loader.columns and 'Y' in df_temp_for_loader.columns:
         df_temp_for_loader.rename(columns={'Y': outcome_var_orig}, inplace=True)

    loader = create_data_loader(df_temp_for_loader, outcome_var_orig, 
                                train_params_local['batch_size_train'], 
                                train_params_local['num_batches_weights'], shuffle=False)
    with torch.no_grad():
        latent_tensor = model_trained.encoder(loader.dataset.data_x.to(device_local)).cpu()
    component_df = pd.DataFrame(latent_tensor.numpy(), columns=[f'Latent{i}' for i in range(latent_size_local)])

    df_with_components = pd.concat([df_data.reset_index(drop=True), component_df.reset_index(drop=True)], axis=1)
    df_with_components['train'] = 1 if is_train_set else 0

    if outcome_var_orig in df_with_components.columns:
        df_with_components.rename(columns={outcome_var_orig: 'Y'}, inplace=True)

    return df_with_components

# %%
def perform_pca_and_prepare_data(df_train_features_unnorm, df_test_features_unnorm, df_train_orig_with_outcome, df_test_orig_with_outcome, n_components, current_seed):
    """
    Perform PCA and prepare dataframes for analysis.
    
    Args:
        df_train_features_unnorm (pd.DataFrame): Unnormalized training features
        df_test_features_unnorm (pd.DataFrame): Unnormalized test features  
        df_train_orig_with_outcome (pd.DataFrame): Original training data with outcome
        df_test_orig_with_outcome (pd.DataFrame): Original test data with outcome
        n_components (int): Number of PCA components to extract
        current_seed (int): Random seed for PCA
    
    Returns:
        tuple: (df_analysis_train_pca, df_analysis_combined_pca, explained_variance_ratio)
    """
    pca = PCA(n_components=n_components, random_state=current_seed)

    pca_train_components = pca.fit_transform(df_train_features_unnorm)
    explained_variance_ratio = pca.explained_variance_ratio_

    pc_train_df = pd.DataFrame(pca_train_components, columns=[f'PC{i}' for i in range(n_components)], index=df_train_features_unnorm.index)

    df_analysis_train_pca = pd.concat([df_train_orig_with_outcome.reset_index(drop=True), pc_train_df.reset_index(drop=True)], axis=1)
    df_analysis_train_pca['train'] = 1
    if TRAIN_HYPERPARAMETERS['outcome_var'] in df_analysis_train_pca.columns:
        df_analysis_train_pca.rename(columns={TRAIN_HYPERPARAMETERS['outcome_var']: 'Y'}, inplace=True)

    df_analysis_combined_pca = df_analysis_train_pca.copy()

    if not df_test_features_unnorm.empty:
        pca_test_components = pca.transform(df_test_features_unnorm)
        pc_test_df = pd.DataFrame(pca_test_components, columns=[f'PC{i}' for i in range(n_components)], index=df_test_features_unnorm.index)

        df_analysis_test_pca_temp = pd.concat([df_test_orig_with_outcome.reset_index(drop=True), pc_test_df.reset_index(drop=True)], axis=1)
        df_analysis_test_pca_temp['train'] = 0
        if TRAIN_HYPERPARAMETERS['outcome_var'] in df_analysis_test_pca_temp.columns:
            df_analysis_test_pca_temp.rename(columns={TRAIN_HYPERPARAMETERS['outcome_var']: 'Y'}, inplace=True)
        df_analysis_combined_pca = pd.concat([df_analysis_train_pca, df_analysis_test_pca_temp], axis=0).reset_index(drop=True)

    return df_analysis_train_pca, df_analysis_combined_pca, explained_variance_ratio

# %%
# =============================================================================
# 3. MAIN ANALYSIS PIPELINE
# =============================================================================
def run_analysis_pipeline(df_input_analysis, analysis_suffix, method_name, outcome_var_y, 
                          component_cols, feature_cols, 
                          analysis_params, train_params, 
                          is_main_method=False, global_ols_coeffs_train=None):
    
    """
    Executes the full analysis pipeline for a single model's output.

    This function is the core analysis engine. For a given set of latent variables
    (from an AE or PCA), it performs the following steps:
    1.  Calculates feature importance (t-test, mutual information) to interpret the components.
    2.  Fits a global Ordinary Least Squares (OLS) model mapping components to the outcome.
    3.  Calculates local regression coefficients (betas) for each patient.
    4.  Dynamically defines subgroups of patients whose local betas deviate significantly
        from the global OLS model's confidence intervals.
    5.  Calculates RMSE statistics to compare the predictive utility of the local vs. global models
        both inside and outside the identified subgroups.
    6.  Generates and saves a suite of plots for visual analysis if `is_main_method` is True.

    Args:
        df_input_analysis (pd.DataFrame): The dataframe containing features, outcome, and components.
        analysis_suffix (str): A string to append to output file names (e.g., seed number).
        method_name (str): Name of the method being analyzed (e.g., "CompositeAE").
        outcome_var_y (str): Name of the outcome variable column.
        component_cols (list): List of column names for the latent components.
        feature_cols (list): List of column names for the original input features.
        analysis_params (dict): Hyperparameters for the analysis stage.
        train_params (dict): Hyperparameters from the training stage.
        is_main_method (bool): If True, generates a full suite of detailed plots.
        global_ols_coeffs_train (pd.DataFrame, optional): Pre-computed OLS coeffs from a training set.

    Returns:
        dict: A dictionary containing all analysis results, including OLS models, local betas,
              subgroup definitions, RMSE stats, and dataframes for stability analysis.
    """

    print(f"\n--- Running Analysis for: {method_name} - {analysis_suffix.upper().replace('_', ' ')} ---")
    data_for_plots = df_input_analysis.copy()

    current_analysis_params = analysis_params.copy()
    selected_comp_for_reporting = None 
    if component_cols: 
        if component_cols[0].startswith("PC"):
            selected_comp_for_reporting = f"PC{train_params['latent_size']-1}"
        else:
            selected_comp_for_reporting = f"Latent{train_params['latent_size']-1}"
        if selected_comp_for_reporting not in component_cols:
            selected_comp_for_reporting = component_cols[-1] 
    current_analysis_params['selected_component_for_reporting'] = selected_comp_for_reporting

    top_n_display = current_analysis_params['ttest_mi_top_n']
    raw_top_ttest_vars_for_stability = {}
    formatted_top_ttest_vars = {}
    all_top_mi_vars_for_subgroup_stability = {}
    all_ttest_results_raw_by_component = {} 

    for comp_dim in component_cols:
        if data_for_plots[comp_dim].nunique() <= 1 or len(data_for_plots[comp_dim].dropna()) < 4:
            formatted_top_ttest_vars[comp_dim] = ["Not enough variance/samples"] * top_n_display
            raw_top_ttest_vars_for_stability[comp_dim] = []
            all_ttest_results_raw_by_component[comp_dim] = {} 
            continue
        comp_median = data_for_plots[comp_dim].median()
        comp_category = np.where(data_for_plots[comp_dim] > comp_median, 1, -1)
        ttest_results_raw = {}
        for var in feature_cols: 
            group1 = data_for_plots.loc[comp_category == 1, var].dropna()
            group2 = data_for_plots.loc[comp_category == -1, var].dropna()
            if len(group1) > 1 and len(group2) > 1 and group1.nunique() > 1 and group2.nunique() > 1:
                t_stat, _ = ttest_ind(group1, group2, equal_var=False, nan_policy='omit')
                ttest_results_raw[var] = t_stat
            else:
                ttest_results_raw[var] = np.nan
            all_ttest_results_raw_by_component[comp_dim] = ttest_results_raw
        sorted_vars_raw_current = sorted([k for k, v in ttest_results_raw.items() if not np.isnan(v)], 
                                       key=lambda x: abs(ttest_results_raw[x]), reverse=True)
        raw_top_ttest_vars_for_stability[comp_dim] = sorted_vars_raw_current[:top_n_display]
        formatted_top_ttest_vars[comp_dim] = [f"{v} (t={ttest_results_raw[v]:.2f})" for v in sorted_vars_raw_current[:top_n_display]]
    ttest_df_display = pd.DataFrame(formatted_top_ttest_vars)
    # Remove these print statements to reduce verbosity (kept for reference analysis only)
    # print(f"\nTop T-Test Variables (Context: median split on components; {method_name} - {analysis_suffix}):")
    # print(ttest_df_display)

    raw_top_mi_vars_for_stability = {} 
    formatted_top_mi_vars = {}
    for comp_dim in component_cols:
        mi_scores_list = []
        valid_vars_for_mi = []
        y_mi = data_for_plots[comp_dim].values.ravel()
        if np.var(y_mi) < 1e-6 or len(y_mi) == 0: 
            formatted_top_mi_vars[comp_dim] = ["Not enough variance/samples"] * top_n_display
            raw_top_mi_vars_for_stability[comp_dim] = []
            all_top_mi_vars_for_subgroup_stability[comp_dim] = [] 
            continue
        for var in feature_cols:
            X_mi_var = data_for_plots[[var]].values 
            valid_mi_indices = ~np.isnan(X_mi_var).any(axis=1) & ~np.isnan(y_mi)
            X_mi_clean = X_mi_var[valid_mi_indices]
            y_mi_clean = y_mi[valid_mi_indices]
            if X_mi_clean.shape[0] > 0 and np.ptp(y_mi_clean) > 1e-9: 
                try:
                    score = mutual_info_regression(X_mi_clean, y_mi_clean, random_state=0)[0]
                    mi_scores_list.append(score)
                    valid_vars_for_mi.append(var)
                except ValueError: pass 
        if valid_vars_for_mi:
            mi_df = pd.DataFrame({'MI': mi_scores_list}, index=valid_vars_for_mi)
            top_vars_mi_raw = mi_df['MI'].nlargest(top_n_display).index.tolist()
            raw_top_mi_vars_for_stability[comp_dim] = top_vars_mi_raw
            all_top_mi_vars_for_subgroup_stability[comp_dim] = mi_df['MI'].nlargest(analysis_params['top_n_vars_forest_plot']).index.tolist()
            formatted_top_mi_vars[comp_dim] = [f"{v} ({mi_df.loc[v, 'MI']:.2f})" for v in top_vars_mi_raw]
        else:
            formatted_top_mi_vars[comp_dim] = ["N/A"] * top_n_display
            raw_top_mi_vars_for_stability[comp_dim] = []
            all_top_mi_vars_for_subgroup_stability[comp_dim] = []
    mi_results_df_display = pd.DataFrame(formatted_top_mi_vars)
    # Print MI variables print for brevity - available in detailed analysis
    # print(f"\nTop Mutual Information Variables (Context: MI with components; {method_name} - {analysis_suffix}):")

    component_data_np = data_for_plots[component_cols].values
    y_data_np = data_for_plots[outcome_var_y].values.astype(float)
    valid_rows_global = ~np.isnan(component_data_np).any(axis=1) & ~np.isnan(y_data_np)

    summary_df_sm = pd.DataFrame() 
    param_names_sm = [] 
    beta_global_for_rmse = None # For storing [intercept, slope1, ...]

    # Local model components
    local_intercepts_list = []
    beta_local_slopes_list = [] 


    if not np.any(valid_rows_global) or component_data_np.shape[1] == 0:
        print(f"Error: No valid data or no components for global OLS model ({method_name} - {analysis_suffix}). Skipping OLS and local betas.")
        placeholder_index = ['const'] + (component_cols if component_cols else [])
        summary_df_sm = pd.DataFrame(np.nan, index=placeholder_index, columns=['Coefficient', 'Lower CI', 'Upper CI', 'P-Value', 'R_squared', 'R_squared_adj'])
        # Initialize beta_local_df as an empty DataFrame with correct columns if component_cols exist
        beta_local_df = pd.DataFrame(columns=component_cols, index=data_for_plots.index) # For slopes
        local_intercepts_s = pd.Series(np.nan, index=data_for_plots.index)
    else:
        ci_alpha = current_analysis_params['ci_alpha']
        X_global_sm = sm.add_constant(component_data_np[valid_rows_global])
        model_sm_fit = sm.OLS(y_data_np[valid_rows_global], X_global_sm).fit()
        param_names_sm = model_sm_fit.model.exog_names 
        summary_df_sm = pd.DataFrame({
            'Coefficient': model_sm_fit.params, 'Lower CI': model_sm_fit.conf_int(alpha=ci_alpha)[:, 0],
            'Upper CI': model_sm_fit.conf_int(alpha=ci_alpha)[:, 1], 'P-Value': model_sm_fit.pvalues,
            'R_squared': model_sm_fit.rsquared, 'R_squared_adj': model_sm_fit.rsquared_adj
        }, index=param_names_sm)
        beta_global_for_rmse = model_sm_fit.params # [intercept, slope1, slope2,...]
        # Skipped for brevity in analysis log
        #print(f"\n Global OLS Coefficients ({method_name} - {analysis_suffix}):")
        #print(summary_df_sm[['Coefficient', 'P-Value', 'R_squared']].head()) 

        # Calculate local betas (intercepts and slopes)
        if method_name != "PCA": # PCA does not have local betas from this process
            sigma_val, k_nearest_val, kernel_val = current_analysis_params['sigma_local_regression'], current_analysis_params['k_nearest_local_regression'], current_analysis_params['kernel_local_regression']
            for i in tqdm(range(len(data_for_plots)), desc=f"Calculating Local Betas ({method_name} - {analysis_suffix})", leave=False):
                if np.isnan(component_data_np[i]).any() or np.isnan(y_data_np[i]): # Skip if target is NaN too
                    local_intercepts_list.append(np.nan)
                    beta_local_slopes_list.append(np.full(len(component_cols), np.nan))
                    continue

                weights_single_np = weights.ss_batch_weights(component_data_np[i], component_data_np, sigma=sigma_val, k_nearest=k_nearest_val, kernel=kernel_val)

                # Ensure y_data_np and weights_single_np are 1D
                y_1d = y_data_np.ravel()
                weights_1d = weights_single_np.ravel()

                # Filter out NaNs from y_data for this specific local regression
                valid_indices_local = ~np.isnan(y_1d) & ~np.isnan(weights_1d) & (weights_1d > 1e-9) # Also ensure weights are non-trivial

                if np.sum(valid_indices_local) < (len(component_cols) + 1): # Need enough points for regression
                    local_intercepts_list.append(np.nan)
                    beta_local_slopes_list.append(np.full(len(component_cols), np.nan))
                    continue

                current_X_local = component_data_np[valid_indices_local]
                current_y_local = y_1d[valid_indices_local]
                current_weights_local = weights_1d[valid_indices_local]

                # Further check for NaNs in current_X_local
                valid_rows_in_current_X = ~np.isnan(current_X_local).any(axis=1)
                if np.sum(valid_rows_in_current_X) < (len(component_cols) +1):
                    local_intercepts_list.append(np.nan)
                    beta_local_slopes_list.append(np.full(len(component_cols), np.nan))
                    continue

                current_X_local_clean = current_X_local[valid_rows_in_current_X]
                current_y_local_clean = current_y_local[valid_rows_in_current_X] # y must be indexed same as X
                current_weights_local_clean = current_weights_local[valid_rows_in_current_X]


                if len(current_X_local_clean) < (len(component_cols) + 1): # Final check after cleaning
                    local_intercepts_list.append(np.nan)
                    beta_local_slopes_list.append(np.full(len(component_cols), np.nan))
                    continue

                beta_single_np = loregs.batch_weighted_regression(X=current_X_local_clean, y=current_y_local_clean, weights=current_weights_local_clean, intercept=True)

                if beta_single_np is not None and not np.all(np.isnan(beta_single_np)) and len(beta_single_np) == len(component_cols) + 1:
                    local_intercepts_list.append(beta_single_np[0])
                    beta_local_slopes_list.append(beta_single_np[1:]) 
                else:
                    local_intercepts_list.append(np.nan)
                    beta_local_slopes_list.append(np.full(len(component_cols), np.nan))
        else: # For PCA, fill with NaNs
            local_intercepts_list = [np.nan] * len(data_for_plots)
            beta_local_slopes_list = [np.full(len(component_cols), np.nan)] * len(data_for_plots)


    local_intercepts_s = pd.Series(local_intercepts_list, index=data_for_plots.index, name="local_intercept")
    beta_local_df = pd.DataFrame(beta_local_slopes_list, columns=component_cols, index=data_for_plots.index) # This is for slopes

    # beta_global_diff_df calculation (uses slopes)
    beta_global_diff_df = pd.DataFrame(columns=component_cols, index=data_for_plots.index)
    if beta_global_for_rmse is not None and len(beta_global_for_rmse) == len(component_cols) + 1:
        beta_global_slopes_only_np = beta_global_for_rmse[1:]
        beta_global_diff_df = beta_local_df.sub(beta_global_slopes_only_np, axis=1)
    else:
        # beta_global_slopes_only_np = np.full(len(component_cols), np.nan) # Not needed if beta_global_for_rmse is None
        beta_global_diff_df = pd.DataFrame(np.nan, index=beta_local_df.index, columns=component_cols)


    # --- Dynamic Subgroup Definition Logic ---
    # Note: Verbose subgroup definition outputs are commented out for manuscript clarity
    # Detailed subgroup definition logic and outputs available in full codebase
    print("\n--- Defining Subgroups Dynamically (Details in Full Analysis) ---")

    num_in_subgroups_dict = {}
    subgroup_configurations = [] # This will be populated with all qualifying dynamic subgroups
    subgroup_patient_ids_dict = {}
    # subgroup_rmse_comparison_stats will be populated later for each sg_config

    all_potential_subgroups = []
    min_dynamic_subgroup_size = 10 # As per your request
    identified_target_ld_orig_name = None # For specific downstream plots (e.g., interaction forest plots)
    subgroup_definition_description_dynamic = "No dynamic subgroups processed." # Default message

    # Ensure prerequisite data is available before attempting subgroup definition
    if summary_df_sm.empty or 'Coefficient' not in summary_df_sm.columns or \
       beta_local_df.empty or not component_cols or not param_names_sm or len(param_names_sm) <=1 :

        print(f"Warning: Prerequisite data for dynamic subgroup definition is missing. Skipping dynamic subgroup definition. Min size set to {min_dynamic_subgroup_size}.")
        # Define a default empty subgroup flag column for downstream code consistency
        dynamic_subgroup_flag_col_name_default = 'subgroup_flag_dynamic_PrereqMissing'
        data_for_plots[dynamic_subgroup_flag_col_name_default] = 0 # Initialize with all zeros
        num_in_subgroups_dict['Dynamic_PrereqMissing'] = 0
        subgroup_definition_description_dynamic = "Prerequisites for dynamic subgroup definition missing."
        # subgroup_configurations remains empty
    else:
        ols_model_component_names = param_names_sm[1:] # These are typically 'x1', 'x2', ...
        for idx, current_comp_dim_orig_name in enumerate(component_cols):
            ols_model_name_for_current_comp_dim = None
            if idx < len(ols_model_component_names):
                ols_model_name_for_current_comp_dim = ols_model_component_names[idx]
            else:
                # This case should ideally not happen if component_cols and OLS model exog_names are aligned
                print(f"  Warning: No corresponding OLS model name for component '{current_comp_dim_orig_name}' (index {idx}). Skipping this dim for subgroup definition.")
                continue

            if current_comp_dim_orig_name not in beta_local_df.columns:
                print(f"  Warning: Local betas for '{current_comp_dim_orig_name}' not found. Skipping.")
                continue
            if ols_model_name_for_current_comp_dim not in summary_df_sm.index:
                print(f"  Warning: Global OLS data for OLS var '{ols_model_name_for_current_comp_dim}' (mapped from '{current_comp_dim_orig_name}') not found in summary_df_sm. Skipping.")
                continue

            global_coeff_val = summary_df_sm.loc[ols_model_name_for_current_comp_dim, 'Coefficient']
            lower_ci_current_dim = summary_df_sm.loc[ols_model_name_for_current_comp_dim, 'Lower CI']
            upper_ci_current_dim = summary_df_sm.loc[ols_model_name_for_current_comp_dim, 'Upper CI']

            local_coeffs_current_dim_series = beta_local_df[current_comp_dim_orig_name]

            if pd.notna(lower_ci_current_dim) and pd.notna(upper_ci_current_dim):
                # Subgroup 1: Local coefficients < Lower CI
                subgroup_below_mask = local_coeffs_current_dim_series < lower_ci_current_dim
                subgroup_below_size = subgroup_below_mask.sum()
                desc_below = f"Local '{current_comp_dim_orig_name}' Coeff < Global LCI ({lower_ci_current_dim:.3f})"
                if subgroup_below_size >= min_dynamic_subgroup_size:
                    all_potential_subgroups.append({
                        'size': subgroup_below_size,
                        'mask': subgroup_below_mask,
                        'description': desc_below,
                        'ld_name': current_comp_dim_orig_name,
                        'ols_ld_name': ols_model_name_for_current_comp_dim,
                        'type': 'below_LCI'
                    })
                    # print(f"    Found potential subgroup: {desc_below}, Size: {subgroup_below_size}") # Can be verbose
                # Subgroup 2: Local coefficients > Upper CI
                subgroup_above_mask = local_coeffs_current_dim_series > upper_ci_current_dim
                subgroup_above_size = subgroup_above_mask.sum()
                desc_above = f"Local '{current_comp_dim_orig_name}' Coeff > Global UCI ({upper_ci_current_dim:.3f})"
                if subgroup_above_size >= min_dynamic_subgroup_size:
                    all_potential_subgroups.append({
                        'size': subgroup_above_size,
                        'mask': subgroup_above_mask,
                        'description': desc_above,
                        'ld_name': current_comp_dim_orig_name,
                        'ols_ld_name': ols_model_name_for_current_comp_dim,
                        'type': 'above_UCI'
                    })
                    # print(f"    Found potential subgroup: {desc_above}, Size: {subgroup_above_size}") # Can be verbose
            else:
                print(f"  Warning: CIs are NaN for OLS var '{ols_model_name_for_current_comp_dim}' (from LD '{current_comp_dim_orig_name}'). Cannot define CI-based subgroups for this dimension.")

        # --- Process all identified potential subgroups that meet criteria ---
        subgroup_definition_description_dynamic_list = []

        if all_potential_subgroups:
            all_potential_subgroups.sort(key=lambda x: x['size'], reverse=True) # Sort by size

            # Set identified_target_ld_orig_name from the largest valid subgroup for specific plots
            if all_potential_subgroups[0]['size'] >= min_dynamic_subgroup_size: # First one is largest
                identified_target_ld_orig_name = all_potential_subgroups[0]['ld_name']
                print(f"\n  Note: 'identified_target_ld_orig_name' for specific interaction plots is set to '{identified_target_ld_orig_name}' (from LD of largest dynamic subgroup: '{all_potential_subgroups[0]['description']}' with size {all_potential_subgroups[0]['size']}).")

            processed_dynamic_sg_count = 0
            for potential_sg_candidate in all_potential_subgroups:
                # We already filtered by min_dynamic_subgroup_size when appending to all_potential_subgroups
                sg_unique_key = f"Dynamic_{potential_sg_candidate['ld_name']}_{potential_sg_candidate['type']}"

                dynamic_subgroup_flag_col_name_current = f'subgroup_flag_{sg_unique_key}'
                data_for_plots[dynamic_subgroup_flag_col_name_current] = potential_sg_candidate['mask'].fillna(False).astype(int)

                current_description = potential_sg_candidate['description']
                subgroup_definition_description_dynamic_list.append(f"{sg_unique_key}: {current_description} (Size: {potential_sg_candidate['size']})")

                num_in_subgroups_dict[sg_unique_key] = potential_sg_candidate['size']

                patient_indices_in_subgroup_current = data_for_plots[potential_sg_candidate['mask'].fillna(False)].index.tolist()
                subgroup_patient_ids_dict[sg_unique_key] = patient_indices_in_subgroup_current

                subgroup_configurations.append({
                    'name': sg_unique_key, 
                    'flag_column': dynamic_subgroup_flag_col_name_current, 
                    'description': current_description,
                    'ld_name_origin': potential_sg_candidate['ld_name'] # Store the originating LD
                })
                processed_dynamic_sg_count += 1
                # print(f"  Registered dynamic subgroup for plotting/analysis: '{sg_unique_key}', Size: {potential_sg_candidate['size']}, Flag: '{dynamic_subgroup_flag_col_name_current}'")

            if processed_dynamic_sg_count == 0:
                subgroup_definition_description_dynamic = f"No dynamic subgroups met the minimum size criterion ({min_dynamic_subgroup_size}) or other criteria after initial scan."
                default_flag_col_name_fallback = 'subgroup_flag_dynamic_NoValidSizeDynamicSGs'
                if default_flag_col_name_fallback not in data_for_plots.columns:
                     data_for_plots[default_flag_col_name_fallback] = 0
                num_in_subgroups_dict[default_flag_col_name_fallback.replace('subgroup_flag_', '')] = 0
                print(f"  {subgroup_definition_description_dynamic}")
            else:
                subgroup_definition_description_dynamic = "; ".join(subgroup_definition_description_dynamic_list)
                # print(f"\n  Total {processed_dynamic_sg_count} dynamic subgroups registered for detailed analysis.")

        else: # No potential subgroups were generated at all
            subgroup_definition_description_dynamic = f"No candidate dynamic subgroups found (e.g., all CIs NaN, no local coeffs outside CIs, or none met min size {min_dynamic_subgroup_size})."
            # identified_target_ld_orig_name remains None
            default_flag_col_name_fallback = 'subgroup_flag_dynamic_NoCandidatesFound'
            if default_flag_col_name_fallback not in data_for_plots.columns:
                data_for_plots[default_flag_col_name_fallback] = 0
            num_in_subgroups_dict[default_flag_col_name_fallback.replace('subgroup_flag_', '')] = 0
            print(f"  {subgroup_definition_description_dynamic}")

    # --- RMSE Comparison Statistic Calculation ---
    # This section will now use the populated subgroup_configurations
    subgroup_rmse_comparison_stats = {} # Initialize here
    if method_name != "PCA" and beta_global_for_rmse is not None and not beta_local_df.empty and not local_intercepts_s.isnull().all():
        X_for_pred_np = data_for_plots[component_cols].values
        Y_true_np = data_for_plots[outcome_var_y].values.astype(float)

        global_intercept_val = beta_global_for_rmse[0]
        global_slopes_np = beta_global_for_rmse[1:]

        Y_pred_global_np = np.full_like(Y_true_np, np.nan)
        valid_X_rows_mask = ~np.isnan(X_for_pred_np).any(axis=1) # Rows in X that are not all NaN
        # Apply predictions only where X is valid
        Y_pred_global_np[valid_X_rows_mask] = global_intercept_val + X_for_pred_np[valid_X_rows_mask] @ global_slopes_np

        Y_pred_local_np = np.full_like(Y_true_np, np.nan)
        for i in range(len(data_for_plots)):
            if not valid_X_rows_mask[i]: continue # Skip if X features for this row are NaN

            current_local_intercept = local_intercepts_s.iloc[i]
            current_local_slopes = beta_local_df.iloc[i].values

            if pd.notna(current_local_intercept) and not np.isnan(current_local_slopes).any():
                Y_pred_local_np[i] = current_local_intercept + X_for_pred_np[i] @ current_local_slopes

        if not subgroup_configurations: # Check if list is empty
            pass # No subgroups to process
        else:
            print(f"\n  Calculating RMSE comparison for {len(subgroup_configurations)} subgroup configuration(s):")

            for sg_config in subgroup_configurations:
                sg_name = sg_config['name']
                sg_flag_col = sg_config['flag_column']

                if sg_flag_col not in data_for_plots.columns:
                    print(f"    Warning: Subgroup flag column {sg_flag_col} not found. Skipping RMSE comparison for {sg_name}.")
                    continue

                current_subgroup_size = data_for_plots[sg_flag_col].sum()
                if current_subgroup_size == 0: # Check if subgroup is empty based on the flag column
                    subgroup_rmse_comparison_stats[sg_name] = {
                        'rmse_global_in_subgroup': np.nan, 'rmse_local_in_subgroup': np.nan,
                        'diff_rmse_in_subgroup': np.nan, 'rmse_global_out_subgroup': np.nan,
                        'rmse_local_out_subgroup': np.nan, 'diff_rmse_out_subgroup': np.nan,
                        'increase_in_rmse_diff_for_subgroup': np.nan,
                        'n_in_subgroup_rmse_calc': 0,
                        'n_out_subgroup_rmse_calc': np.sum(pd.notna(Y_true_np[data_for_plots[sg_flag_col] == 0]) & pd.notna(Y_pred_global_np[data_for_plots[sg_flag_col] == 0]) & pd.notna(Y_pred_local_np[data_for_plots[sg_flag_col] == 0]))
                    }
                    continue

                subgroup_mask = (data_for_plots[sg_flag_col] == 1).values

                # In Subgroup
                y_true_sg = Y_true_np[subgroup_mask]
                y_pred_global_sg = Y_pred_global_np[subgroup_mask]
                y_pred_local_sg = Y_pred_local_np[subgroup_mask]

                valid_pred_sg = pd.notna(y_true_sg) & pd.notna(y_pred_global_sg) & pd.notna(y_pred_local_sg)
                n_valid_sg = np.sum(valid_pred_sg)
                if n_valid_sg > 0:
                    rmse_global_sg = np.sqrt(np.mean((y_true_sg[valid_pred_sg] - y_pred_global_sg[valid_pred_sg])**2))
                    rmse_local_sg = np.sqrt(np.mean((y_true_sg[valid_pred_sg] - y_pred_local_sg[valid_pred_sg])**2))
                    diff_rmse_sg = rmse_global_sg - rmse_local_sg 
                else:
                    rmse_global_sg, rmse_local_sg, diff_rmse_sg = np.nan, np.nan, np.nan

                # Not In Subgroup
                non_subgroup_mask = ~subgroup_mask
                y_true_non_sg = Y_true_np[non_subgroup_mask]
                y_pred_global_non_sg = Y_pred_global_np[non_subgroup_mask]
                y_pred_local_non_sg = Y_pred_local_np[non_subgroup_mask]

                valid_pred_non_sg = pd.notna(y_true_non_sg) & pd.notna(y_pred_global_non_sg) & pd.notna(y_pred_local_non_sg)
                n_valid_non_sg = np.sum(valid_pred_non_sg)
                if n_valid_non_sg > 0:
                    rmse_global_non_sg = np.sqrt(np.mean((y_true_non_sg[valid_pred_non_sg] - y_pred_global_non_sg[valid_pred_non_sg])**2))
                    rmse_local_non_sg = np.sqrt(np.mean((y_true_non_sg[valid_pred_non_sg] - y_pred_local_non_sg[valid_pred_non_sg])**2))
                    diff_rmse_non_sg = rmse_global_non_sg - rmse_local_non_sg
                else:
                    rmse_global_non_sg, rmse_local_non_sg, diff_rmse_non_sg = np.nan, np.nan, np.nan

                increase_in_rmse_diff_for_sg = diff_rmse_sg - diff_rmse_non_sg if pd.notna(diff_rmse_sg) and pd.notna(diff_rmse_non_sg) else np.nan

                subgroup_rmse_comparison_stats[sg_name] = {
                    'rmse_global_in_subgroup': rmse_global_sg,
                    'rmse_local_in_subgroup': rmse_local_sg,
                    'diff_rmse_in_subgroup': diff_rmse_sg,
                    'rmse_global_out_subgroup': rmse_global_non_sg,
                    'rmse_local_out_subgroup': rmse_local_non_sg,
                    'diff_rmse_out_subgroup': diff_rmse_non_sg,
                    'increase_in_rmse_diff_for_subgroup': increase_in_rmse_diff_for_sg,
                    'n_in_subgroup_rmse_calc': n_valid_sg,
                    'n_out_subgroup_rmse_calc': n_valid_non_sg
                }
                # Skipped verbose print statements for brevity in analysis log
                # print(f"    RMSE Comparison for Subgroup '{sg_name}':")
                # print(f"      Benefit of Local Model in Subgroup (N valid: {n_valid_sg}): {diff_rmse_sg:.4f}")
                # print(f"      Benefit of Local Model Out of Subgroup (N valid: {n_valid_non_sg}): {diff_rmse_non_sg:.4f}")
                # print(f"      Increase in Local Model Benefit for Subgroup: {increase_in_rmse_diff_for_sg:.4f}")
    else:
        if not (method_name != "PCA"): print("Skipping RMSE comparison (Method is PCA).")
        if not (beta_global_for_rmse is not None): print("Skipping RMSE comparison (Global betas for RMSE not available).")
        if beta_local_df.empty : print("Skipping RMSE comparison (Local betas df is empty).")
        if local_intercepts_s.isnull().all(): print("Skipping RMSE comparison (Local intercepts series is all NaN).")


    # --- Plotting Logic (remains mostly the same, ensure sg_config is from the updated list) ---
    if is_main_method and subgroup_configurations:
        for sg_config in subgroup_configurations:
            current_sg_name = sg_config['name']
            current_sg_flag_col = sg_config['flag_column']
            current_sg_description = sg_config['description']
            num_in_current_sg = data_for_plots[current_sg_flag_col].sum()
            plot_analysis_suffix = f"{analysis_suffix}_{current_sg_name}" # Use this for unique plot names
            ld_origin_for_current_sg = sg_config.get('ld_name_origin')

            print(f"\n--- Generating Plots for Subgroup: {current_sg_name} ({method_name} - {analysis_suffix}) ---")
            print(f"Description: {current_sg_description}")
            print(f"Number of samples in this subgroup: {num_in_current_sg}")

            #if num_in_current_sg == 0 or (data_for_plots[current_sg_flag_col] == 0).sum() == 0 :
            #    print(f"Skipping plots for subgroup {current_sg_name} as it has no members or no non-members.")
            #    continue

            plot_analysis_suffix = f"{analysis_suffix}_{current_sg_name}"
            plot_type_coeffs = current_analysis_params['beta_coefficient_plot_type']
            if plot_type_coeffs == "Absolute Coefficients" and not beta_local_df.empty:
                color_data_for_plot = beta_local_df
                cbar_title = "Local Coef"
            elif not beta_global_diff_df.empty : 
                color_data_for_plot = beta_global_diff_df
                cbar_title = "OLS Coef Diff"
            elif not beta_local_df.empty: 
                 color_data_for_plot = beta_local_df
                 cbar_title = "Local Coef (Fallback)"
            elif not summary_df_sm.empty and 'const' in summary_df_sm.index: # Check if summary_df_sm is valid
                # Ensure global_coeffs_no_const aligns with component_cols if using them for columns
                global_coeffs_for_plot = []
                if param_names_sm and len(param_names_sm) > 1 : # If OLS names like 'x1', 'x2' exist
                    ols_comp_names_in_summary = param_names_sm[1:] # These are 'x1', 'x2', ...
                    # We need to pick coefficients from summary_df_sm that correspond to component_cols
                    # This requires mapping component_cols to ols_comp_names_in_summary
                    temp_coeffs = []
                    for orig_comp_name in component_cols:
                        try:
                            idx_in_orig = component_cols.index(orig_comp_name)
                            if idx_in_orig < len(ols_comp_names_in_summary):
                                ols_name = ols_comp_names_in_summary[idx_in_orig]
                                if ols_name in summary_df_sm.index:
                                    temp_coeffs.append(summary_df_sm.loc[ols_name, 'Coefficient'])
                                else:
                                    temp_coeffs.append(np.nan) # OLS name not in summary (should not happen if mapping is correct)
                            else:
                                temp_coeffs.append(np.nan) # Original component not covered by OLS names
                        except ValueError: # orig_comp_name not in component_cols (should not happen)
                             temp_coeffs.append(np.nan)
                    global_coeffs_for_plot = pd.Series(temp_coeffs, index=component_cols)

                if not global_coeffs_for_plot.empty and not global_coeffs_for_plot.isnull().all():
                    color_data_for_plot = pd.DataFrame(np.tile(global_coeffs_for_plot.values, (len(data_for_plots), 1)), 
                                                   columns=component_cols, index=data_for_plots.index)
                    cbar_title = "Global OLS Coef"
                else:
                    color_data_for_plot = pd.DataFrame(np.nan, index=data_for_plots.index, columns=component_cols)
                    cbar_title = "N/A (No Global Coeffs)"
            else:
                color_data_for_plot = pd.DataFrame(np.nan, index=data_for_plots.index, columns=component_cols)
                cbar_title = "N/A (No Coeffs)"

            color_data_for_plot = color_data_for_plot.reindex(data_for_plots.index) 

            percentile_val = 0.1 
            if not color_data_for_plot.empty and color_data_for_plot.select_dtypes(include=np.number).size > 0:
                flat_abs_coeffs = np.abs(color_data_for_plot.select_dtypes(include=np.number).values.flatten())
                flat_abs_coeffs = flat_abs_coeffs[~np.isnan(flat_abs_coeffs)] 
                if flat_abs_coeffs.size > 0:
                     percentile_val = np.percentile(flat_abs_coeffs, 95)
            percentile_val = max(percentile_val, 1e-6) 
            cmin_val, cmax_val = -percentile_val, percentile_val
            if cmin_val == cmax_val: cmin_val, cmax_val = -0.1, 0.1 

            num_comp_dims_plot = len(component_cols)
            if num_comp_dims_plot > 0:
                num_cols_fig = min(2, num_comp_dims_plot)
                num_rows_fig = (num_comp_dims_plot + num_cols_fig - 1) // num_cols_fig
                fig_comp_coeffs = make_subplots(rows=num_rows_fig, cols=num_cols_fig, subplot_titles=component_cols)
                train_indicator = data_for_plots['train'] if 'train' in data_for_plots.columns else pd.Series(1, index=data_for_plots.index)
                for i, comp_dim in enumerate(component_cols):
                    row_idx, col_idx = (i // num_cols_fig) + 1, (i % num_cols_fig) + 1
                    current_color_series = color_data_for_plot[comp_dim] if comp_dim in color_data_for_plot else pd.Series(np.nan, index=data_for_plots.index)
                    fig_comp_coeffs.add_trace(go.Scatter(x=data_for_plots[comp_dim], y=data_for_plots[outcome_var_y], mode="markers",
                                                         marker=dict(color=current_color_series, colorscale="Spectral", colorbar=dict(title=cbar_title),
                                                                     cmin=cmin_val, cmax=cmax_val, size=10,
                                                                     line=dict(color=np.where(train_indicator == 0, "red", "black"), 
                                                                               width=np.where(train_indicator == 0, 1.5, 0.5))),
                                                         name=f"{comp_dim}"), row=row_idx, col=col_idx)
                fig_comp_coeffs.update_layout(height=max(400, 300*num_rows_fig), title_text=f"{method_name} Comp. vs Outcome ({plot_analysis_suffix}, Color: {cbar_title})", showlegend=False, plot_bgcolor='rgba(0,0,0,0)')
                fig_comp_coeffs.update_yaxes(showgrid=True, gridcolor='lightgray', gridwidth=0.5, zeroline=True, zerolinewidth=0.5, zerolinecolor='lightgray')
                if 'train_only' in plot_analysis_suffix:
                    fig_comp_coeffs.write_image(os.path.join(OUTPUT_DIR, f"Figure2_latent_coeffs_{method_name}{plot_analysis_suffix}.pdf"))
                    print(f"[6] Saved Figure2_latent_coeffs_{method_name}{plot_analysis_suffix}.pdf")
                else:
                    pass



                cdiff_plot_highlight_data = beta_global_diff_df if not beta_global_diff_df.empty else pd.DataFrame(np.nan, index=data_for_plots.index, columns=component_cols)
                cdiff_plot_highlight_data = cdiff_plot_highlight_data.reindex(data_for_plots.index)
                cdiff_percentile = 0.1 
                if not cdiff_plot_highlight_data.empty and cdiff_plot_highlight_data.select_dtypes(include=np.number).size > 0:
                    flat_abs_diffs = np.abs(cdiff_plot_highlight_data.select_dtypes(include=np.number).values.flatten())
                    flat_abs_diffs = flat_abs_diffs[~np.isnan(flat_abs_diffs)]
                    if flat_abs_diffs.size > 0:
                        cdiff_percentile = np.percentile(flat_abs_diffs, 95)
                cdiff_percentile = max(cdiff_percentile, 1e-6)
                cmin_highlight, cmax_highlight = -cdiff_percentile, cdiff_percentile
                if cmin_highlight == cmax_highlight: cmin_highlight, cmax_highlight = -0.1, 0.1

                fig_subgroup_highlight = make_subplots(rows=num_rows_fig, cols=num_cols_fig, subplot_titles=component_cols)
                for i, comp_dim in enumerate(component_cols):
                    row_idx, col_idx = (i // num_cols_fig) + 1, (i % num_cols_fig) + 1
                    current_color_sg_hl = cdiff_plot_highlight_data[comp_dim] if comp_dim in cdiff_plot_highlight_data else 'lightgrey'
                    fig_subgroup_highlight.add_trace(go.Scatter(x=data_for_plots[comp_dim], y=data_for_plots[outcome_var_y], mode="markers",
                                                                 marker=dict(color=current_color_sg_hl, 
                                                                             colorscale="Spectral", cmin=cmin_highlight, cmax=cmax_highlight, size=8,
                                                                             line=dict(color=np.where(train_indicator==0, 'rgba(255,0,0,0.3)', 'rgba(128,128,128,0.3)'), width=0.5)),
                                                                 opacity=0.3, name=f"All Data {comp_dim}"), row=row_idx, col=col_idx)
                    subgroup_data_plot = data_for_plots[data_for_plots[current_sg_flag_col] == 1]
                    if not subgroup_data_plot.empty:
                        subgroup_color_sg_hl = cdiff_plot_highlight_data.loc[subgroup_data_plot.index, comp_dim] if comp_dim in cdiff_plot_highlight_data else 'blue'
                        subgroup_train_indicator = subgroup_data_plot['train'] if 'train' in subgroup_data_plot.columns else pd.Series(1, index=subgroup_data_plot.index)
                        fig_subgroup_highlight.add_trace(go.Scatter(x=subgroup_data_plot[comp_dim], y=subgroup_data_plot[outcome_var_y], mode="markers",
                                                                     marker=dict(color=subgroup_color_sg_hl, 
                                                                                 colorscale="Spectral", cmin=cmin_highlight, cmax=cmax_highlight, size=10,
                                                                                 line=dict(color=np.where(subgroup_train_indicator==0, 'red', 'black'), width=1.5)),
                                                                     opacity=1.0, name=f"Subgroup {comp_dim}"), row=row_idx, col=col_idx)
                fig_subgroup_highlight.update_layout(height=max(400, 300*num_rows_fig), title_text=f"{method_name} Comp. Space with Subgroup Highlighting ({plot_analysis_suffix})", showlegend=False, plot_bgcolor='rgba(0,0,0,0)')
                fig_subgroup_highlight.update_yaxes(showgrid=True, gridcolor='lightgray', gridwidth=0.5, zeroline=True, zerolinewidth=0.5, zerolinecolor='lightgray')
                if 'train_only' in plot_analysis_suffix:
                    fig_subgroup_highlight.write_image(os.path.join(OUTPUT_DIR, f"Figure3or4_A_latent_subgroup_highlight_{method_name}{plot_analysis_suffix}.pdf"))
                    print(f"[7] Saved Figure3or4_A_latent_subgroup_highlight_{method_name}{plot_analysis_suffix}.pdf")
                else:
                    pass

                if num_in_current_sg > 0 and (data_for_plots[current_sg_flag_col] == 0).sum() > 0: 
                    comp_s1 = data_for_plots.loc[data_for_plots[current_sg_flag_col] == 1, component_cols]
                    y_s1 = data_for_plots.loc[data_for_plots[current_sg_flag_col] == 1, outcome_var_y].astype(float)
                    comp_s0 = data_for_plots.loc[data_for_plots[current_sg_flag_col] == 0, component_cols]
                    y_s0 = data_for_plots.loc[data_for_plots[current_sg_flag_col] == 0, outcome_var_y].astype(float)
                    min_samples_for_ols = len(component_cols) + 2 
                    if len(comp_s1) >= min_samples_for_ols and len(comp_s0) >= min_samples_for_ols:
                        valid_s1 = ~np.isnan(comp_s1).any(axis=1) & ~np.isnan(y_s1)
                        valid_s0 = ~np.isnan(comp_s0).any(axis=1) & ~np.isnan(y_s0)
                        if valid_s1.sum() >= min_samples_for_ols and valid_s0.sum() >= min_samples_for_ols:
                            m_s1_comp = sm.OLS(y_s1[valid_s1], sm.add_constant(comp_s1[valid_s1])).fit()
                            m_s0_comp = sm.OLS(y_s0[valid_s0], sm.add_constant(comp_s0[valid_s0])).fit()
                            plot_vars_comp = ['const'] + component_cols
                            s1_params = pd.Series(m_s1_comp.params, index=m_s1_comp.model.exog_names).reindex(plot_vars_comp)

                            s1_confint_df = m_s1_comp.conf_int() # Get the DataFrame
                            s1_confint_lower = s1_confint_df[0].reindex(plot_vars_comp) # Access column 0 (lower CI)
                            s1_confint_upper = s1_confint_df[1].reindex(plot_vars_comp) # Access column 1 (upper CI)

                            s0_params = pd.Series(m_s0_comp.params, index=m_s0_comp.model.exog_names).reindex(plot_vars_comp)
                            s0_confint_df = m_s0_comp.conf_int()
                            s0_confint_lower = s0_confint_df[0].reindex(plot_vars_comp)
                            s0_confint_upper = s0_confint_df[1].reindex(plot_vars_comp)
                            sum_comb_comp = pd.DataFrame({
                                'Variable': plot_vars_comp, 
                                'Coefficient_1': s1_params.values, 'ConfInt Lower_1': s1_confint_lower.values, 'ConfInt Upper_1': s1_confint_upper.values,
                                'Coefficient_0': s0_params.values, 'ConfInt Lower_0': s0_confint_lower.values, 'ConfInt Upper_0': s0_confint_upper.values
                            }).dropna(subset=['Coefficient_1', 'Coefficient_0'], how='all')
                            if not sum_comb_comp.empty:
                                fig_f_comp = go.Figure()
                                fig_f_comp.add_trace(go.Scatter(x=sum_comb_comp['Coefficient_1'], y=sum_comb_comp['Variable'], error_x=dict(type='data', symmetric=False, array=sum_comb_comp['ConfInt Upper_1']-sum_comb_comp['Coefficient_1'], arrayminus=sum_comb_comp['Coefficient_1']-sum_comb_comp['ConfInt Lower_1']), mode='markers', name='Subgroup (Flag=1)', marker_color='#276DB0'))
                                fig_f_comp.add_trace(go.Scatter(x=sum_comb_comp['Coefficient_0'], y=sum_comb_comp['Variable'], error_x=dict(type='data', symmetric=False, array=sum_comb_comp['ConfInt Upper_0']-sum_comb_comp['Coefficient_0'], arrayminus=sum_comb_comp['Coefficient_0']-sum_comb_comp['ConfInt Lower_0']), mode='markers', name='Non-Subgroup (Flag=0)', marker_color='#008000'))
                                fig_f_comp.add_vline(x=0, line_dash="dash", line_color="grey")
                                fig_f_comp.update_layout(title=f'Coefficients by Subgroup ({method_name} Components, {plot_analysis_suffix})', 
                                                       xaxis_title='Coefficient Value', 
                                                       yaxis=dict(title='Component', showgrid=True, gridcolor='lightgray', gridwidth=0.5, zeroline=True, zerolinewidth=0.5, zerolinecolor='lightgray'), 
                                                       legend_title_text='Group',
                                                       template="plotly_white",
                                                       plot_bgcolor='rgba(0,0,0,0)')
                                # Skipped for brevity of output
                                #fig_f_comp.write_image(os.path.join(OUTPUT_DIR, f"Figure3or4_B_forest_plot_components_{method_name}{plot_analysis_suffix}.pdf"))
                                #print(f"Saved Figure3or4_B_forest_plot_components_{method_name}{plot_analysis_suffix}.pdf")
                            else: print(f"Skipped forest_plot_components for {current_sg_name} as no valid coefficients for subgroups.")
                        else: print(f"Skipped forest_plot_components for {current_sg_name} due to insufficient samples in subgroups after NaN removal.")
                    else: print(f"Skipped forest_plot_components for {current_sg_name} due to insufficient samples in subgroups.")

                    # --- Interaction plots with original features ---
                    top_n_orig = analysis_params['top_n_vars_forest_plot']
                    vars_for_f = []
                    interaction_plot_ld_source_message = "No specific LD source for interaction vars."

                    # Use the LD that originated the current subgroup for selecting vars_for_f
                    if ld_origin_for_current_sg:
                        if ld_origin_for_current_sg in raw_top_ttest_vars_for_stability and \
                           raw_top_ttest_vars_for_stability[ld_origin_for_current_sg]:
                            vars_for_f = raw_top_ttest_vars_for_stability[ld_origin_for_current_sg][:top_n_orig]
                            interaction_plot_ld_source_message = f"using top T-Test variables for its originating LD: {ld_origin_for_current_sg}"
                        elif ld_origin_for_current_sg in all_top_mi_vars_for_subgroup_stability and \
                             all_top_mi_vars_for_subgroup_stability[ld_origin_for_current_sg]:
                            vars_for_f = all_top_mi_vars_for_subgroup_stability[ld_origin_for_current_sg][:top_n_orig]
                            interaction_plot_ld_source_message = f"using top MI variables for its originating LD: {ld_origin_for_current_sg} (T-Test fallback)"
                        else:
                            vars_for_f = [] # Ensure it's an empty list
                            interaction_plot_ld_source_message = f"no T-Test or MI variables found for its originating LD: {ld_origin_for_current_sg}."
                    else:
                        # Fallback if ld_origin_for_current_sg is not available (e.g. for non-dynamic SGs if you add them)
                        # Or if you specifically want to use the overall largest SG's LD context
                        if identified_target_ld_orig_name: # Fallback to the LD of the largest dynamic subgroup
                            if identified_target_ld_orig_name in raw_top_ttest_vars_for_stability and \
                               raw_top_ttest_vars_for_stability[identified_target_ld_orig_name]:
                                vars_for_f = raw_top_ttest_vars_for_stability[identified_target_ld_orig_name][:top_n_orig]
                                interaction_plot_ld_source_message = f"using top T-Test variables for globally identified LD: {identified_target_ld_orig_name} (fallback for SG '{current_sg_name}')"
                            elif identified_target_ld_orig_name in all_top_mi_vars_for_subgroup_stability and \
                                all_top_mi_vars_for_subgroup_stability[identified_target_ld_orig_name]:
                                vars_for_f = all_top_mi_vars_for_subgroup_stability[identified_target_ld_orig_name][:top_n_orig]
                                interaction_plot_ld_source_message = f"using top MI variables for globally identified LD: {identified_target_ld_orig_name} (fallback for SG '{current_sg_name}')"
                            else:
                                vars_for_f = []
                                interaction_plot_ld_source_message = f"no T-Test or MI variables found for globally identified LD: {identified_target_ld_orig_name} (fallback for SG '{current_sg_name}')"
                        else:
                            vars_for_f = []
                            interaction_plot_ld_source_message = f"neither originating LD ('{ld_origin_for_current_sg}') nor global LD ('{identified_target_ld_orig_name}') available/valid for interaction vars for SG '{current_sg_name}'."

                    print(f"  For subgroup '{current_sg_name}', interaction plot variables selection: {interaction_plot_ld_source_message}")

                    # Ensure vars_for_f contains valid column names present in the data
                    vars_for_f = [v for v in vars_for_f if v in feature_cols and v in data_for_plots.columns and pd.notna(v)] # Added notna check

                    if vars_for_f:
                        X_comb_orig_df = data_for_plots[vars_for_f].copy()
                        y_comb_orig = data_for_plots[outcome_var_y].astype(float)
                        sg_flag_orig_series = data_for_plots[current_sg_flag_col] 
                        sum_f_orig_list = []
                        for var in vars_for_f:
                            X_var_int = pd.DataFrame(index=X_comb_orig_df.index)
                            X_var_int[var] = X_comb_orig_df[var]
                            X_var_int[f'{var}_interaction'] = X_comb_orig_df[var] * sg_flag_orig_series
                            X_var_int['subgroup_flag_main_effect'] = sg_flag_orig_series 
                            X_var_int_sm = sm.add_constant(X_var_int, has_constant='add')
                            model_cols = ['const', var, f'{var}_interaction', 'subgroup_flag_main_effect']
                            # model_cols = ['const', var, f'{var}_interaction']
                            current_data_for_model = pd.concat([y_comb_orig, X_var_int_sm[model_cols]], axis=1).dropna()
                            if len(current_data_for_model) > len(model_cols): 
                                try:
                                    m_int = sm.OLS(current_data_for_model[outcome_var_y], current_data_for_model[model_cols]).fit()
                                    conf_int_df = pd.DataFrame(m_int.conf_int(), index=m_int.model.exog_names, columns=[0,1])
                                    sum_f_orig_list.append({
                                        'Variable': var,
                                        'Coefficient_MainEffect': m_int.params.get(var, np.nan),
                                        'ConfInt Lower_MainEffect': conf_int_df.loc[var, 0] if var in conf_int_df.index else np.nan,
                                        'ConfInt Upper_MainEffect': conf_int_df.loc[var, 1] if var in conf_int_df.index else np.nan,
                                        'Coefficient_Interaction': m_int.params.get(f'{var}_interaction', np.nan),
                                        'ConfInt Lower_Interaction': conf_int_df.loc[f'{var}_interaction', 0] if f'{var}_interaction' in conf_int_df.index else np.nan,
                                        'ConfInt Upper_Interaction': conf_int_df.loc[f'{var}_interaction', 1] if f'{var}_interaction' in conf_int_df.index else np.nan,
                                    })
                                except Exception as e: print(f"Error fitting OLS for original feature {var} (subgroup {current_sg_name}): {e}")
                        if sum_f_orig_list:
                            sum_f_orig_df = pd.DataFrame(sum_f_orig_list).dropna(subset=['Coefficient_MainEffect', 'Coefficient_Interaction'], how='all')
                            if not sum_f_orig_df.empty:
                                fig_f_orig = go.Figure()
                                y_labels = sum_f_orig_df['Variable'].tolist()
                                y_num = np.arange(len(y_labels))
                                fig_f_orig.add_trace(go.Scatter(x=sum_f_orig_df['Coefficient_MainEffect'], y=y_num - 0.1, error_x=dict(type='data',symmetric=False,array=sum_f_orig_df['ConfInt Upper_MainEffect']-sum_f_orig_df['Coefficient_MainEffect'],arrayminus=sum_f_orig_df['Coefficient_MainEffect']-sum_f_orig_df['ConfInt Lower_MainEffect']),mode='markers',name='Main Effect',marker_color='#276DB0'))
                                fig_f_orig.add_trace(go.Scatter(x=sum_f_orig_df['Coefficient_Interaction'], y=y_num + 0.1, error_x=dict(type='data',symmetric=False,array=sum_f_orig_df['ConfInt Upper_Interaction']-sum_f_orig_df['Coefficient_Interaction'],arrayminus=sum_f_orig_df['Coefficient_Interaction']-sum_f_orig_df['ConfInt Lower_Interaction']),mode='markers',name='Interaction (Subgroup vs Non-Subgroup)',marker_color='#008000'))
                                fig_f_orig.add_vline(x=0, line_dash="dash", line_color="grey")
                                fig_f_orig.update_layout(title=f'Original Feature Effects by Subgroup ({method_name} - {plot_analysis_suffix})', 
                                                       yaxis=dict(ticktext=y_labels, tickvals=y_num, title='Feature', showgrid=True, gridcolor='lightgray', gridwidth=0.5, zeroline=True, zerolinewidth=0.5, zerolinecolor='lightgray'), 
                                                       xaxis_title='Coefficient Value', 
                                                       legend_title_text='Effect Type',
                                                       template="plotly_white",
                                                       plot_bgcolor='rgba(0,0,0,0)',
                                                       height=max(400, len(y_labels) * 45 + 100))
                                # If 'train_only' is in plot_analysis_suffix string
                                if 'train_only' in plot_analysis_suffix:
                                    fig_f_orig.write_image(os.path.join(OUTPUT_DIR, f"Figure3or4_B_forest_plot_original_features_{method_name}{plot_analysis_suffix}.pdf"))
                                    print(f"[8] Saved Figure3or4_B_forest_plot_original_features_{method_name}{plot_analysis_suffix}.pdf")
                                else:
                                    fig_f_orig.write_image(os.path.join(OUTPUT_DIR, f"Figure5_forest_plot_original_features_{method_name}_test_{plot_analysis_suffix}.pdf"))
                                    print(f"[12] Saved Figure5_forest_plot_original_features_{method_name}_test_{plot_analysis_suffix}.pdf")
                            else: print(f"Skipped forest_plot_original_features for {current_sg_name} as no valid interaction models.")
                    else: print(f"Skipped forest_plot_original_features for {current_sg_name} as no top MI variables found for {selected_comp_for_reporting}.")


                    # --- Interaction Plot with Continuous Local Coefficients (Main + Interaction Effects) ---
                    # Use ld_origin_for_current_sg for selecting the local coefficients series
                    continuous_interaction_ld_to_use = ld_origin_for_current_sg

                    if not continuous_interaction_ld_to_use and identified_target_ld_orig_name : # Fallback
                        continuous_interaction_ld_to_use = identified_target_ld_orig_name
                        print(f"    For continuous LC interaction plot for SG '{current_sg_name}', falling back to use LC from globally largest SG's LD: {identified_target_ld_orig_name}")

                    if continuous_interaction_ld_to_use and \
                       continuous_interaction_ld_to_use in beta_local_df.columns and \
                       vars_for_f: # vars_for_f are now also potentially tied to ld_origin_for_current_sg

                        print(f"  Generating Interaction Plots (Main & Interaction) for SG '{current_sg_name}': Original Features x Continuous Local Coefficients for LD: {continuous_interaction_ld_to_use} ---")

                        local_coeffs_for_interaction = beta_local_df[continuous_interaction_ld_to_use].copy()
                        sanitized_ld_name = continuous_interaction_ld_to_use.replace(' ', '_').replace('.', '_')
                        lc_main_effect_term_name = f"lc_{sanitized_ld_name}" # Defined here

                        continuous_interaction_model_results = [] # Will store dicts for each effect type

                        for original_feature_var in vars_for_f:
                            if original_feature_var not in data_for_plots.columns:
                                # ... (skip message)
                                continue
                            if data_for_plots[original_feature_var].isnull().all() or local_coeffs_for_interaction.isnull().all():
                                # ... (skip message)
                                continue

                            X_interaction_df = pd.DataFrame(index=data_for_plots.index)
                            X_interaction_df[original_feature_var] = data_for_plots[original_feature_var]
                            X_interaction_df[lc_main_effect_term_name] = local_coeffs_for_interaction

                            interaction_product_values = data_for_plots[original_feature_var].values * local_coeffs_for_interaction.values
                            current_interaction_term_name = f"{original_feature_var}_x_{lc_main_effect_term_name}"
                            X_interaction_df[current_interaction_term_name] = interaction_product_values

                            X_interaction_design_matrix = sm.add_constant(X_interaction_df, has_constant='add')
                            y_target_variable = data_for_plots[outcome_var_y].astype(float)
                            model_data_for_fit = pd.concat([y_target_variable, X_interaction_design_matrix], axis=1).dropna()

                            min_samples_for_fit = X_interaction_design_matrix.shape[1] + 1
                            if model_data_for_fit.shape[0] > min_samples_for_fit:
                                try:
                                    condition_number = np.linalg.cond(model_data_for_fit[X_interaction_design_matrix.columns].astype(float))
                                    if condition_number > 1e8: 
                                        # ... (skip multicollinearity message)
                                        continue

                                    ols_model_continuous_interaction = sm.OLS(model_data_for_fit[outcome_var_y], 
                                                                              model_data_for_fit[X_interaction_design_matrix.columns])
                                    results_continuous_interaction = ols_model_continuous_interaction.fit()
                                    conf_int_interaction_terms = pd.DataFrame(results_continuous_interaction.conf_int(), 
                                                                              index=results_continuous_interaction.model.exog_names, 
                                                                              columns=[0,1])

                                    # Store Main Effect of the original_feature_var
                                    if original_feature_var in results_continuous_interaction.params:
                                        continuous_interaction_model_results.append({
                                            'Original_Feature': original_feature_var,
                                            'Effect_Type': 'Main Effect', # Identifier for plotting
                                            'Coefficient': results_continuous_interaction.params.get(original_feature_var),
                                            'Lower_CI': conf_int_interaction_terms.loc[original_feature_var, 0],
                                            'Upper_CI': conf_int_interaction_terms.loc[original_feature_var, 1],
                                            'P_Value': results_continuous_interaction.pvalues.get(original_feature_var, np.nan)
                                        })

                                    # Store Interaction Effect
                                    if current_interaction_term_name in results_continuous_interaction.params:
                                        continuous_interaction_model_results.append({
                                            'Original_Feature': original_feature_var,
                                            'Effect_Type': 'Interaction Effect', # Identifier for plotting
                                            'Coefficient': results_continuous_interaction.params.get(current_interaction_term_name),
                                            'Lower_CI': conf_int_interaction_terms.loc[current_interaction_term_name, 0],
                                            'Upper_CI': conf_int_interaction_terms.loc[current_interaction_term_name, 1],
                                            'P_Value': results_continuous_interaction.pvalues.get(current_interaction_term_name, np.nan)
                                        })
                                except Exception as e:
                                    print(f"  Error fitting OLS for continuous interaction (main+int) with '{original_feature_var}': {e}")
                            else:
                                pass


                        if continuous_interaction_model_results:
                            summary_df_cont_interaction = pd.DataFrame(continuous_interaction_model_results)
                            summary_df_cont_interaction = summary_df_cont_interaction.dropna(subset=['Coefficient', 'Lower_CI', 'Upper_CI'])

                            if not summary_df_cont_interaction.empty:
                                fig_cont_interaction_grouped = go.Figure()

                                # Determine the order of features on Y-axis based on vars_for_f
                                ordered_features_for_plot = [f for f in vars_for_f if f in summary_df_cont_interaction['Original_Feature'].unique()]

                                if not ordered_features_for_plot:
                                    print(f"No features from vars_for_f found in continuous interaction model results. Skipping grouped plot.")
                                else:
                                    y_categories_map = {feature: i for i, feature in enumerate(ordered_features_for_plot)}
                                    y_offset_val = 0.15 # Adjust for visual separation of main and interaction effects

                                    # Filter data for main effects and map y-positions
                                    main_effects_data = summary_df_cont_interaction[summary_df_cont_interaction['Effect_Type'] == 'Main Effect'].copy()
                                    main_effects_data = main_effects_data[main_effects_data['Original_Feature'].isin(ordered_features_for_plot)]
                                    main_effects_data['y_pos'] = main_effects_data['Original_Feature'].map(y_categories_map)
                                    main_effects_data.dropna(subset=['y_pos'], inplace=True)


                                    # Filter data for interaction effects and map y-positions
                                    interaction_effects_data = summary_df_cont_interaction[summary_df_cont_interaction['Effect_Type'] == 'Interaction Effect'].copy()
                                    interaction_effects_data = interaction_effects_data[interaction_effects_data['Original_Feature'].isin(ordered_features_for_plot)]
                                    interaction_effects_data['y_pos'] = interaction_effects_data['Original_Feature'].map(y_categories_map)
                                    interaction_effects_data.dropna(subset=['y_pos'], inplace=True)

                                    if not main_effects_data.empty:
                                        fig_cont_interaction_grouped.add_trace(go.Scatter(
                                            x=main_effects_data['Coefficient'],
                                            y=main_effects_data['y_pos'].astype(float) - y_offset_val,
                                            error_x=dict(type='data', symmetric=False,
                                                         array=main_effects_data['Upper_CI'] - main_effects_data['Coefficient'],
                                                         arrayminus=main_effects_data['Coefficient'] - main_effects_data['Lower_CI']),
                                            mode='markers', name='Main Effect (Feature)', marker_color='#276DB0'
                                        ))

                                    if not interaction_effects_data.empty:
                                        fig_cont_interaction_grouped.add_trace(go.Scatter(
                                            x=interaction_effects_data['Coefficient'],
                                            y=interaction_effects_data['y_pos'].astype(float) + y_offset_val,
                                            error_x=dict(type='data', symmetric=False,
                                                         array=interaction_effects_data['Upper_CI'] - interaction_effects_data['Coefficient'],
                                                         arrayminus=interaction_effects_data['Coefficient'] - interaction_effects_data['Lower_CI']),
                                            mode='markers', name=f'Interaction (Feature x LC-{sanitized_ld_name})', marker_color='#008000'
                                        ))

                                    fig_cont_interaction_grouped.add_vline(x=0, line_dash="dash", line_color="grey")
                                    fig_cont_interaction_grouped.update_layout(
                                        title=f'Main & Interaction Effects with Continuous LC of {identified_target_ld_orig_name}<br>({method_name} - {plot_analysis_suffix})',
                                        yaxis=dict(ticktext=ordered_features_for_plot, 
                                                   tickvals=list(range(len(ordered_features_for_plot))), 
                                                   title='Original Feature', autorange="reversed", 
                                                   showgrid=True, gridcolor='lightgray', gridwidth=0.5, 
                                                   zeroline=True, zerolinewidth=0.5, zerolinecolor='lightgray'), # Align with other plots
                                        xaxis_title='Coefficient Value',
                                        legend_title_text='Effect Type',
                                        template="plotly_white",
                                        plot_bgcolor='rgba(0,0,0,0)',
                                        height=max(400, len(ordered_features_for_plot) * 45 + 100) # Increased height per feature
                                    )
                                    # Skipped saving for brevity of output
                                    #file_path_cont_int_grouped = os.path.join(OUTPUT_DIR, f"forest_plot_cont_interaction_grouped_{method_name}{plot_analysis_suffix}.pdf")
                                    #fig_cont_interaction_grouped.write_image(file_path_cont_int_grouped)
                                    #print(f"Saved grouped continuous interaction forest plot: {file_path_cont_int_grouped}")
                            else:
                                print(f"Skipped grouped continuous interaction forest plot for LD '{identified_target_ld_orig_name}' as no valid model results (after NaN drop).")
                        else:
                            print(f"No continuous interaction (main+int) models were successfully fit or yielded results for LD '{identified_target_ld_orig_name}'.")
                    else:
                        if not identified_target_ld_orig_name:
                            print("Skipping grouped continuous interaction plot: Target latent dimension not identified.")
                        elif identified_target_ld_orig_name not in beta_local_df.columns:
                            print(f"Skipping grouped continuous interaction plot: Local coefficients for {identified_target_ld_orig_name} not found.")
                        elif not vars_for_f:
                            print(f"Skipping grouped continuous interaction plot: No 'vars_for_f' selected.")


                if 'train' not in data_for_plots.columns:
                    print(f"Warning: 'train' column missing. Z-score profiles for {current_sg_name} cannot be generated accurately.")
                else:
                    train_data_for_pop_mean_plots = df_input_analysis[df_input_analysis['train'] == 1]
                    if not train_data_for_pop_mean_plots.empty and not train_data_for_pop_mean_plots[feature_cols].empty:
                        pop_means_z = train_data_for_pop_mean_plots[feature_cols].mean()
                        pop_std_z = train_data_for_pop_mean_plots[feature_cols].std().replace(0,1)
                        sg_current_train_data_z = data_for_plots[(data_for_plots[current_sg_flag_col] == 1) & (data_for_plots['train'] == 1)]
                        if not sg_current_train_data_z.empty and not sg_current_train_data_z[feature_cols].empty:
                            sg_means_z_train = sg_current_train_data_z[feature_cols].mean()
                            z_diff_train = (sg_means_z_train - pop_means_z) / pop_std_z
                            plot_z_train = pd.DataFrame({'Variable': z_diff_train.index, 'Z-score Difference': z_diff_train.values}).dropna()
                            if not plot_z_train.empty:
                                plot_z_train['Cluster'] = plot_z_train['Variable'].apply(lambda x: CLUSTER_NAMES_DICT.get(x, 'Other'))
                                cluster_m_z_train = plot_z_train.groupby('Cluster')['Z-score Difference'].mean().reset_index()
                                fig_z_train = px.bar(cluster_m_z_train, y='Cluster', x='Z-score Difference', orientation='h', color='Z-score Difference', color_continuous_scale='RdBu', range_color=analysis_params['z_profile_range'], title=f"Z-Score Diff (Subgroup Train vs Pop Train, {method_name} - {plot_analysis_suffix})")
                                fig_z_train.update_layout(height=max(400, len(cluster_m_z_train)*20 + 100), template="plotly_white", xaxis_range=analysis_params['z_profile_range'])
                                if 'train_only' in plot_analysis_suffix:
                                    fig_z_train.write_image(os.path.join(OUTPUT_DIR, f"Figure3or4_C_z_profile_{method_name}{plot_analysis_suffix}.pdf"))
                                    print(f"[9] Saved Figure3or4_C_z_profile_{method_name}{plot_analysis_suffix}.pdf")
                                else: 
                                    fig_z_train.write_image(os.path.join(OUTPUT_DIR, f"Figure5_z_profile_{method_name}{plot_analysis_suffix}.pdf"))
                                    print(f"[13] Saved Figure5_z_profile_{method_name}{plot_analysis_suffix}.pdf")

                        if 0 in data_for_plots['train'].unique(): 
                            sg_current_test_data_z = data_for_plots[(data_for_plots[current_sg_flag_col] == 1) & (data_for_plots['train'] == 0)]
                            if not sg_current_test_data_z.empty and not sg_current_test_data_z[feature_cols].empty:
                                sg_means_z_test = sg_current_test_data_z[feature_cols].mean()
                                z_diff_test = (sg_means_z_test - pop_means_z) / pop_std_z
                                plot_z_test = pd.DataFrame({'Variable': z_diff_test.index, 'Z-score Difference': z_diff_test.values}).dropna()
                                if not plot_z_test.empty:
                                    plot_z_test['Cluster'] = plot_z_test['Variable'].apply(lambda x: CLUSTER_NAMES_DICT.get(x, 'Other'))
                                    cluster_m_z_test = plot_z_test.groupby('Cluster')['Z-score Difference'].mean().reset_index()
                                    fig_z_test = px.bar(cluster_m_z_test, y='Cluster', x='Z-score Difference', orientation='h', color='Z-score Difference', color_continuous_scale='RdBu', range_color=analysis_params['z_profile_range'], title=f"Z-Score Diff (Subgroup Test vs Pop Train, {method_name} - {plot_analysis_suffix})")
                                    fig_z_test.update_layout(height=max(400, len(cluster_m_z_test)*20 + 100), template="plotly_white", xaxis_range=analysis_params['z_profile_range'])
                                    #fig_z_test.write_image(os.path.join(OUTPUT_DIR, f"z_profile_test_in_{method_name}{plot_analysis_suffix}.pdf"))
                                    #print(f"Saved z_profile_test_in_{method_name}{plot_analysis_suffix}.pdf")
                    else: print(f"Skipped Z-score profile plots for {current_sg_name} as no training data available for population means or no features.")
            else: 
                 print(f"Skipping component-based plots for {current_sg_name} as there are no component columns.")
    elif not subgroup_configurations and is_main_method: 
        print("No subgroup configurations defined for plotting.")
    elif not is_main_method:
        print("Not the main method, detailed plots are skipped.")

    subgroup_flag_cols_to_return = [col for col in data_for_plots.columns if col.startswith('subgroup_flag_')]
    cols_for_return_df = ['train'] if 'train' in data_for_plots.columns else []
    cols_for_return_df += subgroup_flag_cols_to_return + feature_cols
    seen = set()
    unique_cols_for_return_df = [x for x in cols_for_return_df if not (x in seen or seen.add(x))]

    return_dict = {
            "global_ols_coeffs_df": summary_df_sm,
            "local_betas_df": beta_local_df, # Slopes only
            "local_intercepts_series": local_intercepts_s, # Store local intercepts
            "beta_differences_df": beta_global_diff_df,
            "top_mi_variables_raw": raw_top_mi_vars_for_stability, 
            "top_ttest_variables_raw": raw_top_ttest_vars_for_stability, 
            "all_ttest_stats_by_component": all_ttest_results_raw_by_component,
            "num_in_subgroups_dict": num_in_subgroups_dict, # Contains counts and descriptions
            "subgroup_patient_ids_dict": subgroup_patient_ids_dict,
            "subgroup_rmse_comparison_stats": subgroup_rmse_comparison_stats, # Add new stats
            "data_with_subgroups_and_train_flag": data_for_plots[unique_cols_for_return_df].copy() # Ensure unique_cols_for_return_df is defined earlier
        }
    # ... (ensure unique_cols_for_return_df is correctly defined) ...
    subgroup_flag_cols_to_return = [col for col in data_for_plots.columns if col.startswith('subgroup_flag_')]
    cols_for_return_df = ['train'] if 'train' in data_for_plots.columns else []
    cols_for_return_df += subgroup_flag_cols_to_return + feature_cols # feature_cols might not be what you want to return always, depends on use case.
    # For simplicity, let's just keep it as is.
    if 'Y' in data_for_plots.columns and 'Y' not in cols_for_return_df : cols_for_return_df.append('Y') # Add outcome if not there
    if component_cols: # Add component columns if they exist
        for ccol in component_cols:
            if ccol in data_for_plots.columns and ccol not in cols_for_return_df:
                cols_for_return_df.append(ccol)

    seen = set()
    unique_cols_for_return_df = [x for x in cols_for_return_df if x in data_for_plots.columns and not (x in seen or seen.add(x))]
    return_dict["data_with_subgroups_and_train_flag"] = data_for_plots[unique_cols_for_return_df].copy()


    if 'train' not in return_dict["data_with_subgroups_and_train_flag"].columns: 
        temp_df_ret = return_dict["data_with_subgroups_and_train_flag"]
        temp_df_ret['train'] = 1 # Default to 1 if missing (e.g. if input data was only train)
        final_cols_for_return = ['train'] + [c for c in unique_cols_for_return_df if c != 'train']
        seen_final = set()
        final_unique_cols = [x for x in final_cols_for_return if not (x in seen_final or seen_final.add(x))]
        return_dict["data_with_subgroups_and_train_flag"] = temp_df_ret[final_unique_cols]

    return return_dict

# %%
# =============================================================================
# 4. MAIN EXECUTION BLOCK
# =============================================================================
np.random.seed(BASE_SEED)
seeds_to_run = np.random.choice(1000, N_SEEDS, replace=False).tolist()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n" + "="*80)
print("MANUSCRIPT RESULTS - MULTI-SEED ANALYSIS")
print("="*80)
print(f"Using device: {device}")
print(f"Running analysis for {N_SEEDS} seeds: {seeds_to_run}")

# --- Determine representative seed (median seed for figure generation) ---
representative_seed_idx = len(seeds_to_run) // 2
representative_seed_val = seeds_to_run[representative_seed_idx]
print(f"\n[6-9] Representative seed for figure generation: {representative_seed_val} (index {representative_seed_idx + 1}/{N_SEEDS})")

all_seeds_results_by_method = {
    "CompositeAE": [],
    "VanillaAE": [],
    "PCA": []
}
component_col_names_by_method = {} 

# --- Main Loop Across Seeds ---
for i, current_seed_val in enumerate(seeds_to_run):
    print(f"\nProcessing Seed {i+1}/{N_SEEDS}: {current_seed_val}")
    # Detailed per-seed processing outputs available in full analysis
    set_seed(current_seed_val)

    # Determine if this is the representative seed for figure generation
    is_representative_seed = (current_seed_val == representative_seed_val)
    if is_representative_seed:
        print(f"\n[6-9 & FIGURES] Representative seed {current_seed_val} - generating detailed figures")

    df_train_p, df_test_p, base_features, df_train_unnorm_feat, df_test_unnorm_feat = load_and_process_data(
        DATA_PATH, TRAIN_HYPERPARAMETERS, current_seed_val
    )
    if df_train_p is None:
        print(f"Skipping seed {current_seed_val} due to data error.")
        continue

    outcome_var_y_name = 'Y' 

    # --- Composite Autoencoder ---
    method_tag = "CompositeAE"
    comp_ae_model, \
    comp_fe_recon_loss, \
    comp_fe_null_loss, \
    comp_fe_global_loss, \
    comp_eval_recon_train, \
    comp_eval_recon_test = train_ae_model(
        df_train_p, df_test_p, TRAIN_HYPERPARAMETERS, device, current_seed_val, is_vanilla=False
    )
    if comp_ae_model is None: # Check if training failed
        print(f"Skipping {method_tag} for seed {current_seed_val} due to training error.")
        # Add placeholder result or skip appending
        all_seeds_results_by_method[method_tag].append({'seed': current_seed_val, 'error': 'training failed'})
    else:
        comp_ae_train_df = extract_ae_latent_variables(comp_ae_model, df_train_p, TRAIN_HYPERPARAMETERS['outcome_var'], TRAIN_HYPERPARAMETERS, device, True)
        comp_ae_test_df = extract_ae_latent_variables(comp_ae_model, df_test_p, TRAIN_HYPERPARAMETERS['outcome_var'], TRAIN_HYPERPARAMETERS, device, False) if not df_test_p.empty else pd.DataFrame()
        comp_ae_combined_df = pd.concat([comp_ae_train_df, comp_ae_test_df]).reset_index(drop=True) if not comp_ae_test_df.empty else comp_ae_train_df

        current_comp_cols = [col for col in comp_ae_combined_df.columns if col.startswith('Latent')]
        if method_tag not in component_col_names_by_method: component_col_names_by_method[method_tag] = current_comp_cols

        # Generate figures only for the representative seed
        res_train = run_analysis_pipeline(comp_ae_train_df, f"_seed{current_seed_val}_train_only", method_tag, outcome_var_y_name, current_comp_cols, base_features, ANALYSIS_HYPERPARAMETERS, TRAIN_HYPERPARAMETERS, is_main_method=is_representative_seed)
        res_combined = run_analysis_pipeline(comp_ae_combined_df, f"_seed{current_seed_val}_combined", method_tag, outcome_var_y_name, current_comp_cols, base_features, ANALYSIS_HYPERPARAMETERS, TRAIN_HYPERPARAMETERS, is_main_method=is_representative_seed, global_ols_coeffs_train=res_train['global_ols_coeffs_df'])
        all_seeds_results_by_method[method_tag].append({
            'seed': current_seed_val, 
            'final_epoch_recon_loss_train': comp_fe_recon_loss, 
            'final_epoch_null_loss_train': comp_fe_null_loss,
            'final_epoch_global_loss_train': comp_fe_global_loss,
            'eval_recon_loss_train': comp_eval_recon_train, 
            'eval_recon_loss_test': comp_eval_recon_test, 
            'train_analysis': res_train, 
            'combined_analysis': res_combined
        })

    # --- Vanilla Autoencoder ---
    method_tag = "VanillaAE"
    vanilla_ae_train_hyperparameters = TRAIN_HYPERPARAMETERS.copy()

    print(f"\nTraining {method_tag} for seed {current_seed_val}...")
    van_ae_model, \
    van_fe_recon_loss, \
    van_fe_null_loss, \
    van_fe_global_loss, \
    van_eval_recon_train, \
    van_eval_recon_test = train_ae_model(
        df_train_p, df_test_p, vanilla_ae_train_hyperparameters, device, current_seed_val, is_vanilla=True
    )
    if van_ae_model is None: # Check if training failed
        print(f"Skipping {method_tag} for seed {current_seed_val} due to training error.")
        all_seeds_results_by_method[method_tag].append({'seed': current_seed_val, 'error': 'training failed'})
    else:
        van_ae_train_df = extract_ae_latent_variables(van_ae_model, df_train_p, TRAIN_HYPERPARAMETERS['outcome_var'], TRAIN_HYPERPARAMETERS, device, True)
        van_ae_test_df = extract_ae_latent_variables(van_ae_model, df_test_p, TRAIN_HYPERPARAMETERS['outcome_var'], TRAIN_HYPERPARAMETERS, device, False) if not df_test_p.empty else pd.DataFrame()
        van_ae_combined_df = pd.concat([van_ae_train_df, van_ae_test_df]).reset_index(drop=True) if not van_ae_test_df.empty else van_ae_train_df

        current_comp_cols = [col for col in van_ae_combined_df.columns if col.startswith('Latent')] # Assuming same prefix
        if method_tag not in component_col_names_by_method: component_col_names_by_method[method_tag] = current_comp_cols 

        res_train = run_analysis_pipeline(van_ae_train_df, f"_seed{current_seed_val}_train_only", method_tag, outcome_var_y_name, current_comp_cols, base_features, ANALYSIS_HYPERPARAMETERS, TRAIN_HYPERPARAMETERS, is_main_method=False)
        res_combined = run_analysis_pipeline(van_ae_combined_df, f"_seed{current_seed_val}_combined", method_tag, outcome_var_y_name, current_comp_cols, base_features, ANALYSIS_HYPERPARAMETERS, TRAIN_HYPERPARAMETERS, is_main_method=False, global_ols_coeffs_train=res_train['global_ols_coeffs_df'])
        all_seeds_results_by_method[method_tag].append({
            'seed': current_seed_val, 
            'final_epoch_recon_loss_train': van_fe_recon_loss,
            'final_epoch_null_loss_train': van_fe_null_loss,
            'final_epoch_global_loss_train': van_fe_global_loss,
            'eval_recon_loss_train': van_eval_recon_train,
            'eval_recon_loss_test': van_eval_recon_test,
            'train_analysis': res_train, 
            'combined_analysis': res_combined
        })

    # --- PCA ---
    method_tag = "PCA"
    print(f"\nPerforming {method_tag} for seed {current_seed_val}...")

    pca_train_df, pca_combined_df, pca_exp_var = perform_pca_and_prepare_data(
        df_train_unnorm_feat, df_test_unnorm_feat, df_train_p, df_test_p, 
        TRAIN_HYPERPARAMETERS['latent_size'], current_seed_val
    )
    current_pc_cols = [col for col in pca_combined_df.columns if col.startswith('PC')]
    if method_tag not in component_col_names_by_method: component_col_names_by_method[method_tag] = current_pc_cols

    res_train = run_analysis_pipeline(pca_train_df, f"_seed{current_seed_val}_train_only", method_tag, outcome_var_y_name, current_pc_cols, base_features, ANALYSIS_HYPERPARAMETERS, TRAIN_HYPERPARAMETERS, is_main_method=False)
    res_combined = run_analysis_pipeline(pca_combined_df, f"_seed{current_seed_val}_combined", method_tag, outcome_var_y_name, current_pc_cols, base_features, ANALYSIS_HYPERPARAMETERS, TRAIN_HYPERPARAMETERS, is_main_method=False, global_ols_coeffs_train=res_train['global_ols_coeffs_df'])
    all_seeds_results_by_method[method_tag].append({'seed': current_seed_val, 'pca_explained_variance_sum': np.sum(pca_exp_var), 'pca_explained_variance_per_component': pca_exp_var, 'train_analysis': res_train, 'combined_analysis': res_combined})

# --- Representative Seed Analysis Summary ---
print(f"\nCompleted {N_SEEDS} seeds. Representative seed: {representative_seed_val}")
print(f"\n[MANUSCRIPT FIGURES 2, 3A-C, 5A-B] Figures available: *_seed{representative_seed_val}_*.pdf")

print(f"\n" + "="*80)
print("KEY MANUSCRIPT RESULTS SUMMARY")
print("="*80)

# %%
# =============================================================================
# 5. POST-ANALYSIS AND STABILITY REPORTING
# =============================================================================

# This section of the script is for analyzing the results aggregated from all seeds.
# It requires manual alignment of latent dimensions.

# Manual alignment map for latent dimensions across seeds to conceptual clinical features.
# This map is crucial for comparing "apples to apples" across different runs where, for example,
# "Latent0" in one seed might correspond to "Latent3" in another.
# This alignment is typically done by inspecting the top associated variables (from t-tests/MI)
# for each latent dimension in each seed.

manual_ld_alignment_map = {
    "CompositeAE": {
    521: {
        "Latent0": "Gas_Trapping",
        "Latent1": "Airflow_Obstruction",
        "Latent2": "Predicted_Lung_Volume",
        "Latent3": "Absolute_Lung_Volume"
    },
    737: {
        "Latent0": "Gas_Trapping",
        "Latent1": "Airflow_Obstruction",
        "Latent2": "Absolute_Lung_Volume",
        "Latent3": "Predicted_Lung_Volume"
    },
    740: {
        "Latent0": "Airflow_Obstruction",
        "Latent1": "Predicted_Lung_Volume",
        "Latent2": "Gas_Trapping",
        "Latent3": "Absolute_Lung_Volume"
    },
    660: {
        "Latent0": "Airflow_Obstruction",
        "Latent1": "Predicted_Lung_Volume",
        "Latent2": "Absolute_Lung_Volume",
        "Latent3": "Gas_Trapping"
    },
    411: {
        "Latent0": "Absolute_Lung_Volume",
        "Latent1": "Predicted_Lung_Volume",
        "Latent2": "Gas_Trapping",
        "Latent3": "Airflow_Obstruction"
    },
    678: {
        "Latent0": "Airflow_Obstruction",
        "Latent1": "Absolute_Lung_Volume",
        "Latent2": "Predicted_Lung_Volume",
        "Latent3": "Gas_Trapping"
    },
    626: {
        "Latent0": "Predicted_Lung_Volume",
        "Latent1": "Gas_Trapping",
        "Latent2": "Absolute_Lung_Volume",
        "Latent3": "Airflow_Obstruction"
    },
    513: {
        "Latent0": "Airflow_Obstruction",
        "Latent1": "Gas_Trapping",
        "Latent2": "Predicted_Lung_Volume",
        "Latent3": "Absolute_Lung_Volume"
    },
    859: {
        "Latent0": "Airflow_Obstruction",
        "Latent1": "Absolute_Lung_Volume",
        "Latent2": "Predicted_Lung_Volume",
        "Latent3": "Gas_Trapping"
    },
    136: {
        "Latent0": "Absolute_Lung_Volume",
        "Latent1": "Gas_Trapping",
        "Latent2": "Predicted_Lung_Volume",
        "Latent3": "Airflow_Obstruction"
    },
    811: {
        "Latent0": "Absolute_Lung_Volume",
        "Latent1": "Gas_Trapping",
        "Latent2": "Airflow_Obstruction",
        "Latent3": "Predicted_Lung_Volume"
    },
    76: {
        "Latent0": "Gas_Trapping",
        "Latent1": "Absolute_Lung_Volume",
        "Latent2": "Predicted_Lung_Volume",
        "Latent3": "Airflow_Obstruction"
    },
    636: {
        "Latent0": "Gas_Trapping",
        "Latent1": "Predicted_Lung_Volume",
        "Latent2": "Absolute_Lung_Volume",
        "Latent3": "Airflow_Obstruction"
    },
    973: {
        "Latent0": "Absolute_Lung_Volume",
        "Latent1": "Predicted_Lung_Volume",
        "Latent2": "Airflow_Obstruction",
        "Latent3": "Gas_Trapping"
    },
    938: {
        "Latent0": "Airflow_Obstruction",
        "Latent1": "Predicted_Lung_Volume",
        "Latent2": "Absolute_Lung_Volume",
        "Latent3": "Gas_Trapping"
    }
},
    "VanillaAE": {
    521: {
        "Latent0": "Gas_Trapping",
        "Latent1": "Airflow_Obstruction",
        "Latent2": "Predicted_Lung_Volume",
        "Latent3": "Absolute_Lung_Volume"
    },
    737: {
        "Latent0": "Gas_Trapping",
        "Latent1": "Airflow_Obstruction",
        "Latent2": "Absolute_Lung_Volume",
        "Latent3": "Predicted_Lung_Volume"
    },
    740: {
        "Latent0": "Airflow_Obstruction",
        "Latent1": "Predicted_Lung_Volume",
        "Latent2": "Gas_Trapping",
        "Latent3": "Absolute_Lung_Volume"
    },
    660: {
        "Latent0": "Airflow_Obstruction",
        "Latent1": "Predicted_Lung_Volume",
        "Latent2": "Absolute_Lung_Volume",
        "Latent3": "Gas_Trapping"
    },
    411: {
        "Latent0": "Absolute_Lung_Volume",
        "Latent1": "Predicted_Lung_Volume",
        "Latent2": "Gas_Trapping",
        "Latent3": "Airflow_Obstruction"
    },
    678: {
        "Latent0": "Airflow_Obstruction",
        "Latent1": "Absolute_Lung_Volume",
        "Latent2": "Predicted_Lung_Volume",
        "Latent3": "Gas_Trapping"
    },
    626: {
        "Latent0": "Predicted_Lung_Volume",
        "Latent1": "Gas_Trapping",
        "Latent2": "Absolute_Lung_Volume",
        "Latent3": "Airflow_Obstruction"
    },
    513: {
        "Latent0": "Airflow_Obstruction",
        "Latent1": "Gas_Trapping",
        "Latent2": "Predicted_Lung_Volume",
        "Latent3": "Absolute_Lung_Volume"
    },
    859: {
        "Latent0": "Airflow_Obstruction",
        "Latent1": "Absolute_Lung_Volume",
        "Latent2": "Predicted_Lung_Volume",
        "Latent3": "Gas_Trapping"
    },
    136: {
        "Latent0": "Absolute_Lung_Volume",
        "Latent1": "Gas_Trapping",
        "Latent2": "Predicted_Lung_Volume",
        "Latent3": "Airflow_Obstruction"
    },
    811: {
        "Latent0": "Absolute_Lung_Volume",
        "Latent1": "Gas_Trapping",
        "Latent2": "Airflow_Obstruction",
        "Latent3": "Predicted_Lung_Volume"
    },
    76: {
        "Latent0": "Gas_Trapping",
        "Latent1": "Absolute_Lung_Volume",
        "Latent2": "Predicted_Lung_Volume",
        "Latent3": "Airflow_Obstruction"
    },
    636: {
        "Latent0": "Gas_Trapping",
        "Latent1": "Predicted_Lung_Volume",
        "Latent2": "Absolute_Lung_Volume",
        "Latent3": "Airflow_Obstruction"
    },
    973: {
        "Latent0": "Absolute_Lung_Volume",
        "Latent1": "Predicted_Lung_Volume",
        "Latent2": "Airflow_Obstruction",
        "Latent3": "Gas_Trapping"
    },
    938: {
        "Latent0": "Airflow_Obstruction",
        "Latent1": "Predicted_Lung_Volume",
        "Latent2": "Absolute_Lung_Volume",
        "Latent3": "Gas_Trapping"
    }
}
}


# This assumes patient IDs are consistent. A more robust way is to collect all and take unique.
first_method = list(all_seeds_results_by_method.keys())[0]
first_seed_result = all_seeds_results_by_method[first_method][0]
# A more robust way to get all patient IDs:
all_pids = set()
for method_name, seed_results_list in all_seeds_results_by_method.items():
    for sr in seed_results_list:
        lpdf = sr.get('combined_analysis', {}).get('local_betas_df')
        if lpdf is not None:
            all_pids.update(lpdf.index.tolist())
unique_patient_ids = sorted(list(all_pids))

if not unique_patient_ids:
    print("Error: Could not extract unique patient IDs!")

# 1. Calculate benchmarks and per-seed scale factors
final_averaged_benchmarks, per_seed_scale_factors = calculate_benchmark_inter_patient_variability(
    all_seeds_results_by_method,
    manual_ld_alignment_map, # Your defined map
    component_col_names_by_method, # From your main script
    unique_patient_ids # Extracted list of patient IDs
)

# 2. Calculate patient-specific stabilities, using the per-seed scale factors for normalization
stability_df_patients = analyze_patient_coefficient_stability(
    all_seeds_results_by_method,
    manual_ld_alignment_map,
    component_col_names_by_method,
    unique_patient_ids,
    per_seed_scale_factors # Pass the newly computed scale factors
)

# 3. (Optional) Augment stability_df_patients with the averaged benchmark values for ratio calculation
if not stability_df_patients.empty and final_averaged_benchmarks:
    stability_df_patients['benchmark_std_signed'] = stability_df_patients.apply(
        lambda row: final_averaged_benchmarks.get(
            (row['method'], row['conceptual_ld']), {}).get('benchmark_std_signed_deviation', np.nan),
        axis=1
    )
    stability_df_patients['benchmark_std_abs'] = stability_df_patients.apply(
        lambda row: final_averaged_benchmarks.get(
            (row['method'], row['conceptual_ld']), {}).get('benchmark_std_abs_deviation', np.nan),
        axis=1
    )
    stability_df_patients['stability_ratio_signed'] = \
        stability_df_patients['std_deviation'] / stability_df_patients['benchmark_std_signed']
    stability_df_patients['stability_ratio_abs'] = \
        stability_df_patients['std_abs_deviation'] / stability_df_patients['benchmark_std_abs']

    # The stability analysis outputs above cover the key manuscript results:
    #print("Individual patient rank details available in full analysis but omitted for brevity.")
    #print(stability_df_patients[['patient_id', 'method', 'conceptual_ld',
    #                              'std_deviation', 'benchmark_std_signed', 'stability_ratio_signed',
    #                              'std_abs_deviation', 'benchmark_std_abs', 'stability_ratio_abs',
    #                              'std_norm_dev_signed', 'std_norm_dev_abs' # New normalized SDs
    #                             ]])

# Comment out detailed rank stability outputs to reduce verbosity
# The following sections provide detailed stability analysis but are not directly
# referenced in the manuscript results. They remain available for comprehensive analysis.

# Generate rank stability data for manuscript results
rank_stability_df = analyze_patient_deviation_rank_stability(
    all_seeds_results_by_method,
    manual_ld_alignment_map,
    component_col_names_by_method,
    unique_patient_ids
)




if rank_stability_df.empty:
    print("rank_stability_df is empty. Cannot generate summary per conceptual LD.")
else:
    print("\n[15] Stability Ranks per Conceptual Latent Dimension")

    # Group by method and conceptual_ld
    rank_stability_summary = rank_stability_df.groupby(['method', 'conceptual_ld'])['std_rank'].agg(
        mean_std_rank='mean',
        median_std_rank='median',
        min_std_rank='min',
        max_std_rank='max',
        q25_std_rank=lambda x: x.quantile(0.25),
        q75_std_rank=lambda x: x.quantile(0.75),
        count_patients='count' # Number of patients for whom std_rank was computed for this cLD
    ).reset_index()

    if rank_stability_summary.empty:
        print("  No data to summarize after grouping by method and conceptual_ld.")
    else:
        for _, row in rank_stability_summary.iterrows():
            method_names = row['method']
            conceptual_ld_name = row['conceptual_ld']

            print(f"\n  Method: '{method_names}', Conceptual LD: '{conceptual_ld_name}'")
            print(f"    Mean Std of Ranks:   {row['mean_std_rank']:.3f}")
            print(f"    Median Std of Ranks: {row['median_std_rank']:.3f}")
            print(f"    Min Std of Ranks:    {row['min_std_rank']:.3f}")
            print(f"    Max Std of Ranks:    {row['max_std_rank']:.3f}")
            print(f"    25th Percentile:     {row['q25_std_rank']:.3f}")
            print(f"    75th Percentile:     {row['q75_std_rank']:.3f}")

        # You can also print the DataFrame directly for a tabular view
        print("\n--- Rank Stability Summary Table ---")
        print(rank_stability_summary.to_string())

# %%
# Comment out detailed diverging observations analysis - available in full codebase
'''
# Define the percentile threshold for "most diverging" observations
diverging_percentile_threshold = 0.79 # Top 20% (i.e., observations above the 80th percentile of mean_abs_deviation)

# Detailed diverging observations analysis commented out for manuscript clarity:
# This section analyzed rank stability for the most diverging observations
# (top percentile by mean absolute deviation) but is not directly referenced 
# in the manuscript results. Available in full codebase for comprehensive analysis.
'''

# %%
# Selected method analysis for manuscript results

# %%
subgroup_rank_stability_df = analyze_subgroup_member_rank_stability(
     all_seeds_results_by_method, 
     manual_ld_alignment_map,     
     component_col_names_by_method 
)

# Group by method and subgroup characteristics to get summary statistics
subgroup_rank_summary = subgroup_rank_stability_df.groupby([
    'method', 
    'conceptual_ld_subgroup_defined_by', 
    'subgroup_definition_type'
]).agg({
    'std_rank_in_subgroup_context': ['mean','median']
}).round(3)

# Flatten column names
subgroup_rank_summary.columns = ['_'.join(col).strip() for col in subgroup_rank_summary.columns]

print("\n[11] Subgroup Rank Stability Summary (Aggregated across patients)")
print(subgroup_rank_summary)

# %%  
# Cohort rank stability analysis - COMMENTED OUT FOR BREVITY
# cohort_rank_stability_df = analyze_rank_stability_for_cohort_by_cld(
#     all_seeds_results_by_method,
#     manual_ld_alignment_map,
#     component_col_names_by_method
# )

# print(cohort_rank_stability_df)

# %%
selected_method = "CompositeAE"
selected_seed = representative_seed_val

if selected_method is None:
    print("Please select a method.")
elif selected_seed is None:
    print("Please enter a seed value.")
else:
    found_result = None
    for result in all_seeds_results_by_method.get(selected_method, []):
        if result.get('seed') == selected_seed:
            found_result = result
            break

    if found_result:
        # --- Display Global OLS Coefficients ---
        print(f"\n[2] Table 1: Global OLS Coefficients for {selected_method}, Seed {selected_seed}")
        print("#### Global OLS Coefficients")
        combined_analysis = found_result.get('combined_analysis')
        if combined_analysis and 'global_ols_coeffs_df' in combined_analysis:
            global_ols_df = combined_analysis['global_ols_coeffs_df']
            if not global_ols_df.empty:
                print(global_ols_df)
            else:
                print(f"No global OLS coefficients found for this method and seed. The DataFrame is empty.")
        else:
            print(f"No 'combined_analysis' or 'global_ols_coeffs_df' found for this method and seed.")

        # --- Display Reconstruction Losses / Explained Variance ---
        print(f"\n[5] Representative Seed Train + Test Reconstruction for {selected_method}")
        if selected_method.endswith("AE"): # Check if it's an Autoencoder method
            final_epoch_recon_loss_train = found_result.get('final_epoch_recon_loss_train')
            eval_recon_loss_train = found_result.get('eval_recon_loss_train')
            eval_recon_loss_test = found_result.get('eval_recon_loss_test')

            print(f"- **Final Epoch Average Training Reconstruction Loss:** {final_epoch_recon_loss_train:.4f}" if pd.notna(final_epoch_recon_loss_train) else "- **Final Epoch Average Training Reconstruction Loss:** N/A")
            print(f"- **Full Training Dataset Evaluation Reconstruction Loss:** {eval_recon_loss_train:.4f}" if pd.notna(eval_recon_loss_train) else "- **Full Training Dataset Evaluation Reconstruction Loss:** N/A")
            print(f"- **Full Test Dataset Evaluation Reconstruction Loss:** {eval_recon_loss_test:.4f}" if pd.notna(eval_recon_loss_test) else "- **Full Test Dataset Evaluation Reconstruction Loss:** N/A")
        elif selected_method == "PCA":
            pca_explained_variance_sum = found_result.get('pca_explained_variance_sum')
            pca_explained_variance_per_component = found_result.get('pca_explained_variance_per_component')

            print(f"- **Total Explained Variance:** {pca_explained_variance_sum:.4f}" if pd.notna(pca_explained_variance_sum) else "- **Total Explained Variance:** N/A")
            if pca_explained_variance_per_component is not None:
                print("- **Explained Variance per Component:**")
                for il, varl in enumerate(pca_explained_variance_per_component):
                    print(f"  - PC{il}: {varl:.4f}")
            else:
                print("- **Explained Variance per Component:** N/A")
        else:
            print(f"Reconstruction loss or explained variance is not applicable or available for method `{selected_method}`.")

        # --- Helper function to display RMSE stats ---
        def display_rmse_stats(analysis_data, data_type_label):
            if analysis_data and 'subgroup_rmse_comparison_stats' in analysis_data:
                rmse_stats = analysis_data['subgroup_rmse_comparison_stats']
                if rmse_stats:
                    print(f"\n[10] Subgroup Benefit of Local Models ({data_type_label} Data)")
                    for sg_name, stats in rmse_stats.items():
                        print(f"\n- **Subgroup: `{sg_name}`**")
                        if pd.notna(stats.get('rmse_global_in_subgroup')):
                            print(f"  - RMSE Global (in subgroup, N={stats.get('n_in_subgroup_rmse_calc', 'N/A')}): {stats['rmse_global_in_subgroup']:.4f}")
                        else:
                            print(f"  - RMSE Global (in subgroup): N/A (N={stats.get('n_in_subgroup_rmse_calc', 'N/A')})")

                        if pd.notna(stats.get('rmse_local_in_subgroup')):
                            print(f"  - RMSE Local (in subgroup, N={stats.get('n_in_subgroup_rmse_calc', 'N/A')}): {stats['rmse_local_in_subgroup']:.4f}")
                        else:
                            print(f"  - RMSE Local (in subgroup): N/A (N={stats.get('n_in_subgroup_rmse_calc', 'N/A')})")

                        if pd.notna(stats.get('diff_rmse_in_subgroup')):
                            print(f"  - Benefit of Local Model in Subgroup (RMSE_Global - RMSE_Local): {stats['diff_rmse_in_subgroup']:.4f}")
                        else:
                            print(f"  - Benefit of Local Model in Subgroup: N/A")

                        if pd.notna(stats.get('rmse_global_out_subgroup')):
                            print(f"  - RMSE Global (out of subgroup, N={stats.get('n_out_subgroup_rmse_calc', 'N/A')}): {stats['rmse_global_out_subgroup']:.4f}")
                        else:
                            print(f"  - RMSE Global (out of subgroup): N/A (N={stats.get('n_out_subgroup_rmse_calc', 'N/A')})")

                        if pd.notna(stats.get('rmse_local_out_subgroup')):
                            print(f"  - RMSE Local (out of subgroup, N={stats.get('n_out_subgroup_rmse_calc', 'N/A')}): {stats['rmse_local_out_subgroup']:.4f}")
                        else:
                            print(f"  - RMSE Local (out of subgroup): N/A (N={stats.get('n_out_subgroup_rmse_calc', 'N/A')})")

                        if pd.notna(stats.get('diff_rmse_out_subgroup')):
                            print(f"  - Benefit of Local Model out of Subgroup: {stats['diff_rmse_out_subgroup']:.4f}")
                        else:
                            print(f"  - Benefit of Local Model out of Subgroup: N/A")

                        if pd.notna(stats.get('increase_in_rmse_diff_for_subgroup')):
                            print(f"  - **Increase in Local Model Benefit for Subgroup (vs. non-subgroup): {stats['increase_in_rmse_diff_for_subgroup']:.4f}**")
                        else:
                            print(f"  - **Increase in Local Model Benefit for Subgroup: N/A**")
                else:
                    print(f"No RMSE comparison statistics found for {data_type_label} data. The dictionary is empty.")
            else:
                print(f"No 'subgroup_rmse_comparison_stats' found for {data_type_label} data.")

        # --- Display Subgroup Benefit for Training Data ---
        train_analysis = found_result.get('train_analysis')
        display_rmse_stats(train_analysis, "Training")

        # --- Display Subgroup Benefit for Combined Data ---
        combined_analysis = found_result.get('combined_analysis') # Already retrieved above, but for clarity
        # Skipped for brevity, but you can uncomment to display combined RMSE stats
        # display_rmse_stats(combined_analysis, "Combined")

    else:
        print(f"No results found for method `{selected_method}` with seed `{selected_seed}`. Please check the method and seed value.")

# %%
selected_method_corr = "CompositeAE"
selected_seed_corr = representative_seed_val

if selected_method_corr is None or selected_seed_corr is None:
    print("Please select a method and a seed to view latent dimension correlations.")
else:
    found_result_corr = None
    for results in all_seeds_results_by_method.get(selected_method_corr, []):
        if results.get('seed') == selected_seed_corr:
            found_result_corr = result
            break

    if found_result_corr:
        combined_analysis_data = found_result_corr.get('combined_analysis', {}).get('data_with_subgroups_and_train_flag')

        if combined_analysis_data is not None and not combined_analysis_data.empty:
            # Identify latent dimension columns based on the method
            if selected_method_corr.endswith("AE"):
                latent_cols = [col for col in combined_analysis_data.columns if col.startswith('Latent')]
            elif selected_method_corr == "PCA":
                latent_cols = [col for col in combined_analysis_data.columns if col.startswith('PC')]
            else:
                latent_cols = []

            if latent_cols:
                # Ensure columns are numeric and drop rows with NaNs for correlation calculation
                latent_df = combined_analysis_data[latent_cols].astype(float).dropna()

                if not latent_df.empty:
                    correlation_matrix = latent_df.corr()
                    print(f"\n[17] Table 4: Correlation Matrix of Latent Dimensions for {selected_method_corr}, Seed {selected_seed_corr}")
                    print(correlation_matrix)

                    # Optional: Visualize correlation matrix as a heatmap
                    fig = px.imshow(
                        correlation_matrix,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale=px.colors.sequential.RdBu,
                        range_color=[-1, 1],
                        title=f"Latent Dimension Correlation Heatmap ({selected_method_corr}, Seed {selected_seed_corr})"
                    )
                    fig.update_layout(height=500, width=600)
                    fig
                else:
                    print(f"No valid data for latent dimensions found after dropping NaNs for {selected_method_corr}, Seed {selected_seed_corr}.")
            else:
                print(f"No latent dimension columns (LatentX or PCX) found for {selected_method_corr}, Seed {selected_seed_corr}.")
        else:
            print(f"No combined analysis data found for {selected_method_corr}, Seed {selected_seed_corr}.")
    else:
        print(f"Results for method `{selected_method_corr}` with seed `{selected_seed_corr}` not found.")

# %%
# =============================================================================
# 6. STATISTICAL MODELING PIPELINE WITH AIC
# =============================================================================
# This section implements a statistical modeling pipeline using AIC for variable selection.

# --- Defined Parameters ---
p_value_threshold_univariate = 0.1 # For initial univariate screening
aic_improvement_threshold_backward = 0 # Min AIC improvement to remove a var in backward elimination
aic_improvement_threshold_interaction = 2 # Min AIC improvement to add an interaction                                        

print(f"\n[14] Tables 5, 6, 7: Stepwise Regression Approach")
print(f"--- Running Statistical Modeling Pipeline with AIC ---")
print(f"Univariate p-value threshold: {p_value_threshold_univariate}")
print(f"AIC improvement threshold for backward elimination: {aic_improvement_threshold_backward}")
print(f"AIC improvement threshold for interaction selection: {aic_improvement_threshold_interaction}")
print(f"Note: The analysis is performed only with one p-value threshold, which is {p_value_threshold_univariate}.")

# Prepare data for statsmodels
data_for_standard_model = comp_ae_train_df[base_features + [outcome_var_y_name]].dropna().copy()


if data_for_standard_model.empty:
    print("Error: Data for standard model is empty after dropping NaNs. Cannot proceed.")
else:
    X_full = data_for_standard_model[base_features]
    y_full = data_for_standard_model[outcome_var_y_name]

    # --- Step 1: Univariate Selection ---
    # print("\nStep 1: Univariate Selection...")
    selected_univariate_vars = []
    # print(f"Testing {len(base_features)} features...")

    for feature in base_features:
        X_uni = sm.add_constant(X_full[[feature]])
        try:
            model_uni = sm.OLS(y_full, X_uni).fit()
            if model_uni.pvalues[1] <= p_value_threshold_univariate:
                selected_univariate_vars.append(feature)
        except Exception as e:
            print(f"Could not fit univariate model for {feature}: {e}")
            pass

    # print(f"Selected {len(selected_univariate_vars)} variables based on univariate p <= {p_value_threshold_univariate}:")
    # print(selected_univariate_vars)

    if not selected_univariate_vars:
        print("\nNo variables selected in univariate step. Cannot proceed.")
    else:
        # --- Step 2: Backward Elimination using AIC ---
        # print("\nStep 2: Backward Elimination using AIC...")
        current_vars_backward = selected_univariate_vars[:]

        if not current_vars_backward:
            # print("No variables to start backward elimination with.")
            final_vars_after_backward = []
        else:
            while True:
                if not current_vars_backward:
                    # print("No variables remaining in backward elimination.")
                    break

                X_multi = sm.add_constant(X_full[current_vars_backward])
                try:
                    model_multi = sm.OLS(y_full, X_multi).fit()
                    current_aic = model_multi.aic
                    # print(f"Current model AIC: {current_aic:.4f} with variables: {current_vars_backward}")
                except Exception as e:
                    # print(f"Error fitting model in backward elimination with vars {current_vars_backward}: {e}")
                    break # Stop if model fitting fails

                if len(current_vars_backward) == 1: # Cannot remove the last variable if it's just one
                    # print("Only one variable remaining, stopping backward elimination.")
                    break

                best_aic_after_removal = float('inf')
                var_to_remove = None

                for var in current_vars_backward:
                    temp_vars = [v for v in current_vars_backward if v != var]
                    if not temp_vars: # Should not happen if len(current_vars_backward) > 1
                        continue

                    X_temp = sm.add_constant(X_full[temp_vars])
                    try:
                        temp_model = sm.OLS(y_full, X_temp).fit()
                        if temp_model.aic < best_aic_after_removal:
                            best_aic_after_removal = temp_model.aic
                            var_to_remove = var
                    except Exception as e:
                        pass

                # Check if removing the best candidate variable improves AIC sufficiently
                if var_to_remove and (current_aic - best_aic_after_removal) > aic_improvement_threshold_backward:
                    # print(f"Removing '{var_to_remove}' (AIC improves from {current_aic:.4f} to {best_aic_after_removal:.4f}, improvement: {(current_aic - best_aic_after_removal):.4f})")
                    current_vars_backward.remove(var_to_remove)
                else:
                    # if var_to_remove:
                        # print(f"No variable removal improves AIC by more than {aic_improvement_threshold_backward}. Best possible improvement by removing '{var_to_remove}' is {(current_aic - best_aic_after_removal):.4f}.")
                    # else:
                        # print("No variable removal leads to a valid model or AIC improvement.")
                    # print("Stopping backward elimination.")
                    break

        final_vars_after_backward = current_vars_backward[:]
        # print(f"Final variables after backward elimination ({len(final_vars_after_backward)}):")
        # print(final_vars_after_backward)

        if not final_vars_after_backward:
            print("\nNo variables remaining after backward elimination. Cannot proceed with interaction selection.")
            final_model_standard = None # No model to show
        else:
            X_final_main_effects = sm.add_constant(X_full[final_vars_after_backward])
            final_model_main_effects = sm.OLS(y_full, X_final_main_effects).fit()
            # print("\nModel Summary after Backward Elimination (Main Effects Only):")
            # print(final_model_main_effects.summary())
            current_aic_main_effects = final_model_main_effects.aic

            # --- Step 3: Forward Selection for Pairwise Interactions using AIC ---
            # print("\nStep 3: Forward Selection for Pairwise Interactions using AIC...")
            current_model_aic = current_aic_main_effects
            current_vars_with_interactions_names = final_vars_after_backward[:] # List of feature names (strings)

            # Keep track of actual columns to use, including created interaction terms
            X_current_interaction_model = X_full[final_vars_after_backward].copy()
            added_interactions_terms = [] # list of interaction term names like "v1:v2"

            potential_interactions_pairs = list(itertools.combinations(final_vars_after_backward, 2))
            # print(f"Considering {len(potential_interactions_pairs)} potential pairwise interactions from {len(final_vars_after_backward)} main effects.")

            while True:
                best_aic_after_addition = float('inf')
                best_interaction_to_add_pair = None
                best_interaction_to_add_name = None

                candidate_interactions_this_step = [
                    (v1, v2) for (v1, v2) in potential_interactions_pairs
                    if f"{v1}:{v2}" not in added_interactions_terms and f"{v2}:{v1}" not in added_interactions_terms
                ]

                if not candidate_interactions_this_step:
                    # print("No more unique potential interactions to check.")
                    break

                for v1, v2 in candidate_interactions_this_step:
                    interaction_name = f"{v1}:{v2}"

                    # Create a temporary DataFrame for this iteration
                    X_temp_interaction = X_current_interaction_model.copy()
                    X_temp_interaction[interaction_name] = X_full[v1] * X_full[v2]

                    X_temp_sm = sm.add_constant(X_temp_interaction)

                    try:
                        model_temp = sm.OLS(y_full, X_temp_sm).fit()
                        if model_temp.aic < best_aic_after_addition:
                            best_aic_after_addition = model_temp.aic
                            best_interaction_to_add_pair = (v1, v2)
                            best_interaction_to_add_name = interaction_name
                    except Exception as e:
                        # print(f"Could not fit model with interaction {interaction_name}: {e}") # Optional: for debugging
                        pass

                # Check if adding the best candidate interaction improves AIC sufficiently
                if best_interaction_to_add_name and (current_model_aic - best_aic_after_addition) > aic_improvement_threshold_interaction:
                    # print(f"Adding interaction '{best_interaction_to_add_name}' (AIC improves from {current_model_aic:.4f} to {best_aic_after_addition:.4f}, improvement: {(current_model_aic - best_aic_after_addition):.4f})")

                    # Add to list of terms and update current X matrix for model
                    added_interactions_terms.append(best_interaction_to_add_name)
                    v1_added, v2_added = best_interaction_to_add_pair
                    X_current_interaction_model[best_interaction_to_add_name] = X_full[v1_added] * X_full[v2_added]

                    current_model_aic = best_aic_after_addition
                else:
                    # if best_interaction_to_add_name:
                        # print(f"No interaction addition improves AIC by more than {aic_improvement_threshold_interaction}. Best possible improvement by adding '{best_interaction_to_add_name}' is {(current_model_aic - best_aic_after_addition):.4f}.")
                    # else:
                        # print("No interaction addition leads to a valid model or AIC improvement.")
                    # print("Stopping forward selection of interactions.")
                    break

            # Fit the final model with selected main effects and interactions
            if not X_current_interaction_model.empty:
                X_final_sm = sm.add_constant(X_current_interaction_model)
                final_model_standard = sm.OLS(y_full, X_final_sm).fit()
                print("\n[14] Final Model Summary (Standard Approach with AIC)")
                print(final_model_standard.summary())
            else: # Should not happen if final_vars_after_backward was not empty
                print("\nNo variables in the final model construction for interactions.")
                final_model_standard = final_model_main_effects # Fallback to main effects model
                print("\n--- Final Model Summary (Fallback to Main Effects Model) ---")
                print(final_model_standard.summary())


# %%
def analyze_and_report_stability(all_seeds_results_by_method, component_col_names_by_method, analysis_params):
        print("\n[MANUSCRIPT RESULTS 3, 4, 10, 15] STABILITY ANALYSIS ACROSS SEEDS & METHODS")
        print("-" * 80)

        for method_name, all_seeds_results_raw in all_seeds_results_by_method.items():
            # Filter out results with errors first
            all_seeds_results = [r for r in all_seeds_results_raw if 'error' not in r]

            if not all_seeds_results:
                print(f"\nNo valid results for method: {method_name} after filtering errors.")
                continue

            print(f"\n--- Method: {method_name} ---")
            current_component_cols = component_col_names_by_method.get(method_name, [])

            # Losses for AE methods - 3
            if method_name.endswith("AE"):
                # Full dataset evaluation reconstruction losses (post-training)
                eval_recon_losses_train = [r['eval_recon_loss_train'] for r in all_seeds_results if 'eval_recon_loss_train' in r and pd.notna(r['eval_recon_loss_train'])]
                eval_recon_losses_test = [r['eval_recon_loss_test'] for r in all_seeds_results if 'eval_recon_loss_test' in r and pd.notna(r['eval_recon_loss_test'])]

                print(f"\n[3] Reconstruction Loss Multi-Seed Results (Lower is Better):")
                if eval_recon_losses_train:
                    print(f"  Train Reconstruction Loss: Mean={np.mean(eval_recon_losses_train):.4f}, SD={np.std(eval_recon_losses_train):.4f} (N={len(eval_recon_losses_train)} seeds)")
                else:
                    print("  Train Reconstruction Loss: Not available.")

                if eval_recon_losses_test:
                    print(f"  Test Reconstruction Loss:  Mean={np.mean(eval_recon_losses_test):.4f}, SD={np.std(eval_recon_losses_test):.4f} (N={len(eval_recon_losses_test)} seeds)")
                else:
                    print("  Test Reconstruction Loss:  Not available.")

            elif method_name == "PCA":
                explained_variances = [r['pca_explained_variance_sum'] for r in all_seeds_results if 'pca_explained_variance_sum' in r and pd.notna(r['pca_explained_variance_sum'])]
                if explained_variances:
                    print(f"\n PCA Explained Variance Multi-Seed Results (Higher is Better):")
                    print(f"  Explained Variance: Mean={np.mean(explained_variances):.4f}, SD={np.std(explained_variances):.4f} (N={len(explained_variances)} seeds)")
                else:
                    print(f"\n PCA Explained Variance: Not available.")

            # Global OLS R-squared (Components vs. Outcome 'Y') - 4
            r_squared_combined_list = []
            num_valid_r_squared = 0
            for r in all_seeds_results:
                if ('combined_analysis' in r and 
                    r['combined_analysis'] is not None and 
                    'global_ols_coeffs_df' in r['combined_analysis'] and 
                    not r['combined_analysis']['global_ols_coeffs_df'].empty and
                    'R_squared' in r['combined_analysis']['global_ols_coeffs_df'].columns and
                    pd.notna(r['combined_analysis']['global_ols_coeffs_df']['R_squared'].iloc[0])):
                    r_squared_combined_list.append(r['combined_analysis']['global_ols_coeffs_df']['R_squared'].iloc[0])
                    num_valid_r_squared +=1

            if r_squared_combined_list:
                print(f"\n[4] Average R² Across Seeds (Components vs Y, Higher is Better):")
                print(f"  R²: Mean={np.mean(r_squared_combined_list):.4f}, SD={np.std(r_squared_combined_list):.4f} (N={len(r_squared_combined_list)} seeds)")
            else:
                print(f"\n[4] Average R² Across Seeds: Not available.")

            # Subgroup Benefit Analysis - 10
            if method_name != "PCA":
                # print(f"\n Subgroup Benefit Over Global Model ({method_name}):")
                # Determine primary subgroup key, e.g., "LargestDynamic" or first from a sorted list

                # Collect all subgroup keys that have RMSE stats from the first valid seed result
                first_valid_seed_with_rmse_stats = next((r for r in all_seeds_results if 'combined_analysis' in r and r['combined_analysis'] and 'subgroup_rmse_comparison_stats' in r['combined_analysis']), None)

                if first_valid_seed_with_rmse_stats:
                    available_rmse_sg_keys = list(first_valid_seed_with_rmse_stats['combined_analysis']['subgroup_rmse_comparison_stats'].keys())

                    for sg_key_for_rmse in available_rmse_sg_keys:
                        rmse_benefit_stats_list = []
                        num_seeds_with_stat = 0
                        for r_val in all_seeds_results:
                            if ('combined_analysis' in r_val and r_val['combined_analysis'] and 
                                'subgroup_rmse_comparison_stats' in r_val['combined_analysis'] and
                                sg_key_for_rmse in r_val['combined_analysis']['subgroup_rmse_comparison_stats'] and
                                pd.notna(r_val['combined_analysis']['subgroup_rmse_comparison_stats'][sg_key_for_rmse].get('increase_in_rmse_diff_for_subgroup'))):
                                rmse_benefit_stats_list.append(r_val['combined_analysis']['subgroup_rmse_comparison_stats'][sg_key_for_rmse]['increase_in_rmse_diff_for_subgroup'])
                                num_seeds_with_stat +=1

                        if rmse_benefit_stats_list:
                            mean_stat = np.mean(rmse_benefit_stats_list)
                            std_stat = np.std(rmse_benefit_stats_list)
                            # print(f"  Subgroup '{sg_key_for_rmse}' RMSE Benefit: Mean={mean_stat:.4f}, SD={std_stat:.4f} (N={len(rmse_benefit_stats_list)} seeds)")
                        # Remove verbose output for non-available results
                else:
                    print(f"  No subgroup RMSE comparison stats available for method {method_name}.")

            # Comment out verbose sections - keep only essential manuscript results
            # Detailed stability analysis outputs are commented out for brevity
            # Original detailed outputs can be found in the source code but are not needed for manuscript mapping
            # Comment out detailed stability analysis outputs - keep only essential manuscript results
            # The following sections contain detailed analysis that are not directly referenced in the manuscript
            # but are kept for comprehensive analysis. They are commented out to reduce log verbosity.
            
            print(f"\n  --> Detailed analysis outputs for {method_name} available in full codebase but omitted for manuscript clarity.")
            print("      (Includes: coefficient stability, patient consistency, subgroup sizes, variable frequencies)")

# %%
analyze_and_report_stability(all_seeds_results_by_method, component_col_names_by_method, ANALYSIS_HYPERPARAMETERS)

# %%
def print_top_ttest_results_with_stats_per_seed(all_seeds_results_by_method, top_n=5):
    """
    Prints the top N t-test associated variables and their t-statistics for each component,
    for each seed and each method, from the 'combined_analysis' results.

    Args:
        all_seeds_results_by_method (dict): The main results dictionary.
            Expected structure: {method_name: [seed_result_1, seed_result_2, ...]}
            Each seed_result should contain 'combined_analysis', which in turn should
            contain 'all_ttest_stats_by_component' (after modifying run_analysis_pipeline).
        top_n (int): The number of top variables to display for each component.
    """
    print("\n[16] Tables 2 & 3: Associated Variables per Latent Dimension")

    for method_name, all_seeds_results_raw_for_method in all_seeds_results_by_method.items():
        # Filter out results that might have encountered an error during generation
        all_seeds_results_for_method = [r for r in all_seeds_results_raw_for_method if 'error' not in r]

        if not all_seeds_results_for_method:
            print(f"\nNo valid results for method: {method_name} to display T-Test results.")
            continue

        print(f"\n--- Method: {method_name} ---")

        for seed_result in all_seeds_results_for_method:
            seed_val = seed_result.get('seed', 'Unknown Seed')
            print(f"\n  Seed: {seed_val}")

            combined_analysis_results = seed_result.get('combined_analysis')
            if not combined_analysis_results:
                print("    No 'combined_analysis' results found for this seed.")
                continue

            all_ttest_stats_by_comp = combined_analysis_results.get('all_ttest_stats_by_component')

            if not all_ttest_stats_by_comp: # Handles None or empty dict
                print("    'all_ttest_stats_by_component' not found or is empty in combined_analysis for this seed.")
                print("    Ensure 'run_analysis_pipeline' is modified to return this key with the t-test statistics.")
                continue

            if not isinstance(all_ttest_stats_by_comp, dict):
                print(f"    'all_ttest_stats_by_component' is not a dictionary for this seed (type: {type(all_ttest_stats_by_comp)}). Skipping.")
                continue


            for component_dim, ttest_stats_for_comp in all_ttest_stats_by_comp.items():
                print(f"    Component: {component_dim}")
                if not ttest_stats_for_comp or not isinstance(ttest_stats_for_comp, dict):
                    print(f"      - T-Test stats for this component are missing, not a dictionary, or empty.")
                    continue

                # Filter out NaN t-statistics before sorting
                valid_vars_with_stats = {
                    var: stat for var, stat in ttest_stats_for_comp.items() if pd.notna(stat)
                }

                if not valid_vars_with_stats:
                    print(f"      - No valid (non-NaN) T-Test stats found for this component.")
                    continue

                # Sort variables by the absolute value of their t-statistic in descending order
                sorted_vars = sorted(
                    valid_vars_with_stats.items(),
                    key=lambda item: abs(item[1]), # item[1] is the t-statistic
                    reverse=True
                )

                if not sorted_vars:
                    print("      - No T-Test variables found after sorting (list is empty).")
                else:
                    for var_name, t_stat in sorted_vars[:top_n]:
                        print(f"      - {var_name}: t-statistic = {t_stat:.2f}")
            if not all_ttest_stats_by_comp : # If the outer loop didn't run because the dict was empty
                 print("    No components with T-Test stats found for this seed.")

# %%
print_top_ttest_results_with_stats_per_seed(all_seeds_results_by_method, top_n=10) 

# %%
print(f"\n" + "="*80)
print("MANUSCRIPT RESULTS SUMMARY COMPLETE")
print("="*80)
print("Key results mapped to manuscript results:")
print("[1] Outcome Mean + SD")
print("[2] Table 1: Global Model Coefficients + R²") 
print("[3] Reconstruction train/test + SD from multi-seed")
print("[4] Average R² across seeds + SD")
print("[5] Representative seed train + test reconstruction")
print("[6-9] Figure 2 & 3A-C: Plots (generated for representative seed)")
print("[10] Subgroup benefit over global model")
print("[11] Subgroup stability ranks")
print("[12-13] Figure 5A-B: Test plots (generated for representative seed)")
print("[14] Tables 5,6,7: Stepwise regression approach for one p-value threshold")
print("[15] Patient ranks across methods")
print("[16] Tables 2+3: Associated variables per latent dimension")
print("[17] Table 4: Correlation of latent dimensions")
print("\nDetailed analysis outputs have been omitted for easier mapping.")
print("Full analysis available in complete codebase.")
print("="*80) 
