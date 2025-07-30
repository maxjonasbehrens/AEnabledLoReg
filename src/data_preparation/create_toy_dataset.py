#!/usr/bin/env python3
"""
Toy Dataset Generator for COPD Study - AEnabledLoReg

This script creates a synthetic dataset that mimics the structure and statistical
properties of the original PREVENT COPD dataset for reproducibility testing.
The synthetic data maintains realistic relationships between variables while
protecting patient privacy.

The dataset includes:
- 76 predictor variables organized by clinical categories
- SGRQ outcome variable at follow-up
- Multiple visits per patient (V0-V6, URTI)
- Realistic missing data patterns
- Clinical meaningful ranges and correlations

This addresses the Biometrical Journal reproducibility checklist requirement
for synthetic data when original data cannot be shared.
"""

import pandas as pd
import numpy as np
import os
from scipy import stats
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)

def create_toy_copd_dataset(n_patients=217, n_visits_per_patient=4):
    """
    Create a synthetic COPD dataset with realistic clinical relationships.
    
    Parameters:
    -----------
    n_patients : int
        Number of unique patients to generate
    n_visits_per_patient : int
        Average number of visits per patient (will vary randomly)
    
    Returns:
    --------
    pd.DataFrame
        Synthetic dataset with structure matching original PREVENT data
    """
    
    # Define variable categories and their expected ranges/distributions
    variable_definitions = {
        # Age and Demographics
        'alter_W': {'range': (40, 85), 'type': 'normal', 'mean': 65, 'std': 10},
        'HEIGHT_W': {'range': (150, 190), 'type': 'normal', 'mean': 170, 'std': 10},
        'WEIGHT_W': {'range': (45, 120), 'type': 'normal', 'mean': 75, 'std': 15},
        
        # Respiratory Function - Lung Volume Parameters
        'FVCL_W': {'range': (1.5, 5.5), 'type': 'normal', 'mean': 3.2, 'std': 0.8},
        'FVCLP_W': {'range': (40, 120), 'type': 'normal', 'mean': 85, 'std': 20},
        'VCL_W': {'range': (1.8, 6.0), 'type': 'normal', 'mean': 3.5, 'std': 0.9},
        'VCLP_W': {'range': (45, 125), 'type': 'normal', 'mean': 90, 'std': 22},
        'VCMAXL_W': {'range': (2.0, 6.5), 'type': 'normal', 'mean': 4.0, 'std': 1.0},
        'VCMAXLP_W': {'range': (50, 130), 'type': 'normal', 'mean': 95, 'std': 25},
        
        # Lung Capacity Percentages
        'FVCP_W': {'range': (30, 120), 'type': 'normal', 'mean': 75, 'std': 20},
        'FVCPP_W': {'range': (35, 125), 'type': 'normal', 'mean': 80, 'std': 22},
        'VCP_W': {'range': (40, 130), 'type': 'normal', 'mean': 85, 'std': 25},
        'VCPP_W': {'range': (45, 135), 'type': 'normal', 'mean': 90, 'std': 27},
        'VCMAXP_W': {'range': (50, 140), 'type': 'normal', 'mean': 95, 'std': 30},
        'VCMAXPP_W': {'range': (55, 145), 'type': 'normal', 'mean': 100, 'std': 32},
        
        # Diffusion Capacity
        'DLCOMP_W': {'range': (30, 120), 'type': 'normal', 'mean': 75, 'std': 20},
        'DLCOPP_W': {'range': (35, 125), 'type': 'normal', 'mean': 80, 'std': 22},
        'DLCOVAMP_W': {'range': (3, 7), 'type': 'normal', 'mean': 5.0, 'std': 1.0},
        'DLCOVAPP_W': {'range': (60, 140), 'type': 'normal', 'mean': 100, 'std': 20},
        'FEVL_W': {'range': (0.8, 4.0), 'type': 'normal', 'mean': 2.2, 'std': 0.6},
        'FEVLP_W': {'range': (25, 110), 'type': 'normal', 'mean': 65, 'std': 20},
        'FEVP_W': {'range': (30, 115), 'type': 'normal', 'mean': 70, 'std': 22},
        'FEVPP_W': {'range': (35, 120), 'type': 'normal', 'mean': 75, 'std': 25},
        'FEVVCP_W': {'range': (40, 90), 'type': 'normal', 'mean': 65, 'std': 15},
        'FEVVCPP_W': {'range': (45, 95), 'type': 'normal', 'mean': 70, 'std': 17},
        
        # Flow Measurements
        'MEF25P_W': {'range': (0.2, 3.0), 'type': 'normal', 'mean': 1.2, 'std': 0.5},
        'MEF25PP_W': {'range': (20, 120), 'type': 'normal', 'mean': 60, 'std': 25},
        'MEF50LS_W': {'range': (0.5, 4.0), 'type': 'normal', 'mean': 2.0, 'std': 0.8},
        'MEF50LSP_W': {'range': (25, 125), 'type': 'normal', 'mean': 70, 'std': 30},
        'MEF50P_W': {'range': (0.6, 4.5), 'type': 'normal', 'mean': 2.2, 'std': 0.9},
        'MEF50PP_W': {'range': (30, 130), 'type': 'normal', 'mean': 75, 'std': 32},
        'MEF75LS_W': {'range': (0.3, 2.5), 'type': 'normal', 'mean': 1.0, 'std': 0.4},
        'MEF75LSP_W': {'range': (15, 100), 'type': 'normal', 'mean': 50, 'std': 20},
        'MEF75P_W': {'range': (0.4, 2.8), 'type': 'normal', 'mean': 1.2, 'std': 0.5},
        'MEF75PP_W': {'range': (20, 105), 'type': 'normal', 'mean': 55, 'std': 22},
        'PEFLS_W': {'range': (3, 12), 'type': 'normal', 'mean': 7.0, 'std': 2.0},
        'PEFLSP_W': {'range': (40, 140), 'type': 'normal', 'mean': 85, 'std': 25},
        'PEFP_W': {'range': (3.5, 13), 'type': 'normal', 'mean': 7.5, 'std': 2.2},
        'PEFPP_W': {'range': (45, 145), 'type': 'normal', 'mean': 90, 'std': 27},
        
        # Lung Intrathoracic Gas Volume
        'ITGVL_W': {'range': (2.0, 8.0), 'type': 'normal', 'mean': 4.5, 'std': 1.2},
        'ITGVLP_W': {'range': (80, 200), 'type': 'normal', 'mean': 130, 'std': 30},
        'ITGVP_W': {'range': (85, 205), 'type': 'normal', 'mean': 135, 'std': 32},
        'ITGVPP_W': {'range': (90, 210), 'type': 'normal', 'mean': 140, 'std': 35},
        'RVL_W': {'range': (1.2, 5.0), 'type': 'normal', 'mean': 2.8, 'std': 0.8},
        'RVLP_W': {'range': (70, 180), 'type': 'normal', 'mean': 115, 'std': 25},
        'RVP_W': {'range': (75, 185), 'type': 'normal', 'mean': 120, 'std': 27},
        'RVPP_W': {'range': (80, 190), 'type': 'normal', 'mean': 125, 'std': 30},
        'RVTLCP_W': {'range': (25, 60), 'type': 'normal', 'mean': 40, 'std': 8},
        'RVTLCPP_W': {'range': (30, 65), 'type': 'normal', 'mean': 45, 'std': 10},
        'TLCL_W': {'range': (4.0, 9.0), 'type': 'normal', 'mean': 6.2, 'std': 1.2},
        'TLCLP_W': {'range': (80, 150), 'type': 'normal', 'mean': 110, 'std': 20},
        'TLCP_W': {'range': (85, 155), 'type': 'normal', 'mean': 115, 'std': 22},
        'TLCPP_W': {'range': (90, 160), 'type': 'normal', 'mean': 120, 'std': 25},
        
        # Expiratory Reserve
        'ERVL_W': {'range': (0.5, 2.5), 'type': 'normal', 'mean': 1.3, 'std': 0.4},
        'ERVLP_W': {'range': (40, 140), 'type': 'normal', 'mean': 85, 'std': 25},
        'ERVP_W': {'range': (45, 145), 'type': 'normal', 'mean': 90, 'std': 27},
        'ERVPP_W': {'range': (50, 150), 'type': 'normal', 'mean': 95, 'std': 30},
        
        # Ventilation Efficiency
        'VALP_W': {'range': (15, 45), 'type': 'normal', 'mean': 28, 'std': 6},
        'VAPP_W': {'range': (20, 50), 'type': 'normal', 'mean': 32, 'std': 8},
        
        # Blood Pressure
        'BPDIA_W': {'range': (60, 100), 'type': 'normal', 'mean': 80, 'std': 10},
        'BPSYS_W': {'range': (110, 180), 'type': 'normal', 'mean': 135, 'std': 15},
        
        # Heart Rate
        'HR_W': {'range': (60, 120), 'type': 'normal', 'mean': 85, 'std': 15},
        'HHR_W': {'range': (100, 180), 'type': 'normal', 'mean': 140, 'std': 20},
        'HRREST_W': {'range': (55, 100), 'type': 'normal', 'mean': 75, 'std': 12},
        
        # Respiratory and Exercise
        'BREATH_W': {'range': (12, 30), 'type': 'normal', 'mean': 18, 'std': 4},
        'DIST_W': {'range': (100, 600), 'type': 'normal', 'mean': 350, 'std': 100},
        'BORG_W': {'range': (0, 10), 'type': 'normal', 'mean': 4, 'std': 2},
        
        # Oxygen Levels
        'POXSAT_W': {'range': (88, 99), 'type': 'normal', 'mean': 95, 'std': 3},
        'POXSAT_WDT6_W': {'range': (85, 98), 'type': 'normal', 'mean': 93, 'std': 4},
        'LOXAT_W': {'range': (0, 15), 'type': 'normal', 'mean': 4, 'std': 3},
        
        # Disease Duration and History
        'COPDDIA_W': {'range': (0, 25), 'type': 'normal', 'mean': 8, 'std': 5},
        'COPDSYM_W': {'range': (0, 30), 'type': 'normal', 'mean': 10, 'std': 6},
        'PY_W': {'range': (0, 80), 'type': 'normal', 'mean': 35, 'std': 20},
        
        # Inflammatory Markers
        'FENO': {'range': (10, 80), 'type': 'normal', 'mean': 35, 'std': 15},
        'NO1_W': {'range': (5, 50), 'type': 'normal', 'mean': 20, 'std': 10},
        
        # Medication
        'MEDDIS_W': {'range': (0, 15), 'type': 'poisson', 'lambda': 3}
    }
    
    # Visit types
    visit_types = ['Baseline (V0)', 'Visit 1 (V1)', 'Visit 2 (V2)', 'Visit 3 (V3)', 
                   'Visit 4 (V4)', 'Visit 5 (V5)', 'Visit 6 (V6)', 'URTI']
    
    # Center IDs (2 centers as in original study, data was pooled)
    centers = ['CENTER_01', 'CENTER_02']
    
    # Generate patient data
    all_data = []
    
    for pid in range(1, n_patients + 1):
        # Assign patient to center
        center = np.random.choice(centers)
        
        # Generate baseline patient characteristics (age-related)
        age = np.clip(np.random.normal(65, 10), 40, 85)
        
        # Age influences disease severity (older = more severe)
        severity_factor = (age - 40) / 45  # 0 to 1 scale
        
        # Generate number of visits for this patient (more severe patients have more visits)
        n_visits = np.random.poisson(n_visits_per_patient + severity_factor * 2)
        n_visits = max(2, min(n_visits, 8))  # At least baseline + 1 visit, max 8
        
        # Select which visits this patient has
        patient_visits = ['Baseline (V0)']  # Everyone has baseline
        remaining_visits = [v for v in visit_types if v != 'Baseline (V0)']
        patient_visits.extend(np.random.choice(remaining_visits, 
                                             size=min(n_visits-1, len(remaining_visits)), 
                                             replace=False))
        
        # Generate visit dates (sorted chronologically)
        base_date = pd.Timestamp('2020-01-01') + pd.Timedelta(days=np.random.randint(0, 365))
        visit_dates = sorted([base_date + pd.Timedelta(days=np.random.randint(0, max(1, 30*i))) 
                             for i in range(len(patient_visits))])
        
        # Generate data for each visit
        for visit_idx, (visit, visit_date) in enumerate(zip(patient_visits, visit_dates)):
            row_data = {
                'PID': f'P{pid:04d}',
                'CENTER': center,
                'VISIT': visit,
                'VISITDC_W': visit_date,
                'alter_W': age  # Age stays constant
            }
            
            # Generate clinical variables with realistic correlations
            for var_name, var_info in variable_definitions.items():
                if var_name == 'alter_W':
                    continue  # Already set
                
                # Base value generation
                if var_info['type'] == 'normal':
                    value = np.random.normal(var_info['mean'], var_info['std'])
                elif var_info['type'] == 'poisson':
                    value = np.random.poisson(var_info['lambda'])
                else:
                    value = np.random.uniform(var_info['range'][0], var_info['range'][1])
                
                # Apply age and severity influences
                if 'Lung' in var_name or 'FEV' in var_name or 'FVC' in var_name:
                    # Lung function decreases with age and severity
                    value *= (1 - severity_factor * 0.3)
                elif 'BORG' in var_name or 'BREATH' in var_name:
                    # Symptoms increase with severity
                    value *= (1 + severity_factor * 0.5)
                elif 'DLCO' in var_name:
                    # Diffusion capacity decreases with severity
                    value *= (1 - severity_factor * 0.4)
                
                # Add visit-to-visit variability (disease progression)
                if visit_idx > 0:
                    # Small random change from previous visits
                    progression = np.random.normal(0, 0.05)  # 5% variability
                    value *= (1 + progression)
                
                # Clip to realistic ranges
                if 'range' in var_info:
                    value = np.clip(value, var_info['range'][0], var_info['range'][1])
                
                row_data[var_name] = value
            
            # Generate SGRQ outcome (correlated with lung function and symptoms)
            # SGRQ ranges from 0-100, higher = worse quality of life
            # Typical COPD patients have SGRQ scores distributed across the range with mean ~45-50
            
            # Normalize lung function metrics to 0-1 scale (lower = worse)
            lung_function_fevp = np.clip((row_data.get('FEVP_W', 70) - 30) / (115 - 30), 0, 1)
            lung_function_fvcp = np.clip((row_data.get('FVCP_W', 75) - 30) / (120 - 30), 0, 1)  
            lung_function_dlco = np.clip((row_data.get('DLCOPP_W', 80) - 35) / (125 - 35), 0, 1)
            
            lung_function_avg = np.mean([lung_function_fevp, lung_function_fvcp, lung_function_dlco])
            
            # Normalize symptoms to 0-1 scale (higher = worse)  
            borg_normalized = np.clip(row_data.get('BORG_W', 4) / 10, 0, 1)
            breath_normalized = np.clip((row_data.get('BREATH_W', 18) - 12) / (30 - 12), 0, 1)
            
            symptoms_avg = np.mean([borg_normalized, breath_normalized])
            
            # Create more realistic SGRQ score distribution
            # Use a base score that varies more naturally
            base_score = np.random.normal(35, 15)  # Start with wider base distribution
            
            # Lung function impact: worse function increases SGRQ (inverse relationship)
            lung_impact = (1 - lung_function_avg) * 30  # Scale impact
            
            # Symptoms impact: more symptoms increase SGRQ  
            symptom_impact = symptoms_avg * 25
            
            # Age impact (moderate)
            age_impact = (age - 40) / 45 * 10  # Max 10 points for oldest patients
            
            # Combine all factors
            sgrq_base = base_score + lung_impact + symptom_impact + age_impact
            
            # Add visit-specific random variation
            sgrq_base += np.random.normal(0, 8)  # Random noise
            
            # Apply soft ceiling to prevent too many values at 100
            # Use a sigmoid-like transformation to compress high values
            if sgrq_base > 85:
                # Compress values above 85 more strongly
                excess = sgrq_base - 85
                compressed_excess = 15 * (1 - np.exp(-excess / 10))  # Asymptotic approach to 100
                sgrq_base = 85 + compressed_excess
            
            # Final clipping with realistic bounds
            sgrq_base = np.clip(sgrq_base, 0, 98)  # Keep max at 98 to avoid ceiling effect
            
            row_data['Y'] = sgrq_base  # Y is the outcome variable name used in the code
            
            all_data.append(row_data)
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Add realistic missing data patterns
    # Clinical variables have different missing rates
    missing_patterns = {
        'FENO': 0.15,  # More specialized test
        'NO1_W': 0.12,
        'DLCOMP_W': 0.08,
        'DLCOPP_W': 0.08,
        'LOXAT_W': 0.10,
        'POXSAT_WDT6_W': 0.07
    }
    
    for var, missing_rate in missing_patterns.items():
        if var in df.columns:
            n_missing = int(len(df) * missing_rate)
            missing_indices = np.random.choice(df.index, n_missing, replace=False)
            df.loc[missing_indices, var] = np.nan
    
    # Sort by patient and visit date
    df = df.sort_values(['PID', 'VISITDC_W']).reset_index(drop=True)
    
    return df

def save_toy_dataset():
    """Generate and save the toy dataset in the expected format."""
    
    print("Generating synthetic COPD dataset...")
    
    # Create the synthetic dataset
    df_synthetic = create_toy_copd_dataset(n_patients=217, n_visits_per_patient=4)
    
    print(f"Generated dataset with {len(df_synthetic)} observations from {df_synthetic['PID'].nunique()} patients")
    print(f"Variables included: {len(df_synthetic.columns)} columns")
    print(f"Visit distribution:\n{df_synthetic['VISIT'].value_counts()}")
    
    # Create output directory if it doesn't exist
    # Use absolute path to ensure we save in the correct location
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(project_root, 'data', 'raw')
    os.makedirs(data_dir, exist_ok=True)
    
    # Save as SAS dataset equivalent (CSV format for compatibility)
    output_path = os.path.join(data_dir, 'prevent_st2_synthetic.csv')
    df_synthetic.to_csv(output_path, index=False)
    
    print(f"\nSynthetic dataset saved to: {output_path}")
    
    # Display summary statistics
    print("\n--- Summary Statistics (first few variables) ---")
    numeric_cols = df_synthetic.select_dtypes(include=[np.number]).columns[:10]
    print(df_synthetic[numeric_cols].describe().round(2))
    
    # Check for missing data
    missing_data = df_synthetic.isnull().sum()
    missing_vars = missing_data[missing_data > 0]
    if len(missing_vars) > 0:
        print(f"\n--- Missing Data Patterns ---")
        for var in missing_vars.index:
            pct_missing = (missing_vars[var] / len(df_synthetic)) * 100
            print(f"{var}: {missing_vars[var]} ({pct_missing:.1f}%)")
    
    return df_synthetic

if __name__ == "__main__":
    # Generate the synthetic dataset
    synthetic_data = save_toy_dataset()
    
    print(f"\n✓ Synthetic COPD dataset created successfully!")
    print(f"✓ Dataset includes all 76 predictor variables from the original study")
    print(f"✓ Realistic clinical relationships and missing data patterns included")
    print(f"✓ Ready for reproducibility testing with your AEnabledLoReg pipeline")
