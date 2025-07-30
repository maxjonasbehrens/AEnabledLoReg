# Synthetic COPD Dataset Documentation

## Overview

This synthetic dataset was created for providing synthetic pseudo-data when the original clinical data cannot be shared. The synthetic dataset maintains a similar structure, variable relationships, and statistical properties as the original COPD dataset while protecting patient privacy.

## Dataset Characteristics

### Study Design
- **Population**: Synthetic COPD patients (N=217)
- **Study Type**: Observational multi-site clinical trial
- **Centers**: 2 simulated study centers (CENTER_01 and CENTER_02, data pooled)
- **Variables**: 76 clinical predictors + 1 outcome variable (SGRQ)

## Variable Categories and Clinical Relationships

The synthetic dataset includes 76 predictor variables organized into clinically meaningful categories:

### 1. Demographics and Anthropometrics (3 variables)
- `alter_W`: Age (40-85 years, mean: 65)
- `HEIGHT_W`: Height in cm (150-190 cm)
- `WEIGHT_W`: Weight in kg (45-120 kg)

### 2. Respiratory Function Tests

#### Lung Volume Parameters (6 variables)
- `FVCL_W`, `FVCLP_W`: Forced Vital Capacity (absolute and % predicted)
- `VCL_W`, `VCLP_W`: Vital Capacity (absolute and % predicted)
- `VCMAXL_W`, `VCMAXLP_W`: Maximum Vital Capacity (absolute and % predicted)

#### Lung Capacity Percentages (6 variables)
- `FVCP_W`, `FVCPP_W`: FVC percentages
- `VCP_W`, `VCPP_W`: VC percentages
- `VCMAXP_W`, `VCMAXPP_W`: Maximum VC percentages

#### Diffusion Capacity (16 variables)
- `DLCOMP_W`, `DLCOPP_W`: DLCO measurements (absolute and % predicted)
- `DLCOVAMP_W`, `DLCOVAPP_W`: DLCO/VA measurements
- `FEVL_W`, `FEVLP_W`, `FEVP_W`, `FEVPP_W`: FEV1 measurements
- `FEVVCP_W`, `FEVVCPP_W`: FEV1/FVC ratios
- Various MEF (Mid-Expiratory Flow) and PEF (Peak Expiratory Flow) measurements

#### Lung Flow Measurements (2 variables)
- `MEF25P_W`, `MEF25PP_W`: Mid-expiratory flow at 25% FVC

### 3. Lung Intrathoracic Gas Volume (12 variables)
- Residual Volume (RV): `RVL_W`, `RVLP_W`, `RVP_W`, `RVPP_W`
- Total Lung Capacity (TLC): `TLCL_W`, `TLCLP_W`, `TLCP_W`, `TLCPP_W`
- Intrathoracic Gas Volume (ITGV): `ITGVL_W`, `ITGVLP_W`, `ITGVP_W`, `ITGVPP_W`

### 4. Expiratory Reserve (4 variables)
- `ERVL_W`, `ERVLP_W`, `ERVP_W`, `ERVPP_W`: Expiratory Reserve Volume

### 5. Ventilation Efficiency (2 variables)
- `VALP_W`, `VAPP_W`: Ventilation efficiency measures

### 6. Cardiovascular Measures (5 variables)
- Blood Pressure: `BPDIA_W` (diastolic), `BPSYS_W` (systolic)
- Heart Rate: `HR_W` (exercise), `HHR_W` (peak exercise), `HRREST_W` (resting)

### 7. Clinical Symptoms and Exercise (3 variables)
- `BREATH_W`: Respiratory rate
- `DIST_W`: Exercise capacity (6-minute walk distance)
- `BORG_W`: Dyspnea score (0-10 scale)

### 8. Oxygenation (3 variables)
- `POXSAT_W`: Baseline oxygen saturation
- `POXSAT_WDT6_W`: Oxygen saturation during 6-minute walk
- `LOXAT_W`: Exercise oxygen desaturation

### 9. Disease History and Lifestyle (3 variables)
- `COPDDIA_W`: COPD diagnosis duration (years)
- `COPDSYM_W`: COPD symptom duration (years)
- `PY_W`: Smoking history (pack-years)

### 10. Inflammatory Markers (2 variables)
- `FENO`: Fractional exhaled nitric oxide
- `NO1_W`: Nitric oxide measurement

### 11. Medication (1 variable)
- `MEDDIS_W`: Number of medications

### 12. Outcome Variable
- `Y`: St. George's Respiratory Questionnaire (SGRQ) score at follow-up (0-100 scale, higher = worse quality of life)

## Realistic Clinical Relationships

The synthetic data incorporates clinically meaningful relationships:

1. **Age Effects**: Older patients have:
   - Reduced lung function parameters
   - Higher symptom scores
   - More medications

2. **Disease Severity**: Patients with more severe COPD have:
   - Lower FEV1 and FVC values
   - Higher residual volumes (air trapping)
   - Worse exercise capacity
   - Higher SGRQ scores (worse quality of life)

3. **Longitudinal Changes**: 
   - Visit-to-visit variability (±5%)
   - Disease progression patterns
   - Correlated changes across related variables

4. **Outcome Relationships**: SGRQ scores are correlated with:
   - Lung function (inverse relationship)
   - Symptom scores (positive relationship)
   - Exercise capacity (inverse relationship)

## Missing Data Patterns

Realistic missing data patterns are implemented:
- **FENO**: 15% missing (specialized test)
- **NO1_W**: 12% missing
- **DLCO measurements**: 8% missing
- **Exercise oxygenation**: 7-10% missing
- Other variables: <5% missing

## Data Processing Pipeline

The synthetic dataset follows the same processing pipeline as the original:

1. **create_toy_dataset.py**: Generates synthetic raw data
2. **prevent_dataprep_synthetic.R**: Processes data using same logic as original
3. **Output**: Two files ready for analysis
   - `prevent_num_imp_varL2.csv` (unscaled)
   - `prevent_num_imp_varL2_std.csv` (scaled)

## Usage Instructions

### Step 1: Generate Synthetic Data
```bash
python src/data_preparation/create_toy_dataset.py
```

### Step 2: Process Data
```bash
Rscript src/data_preparation/prevent_dataprep_synthetic.R
```

### Step 3: Rename for Analysis
```bash
# Choose scaled or unscaled version
cp data/processed/prevent_num_imp_varL2_std.csv data/processed/prevent_direct_train_data.csv
```

### Step 4: Run Main Analysis
```bash
python src/singlesite_prevent.script.py
```

## Limitations and Considerations

1. **Simplified Relationships**: While clinically informed, the synthetic data uses simplified statistical relationships compared to real patient data.

2. **Population Characteristics**: The synthetic population may not fully capture the complexity of real COPD patient heterogeneity.

3. **Temporal Patterns**: Disease progression patterns are simplified compared to real longitudinal clinical data.

4. **Center Effects**: Study center differences are minimal in the synthetic data.

5. **Validation Results**: Results from the synthetic data should be interpreted as methodological validation rather than clinical findings.

---

**Note**: This synthetic dataset is provided solely for reproducibility validation and methodological evaluation. It should not be used for clinical research or to draw medical conclusions.
