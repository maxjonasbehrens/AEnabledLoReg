# Autoencoder-Enabled Localized Regression (AEnabledLoReg)

## Abstract 

This repository contains the source code for the paper, "Contrasting Global and Patient-Specific Regression Models via a Neural Network Representation". We introduce a novel methodology to contrast global and local modeling strategies by enabling localized regression models through an autoencoder neural network. Our primary model is trained on clinical data to learn a meaningful latent space representation. This latent space is then used to define local patient neighborhoods, allowing us to fit localized regression models. We analyze the stability of these models and the resulting patient subgroups across multiple random seeds, comparing our approach against Principal Component Analysis (PCA) and a standard Vanilla Autoencoder. The goal is to identify stable, interpretable patient subgroups characterized by deviations in their local model coefficients from global trends.

## Methodology

The core of our approach is a novel Composite Autoencoder that is optimized with a composite loss function:

- Reconstruction Loss: Standard autoencoder loss to ensure the latent space captures the original data's variance.
- Prediction Likelihood Loss: A localized regression loss that encourages the latent space to be predictive of a clinical outcome. It pushes the model to create a latent space where patient similarity is tied to local relationships with the outcome.

Patient subgroups are subsequently identified by analyzing the coefficients of local regression models. For each patient, a weighted regression is performed using their neighbors in the latent space. Patients whose local coefficients deviate significantly from a global regression model are grouped together for further clinical interpretation.

## Repository Structure

<pre>
```
AEnabledLoReg/
├── data/
│   ├── raw/ (Input raw data, e.g., prevent_st2.sas7bdat)
│   └── processed/ (Output of the R script, e.g., prevent_direct_train_data.csv)
├── results/
│   └── figures/ (Output plots from the analysis)
└── src/
    ├── data_preparation/
    │   └── prevent_dataprep_allinone.R   # R script for data preprocessing
    ├── methods/
    │   └── single_site_ae_loreg_module.py # PyTorch module for the AE model and training
    ├── utilities/
    │   ├── loregs.py                     # Weighted regression functions
    │   └── weights.py                    # Kernel weighting functions
    └── singlesite_prevent.script.py      # Main Python script to run the analysis
```
</pre>


## How to Run the Analysis

### For Reproducibility Testing

We provide a complete pipeline with synthetic data to comply with the reproducibility requirements. This allows reviewers and researchers to validate our methodology without access to the original clinical data.

#### Full Reproducibility Validation (1-2 hours)
```bash
./run_reproducibility_pipeline.sh full
```

#### Quick Validation for Testing (10-15 minutes)
```bash
./run_reproducibility_pipeline.sh quick
```

The pipeline will:
1. Generate a synthetic COPD dataset with 76 predictors and realistic clinical relationships
2. Process the data using the same preprocessing pipeline
3. Run the complete multi-seed analysis (15 seeds and 300 epochs for full mode, 3 seeds and 50 epochs for quick mode)
4. Generate all figures and results
5. Create a detailed reproducibility report

#### Checking Results and Output

After running the pipeline, you can examine the generated results in several ways:

**1. Generated Figures (in `results/figures/`)**
The analysis produces several key figures corresponding to the manuscript:
- `Figure2_*`: Latent coefficient analysis showing how local regression coefficients deviate from global trends
- `Figure3or4_A_*`: Patient subgroup highlighting in the latent space
- `Figure3or4_B_*`: Forest plots showing original feature coefficients for identified subgroups
- `Figure3or4_C_*`: Latent space profiles (z-profiles) characterizing patient subgroups
- `Figure5_*`: Test results showing generalization of subgroup patterns

**2. Log Files (in `logs/`)**
- `main_analysis.log`: Comprehensive analysis log with numbered references [2-17] corresponding to manuscript findings. An overview of all results is provided at the end of this log file. Specific results can be searched using "[X]" (replace X with the result number, e.g., "[3]" for result 3)
- `data_processing.log`: Data preprocessing pipeline log with outcome summary result [1]

**3. Processed Data (in `data/processed/`)**
- `prevent_direct_train_data.csv`: Final processed dataset used for analysis
- `prevent_num_imp_varL2.csv` and `prevent_num_imp_varL2_std.csv`: Intermediate processing outputs

**4. Reproducibility Report**
The pipeline automatically generates `REPRODUCIBILITY_REPORT.md` containing:
- Execution summary and system information
- File verification results
- Analysis parameter confirmation
- Quick validation checklist

**Note**: The main analysis produces a detailed log file documenting the entire process. Manuscript results can be searched within the log using reference numbers [1-17], which correspond to specific analyses and findings presented in the paper.

### For Original Data Analysis

If you have access to the original PREVENT study data, follow these steps:

#### Step 1: Prepare the Data

Place your raw data file (e.g., prevent_st2.sas7bdat) into the data/raw/ directory.
Run the R script to process the data. This script will handle missing data imputation and feature scaling.

```bash
Rscript src/data_preparation/prevent_dataprep_allinone.R
```

The script will output two files: prevent_num_imp_varL2.csv and prevent_num_imp_varL2_std.csv in the data/processed/ directory. Rename the file you wish to use to prevent_direct_train_data.csv, as this is the filename expected by the main Python script.

#### Step 2: Run the Multi-Seed Analysis

The main analysis is executed by the singlesite_prevent.script.py script.

```bash
python src/singlesite_prevent.script.py
```

The script will train the models (CompositeAE, VanillaAE, PCA) across 15 different random seeds, perform stability analysis, and save results and plots to the results/figures/ directory.

#### Step 3: Interpreting the Results

The script performs a comprehensive stability analysis. A critical part of this process is the manual alignment of latent dimensions across different seeds, which is done in the manual_ld_alignment_map dictionary within the script. The latent dimensions are given conceptual names (e.g., "Gas_Trapping", "Airflow_Obstruction") based on their correlation with the original clinical features. You must perform this interpretation and update the map if you retrain the models on new seeds or data.

## Reproducibility and Data Availability

### Synthetic Dataset

Since the original PREVENT study data contains sensitive patient information and cannot be shared, we provide a synthetic dataset. The synthetic dataset includes:

- 500 synthetic COPD patients
- 76 clinical predictor variables across all major categories
- Realistic missing data patterns
- Appropriate correlations between variables
- SGRQ outcome variable with clinically meaningful relationships

See `SYNTHETIC_DATA_DOCUMENTATION.md` for detailed information about the synthetic dataset.

### Installation Requirements

#### System Requirements

**Python**: Version 3.8 or higher (tested with Python 3.8-3.11)
**R**: Version 4.0 or higher (tested with R 4.0-4.3)

#### Python Dependencies
```bash
pip install -r requirements.txt
```

The analysis requires Python 3.8+ for compatibility with PyTorch and modern scientific computing packages. All package versions are specified in `requirements.txt` to ensure reproducibility.

#### R Dependencies
```r
install.packages(c("dplyr", "haven", "VIM", "DataExplorer"))
```

R 4.0+ is recommended for optimal performance with the data preprocessing pipeline and statistical functions used in the analysis.