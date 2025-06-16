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
│   └── figures/
│       └── paper02/ (Output plots from the analysis)
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

Follow these steps to train our method:

### Step 1: Prepare the Data

Place your raw data file (e.g., prevent_st2.sas7bdat) into the data/raw/ directory.
Run the R script to process the data. This script will handle missing data imputation and feature scaling.

<pre>
```bash
Rscript src/data_preparation/prevent_dataprep_allinone.R
```
</pre>


The script will output two files: prevent_num_imp_varL2.csv and prevent_num_imp_varL2_std.csv in the data/processed/ directory. Rename the file you wish to use to prevent_direct_train_data.csv, as this is the filename expected by the main Python script.

### Step 2: Run the Multi-Seed Analysis

The main analysis is executed by the singlesite_prevent.script.py script.

<pre>
```bash
python src/singlesite_prevent.script.py
```
</pre>

The script will train the models (CompositeAE, VanillaAE, PCA) across 15 different random seeds, perform stability analysis, and save results and plots to the results/figures/paper02/ directory.

### Step 3: Interpreting the Results

The script performs a comprehensive stability analysis. A critical part of this process is the manual alignment of latent dimensions across different seeds, which is done in the manual_ld_alignment_map dictionary within the script. The latent dimensions are given conceptual names (e.g., "Gas_Trapping", "Airflow_Obstruction") based on their correlation with the original clinical features. You must perform this interpretation and update the map if you retrain the models on new seeds or data.