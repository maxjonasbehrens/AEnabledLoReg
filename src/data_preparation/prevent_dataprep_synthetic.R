library(haven)
library(DataExplorer)
library(dplyr)

# Modified data preparation script for synthetic COPD dataset
# This script processes the synthetic dataset created by create_toy_dataset.py
# to match the expected format for the AEnabledLoReg pipeline

# Load synthetic data (CSV format instead of SAS)
prevent_file = "./data/raw/prevent_st2_synthetic.csv"

if (!file.exists(prevent_file)) {
  stop("Synthetic dataset not found. Please run create_toy_dataset.py first to generate the synthetic data.")
}

prevent <- read.csv(prevent_file, stringsAsFactors = FALSE)

# Convert date column
prevent$VISITDC_W <- as.Date(prevent$VISITDC_W)

# Replace empty strings with NA (if any)
prevent[prevent == ""] <- NA

# Display basic info about the synthetic dataset
cat("Synthetic dataset loaded successfully!\n")
cat("Number of patients:", n_distinct(prevent$PID), "\n")
cat("Total observations:", nrow(prevent), "\n")

prevent %>%
  group_by(CENTER) %>%
  summarise(cnt = n_distinct(PID)) %>%
  print()

# Step 1: Filter out patients where `URTI` occurs before the first `Visit 1`
# (This filtering logic is maintained for consistency with original script)
filtered_data <- prevent %>%
  group_by(PID) %>%
  arrange(PID, VISITDC_W) %>%  # Order by patient and visit date
  mutate(
    first_visit1_date = min(VISITDC_W[grepl("V1", VISIT)], na.rm = TRUE),
    first_visit2_date = min(VISITDC_W[grepl("V2", VISIT)], na.rm = TRUE),
    first_visit3_date = min(VISITDC_W[grepl("V3", VISIT)], na.rm = TRUE) # Find the first Visit 1 date using regex
  ) %>%
  filter(!(any(VISIT == "URTI" & VISITDC_W < first_visit1_date))) %>%
  filter(!(any(VISIT == "URTI" & VISITDC_W < first_visit2_date))) %>%
  filter(!(any(VISIT == "URTI" & VISITDC_W < first_visit3_date))) %>% # Exclude patients with URTI before Visit 1
  ungroup()

# Step 2: Keep the filtered data (synthetic data already has appropriate visits)
final_filtered_data <- filtered_data

cat("After filtering - Number of patients:", n_distinct(final_filtered_data$PID), "\n")

prevent <- final_filtered_data

# Create identifiers for merging later
prevent$ID <- seq_len(nrow(prevent))
id_data <- prevent[, c("ID", "CENTER", "PID", "VISIT")]

prevent$PID <- as.character(prevent$PID)

# Filter rows based on visit codes
prevent_visits <- prevent[grepl("V[0-6]", prevent$VISIT),]

# Define columns
id_col <- "PID"
visit_col <- "VISIT"
baseline_visit <- "Baseline (V0)"

# Function to fill baseline values for each PID
fill_baseline_values <- function(data, id_col, visit_col, baseline_visit) {
  # Get unique PIDs
  unique_ids <- unique(data[[id_col]])

  # Loop through each PID
  for (pid in unique_ids) {
    # Subset data for the current PID
    pid_data <- data[data[[id_col]] == pid, ]

    # Identify baseline values
    baseline_values <- pid_data[pid_data[[visit_col]] == baseline_visit, ]
    
    if (nrow(baseline_values) == 0) next  # Skip if no baseline visit

    # Loop through each column and fill values
    for (col in names(data)) {
      if (!col %in% c(id_col, visit_col) && 
          nrow(baseline_values) > 0 && 
          !is.na(baseline_values[[col]][1])) {
        data[data[[id_col]] == pid & is.na(data[[col]]), col] <- baseline_values[[col]][1]
      }
    }
  }

  return(data)
}

# Apply the function to fill forward baseline values
prevent_visits <- fill_baseline_values(prevent_visits, id_col, visit_col, baseline_visit)

# Calculate the frequency of non-missing values by column and row
prevent_visits_nafreq_cols <- 1 - (colSums(is.na(prevent_visits)) / nrow(prevent_visits))
prevent_visits_nafreq_rows <- 1 - (rowSums(is.na(prevent_visits)) / ncol(prevent_visits))

cat("Data quality before filtering:\n")
cat("Columns with >80% complete data:", sum(prevent_visits_nafreq_cols > 0.8), "/", length(prevent_visits_nafreq_cols), "\n")
cat("Rows with >10% complete data:", sum(prevent_visits_nafreq_rows > 0.10), "/", length(prevent_visits_nafreq_rows), "\n")

# Filter data based on non-missing values
prevent_visits <- prevent_visits[prevent_visits_nafreq_rows > 0.10,]
prevent_x <- prevent_visits[, prevent_visits_nafreq_cols > 0.8]

# Save row identifiers for later use
row_ids <- prevent_x$ID

# Select only numeric columns for processing
numeric_columns <- sapply(prevent_x, is.numeric)
prevent_x_num <- prevent_x[, numeric_columns]

cat("Numeric variables selected:", ncol(prevent_x_num), "\n")

# Apply k-NN imputation for remaining missing values
library(VIM)
prevent_x_num <- kNN(prevent_x_num, k = 5, imp_var = FALSE)

# Calculate variance for variable selection
var_prevent_x_num_imp <- apply(prevent_x_num, 2, var, na.rm = TRUE)

cat("Variables before variance filtering:", length(var_prevent_x_num_imp), "\n")

# Filter columns based on variance (keep variables with variance > 0.2)
prevent_x_num_imp_varL2 <- prevent_x_num[, var_prevent_x_num_imp > 0.2]

cat("Variables after variance filtering:", ncol(prevent_x_num_imp_varL2), "\n")

# Create scaled version
prevent_x_num_imp_varL2_std <- as.data.frame(scale(prevent_x_num_imp_varL2))

# Add back the ID column for merging
prevent_x_num_imp_varL2_std$ID <- prevent_x_num_imp_varL2$ID

# Merge the scaled numeric data with the identifier columns, using 'ID' to align data
result_data_std <- merge(id_data, prevent_x_num_imp_varL2_std, by = "ID")
result_data <- merge(id_data, prevent_x_num_imp_varL2, by = "ID")

# Remove the ID column as it's no longer needed
result_data_std$ID <- NULL
result_data$ID <- NULL

# Function to drop columns with less than x distinct values, except specified columns
drop_columns <- function(data, threshold, keep_cols) {
  # Get the number of distinct values for each column
  distinct_counts <- sapply(data, dplyr::n_distinct)

  # Identify columns to keep (those with >= threshold distinct values or in keep_cols)
  cols_to_keep <- names(distinct_counts)[distinct_counts >= threshold | names(distinct_counts) %in% keep_cols]

  # Select only those columns
  data %>% dplyr::select(all_of(cols_to_keep))
}

# Apply the function to filter low-variance categorical variables
filtered_df <- drop_columns(result_data, 10, c("CENTER","VISIT"))
filtered_df_std <- drop_columns(result_data_std, 10, c("CENTER","VISIT"))

cat("Final dataset dimensions:\n")
cat("Unscaled:", nrow(filtered_df), "observations x", ncol(filtered_df), "variables\n")
cat("Scaled:", nrow(filtered_df_std), "observations x", ncol(filtered_df_std), "variables\n")

# Display variable summary
cat("\nFinal variables included:\n")
var_names <- setdiff(names(filtered_df), c("CENTER", "PID", "VISIT"))
cat("Clinical variables:", length(var_names), "\n")

# Create processed data directory
dir.create("data/processed", showWarnings = FALSE, recursive = TRUE)

# Save the processed datasets
write.csv(filtered_df, "./data/processed/prevent_num_imp_varL2.csv", row.names = FALSE)
write.csv(filtered_df_std, "./data/processed/prevent_num_imp_varL2_std.csv", row.names = FALSE)

cat("\n✓ Processed synthetic datasets saved successfully!\n")
cat("✓ Files created:\n")
cat("  - data/processed/prevent_num_imp_varL2.csv (unscaled)\n")
cat("  - data/processed/prevent_num_imp_varL2_std.csv (scaled)\n")
cat("\nNext steps:\n")
cat("1. Rename one of these files to 'prevent_direct_train_data.csv'\n")
cat("2. Run the main Python analysis script: python src/singlesite_prevent.script.py\n")

# Display some summary statistics for verification
cat("\n--- Dataset Summary ---\n")
cat("Patients by center:\n")
filtered_df %>%
  group_by(CENTER) %>%
  summarise(n_observations = n()) %>%
  print()

cat("\n[1] Outcome variable (Y) summary:\n")
if ("Y" %in% names(filtered_df)) {
  print(summary(filtered_df$Y))
} else {
  cat("Warning: Outcome variable 'Y' not found in final dataset\n")
}
