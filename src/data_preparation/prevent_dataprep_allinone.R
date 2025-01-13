library(haven)
library(DataExplorer)
library(dplyr)

# Load data
prevent_file = "./data/raw/prevent_st2.sas7bdat"
prevent <- read_sas(prevent_file)

# Replace empty strings with NA
prevent[prevent == ""] = NA

prevent |>
  group_by(CENTER) |>
  summarise(cnt = n_distinct(PID))

# Step 1: Filter out patients where `URTI` occurs before the first `Visit 1`
filtered_data <- prevent %>%
  group_by(PID) %>%
  arrange(PID, VISITDC_W) %>%  # Order by patient and visit date
  mutate(
    first_visit1_date = min(VISITDC_W[grepl("V1", VISIT)], na.rm = TRUE) ,
    first_visit2_date = min(VISITDC_W[grepl("V2", VISIT)], na.rm = TRUE) ,
    first_visit3_date = min(VISITDC_W[grepl("V3", VISIT)], na.rm = TRUE) # Find the first Visit 1 date using regex
  ) %>%
  filter(!(any(VISIT == "URTI" & VISITDC_W < first_visit1_date))) %>%
  filter(!(any(VISIT == "URTI" & VISITDC_W < first_visit2_date))) %>%
  filter(!(any(VISIT == "URTI" & VISITDC_W < first_visit3_date))) %>% # Exclude patients with URTI before Visit 1
  ungroup()

# Step 2: Filter only for patients with both a baseline visit (contains "V0") and visit 1 (contains "V1")
final_filtered_data <- filtered_data

n_distinct(final_filtered_data$PID)

prevent <- final_filtered_data

# Create identifiers for merging later
prevent$ID = seq_len(nrow(prevent))
id_data = prevent[, c("ID", "CENTER", "PID", "VISIT")]

prevent$PID  <- as.character(prevent$PID)

# Filter rows based on visit codes
prevent_visits = prevent[grepl("V[0-6]", prevent$VISIT),]

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

    # Loop through each column and fill values
    for (col in names(data)) {
      if (!col %in% c(id_col, visit_col) && !is.na(baseline_values[[col]])) {
        data[data[[id_col]] == pid & is.na(data[[col]]), col] <- baseline_values[[col]]
      }
    }
  }

  return(data)
}

# Apply the function to fill forward baseline values
prevent_visits <- fill_baseline_values(prevent_visits, id_col, visit_col, baseline_visit)

# Calculate the frequency of non-missing values by column and row
prevent_visits_nafreq_cols = 1 - (colSums(is.na(prevent_visits)) / nrow(prevent_visits))
prevent_visits_nafreq_rows = 1 - (rowSums(is.na(prevent_visits)) / ncol(prevent_visits))

# Filter data based on non-missing values
prevent_visits = prevent_visits[prevent_visits_nafreq_rows > 0.10,]
prevent_x = prevent_visits[, prevent_visits_nafreq_cols > 0.8]

# Save row names for later use
row_ids = prevent_x$ID

# Select only numeric columns for processing
numeric_columns = sapply(prevent_x, is.numeric)
prevent_x_num = prevent_x[, numeric_columns]

library(VIM)
prevent_x_num <- kNN(prevent_x_num, k = 5, imp_var = FALSE)

# Calculate variance
var_prevent_x_num_imp = apply(prevent_x_num, 2, var)

# Filter columns based on variance
prevent_x_num_imp_varL2 = prevent_x_num[, var_prevent_x_num_imp > 0.2]

# Scaled version
prevent_x_num_imp_varL2_std = as.data.frame(scale(prevent_x_num_imp_varL2))

prevent_x_num_imp_varL2_std$ID <- prevent_x_num_imp_varL2$ID

# Merge the scaled numeric data with the identifier columns, using 'ID' to align data
result_data_std = merge(id_data, prevent_x_num_imp_varL2_std, by = "ID")
result_data = merge(id_data, prevent_x_num_imp_varL2, by = "ID")

# Optionally, clean up the 'ID' column if it's no longer needed
result_data_std$ID = NULL
result_data$ID = NULL

# Function to drop columns with less than x distinct values, except 'CENTER'
drop_columns <- function(data, threshold, keep_cols) {
  # Get the number of distinct values for each column
  distinct_counts <- sapply(data, dplyr::n_distinct)

  # Identify columns to keep (those with >= threshold distinct values or the CENTER column)
  cols_to_keep <- names(distinct_counts)[distinct_counts >= threshold | names(distinct_counts) %in% keep_cols]

  # Select only those columns
  data |>  dplyr::select(all_of(cols_to_keep))
}

# Apply the function to the sample data frame
filtered_df <- drop_columns(result_data, 10, c("CENTER","VISIT"))
filtered_df_std <- drop_columns(result_data_std, 10, c("CENTER","VISIT"))

# Save or process your result_data as needed
write.csv(filtered_df, "./data/processed/prevent_num_imp_varL2.csv", row.names = FALSE)
write.csv(filtered_df_std, "./data/processed/prevent_num_imp_varL2_std.csv", row.names = FALSE)
