import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_plotly_events import plotly_events
import sys
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression, LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import RFE
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from scipy.stats import ttest_ind
import statsmodels.api as sm
from sklearn.tree import DecisionTreeRegressor

# Call the function
sys.path.append('src/utilities')  # Add the directory to the module search path

import weights  
import loregs

import plotly.io as pio
pio.templates.default = "plotly"

# Title of the app
st.title('Autoencoder Evaluation')

# File uploader widget
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")


if uploaded_file is not None:

    # Extract hyperparameters from file name
    filename = uploaded_file.name
    hyperparameters = filename.split('_')

    def extract_hyperparameters(hyperparams):
        extracted_params = {}
        for param in hyperparams:
            # Check if the parameter contains any digit
            if any(char.isdigit() for char in param):
                # Find the position where the number starts
                for i, char in enumerate(param):
                    if char.isdigit():
                        param_name = param[:i]
                        param_value = param[i:].rstrip('.csv')  # Remove file extension if present
                        extracted_params[param_name] = param_value
                        break
        return extracted_params
    
    extracted_hyperparameters = extract_hyperparameters(hyperparameters)
    st.write(extracted_hyperparameters)

    # Read the CSV file
    data = pd.read_csv(uploaded_file)

    # Decide if only training or all data should be used
    use_all_data = False
    use_all_data = st.checkbox("Use all data (including test data)")

    if use_all_data:
        pass
    else:
        data = data[data['train'] == 1]

    # Select only numeric columns
    numeric_data = data.select_dtypes(include=[np.number])

    # Dynamically determine latent size
    latent_columns = [col for col in data.columns if col.startswith('Latent')]
    latent_size = len(latent_columns)

    # Initialize dictionaries to store results
    top_ttest_dict = {latent_dim: [] for latent_dim in latent_columns}
    top_ttest_vars_dict = {latent_dim: [] for latent_dim in latent_columns}

    # Show correlation matrix of latent dimensions
    st.write("Correlation Matrix of Latent Dimensions")
    st.write(data[latent_columns].corr())

    # Loop through each latent dimension
    for latent_dim in latent_columns:
        # Determine median for categorization
        latent_median = data[latent_dim].median()
        # latent_median = 0
        latent_category = np.where(data[latent_dim] > latent_median, 1, -1)

        # Initialize a dictionary to store T-statistics and p-values
        ttest_results = {}

        # Loop through each numeric variable (excluding latent dimensions)
        for var in numeric_data.columns.drop(latent_columns + ['Y']):
            # Perform a t-test
            observed_variable = data[var]
            t_stat, p_value = ttest_ind(observed_variable[latent_category == 1], 
                                        observed_variable[latent_category == -1], 
                                        equal_var=False)
            
            # Store the t-statistic and p-value in the dictionary
            ttest_results[var] = (t_stat, p_value)
        
        # Sort by absolute t-statistic and get top 10 variables
        sorted_vars = sorted(ttest_results, key=lambda x: abs(ttest_results[x][0]), reverse=True)[:10]
        sorted_vars_all = sorted(ttest_results, key=lambda x: abs(ttest_results[x][0]), reverse=True)
        top_ttest_dict[latent_dim] = [f"{var} (t={ttest_results[var][0]:.2f})" for var in sorted_vars]
        top_ttest_vars_dict[latent_dim] = sorted_vars_all

    # Calculate unique variables for each latent dimension
    unique_vars_dict = {latent_dim: set(vars_list) for latent_dim, vars_list in top_ttest_vars_dict.items()}

    # Identify variables that appear in more than one latent dimension
    non_unique_vars = set(var for vars_list in top_ttest_vars_dict.values() for var in vars_list if sum(var in v for v in top_ttest_vars_dict.values()) > 1)

    # Remove non-unique variables from each set in unique_vars_dict
    for latent_dim in unique_vars_dict:
        unique_vars_dict[latent_dim] -= non_unique_vars

    # Create a DataFrame from the top_ttest_dict
    top_ttest_df = pd.DataFrame(top_ttest_dict)
    top_ttest_vars = pd.DataFrame(top_ttest_vars_dict)

    # Function to apply highlighting
    def highlight_unique_vars(s):
        unique_vars = unique_vars_dict[s.name]
        return ['background-color: green' if var.split(' ')[0] in unique_vars else '' for var in s]

    # Apply the highlighting function
    styled_df = top_ttest_df.style.apply(highlight_unique_vars, axis=0)

    # Display the results in a color-coded table
    st.write("Top 10 T-Test Variables for Each Latent Dimension:")
    st.dataframe(styled_df)

    # Initialize a dictionary to store the results
    top_mi_dict = {latent_dim: [] for latent_dim in latent_columns}
    top_mi_vars_dict = {latent_dim: [] for latent_dim in latent_columns}

    # Loop through each latent dimension
    for latent_dim in latent_columns:
        # Calculate mutual information of the current latent dimension with other variables
        mi_scores = [
            mutual_info_regression(
                numeric_data[[var]].values, numeric_data[latent_dim].values
            )[0]
            for var in numeric_data.columns if var not in latent_columns + ['Y']
        ]
        
        # Create a DataFrame with MI scores for each variable
        mi_df = pd.DataFrame(mi_scores, index=[var for var in numeric_data.columns if var not in latent_columns + ['Y']], columns=['MI'])

        # Get top 10 variables based on MI score
        top_10_mi_vars = mi_df['MI'].nlargest(100).index.tolist()
        
        # Store the top 10 MI variables with scores in the dictionary
        top_mi_dict[latent_dim] = [
            f"{var} ({mi_df.loc[var, 'MI']:.2f})" for var in top_10_mi_vars
        ]
        
        top_mi_vars_dict[latent_dim] = top_10_mi_vars

    # Calculate unique variables for each latent dimension
    unique_vars_dict = {latent_dim: set(vars_list) for latent_dim, vars_list in top_mi_vars_dict.items()}

    # Identify variables that appear in more than one latent dimension
    non_unique_vars = set(var for vars_list in top_mi_vars_dict.values() for var in vars_list if sum(var in v for v in top_mi_vars_dict.values()) > 1)

    # Remove non-unique variables from each set in unique_vars_dict
    for latent_dim in unique_vars_dict:
        unique_vars_dict[latent_dim] -= non_unique_vars

    # Create a DataFrame from the top_mi_dict
    top_mi_df = pd.DataFrame(top_mi_dict)

    # Function to apply highlighting
    def highlight_unique_vars(s):
        unique_vars = unique_vars_dict[s.name]
        return ['background-color: green' if var.split(' ')[0] in unique_vars else '' for var in s]

    # Apply the highlighting function
    styled_df = top_mi_df.style.apply(highlight_unique_vars, axis=0)

    # Display the results in a color-coded table
    st.write("Top 10 Variables with Highest MI for Each Latent Dimension:")
    st.dataframe(styled_df)

    #######
    # Add a section to display the latent space patterns
    #######

    # Print global coefficients
    st.subheader("Global Coefficients")

    # Get the latent space for target and external data
    latent_data = data[latent_columns].values

    # Fit a global regression model with OLS
    X_global = sm.add_constant(latent_data)
    y_global = data['Y'].values
    model = sm.OLS(y_global, X_global)
    results = model.fit()

    st.write(results.summary())
    
    # Extract coefficients and confidence intervals
    summary_df = pd.DataFrame({
        'Coefficient': results.params,
        'Lower CI': results.conf_int()[:, 0],
        'Upper CI': results.conf_int()[:, 1]
    })

    st.dataframe(summary_df)

    st.header("Latent Space Patterns")

    # Add widgets for sigma, k_nearest, and kernel
    sigma = st.slider("Sigma", min_value=0.1, max_value=2.0, value=float(extracted_hyperparameters['sigma']), step=0.1)
    k_nearest = st.slider("k Nearest", min_value=0.0, max_value=1.0, value=float(extracted_hyperparameters['nearest']), step=0.1)
    kernel = "gaussian"

    selected_latent_dim = st.selectbox("Select Latent Dimension", latent_columns)
    
    # get last character of the selected_latent_dim
    selected_latent_num = int(selected_latent_dim[-1])

    ############################
    # Display scatter plot of latent dimensions vs Y
    ############################

    y_axis_var = 'Y'

    # Initialize lists to store absolute coefficients
    beta_abs_all = []

    # Loop over each latent dimension and each point in the data
    for i in range(len(latent_data)):
        beta_abs_per_sample = []
        for latent_num in range(len(latent_columns)):
                
            # Compute weights for the current observation
            weights_single = weights.ss_batch_weights(latent_data[i], latent_data, sigma=sigma, k_nearest=k_nearest, kernel=kernel)
        
            # Fit regression for the current latent dimension
            beta_single = loregs.batch_weighted_regression(X=latent_data, y=data['Y'].values, weights=weights_single, intercept=True)
            
            # Store the absolute beta coefficient
            beta_abs_per_sample.append(beta_single[latent_num+1])
        
        # After looping over all latent dimensions for this sample, store the absolute coefficients
        beta_abs_all.append(beta_abs_per_sample)


    # Fit global only on training data
    latent_data_filtered = latent_data[data['train'] == 1]
    data_filtered = data[data['train'] == 1]

    # Make a vector of weights of 1 for the global regression model
    weights_global = np.ones(len(latent_data))
    
    # Fit a global regression model
    beta_global = loregs.batch_weighted_regression(X=latent_data, y=data['Y'].values, weights=weights_global, intercept=True)

    # Convert beta coefficients to a DataFrame
    beta_abs_df = pd.DataFrame(beta_abs_all, columns=latent_columns)

    # Difference to global coefficients
    beta_global_diff = beta_abs_df - beta_global[1:]

    # Transform to z-scores with mean 0
    z_min_abs = beta_abs_df[selected_latent_dim].min().min()
    z_max_abs = beta_abs_df[selected_latent_dim].max().max()
    z_min_diff = beta_global_diff[selected_latent_dim].min().min()
    z_max_diff = beta_global_diff[selected_latent_dim].max().max()

    # Train a regression tree with user-defined max depth
    max_depth = st.slider("Select Max Depth of the Tree", 1, 10, 4)
    show_what = st.selectbox("Show", ["Absolute Coefficients", "Difference to Global Coefficients"])

    if show_what == "Absolute Coefficients":
        # Train the decision tree on the chosen outcome
        tree_model = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=20)
        tree_model.fit(data[latent_columns + [y_axis_var]], beta_abs_df)
    elif show_what == "Difference to Global Coefficients":
        # Train the decision tree on the chosen outcome
        tree_model = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=20)
        tree_model.fit(data[latent_columns + [y_axis_var]], beta_global_diff)

    # Apply the trained tree to the data
    tree_regions = tree_model.apply(data[latent_columns + [y_axis_var]])

    # Add the tree regions to the data frame
    data['Tree_Region'] = tree_regions.astype(str)

    from plotly.subplots import make_subplots
    from math import ceil

    # Determine the number of latent dimensions
    num_latent_dims = len(latent_columns)

    # Calculate the number of rows and columns for the subplot
    num_cols = 2
    num_rows = ceil(num_latent_dims / num_cols)

    # Create subplots
    fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=latent_columns, vertical_spacing=0.1)

    # Center color gradient around zero for absolute coefficients and differences
    cabs = 0.1
    cdiff = 0.1

    # Loop through latent dimensions to create plots
    for i, latent_dim in enumerate(latent_columns):
        row = (i // num_cols) + 1
        col = (i % num_cols) + 1

        if show_what == "Absolute Coefficients":
            fig.add_trace(
                go.Scatter(
                    x=data[latent_dim], 
                    y=data[y_axis_var],
                    mode="markers",
                    marker=dict(
                        color=beta_abs_df[latent_dim].values,
                        colorscale="Spectral",
                        colorbar=dict(title="Coef", x=1.0, y=0.55),
                        cmin=-cabs,
                        cmax=cabs,
                        size=12,
                        # Contour-like line based on train column
                        line=dict(
                            color=np.where(data['train'] == 0, "red", "black"),  # Add black outline for train == 0
                            width=np.where(data['train'] == 0, 2, 0)  # Adjust the width for train == 0
                        ),
                    ),
                    name=f"Latent: {latent_dim}"
                ),
                row=row,
                col=col
            )
        elif show_what == "Difference to Global Coefficients":
            fig.add_trace(
                go.Scatter(
                    x=data[latent_dim],
                    y=data[y_axis_var],
                    mode="markers",
                    marker=dict(
                        color=beta_global_diff[latent_dim].values,
                        colorscale="Spectral",
                        colorbar=dict(title="Coef Diff", x=1.0, y=0.55),
                        cmin=-cdiff,
                        cmax=cdiff,
                        size=12,
                        line=dict(
                            color=np.where(data['train'] == 0, "red", "black"),  # Add black outline for train == 0
                            width=np.where(data['train'] == 0, 2, 0)  # Adjust the width for train == 0
                        ),
                    ),
                    name=f"Latent: {latent_dim}"
                ),
                row=row,
                col=col
            )

    # Adjust layout
    fig.update_layout(
        height=400 * num_rows,  # Adjust height dynamically based on the number of rows
        width=800,
        title_text="Latent Dimensions and Their Effects",
        showlegend=False,
        # no gray background
        plot_bgcolor='rgba(0,0,0,0)',
    )

    fig.update_yaxes(
        showgrid=True,          # Enable horizontal gridlines
        gridcolor='lightgray',  # Set the color of the gridlines
        gridwidth=0.5           # Set the width of the gridlines
    )
 
    fig.update_yaxes(
        zeroline=True,          # Show the line at y=0
        zerolinewidth=0.5,        # Adjust width
        zerolinecolor='lightgray'    # Customize color
    )

    fig.write_image("../results/figures/paper_temp/latent_space_patterns.pdf")
    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)
                                                                                                                                   

    # Multi-select box to select data based on tree region
    selected_tree_region = st.multiselect("Select Tree Region", options=data['Tree_Region'].unique())
    data_beta_global = pd.concat([data, beta_global_diff.add_suffix('_beta')], axis=1)

    # Create subplots
    fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=latent_columns, vertical_spacing=0.1)

    beta_diff_threshold = st.slider("Beta Difference Threshold", min_value=-0.1, max_value=0.1, value=0.0, step=0.005)

    # check if filter only based on beta difference
    only_beta_diff = st.checkbox("Filter only based on beta difference")

    # Loop through each latent dimension to create individual plots
    for i, latent_dim in enumerate(latent_columns):
        row = (i // num_cols) + 1
        col = (i % num_cols) + 1

        # Base scatter plot
        fig.add_trace(
            go.Scatter(
                x=data[latent_dim],
                y=data[y_axis_var],
                mode="markers",
                marker=dict(
                    color=data_beta_global[latent_dim + '_beta'].values,
                    colorscale="Spectral",
                    cmin=-cdiff,
                    cmax=cdiff,
                    size=12,
                    line=dict(
                            color=np.where(data['train'] == 0, "red", "black"),  # Add black outline for train == 0
                            width=np.where(data['train'] == 0, 2, 0)  # Adjust the width for train == 0
                        ),
                ),
                opacity=0.6,
                name=f"Latent {latent_dim}",
            ),
            row=row,
            col=col,
        )

        # Add contour for selected tree regions
        if selected_tree_region and not only_beta_diff:
            unique_regions_target = list(set(selected_tree_region))

            for region in unique_regions_target:
                # Filter data for the current region
                region_data = data[data['Tree_Region'] == region]

                st.write(len(region_data),"in region", region)

                # Filter data beta global to select tree region and beta difference
                data_beta_global_filtered = data_beta_global[data['Tree_Region'] == region]

                if beta_diff_threshold > 0:
                    region_data_filtered = region_data[
                        (data_beta_global[data['Tree_Region'] == region][selected_latent_dim + '_beta'] > beta_diff_threshold)
                    ]
                    data_beta_global_filtered = data_beta_global_filtered[
                        (data_beta_global[selected_latent_dim + '_beta'] > beta_diff_threshold)
                    ]

                else:
                    region_data_filtered = region_data[
                        (data_beta_global[data['Tree_Region'] == region][selected_latent_dim + '_beta'] < beta_diff_threshold)
                    ]
                    data_beta_global_filtered = data_beta_global_filtered[
                        (data_beta_global[selected_latent_dim + '_beta'] < beta_diff_threshold)
                    ]


                fig.add_trace(
                    go.Scatter(
                        x=region_data_filtered[latent_dim],
                        y=region_data_filtered[y_axis_var],
                        mode="markers",
                        marker=dict(
                            color=data_beta_global_filtered[latent_dim + '_beta'].values,
                            colorscale="Spectral",
                            cmin=-cdiff,
                            cmax=cdiff,
                            size=12,
                            line=dict(
                                color=np.where(region_data_filtered['train'] == 0, "red", "black"),  # Add black outline for train == 0
                                width=np.where(region_data_filtered['train'] == 0, 2, 1)  # Adjust the width for train == 0
                            ),
                        ),
                        opacity=1.0,  # Full opacity for selected data
                        name=f"Tree Region {region}",
                    ),
                    row=row,
                    col=col,
                )

        if only_beta_diff:

            if beta_diff_threshold > 0:
                data_filtered = data[
                    (data_beta_global[selected_latent_dim + '_beta'] > beta_diff_threshold)
                ]
                data_beta_global_filtered = data_beta_global[
                    (data_beta_global[selected_latent_dim + '_beta'] > beta_diff_threshold)
                ]

            else:
                data_filtered = data[
                    (data_beta_global[selected_latent_dim + '_beta'] < beta_diff_threshold)
                ]
                data_beta_global_filtered = data_beta_global[
                    (data_beta_global[selected_latent_dim + '_beta'] < beta_diff_threshold)
                ]

            fig.add_trace(
                go.Scatter(
                    x=data_filtered[latent_dim],
                    y=data_filtered[y_axis_var],
                    mode="markers",
                    marker=dict(
                        color=data_beta_global_filtered[latent_dim + '_beta'].values,
                        colorscale="Spectral",
                        cmin=-cdiff,
                        cmax=cdiff,
                        size=12,
                        line=dict(
                            color=np.where(data_filtered['train'] == 0, "red", "black"),  # Add black outline for train == 0
                            width=np.where(data_filtered['train'] == 0, 2, 2)  # Adjust the width for train == 0
                        ),
                    ),
                    opacity=1.0,
                    name=f"Latent {latent_dim}",
                ),
                row=row,
                col=col,
            )

    # Adjust layout
    fig.update_layout(
        height=400 * num_rows,  # Adjust height dynamically based on the number of rows
        width=800,
        title_text="Latent Dimensions with Tree Region Contours",
        showlegend=False,
        # no gray background
        plot_bgcolor='rgba(0,0,0,0)',
    )

    fig.update_yaxes(
        showgrid=True,          # Enable horizontal gridlines
        gridcolor='lightgray',  # Set the color of the gridlines
        gridwidth=0.5           # Set the width of the gridlines
    )
 
    fig.update_yaxes(
        zeroline=True,          # Show the line at y=0
        zerolinewidth=0.5,        # Adjust width
        zerolinecolor='lightgray'    # Customize color
    )

    fig.write_image("../results/figures/paper_temp/latent_space_groups.pdf")

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Determine if an index is selected via clicking or input box or a tree region is selected
    if len(selected_tree_region) > 0 or only_beta_diff:

        if not only_beta_diff:
            selected_data = data[data['Tree_Region'].isin(selected_tree_region)]
        else:
            selected_data = data
        
        if beta_diff_threshold > 0:
            selected_data = selected_data[
                data_beta_global[selected_latent_dim + '_beta'] > beta_diff_threshold
            ]
        else:
            selected_data = selected_data[
                data_beta_global[selected_latent_dim + '_beta'] < beta_diff_threshold
            ]

        # Display number of points selected
        st.write(f"Number of points selected: {len(selected_data)}")

        selected_data['subgroup_flag'] = 1

        # non-selected data
        non_selected_data = data.loc[~data.index.isin(selected_data.index)]
        non_selected_data['subgroup_flag'] = 0

        # Combine selected and non-selected data
        combined_data = pd.concat([selected_data, non_selected_data])
        latent_combined = combined_data[latent_columns]
        X_combined = combined_data[top_10_mi_vars]
        y = combined_data['Y'].values

        latent_subgroup = latent_combined[combined_data['subgroup_flag'] == 1]
        latent_nonsubgroup = latent_combined[combined_data['subgroup_flag'] == 0]

        y_subgroup = y[combined_data['subgroup_flag'] == 1]
        y_nonsubgroup = y[combined_data['subgroup_flag'] == 0]

        # Add constant
        latent_subgroup = sm.add_constant(latent_subgroup)
        latent_nonsubgroup = sm.add_constant(latent_nonsubgroup)

        # Fit linear regresssion model
        model_subgroup = sm.OLS(y_subgroup, latent_subgroup).fit()
        model_nonsubgroup = sm.OLS(y_nonsubgroup, latent_nonsubgroup).fit()

        # Extract coefficients and confidence intervals
        coefficients_subgroup = model_subgroup.params
        conf_int_subgroup = model_subgroup.conf_int()

        coefficients_non_subgroup = model_nonsubgroup.params
        conf_int_non_subgroup = model_nonsubgroup.conf_int()

        summary_combined = pd.DataFrame({
            'Variable': coefficients_subgroup.index,
            'Coefficient_1': coefficients_subgroup.values,
            'ConfInt Lower_1': conf_int_subgroup[0].values,
            'ConfInt Upper_1': conf_int_subgroup[1].values,
            'Coefficient_0': coefficients_non_subgroup.values,
            'ConfInt Lower_0': conf_int_non_subgroup[0].values,
            'ConfInt Upper_0': conf_int_non_subgroup[1].values
        })

        fig_forest = go.Figure()

        # Add traces for subgroup model
        fig_forest.add_trace(go.Scatter(
            x=summary_combined['Coefficient_1'],
            y=summary_combined['Variable'],
            error_x=dict(
                type='data',
                symmetric=False,
                array=summary_combined['ConfInt Upper_1'] - summary_combined['Coefficient_1'],
                arrayminus=summary_combined['Coefficient_1'] - summary_combined['ConfInt Lower_1']
            ),
            mode='markers',
            name='Subgroup 1',
            marker=dict(color='blue')
        ))

        # Add traces for non-subgroup model
        fig_forest.add_trace(go.Scatter(
            x=summary_combined['Coefficient_0'],
            y=summary_combined['Variable'],
            error_x=dict(
                type='data',
                symmetric=False,
                array=summary_combined['ConfInt Upper_0'] - summary_combined['Coefficient_0'],
                arrayminus=summary_combined['Coefficient_0'] - summary_combined['ConfInt Lower_0']
            ),
            mode='markers',
            name='Subgroup 0',
            marker=dict(color='red')
        ))

        # Add vertical line at zero
        fig_forest.add_shape(
            dict(
                type="line",
                x0=0,
                y0=-1,
                x1=0,
                y1=len(summary_combined),
                line=dict(color="black", width=2, dash="dashdot")
            )
        )

        fig_forest.update_layout(
            title='Coefficients by Subgroup',
            xaxis_title='Coefficient',
            yaxis_title='Variables',
            showlegend=True
        )

        # Display the plot
        st.plotly_chart(fig_forest)

        # Select top 3 variables based on some criteria (e.g., top_ttest_vars)
        top_3_vars = top_ttest_vars[selected_latent_dim][:5]  # Replace with the actual selection

        # top_3_vars = top_mi_vars_dict[selected_latent_dim][:5] # Replace with the actual selection
        X_top = X_combined[top_3_vars]

        # Split data based on subgroup_flag
        subgroup_1_mask = combined_data['subgroup_flag'] == 1
        subgroup_0_mask = combined_data['subgroup_flag'] == 0

        # Initialize summary dataframe
        summary_combined = pd.DataFrame()

        # Loop through each variable in top_3_vars
        for var in top_3_vars:
            X_var_interaction = X_combined[[var]]
            X_var_interaction[var + '_interaction'] = X_var_interaction[var] * combined_data['subgroup_flag']
            # Main effect of subgroup
            X_var_interaction['subgroup_flag'] = combined_data['subgroup_flag']
            X_var_interaction = sm.add_constant(X_var_interaction)
            model_interaction = sm.OLS(combined_data['Y'], X_var_interaction).fit()

            # Fit model for Subgroup 1
            X_var_subgroup_1 = sm.add_constant(X_combined[[var]].loc[subgroup_1_mask])
            model_1 = sm.OLS(combined_data['Y'].loc[subgroup_1_mask], X_var_subgroup_1).fit()
            
            # Fit model for Subgroup 0
            X_var_subgroup_0 = sm.add_constant(X_combined[[var]].loc[subgroup_0_mask])
            model_0 = sm.OLS(combined_data['Y'].loc[subgroup_0_mask], X_var_subgroup_0).fit()
            
            # Collect results
            summary_combined = pd.concat([summary_combined, pd.DataFrame({
                'Variable': [var],
                'Coefficient_1': [model_interaction.params[var]],
                'ConfInt Lower_1': [model_interaction.conf_int().loc[var, 0]],
                'ConfInt Upper_1': [model_interaction.conf_int().loc[var, 1]],
                'Coefficient_0': [model_0.params[var]],
                'ConfInt Lower_0': [model_0.conf_int().loc[var, 0]],
                'ConfInt Upper_0': [model_0.conf_int().loc[var, 1]],
                'Coefficient_Interaction': [model_interaction.params[var + '_interaction']],
                'ConfInt Lower_Interaction': [model_interaction.conf_int().loc[var + '_interaction', 0]],
                'ConfInt Upper_Interaction': [model_interaction.conf_int().loc[var + '_interaction', 1]]
            })], ignore_index=True)

        # Create forest plot
        fig_forest = go.Figure()

        summary_combined['y_numeric'] = pd.factorize(summary_combined['Variable'])[0]

        # Apply a slight jitter
        jitter_main = -0.1
        jitter_interaction = 0.1

        st.write(summary_combined)

        # Add traces for each subgroup with distinct colors
        fig_forest.add_trace(go.Scatter(
            x=summary_combined['Coefficient_1'],
            y=summary_combined['y_numeric'] + jitter_main,
            error_x=dict(
                type='data',
                symmetric=False,
                array=summary_combined['ConfInt Upper_1'] - summary_combined['Coefficient_1'],
                arrayminus=summary_combined['Coefficient_1'] - summary_combined['ConfInt Lower_1']
            ),
            mode='markers',
            name='Main Effect',
            marker=dict(color='blue')
        ))

        fig_forest.add_trace(go.Scatter(
            x=summary_combined['Coefficient_Interaction'],
            y=summary_combined['y_numeric'] - jitter_main,
            error_x=dict(
                type='data',
                symmetric=False,
                array=summary_combined['ConfInt Upper_Interaction'] - summary_combined['Coefficient_Interaction'],
                arrayminus=summary_combined['Coefficient_Interaction'] - summary_combined['ConfInt Lower_Interaction']
            ),
            mode='markers',
            name='Interaction',
            marker=dict(color='green')
        ))

        # Add vertical line at zero
        fig_forest.add_shape(
            dict(
                type="line",
                x0=0,
                y0=-1,
                x1=0,
                y1=len(summary_combined),
                line=dict(color="black", width=2, dash="dashdot")
            )
        )

        fig_forest.update_yaxes(
            showgrid=True,          # Enable horizontal gridlines
            gridcolor='lightgray',  # Set the color of the gridlines
            gridwidth=0.5           # Set the width of the gridlines
        )
    
        fig_forest.update_yaxes(
            zeroline=True,          # Show the line at y=0
            zerolinewidth=0.5,        # Adjust width
            zerolinecolor='lightgray'    # Customize color
        )

        fig_forest.update_layout(
            title='Coefficients by Subgroup',
            xaxis_title='Coefficient',
            yaxis_title='Variables',
            showlegend=True,
            # no gray background
            plot_bgcolor='rgba(0,0,0,0)',
        )

        fig_forest.write_image("../results/figures/paper_temp/interaction_forest.pdf")

        # Display the plot
        st.plotly_chart(fig_forest)

        # Use hierarchical clustering to group correlated variables
        corr_matrix = X_combined.corr()
        
        # convert X_Comined to numpy array
        X_combined_scaled = X_combined.to_numpy()
        
        # Population means
        population_means = X_combined.mean()
        
        # Subgroup means
        # subgroup_means = X_combined[combined_data['subgroup_flag'] == 1].mean(axis = 0)
        subgroup_means = X_combined[(combined_data['subgroup_flag'] == 1) & (combined_data['train'] == 1)].mean(axis=0)
        st.write("No. of train data in subgroup: ", len(X_combined[(combined_data['subgroup_flag'] == 1) & (combined_data['train'] == 1)]))
        st.write("No. of test data in subgroup: ", len(X_combined[(combined_data['subgroup_flag'] == 1) & (combined_data['train'] == 0)]))
        mean_diff = subgroup_means - population_means

        # Plotting
        # Create a DataFrame for Plotly with variable name and Z-score difference
        plot_data = pd.DataFrame({
            'Variable': X_combined.columns,
            'Z-score Difference': mean_diff,
            'Cluster': 'test',
        })

        cluster_names = {
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

        plot_data['Cluster'] = plot_data['Variable'].map(cluster_names)

        # Average Z-score difference for each cluster
        cluster_means = plot_data.groupby('Cluster')['Z-score Difference'].mean().reset_index()

        # Plot the average Z-score difference for each cluster
        fig_cluster = px.bar(
            cluster_means,
            y='Cluster',
            x='Z-score Difference',
            orientation='h',
            color='Z-score Difference',
            color_continuous_scale='RdBu',
            # Range of x-axis
            range_x=[-1, 1],
            # Range of color
            range_color=[-1, 1],
            title="Average Z-Score Difference for Each Cluster",
            labels={'Z-score Difference': 'Average Z-score Difference', 'Cluster': 'Cluster'},
        )

        fig_cluster.update_layout(
            height=800,
            width=800,
            yaxis_title="Cluster",
            xaxis_title="Average Z-score Difference",
            template="plotly_white",
        )

        fig_cluster.write_image("../results/figures/paper_temp/z_profile.pdf")

        # Display the plot in Streamlit
        st.plotly_chart(fig_cluster)

        # Make same plot for test data only

        # Subgroup means
        subgroup_means_test = X_combined[(combined_data['subgroup_flag'] == 1) & (combined_data['train'] == 0)].mean(axis=0)

        mean_diff_test = subgroup_means_test - population_means

        # Plotting
        # Create a DataFrame for Plotly with variable name and Z-score difference
        plot_data_test = pd.DataFrame({
            'Variable': X_combined.columns,
            'Z-score Difference': mean_diff_test,
            'Cluster': 'test'
        })

        plot_data_test['Cluster'] = plot_data_test['Variable'].map(cluster_names)

        # Average Z-score difference for each cluster
        cluster_means_test = plot_data_test.groupby('Cluster')['Z-score Difference'].mean().reset_index()

        # Plot the average Z-score difference for each cluster
        fig_cluster_test = px.bar(
            cluster_means_test,
            y='Cluster',
            x='Z-score Difference',
            orientation='h',
            color='Z-score Difference',
            color_continuous_scale='RdBu',
            # Range of x-axis
            range_x=[-1, 1],
            # Range of color
            range_color=[-1, 1],
            title="Average Z-Score Difference for Test Data",
            labels={'Z-score Difference': 'Average Z-score Difference', 'Cluster': 'Cluster'},
        )

        fig_cluster_test.update_layout(
            height=800,
            width=800,
            yaxis_title="Cluster",
            xaxis_title="Average Z-score Difference",
            template="plotly_white",
        )

        fig_cluster_test.write_image("../results/figures/paper_temp/z_profile_test.pdf")


        st.plotly_chart(fig_cluster_test)

        # Write section header
        st.subheader("Localized Regression for Single Latent Dimension")

        selected_point = 2

        # Compute weights for one observation in the target data
        weights_target_point = weights.ss_batch_weights(latent_data[selected_point], latent_data, sigma=sigma, k_nearest=k_nearest, kernel=kernel)

        dynamic_data = data.copy()

        # Merge with latent__df
        dynamic_data['Weights'] = weights_target_point

        # Perform multivariate regression with values from selected point
        selected_point_values = dynamic_data.loc[selected_point][latent_columns].values

        # Generate x_matrix with fixed values except for the chosen latent dimension
        x_matrix = np.tile(selected_point_values, (100, 1))
        x_matrix[:, selected_latent_num] = np.linspace(dynamic_data[selected_latent_dim].min(), dynamic_data[selected_latent_dim].max(), 100)
        # Add intercept
        x_matrix_with_intercept = np.hstack((np.ones((100, 1)), x_matrix))

        # Create the plot
        # filter out external data from dynamic_data
        fig = px.scatter(dynamic_data, x=selected_latent_dim, y='Y', size='Weights')

        # Highlight the selected point
        x_selected = dynamic_data[selected_latent_dim].values[selected_point]
        y_selected = dynamic_data['Y'].values[selected_point]

        fig.add_trace(go.Scatter(
            x=[x_selected],
            y=[y_selected],
            mode='markers',
            marker=dict(color='gold', size=15, line=dict(color='black', width=2)),
            name='Selected Point'
        ))

        # Fix y-axis range to min and max of Y
        fig.update_yaxes(range=[dynamic_data['Y'].min()-1, dynamic_data['Y'].max()+1])

        # Remove legend
        fig.update_layout(showlegend=False,
                          width = 500,
                          height = 500)

        st.plotly_chart(fig)

        # Fit Linear Model Tree as Comparison on original data
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.base import BaseEstimator, RegressorMixin
        from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

        class LinearModelTree(BaseEstimator, RegressorMixin):
            def __init__(self, max_depth=3, min_samples_leaf=5):
                self.max_depth = max_depth
                self.min_samples_leaf = min_samples_leaf
                self.tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf)
                self.leaf_models = {}

            def fit(self, X, y):
                # Ensure inputs are numpy arrays
                X, y = check_X_y(X, y)

                # Fit the decision tree
                self.tree.fit(X, y)

                # Train linear models for each leaf
                self.leaf_models = {}
                leaf_indices = self.tree.apply(X)
                for leaf in np.unique(leaf_indices):
                    leaf_data = X[leaf_indices == leaf]
                    leaf_targets = y[leaf_indices == leaf]

                    # Fit a linear model for this leaf
                    model = LinearRegression()
                    model.fit(leaf_data, leaf_targets)
                    self.leaf_models[leaf] = model

                return self

            def predict(self, X):
                # Ensure input is a numpy array
                X = check_array(X)

                # Follow the decision tree and use the appropriate linear model
                leaf_indices = self.tree.apply(X)
                predictions = np.zeros(X.shape[0])

                for i, leaf in enumerate(leaf_indices):
                    predictions[i] = self.leaf_models[leaf].predict(X[i].reshape(1, -1))

                return predictions

        X_train = dynamic_data[top_10_mi_vars]
        y_train = dynamic_data['Y']

        # Train the linear model tree
        lmt = LinearModelTree(max_depth=3, min_samples_leaf=5)
        lmt.fit(X_train.values, y_train.values)

        # Make predictions
        predictions = lmt.predict(X_train.values)

        leaf_indices = lmt.tree.apply(X_combined.values)

        # Create a DataFrame for easy Plotly handling
        latent_combined["Leaf"] = leaf_indices

        # Assign unique colors to each leaf for consistent visualization
        unique_leaves = np.unique(leaf_indices)
        color_mapping = {leaf: f"Leaf {leaf}" for leaf in unique_leaves}

        fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=latent_columns, vertical_spacing=0.1)

        fig.add_trace(
            go.Scatter(
            x=latent_combined['Latent1'],  # Assuming first latent dimension is "Latent Dimension 1"
            y=combined_data['Y'],  # Assuming sixth latent dimension is "Latent Dimension 2"
            mode='markers',
            marker=dict(
                size=np.where(combined_data['subgroup_flag']==0,12,0),
                color=latent_combined["Leaf"],
                colorscale='Spectral',
                line=dict(
                            color=np.where(combined_data['subgroup_flag'] == 0, "red", "black"),  # Add black outline for train == 0
                            width=np.where(combined_data['subgroup_flag'] == 0, 0, 2)  # Adjust the width for train == 0
                        ),
            ),
            opacity=0.6
            ),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(
            x=latent_combined['Latent1'],  # Assuming first latent dimension is "Latent Dimension 1"
            y=combined_data['Y'],  # Assuming sixth latent dimension is "Latent Dimension 2"
            mode='markers',
            marker=dict(
                size=np.where(combined_data['subgroup_flag']==0,0,12),
                color=latent_combined["Leaf"],
                colorscale='Spectral',
                line=dict(
                            color=np.where(combined_data['subgroup_flag'] == 0, "red", "black"),  # Add black outline for train == 0
                            width=np.where(combined_data['subgroup_flag'] == 0, 0, 2)  # Adjust the width for train == 0
                        ),
            ),
            opacity=1.0
            ),
            row=1, col=2
        )

        fig.update_layout(
            height=800,  # Adjust height dynamically based on the number of rows
            width=800,
            title_text="Latent Dimensions with Tree Region Contours",
            showlegend=False,
            # no gray background
            plot_bgcolor='rgba(0,0,0,0)',
        )

        fig.update_yaxes(
            showgrid=True,          # Enable horizontal gridlines
            gridcolor='lightgray',  # Set the color of the gridlines
            gridwidth=0.5           # Set the width of the gridlines
        )
    
        fig.update_yaxes(
            zeroline=True,          # Show the line at y=0
            zerolinewidth=0.5,        # Adjust width
            zerolinecolor='lightgray'    # Customize color
        )

        fig.write_image("../results/figures/paper_temp/latent_space_tree_grous.pdf")


        # Show the plot
        st.plotly_chart(fig)

        import matplotlib.pyplot as plt
        from sklearn import tree

        plt.figure(figsize=(20, 10))
        tree.plot_tree(
            lmt.tree,  # The decision tree used in the Linear Model Tree
            feature_names=X_train.columns,  # Replace with your feature column names
            filled=True, 
            rounded=True,
            fontsize=12
        )

        plt.title("Decision Tree Visualization for Linear Model Tree")
        st.pyplot(plt)

        
