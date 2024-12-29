import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import os

# Function to calculate the Congruence Correlation Coefficient (ρ_c)
def congruence_correlation(x, y):
    # Standardize the features
    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)
    
    # Compute the congruence correlation coefficient (ρ_c)
    rho_c = np.corrcoef(x, y)[0, 1]
    return rho_c

# Piecewise regression function
def piecewise_regression(X, y):
    # Split the data into two segments (simplified version of piecewise regression)
    mid_point = len(X) // 2
    X1, y1 = X[:mid_point], y[:mid_point]
    X2, y2 = X[mid_point:], y[mid_point:]
    
    # Perform regression on each segment
    reg1 = LinearRegression().fit(X1, y1)
    reg2 = LinearRegression().fit(X2, y2)
    
    # Predict values for both segments
    y_pred1 = reg1.predict(X1)
    y_pred2 = reg2.predict(X2)
    
    # Calculate R^2 values for both segments
    r2_1 = r2_score(y1, y_pred1)
    r2_2 = r2_score(y2, y_pred2)
    
    return r2_1, r2_2

# Function to perform feature selection based on congruence correlation and piecewise regression
def feature_selection(df, target_column='defects', r_squared_threshold=0.1, corr_threshold=0.5, min_features=2):
    # Handle missing values: Impute missing values with the mean
    imputer = SimpleImputer(strategy='mean')  # You can also try 'median' or 'most_frequent'
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    relevant_features = []
    irrelevant_features = []
    
    y = df_imputed[target_column].values  # Target variable
    
    # Create directory for saving visualizations
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')

    # Calculate number of features (excluding target column)
    features = df_imputed.drop(columns=[target_column]).columns
    num_features = len(features)
    
    # Divide features into two halves
    mid_point = num_features // 2
    first_half_features = features[:mid_point]
    second_half_features = features[mid_point:]
    
    # Function to create and save scatter plots only (excluding distributions)
    def create_feature_plots(features_to_plot, filename):
        num_columns = 3  # Number of columns for subplots (you can adjust this)
        num_rows = (len(features_to_plot) + num_columns - 1) // num_columns  # Ensure enough rows for all subplots
        fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, 5 * num_rows))  # Adjust size based on number of features
        axes = axes.flatten()  # Flatten the axes array for easy indexing

        plot_idx = 0  # Plot index for subplots
        for feature in features_to_plot:
            # Scatter plot of feature vs target variable
            axes[plot_idx].scatter(df_imputed[feature], y, color='red', alpha=0.5)
            axes[plot_idx].set_title(f'{feature} vs Target ({target_column})')
            axes[plot_idx].set_xlabel(feature)
            axes[plot_idx].set_ylabel(target_column)
            axes[plot_idx].grid(True)
            plot_idx += 1

        # Remove any extra axes (those beyond the number of features)
        for i in range(plot_idx, len(axes)):
            fig.delaxes(axes[i])

        # Adjust layout and save the chart
        plt.tight_layout()
        plt.savefig(f'visualizations/{filename}')
        plt.close()  # Close the plot to free memory

    # Plot first half of features (scatter plots vs target)
    create_feature_plots(first_half_features[:10], 'first_half_features_vs_target.png')  # Plot only 10 features in the first chart
    
    # Plot second half of features (scatter plots vs target)
    if len(second_half_features) > 0:
        create_feature_plots(second_half_features[:11], 'second_half_features_vs_target.png')  # Plot only 11 features in the second chart

    # Feature selection based on correlation and R^2 values
    for feature in df_imputed.drop(columns=[target_column]).columns:
        X = df_imputed[feature].values.reshape(-1, 1)  # Feature to check
        
        # Calculate the congruence correlation coefficient (ρ_c)
        rho_c = congruence_correlation(X.flatten(), y)
        
        # Apply piecewise regression
        r2_1, r2_2 = piecewise_regression(X, y)
        
        # Combine R^2 values and congruence correlation
        avg_r2 = (r2_1 + r2_2) / 2
        
        # Print the results
        print(f"Feature: {feature}, Congruence Correlation: {rho_c:.4f}, R^2 (segment 1): {r2_1:.4f}, R^2 (segment 2): {r2_2:.4f}")
        
        # Feature selection criteria based on congruence correlation and R^2
        if rho_c > corr_threshold and avg_r2 > r_squared_threshold:
            relevant_features.append(feature)
        else:
            irrelevant_features.append(feature)
    
    # Ensure at least 'min_features' are selected
    if len(relevant_features) < min_features:
        print(f"\nWarning: Fewer than {min_features} features selected. Adding more features.")
        # If fewer than 'min_features' relevant features, relax thresholds and select top features
        # Forcing additional feature selection
        for feature in df_imputed.drop(columns=[target_column]).columns:
            if feature not in relevant_features:
                relevant_features.append(feature)
            if len(relevant_features) >= min_features:
                break
    
    # Return the relevant and irrelevant features
    print("\nRelevant Features:", relevant_features)
    print("Irrelevant Features:", irrelevant_features)
    
    return relevant_features, irrelevant_features

# Example usage with a dataset
# Load your dataset (replace with actual dataset path)
df = pd.read_csv("paper3_main_dataset.csv")

# Perform feature selection
relevant_features, irrelevant_features = feature_selection(df, target_column='defects', r_squared_threshold=0.1, corr_threshold=0.5, min_features=3)


# Optionally, save the selected features
df_selected = df[relevant_features + ['defects']]  # Include the target column
df_selected.to_csv("paper3_relevant_features.csv", index=False)

print(f"Relevant features saved to 'paper3_relevant_features.csv'")
