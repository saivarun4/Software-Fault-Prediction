import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.optimize import least_squares

# Assume Camargo's Index CI function (as placeholder)
def camargos_index(training_feature, testing_feature):
    # Placeholder function for Camargo's Index
    return np.abs(np.dot(training_feature, testing_feature))

# Hardlimit Activation Function
def hardlimit_activation(ci, threshold=0.5):
    if ci > threshold:
        return 1
    else:
        return 0

# Levenberg-Marquardt optimization (for minimizing least squares error)
def levenberg_marquardt(X, y):
    # Placeholder for Levenberg-Marquardt, assuming it optimizes a model
    def model(params, X):
        return np.dot(X, params)

    def objective(params):
        return model(params, X) - y

    result = least_squares(objective, np.zeros(X.shape[1]))
    return result.x

# Main Function for the SILEL-C Algorithm
def silelc_algorithm(dataset):
    print("Loading dataset...")
    # Load the dataset using pandas to handle missing values
    df = pd.read_csv(dataset)
    
    print("Initial dataset shape:", df.shape)

    # Check and clean the data: Remove rows with missing values or replace them
    df = df.dropna()  # Removes rows with missing values
    print("Dataset shape after removing missing values:", df.shape)
    # Alternatively, you could replace missing values with the mean or median:
    # df.fillna(df.mean(), inplace=True)
    
    # Convert all data to numeric (force errors to NaN, which we can then drop or replace)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()  # Drop rows where conversion failed
    print("Dataset shape after converting to numeric and cleaning:", df.shape)

    # Separate into features and labels
    X = df.iloc[:, :-1].values  # All columns except the last one (features)
    y = df.iloc[:, -1].values   # Last column (labels)
    
    print("Number of features:", X.shape[1])
    print("Number of samples:", X.shape[0])

    # Split into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print("Training set size:", X_train.shape[0], "Testing set size:", X_test.shape[0])
    
    # Hidden Layer: Camargo's Index Calculation for each pair of training and testing data
    print("Calculating Camargo's Index for training and testing data...")
    ci_results = []
    for i in range(X_train.shape[0]):  # For each training data D_i
        for j in range(X_test.shape[0]):  # For each testing feature D_j
            ci = camargos_index(X_train[i], X_test[j])
            ci_results.append(ci)

    print("Total Camargo's Index calculations completed:", len(ci_results))
    
    # Apply Hardlimit activation function
    print("Applying hardlimit activation function...")
    threshold = 0.5
    predictions = []
    for ci in ci_results:
        prediction = hardlimit_activation(ci, threshold)
        predictions.append(prediction)
    
    print("Hardlimit activation completed.")
    
    # Reshape predictions to match the number of test samples
    predictions = np.array(predictions).reshape(X_test.shape[0], -1)
    
    # Output Layer: Apply Levenberg–Marquardt algorithm to minimize least square error
    print("Optimizing weights using Levenberg-Marquardt algorithm...")
    optimized_weights = levenberg_marquardt(X_train, y_train)
    print("Optimized weights:", optimized_weights)
    
    # Use the optimized weights to predict output (classification)
    print("Predicting output using optimized weights...")
    y_pred = np.dot(X_test, optimized_weights)
    
    # Apply Hardlimit Activation to predictions
    print("Applying hardlimit activation to final predictions...")
    final_predictions = np.array([hardlimit_activation(val) for val in y_pred])
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, final_predictions)
    print(f"Attack Detection Accuracy: {accuracy*100:.2f}%")
    
# Assuming the dataset is named 'paper3_main_dataset.csv'
silelc_algorithm('paper3_main_dataset.csv')
