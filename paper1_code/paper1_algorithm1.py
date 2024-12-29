import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/home/saivarun/sureka_madam_works/main_dataset.csv'
print("Loading dataset from:", file_path)
data = pd.read_csv(file_path)
print("Dataset loaded successfully. First 5 rows:")
print(data.head())

# Preprocess the data
print("\nPreprocessing the data...")
data.replace('?', np.nan, inplace=True)
data = data.apply(pd.to_numeric, errors='coerce')
print("Converted data to numeric. Any remaining missing values?")
print(data.isnull().sum())
data.fillna(data.mean(), inplace=True)
print("Filled missing values with column means.")

features = data.iloc[:, :-1]
target = data.iloc[:, -1]
print("\nFeatures and target separated.")
print(f"Features shape: {features.shape}")
print(f"Target shape: {target.shape}")

# Encode target if it is categorical
if target.dtypes == 'object':
    print("\nTarget is categorical. Encoding...")
    le = LabelEncoder()
    target = le.fit_transform(target)
    print("Target encoded successfully.")

# Gaussian similarity matrix
gamma = 0.0001
print(f"\nComputing Gaussian similarity matrix with gamma={gamma}...")
gaussian_similarity = rbf_kernel(features, gamma=gamma)
print("Gaussian similarity matrix computed.")

# Feature relevance scoring
threshold = 0.4
print(f"\nScoring features with a similarity threshold of {threshold}...")
feature_scores = []

for i in range(features.shape[1]):
    neighbor_count = 0
    for j in range(features.shape[1]):
        if i != j and gaussian_similarity[i, j] > threshold:
            neighbor_count += 1
    feature_scores.append(neighbor_count)
print("Feature scores calculated:", feature_scores)

# Select relevant features
print("\nSelecting relevant features based on scores...")
relevant_features = [features.columns[i] for i, score in enumerate(feature_scores) if score > 0]
print(f"Relevant features selected ({len(relevant_features)}): {relevant_features}")

# Save relevant features to a new dataset
print("\nSaving filtered dataset...")
filtered_data = data[relevant_features + [target.name]]
filtered_data.to_csv('paper1_relevant_features.csv', index=False)
print("Filtered dataset saved as 'paper1_relevant_features.csv'.")

# Visualization
print("\nCreating visualization for feature relevance...")
plt.figure(figsize=(12, 6))
plt.bar(features.columns, feature_scores, color='skyblue', alpha=0.8)
plt.axhline(y=0, color='k', linewidth=0.8, linestyle='--')  # Add a baseline
plt.axhline(y=threshold * features.shape[1], color='red', linestyle='--', label="Threshold")  # Add threshold line

# Highlight relevant features
for idx, feature in enumerate(features.columns):
    if feature in relevant_features:
        plt.bar(feature, feature_scores[idx], color='orange')

plt.xlabel('Features')
plt.ylabel('Feature Scores (Neighbor Count)')
plt.title('Feature Relevance Visualization')
plt.xticks(rotation=45, ha='right')
plt.legend(['Threshold', 'Relevant Features'], loc='upper right')
plt.tight_layout()

# Save the visualization
plt.savefig('feature_relevance_visualization.png')
plt.show()
print("Visualization saved as 'feature_relevance_visualization.png'.")
print("\nProcess completed!")
