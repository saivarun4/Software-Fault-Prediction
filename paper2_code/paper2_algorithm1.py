import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
# Replace 'main_dataset.csv' with the actual dataset file name
dataset = pd.read_csv('main_dataset.csv')
print(f"Dataset loaded with shape: {dataset.shape}")

# Assuming the target column is named 'defects' and other columns are features
X = dataset.drop(columns=['defects'])  # Features
y = dataset['defects']  # Target variable
print(f"Features selected: {X.columns.tolist()}")
print(f"Target variable: {y.name}")

# Step 2 to Step 13: Implementing the algorithm
def torgerson_gower_feature_scaling(X):
    n_features = X.shape[1]
    feature_names = X.columns
    feature_scores = pd.DataFrame(index=feature_names, columns=feature_names)  # Matrix for similarity scores
    relevant_features = []  # List to store relevant features
    irrelevant_features = []  # List to store irrelevant features

    print("Starting the feature scaling process...")
    
    # Iterate over all feature pairs
    for i in range(n_features):
        for j in range(i + 1, n_features):
            feature_i = feature_names[i]
            feature_j = feature_names[j]
            
            # Step 3: Apply nonparametric statistical test (Spearman correlation)
            correlation, _ = spearmanr(X[feature_i], X[feature_j])
            
            # Step 4: Measure similarity
            similarity = abs(correlation)  # Use absolute value for similarity
            
            # Store the similarity score
            feature_scores.loc[feature_i, feature_j] = similarity
            feature_scores.loc[feature_j, feature_i] = similarity
            
            # Step 5-6: Check similarity threshold
            if similarity >= 0.8:  # Threshold for mutual dependence
                if feature_i not in relevant_features:
                    relevant_features.append(feature_i)
            else:
                if feature_i not in irrelevant_features:
                    irrelevant_features.append(feature_i)

    # Step 12-13: Remove irrelevant features
    relevant_features = list(set(relevant_features))
    irrelevant_features = list(set(irrelevant_features))
    
    # Filter the dataset to include only relevant features
    X_relevant = X[relevant_features]
    
    print(f"Relevant features identified: {relevant_features}")
    print(f"Irrelevant features identified: {irrelevant_features}")
    
    # Add the target variable 'defects' to the relevant features
    X_relevant['defects'] = y  # Add the target variable to the relevant features

    return X_relevant, relevant_features, irrelevant_features, feature_scores

# Apply the algorithm
X_relevant, relevant_features, irrelevant_features, feature_scores = torgerson_gower_feature_scaling(X)

# Visualization of feature scores
print("Visualizing feature similarity scores...")
plt.figure(figsize=(10, 8))
sns.heatmap(feature_scores.astype(float), annot=True, cmap="coolwarm", cbar=True, fmt=".2f")
plt.title("Feature Similarity Scores (Spearman Correlation)")
plt.xlabel("Features")
plt.ylabel("Features")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save the visualization
plt.savefig('feature_similarity_scores.png')
print("Feature similarity heatmap saved as 'feature_similarity_scores.png'")

# Bar plot of relevance
print("Visualizing the relevance of features...")
plt.figure(figsize=(12, 6))
relevant_scores = feature_scores.loc[relevant_features, relevant_features].mean(axis=1).sort_values(ascending=False)
relevant_scores.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Nonparametric Statistical Torgerson–Gower Feature Scaling")
plt.ylabel("Average Similarity Score")
plt.xlabel("Features")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save the bar plot
plt.savefig('relevant_features_scores.png')
print("Bar plot of relevant features saved as 'relevant_features_scores.png'")

# Print results
print("Relevant Features:", relevant_features)
print("Irrelevant Features:", irrelevant_features)

# Save the dataset with only relevant features and target variable
X_relevant.to_csv('relevant_features_with_target.csv', index=False)
print("Dataset with relevant features and target variable saved as 'paper2_relevant_features.csv'")
