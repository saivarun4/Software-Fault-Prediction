import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from sklearn.metrics import accuracy_score

data = pd.read_csv('/home/saivarun/sureka_madam_works/paper1_relevant_features.csv')


X = data.drop('defects', axis=1).values 
y = data['defects'].values  

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
weights = np.random.rand(X_train.shape[1]) 
bias = np.random.rand()  

learning_rate = 0.001
epochs = 250


def step_function(x):
    return 1 if x >= 0 else 0

def kmo_test(X):
    correlation_matrix = np.corrcoef(X.T)
    kmo = np.linalg.det(correlation_matrix) / (np.prod(np.diagonal(np.linalg.inv(correlation_matrix))))
    return kmo

for epoch in range(epochs):
    for i in range(len(X_train)):
        x_i = X_train[i]
        y_i = y_train[i]

        R_t = np.dot(x_i, weights) + bias

        ρ_kmo = kmo_test(X_train)

        if ρ_kmo == 1:
            prediction = step_function(R_t)
        else:
            prediction = 0

        error_rate = (y_i - prediction) ** 2

        weights += learning_rate * error_rate * x_i
        bias += learning_rate * error_rate

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Weights: {weights}, Bias: {bias}")

y_pred = []

for i in range(len(X_test)):
    x_i = X_test[i]
    R_t = np.dot(x_i, weights) + bias
    ρ_kmo = kmo_test(X_test)

    if ρ_kmo == 1:
        prediction = step_function(R_t)
    else:
        prediction = 0

    y_pred.append(prediction)

accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
