import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import regularizers


dataset = pd.read_csv('/home/saivarun/sureka_madam_works/paper2_relevant_features.csv')
dataset.fillna(dataset.mean(), inplace=True)

scaler = StandardScaler()
X = scaler.fit_transform(dataset.drop('defects', axis=1))  
y = dataset['defects'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def build_cnn_model(input_shape):
    model = models.Sequential()
    model.add(layers.Reshape((input_shape[0], 1), input_shape=input_shape)) 
    model.add(layers.Conv1D(64, 5, activation='relu', kernel_regularizer=regularizers.l2(0.01), padding='same'))
    model.add(layers.MaxPooling1D(pool_size=1))  
    model.add(layers.Conv1D(128, 5, activation='relu', kernel_regularizer=regularizers.l2(0.01), padding='same'))
    model.add(layers.MaxPooling1D(pool_size=1))  
    model.add(layers.Conv1D(256, 5, activation='relu', kernel_regularizer=regularizers.l2(0.01), padding='same'))
    model.add(layers.MaxPooling1D(pool_size=1)) 
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Dropout(0.5))  
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  
    
    return model

model = build_cnn_model(X_train.shape[1:])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])


lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping, lr_scheduler])
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")

def feature_mapping_correlation(X):
    correlation_matrix = np.corrcoef(X, rowvar=False)
    return correlation_matrix

correlation_matrix = feature_mapping_correlation(X_train)

def censored_regression(correlation_matrix, beta=0.8):
    selected_features = np.where(np.abs(correlation_matrix) > beta)
    return selected_features

selected_features = censored_regression(correlation_matrix)


def softstep_activation(x):
    return 1 / (1 + np.exp(-x))  

dense_layer_output = softstep_activation(np.dot(X_train, np.random.rand(X_train.shape[1], 1)))

def quadratic_loss(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2)

def nelder_mead_optimization(loss_function, x0):
    result = minimize(loss_function, x0, method='Nelder-Mead')
    return result

initial_guess = np.random.rand(X_train.shape[1]) 
loss_function = lambda x: quadratic_loss(y_train, softstep_activation(np.dot(X_train, x)))
optimization_result = nelder_mead_optimization(loss_function, initial_guess)
final_predictions = softstep_activation(np.dot(X_test, optimization_result.x))
final_predictions_binary = np.round(final_predictions)


print("Final Fault Predictions: ", final_predictions_binary)

accuracy = np.mean(final_predictions_binary == y_test)
print(f"Accuracy of the final model: {accuracy:.2f}")
