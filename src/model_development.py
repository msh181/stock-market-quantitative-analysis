"""
Model Development for Stock Market Forecasting

This script outlines the creation and training of a machine learning model to forecast stock prices.
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score, GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np
import pandas as pd

def load_dataset(filename):
    """
    Loads the dataset for modeling.
    
    Parameters:
        filename (str): Path to the dataset CSV file.
        
    Returns:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Target variable array.
    """
    data = pd.read_csv(filename)
    X = data.drop(['Date', 'Close'], axis=1).values
    y = data['Close'].values
    return X, y

def train_test_split_data(X, y, test_size=0.2):
    """
    Splits the dataset into training and testing sets.
    
    Parameters:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Target variable array.
        test_size (float): Proportion of the dataset to include in the test split.
        
    Returns:
        X_train, X_test, y_train, y_test (tuple): Training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

def build_linear_regression_model(X_train, y_train):
    """
    Builds and trains a linear regression model.
    
    Parameters:
        X_train (numpy.ndarray): Training feature matrix.
        y_train (numpy.ndarray): Training target variable array.
        
    Returns:
        model (LinearRegression): Trained linear regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluates the performance of the model on the test set.
    
    Parameters:
        model (LinearRegression): The trained model.
        X_test (numpy.ndarray): Testing feature matrix.
        y_test (numpy.ndarray): Testing target variable array.
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")

# Assuming X and y are already defined and preprocessed for Linear Regression and Random Forest
# For LSTM, we will need to reshape the data into 3D format [samples, time steps, features]

def build_random_forest_model(X_train, y_train):
    """
    Builds and trains a Random Forest model.
    
    Parameters:
        X_train (numpy.ndarray): Training feature matrix.
        y_train (numpy.ndarray): Training target variable array.
        
    Returns:
        model (RandomForestRegressor): Trained Random Forest model.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def build_lstm_model(X_train, y_train, input_shape):
    """
    Builds and trains an LSTM model.
    
    Parameters:
        X_train (numpy.ndarray): Training feature matrix.
        y_train (numpy.ndarray): Training target variable array.
        input_shape (tuple): Shape of the input data (time steps, features).
        
    Returns:
        model (Sequential): Trained LSTM model.
    """
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=20, batch_size=32)
    return model

def evaluate_models(models, X_test, y_test):
    """
    Evaluates the performance of multiple models on the test set and compares them.
    
    Parameters:
        models (dict): A dictionary of trained models.
        X_test (numpy.ndarray): Testing feature matrix.
        y_test (numpy.ndarray): Testing target variable array.
    """
    for name, model in models.items():
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        print(f"{name} - Mean Squared Error: {mse}, Mean Absolute Error: {mae}")


def tune_linear_regression(X_train, y_train):
    """
    Since Linear Regression has fewer hyperparameters, this function simply applies cross-validation to evaluate its performance.
    
    Parameters:
        X_train (numpy.ndarray): Training feature matrix.
        y_train (numpy.ndarray): Training target variable array.
        
    Returns:
        float: The mean score of the model across all folds.
    """
    model = LinearRegression()
    cv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
    return scores.mean()

def create_lstm_model(input_shape):
    """
    Function to create LSTM model, required for KerasRegressor wrapper.
    
    Parameters:
        input_shape (tuple): Shape of the input data (samples, time steps, features).
        
    Returns:
        Compiled Keras model.
    """
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def perform_cross_validation(model, X, y, cv=5):
    """
    Performs cross-validation on a given model.
    
    Parameters:
        model: The machine learning model to evaluate.
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Target variable array.
        cv (int): Number of folds in cross-validation.
        
    Returns:
        float: The mean score of the model across all folds.
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    return scores.mean()

def tune_hyperparameters(X_train, y_train):
    """
    Tunes hyperparameters of the Random Forest model using GridSearchCV.
    
    Parameters:
        X_train (numpy.ndarray): Training feature matrix.
        y_train (numpy.ndarray): Training target variable array.
        
    Returns:
        model: The best model found by GridSearchCV.
    """
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def tune_lstm_model(X_train, y_train, input_shape):
    """
    Applies hyperparameter tuning to the LSTM model using KerasRegressor and GridSearchCV.
    
    Parameters:
        X_train (numpy.ndarray): Training feature matrix reshaped for LSTM.
        y_train (numpy.ndarray): Training target variable array.
        input_shape (tuple): Input shape for the LSTM model.
        
    Returns:
        model: The best LSTM model found by GridSearchCV.
    """
    model = KerasRegressor(build_fn=create_lstm_model, input_shape=input_shape, epochs=100, batch_size=32, verbose=0)
    param_grid = {
        'epochs': [50, 100],
        'batch_size': [16, 32]
    }
    cv = TimeSeriesSplit(n_splits=3)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def main(filename):
    """
    Main function to run the model development process.
    
    Parameters:
        filename (str): Path to the dataset CSV file.
    """
    X, y = load_dataset(filename)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    model = build_linear_regression_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    
# Ensure the correct environment setup for running this script, with necessary libraries installed and data available.
