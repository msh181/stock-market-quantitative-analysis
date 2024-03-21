import streamlit as st
import numpy as np
import pandas as pd
import joblib
import yfinance as yf
from tensorflow.keras.models import load_model

import sys
sys.path.append('C:/Users/marah/Documents/GitHub/stock-market-quantitative-analysis/src')

from preprocess import cleanData, featureEngineering
from prepareData import createSequences
from datetime import datetime, timedelta


# Function to load models (adjust paths as needed)
def load_models():
    models = {
        "Linear Regression": joblib.load('models/linearRegressor.joblib'),
        "Random Forest": joblib.load('models/randomForest.joblib'),
        "RNN": load_model('models/RNN.keras'),
        "LSTM": load_model('models/LSTM.keras')
    }
    return models

# Placeholder for a function to fetch data based on ticker
# This function should return a DataFrame with the data needed for prediction
def pullData(ticker):
    endDate = datetime.now().date()  # Using today's date
    startDate = endDate - timedelta(days=5*365)  # Adjust for 5 years of data
    startDate, endDate = startDate.strftime('%Y-%m-%d'), endDate.strftime('%Y-%m-%d')  # Format dates as strings

    tkr = yf.Ticker(ticker)
    stockData = tkr.history(start=startDate, end=endDate, interval='1d')

    return stockData

# Placeholder for a function to preprocess data for a given model
def preprocessPrepareData(data, model_name, ticker):
    # Implement preprocessing logic specific to each model
    df = pd.DataFrame(data)
    df['ticker'] = ticker
        
    # Ensure 'Date' is a datetime type and set as index
    #df['Date'] = pd.to_datetime(df['Date'], utc=True)
    #df.set_index('Date', inplace=True)
    df = cleanData(df)
    df = featureEngineering(df)
    df = df.dropna()
    
    scaler = joblib.load('models/scaler.joblib')
    encoder = joblib.load('models/encoder.joblib')
    # Encode 'ticker' column for non-sequential models
    if model_name in ['Linear Regression', 'Random Forest']:
        ticker_encoded = encoder.transform(df[['ticker']]).toarray()
        ticker_encoded_df = pd.DataFrame(ticker_encoded, columns=encoder.get_feature_names_out(['ticker']))
        data = pd.concat([df.drop('ticker', axis=1).reset_index(drop=True), ticker_encoded_df], axis=1)
        
        # Scale the features
        features = data.drop(['Close', 'Date'], axis=1).values  # Assuming 'Close' and 'Date' are not used as features
        scaled_features = scaler.transform(features)
        
        return scaled_features
    
    # Prepare sequence data for sequential models
    elif model_name in ['RNN', 'LSTM']:
        # Map ticker to tickerId as done during initial preparation
        # This assumes tickerIds and sequenceLength are defined globally or passed to this function
        tickerIds = {ticker: i for i, ticker in enumerate(df['ticker'].unique())}
        df['tickerId'] = df['ticker'].map(tickerIds)
        
        # Drop 'ticker' and 'Date', convert to values
        sequenceLength = 10
        features = df.drop(['ticker', 'Date', 'Close'], axis=1).values  # Assuming 'Close' is the target
        targets = df['Close'].values
        
        # Create sequences
        sequences, sequence_targets = createSequences(features, targets, sequenceLength)
        
        # No scaling is applied in this placeholder, adjust as necessary
        return sequences
    
    else:
        raise ValueError("Model name not recognized.")


# Initialize Streamlit app
st.title("Stock Price Prediction for Everyday Investors")

# Load the pre-trained models
models = load_models()

# User inputs
tickers = st.text_input("Enter the ticker symbols (comma-separated for multiple):").split(',')
model_choice = st.selectbox("Choose a model for prediction:", options=list(models.keys()))

# Display button to make predictions
if st.button("Predict"):
    if tickers:
        results = {}
        for ticker in tickers:
            # Fetch data for the ticker
            data = pullData(ticker.strip().upper())
            
            # Preprocess data for the selected model
            preprocessedData = preprocessPrepareData(data, model_choice, ticker)
            
            # Predict using the selected model
            model = models[model_choice]
            prediction = model.predict(preprocessedData)  # Adjust this call based on the model's expected input format
            
            # Store predictions in results
            results[ticker] = prediction
        
        # Display predictions
        for ticker, prediction in results.items():
            st.write(f"Prediction for {ticker}: {prediction}")
            
        # Optional: Display last registered open and close, and a summary of the current state analysis
        # This could involve fetching additional data for each ticker and displaying it
    else:
        st.error("Please enter at least one ticker symbol.")

print('hello world')
# Optional: Implement additional app features such as historical data visualization, comparing multiple tickers, etc.

