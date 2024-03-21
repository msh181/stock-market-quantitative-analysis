"""
This is the main model training script that executes the scripts outlined below. 
This script:
    1. Fetches the data using the yfinance API (fetchData.py)
    2. Combines the data (combineData.py)
    3. Preprocesses the data (preprocess.py) - this is where feature engineering is computed, currently the additional features are:
        a. daily price change
        b. moving averages for specified window sizes
        c. Relative Strength Index (RSI) for given period
        d. Bollinger Bands
        e. Volume Change Rate
    4. Prepares the data for training the models (prepareData.py)
    5. Trains the models. Currently the models included are:
        a. Linear Regression (linearRegression.py)
        b. Random Forest (randomForest.py)
        c. Long Short-Term Memory (LSTM) network (LSTM.py)
        d. Recurrent Neural Network (RNN) (RNN.py)
    6. All trained models, the encoder and scaler are saved in the "models/" folder
"""

import sys
sys.path.append('C:/Users/marah/Documents/GitHub/stock-market-quantitative-analysis/src')

