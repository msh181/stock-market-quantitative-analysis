"""
This script preprocesses the fetched stock data.

It includes handling missing values, feature scaling, and creating new features like moving averages.

Usage:
    python preprocess.py --input "aapl_data.csv" --output "aapl_preprocessed.csv"
"""

import pandas as pd
import os
from datetime import datetime
import argparse

def loadAndCombineDatasets(dataDirectory):
    combinedData = pd.DataFrame()
    
    csvFiles = [f for f in os.listdir(dataDirectory) if f.endswith('.csv')]
    
    for csvFile in csvFiles:
        filePath = os.path.join(dataDirectory, csvFile)
        
        # Extract symbol from filename
        symbol = csvFile.split('_')[0] 
        
        df = pd.read_csv(filePath)
        df['ticker'] = symbol
        
        # Ensure 'Date' is a datetime type and set as index
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        df.set_index('Date', inplace=True)
        
        combinedData = pd.concat([combinedData, df])

    return combinedData

dataDirectory = 'data'
combinedData = loadAndCombineDatasets(dataDirectory)


combinedData.to_csv('data/masterDF.csv')