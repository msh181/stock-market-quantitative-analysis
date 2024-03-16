"""
This script preprocesses the fetched stock data.

It includes handling missing values, feature scaling, and creating new features like moving averages.

Usage:
    python preprocess.py --input "aapl_data.csv" --output "aapl_preprocessed.csv"
"""

import pandas as pd
import argparse

def load_data(filename):
    """
    Loads the stock data from a CSV file.

    Parameters:
        filename (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The loaded stock data.
    """
    data = pd.read_csv(filename)
    return data

def handle_missing_values(data):
    """
    Handles missing values in the stock data, if any.

    Parameters:
        data (pandas.DataFrame): The stock data.

    Returns:
        pandas.DataFrame: The stock data with missing values handled.
    """
    # For simplicity, we'll fill missing values with the previous value (forward fill)
    data_filled = data.fillna(method='ffill')
    return data_filled

def add_moving_averages(data, window_sizes=[5, 20]):
    """
    Adds moving average columns to the stock data.

    Parameters:
        data (pandas.DataFrame): The stock data.
        window_sizes (list of int): The window sizes for the moving averages.

    Returns:
        pandas.DataFrame: The stock data with moving average columns added.
    """
    for window in window_sizes:
        data[f'moving_avg_{window}'] = data['Close'].rolling(window=window).mean()
    return data

def preprocess_data(input_file, output_file):
    """
    Preprocesses the stock data.

    Parameters:
        input_file (str): The input CSV file containing the stock data.
        output_file (str): The output CSV file for the preprocessed data.
    """
    data = load_data(input_file)
    data = handle_missing_values(data)
    data = add_moving_averages(data)
    
    data.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Preprocess stock data.')
    parser.add_argument('--input', type=str, help='Input CSV file containing the stock data', required=True)
    parser.add_argument('--output', type=str, help='Output CSV file for the preprocessed data', required=True)

    args = parser.parse_args()

    preprocess_data(args.input, args.output)

# Note: The execution of this script will be outside this environment. The main function is defined for completeness.
