"""
This script fetches historical stock data using the yfinance library.

Usage:
    python fetch_data.py --ticker "AAPL" --start "2020-01-01" --end "2021-01-01" --filename "aapl_data.csv"
"""

import yfinance as yf
import argparse
import os

def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetches historical stock data for a given ticker from start_date to end_date.

    Parameters:
        ticker (str): The stock ticker symbol.
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.

    Returns:
        pandas.DataFrame: The historical stock data.
    """
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def save_to_csv(data, filename):
    """
    Saves the fetched data to a CSV file.

    Parameters:
        data (pandas.DataFrame): The data to save.
        filename (str): The filename for the saved CSV file.
    """
    data.to_csv(filename)
    print(f"Data saved to {filename}")

def main():
    # Setting up argument parsing
    parser = argparse.ArgumentParser(description='Fetch and save historical stock data.')
    parser.add_argument('--ticker', type=str, help='Stock ticker symbol', required=True)
    parser.add_argument('--start', type=str, help='Start date in YYYY-MM-DD format', required=True)
    parser.add_argument('--end', type=str, help='End date in YYYY-MM-DD format', required=True)
    parser.add_argument('--filename', type=str, help='Filename for the output CSV file', required=True)

    args = parser.parse_args()

    # Fetching the data
    data = fetch_stock_data(args.ticker, args.start, args.end)

    # Saving the data to CSV
    save_to_csv(data, args.filename)

if __name__ == "__main__":
    main()
