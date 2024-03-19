import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def readTickers(filePath='data/trainingTickers.xlsx'):
    """
    Reads stock tickers from an Excel file.

    Parameters:
    - filePath (str): Path to the Excel file containing the stock tickers.

    Returns:
    - list: A list of stock tickers.
    """
    df = pd.read_excel(filePath)
    return df[df.columns[0]].tolist()

def fetchHistoricalData(stockSymbols, startDate, endDate):
    """
    Fetches historical stock data for the given symbols from startDate to endDate.

    Parameters:
    - stockSymbols (list of str): List of stock symbols to fetch data for.
    - startDate (str): The start date for the data fetch in 'YYYY-MM-DD' format.
    - endDate (str): The end date for the data fetch in 'YYYY-MM-DD' format.

    Returns:
    - dict: A dictionary where keys are stock symbols and values are DataFrames with the historical data.
    """
    stockData = {}
    for symbol in stockSymbols:
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=startDate, end=endDate, interval='1d')
        stockData[symbol] = data
    return stockData

def saveData(stockData, directory='data'):
    """
    Saves each DataFrame in stockData to a CSV file in the specified directory.

    Parameters:
    - stockData (dict): Dictionary with stock symbols as keys and DataFrames as values.
    - directory (str): Directory path where the CSV files will be saved.
    """
    for symbol, data in stockData.items():
        filename = f"{directory}/{symbol}_{startDate.replace('-', '')}_{endDate.replace('-', '')}.csv"
        data.to_csv(filename)

if __name__ == '__main__':
    stockSymbols = readTickers()  # Reads stock tickers from the Excel file
    endDate = datetime.now().date()  # Using today's date
    startDate = endDate - timedelta(days=5*365)  # Adjust for 5 years of data
    startDate, endDate = startDate.strftime('%Y-%m-%d'), endDate.strftime('%Y-%m-%d')  # Format dates as strings

    stockData = fetchHistoricalData(stockSymbols, startDate, endDate)
    saveData(stockData)
