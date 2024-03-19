import pandas as pd

def loadData(filePath):
    """Loads the master dataset from a CSV file."""
    df = pd.read_csv(filePath, index_col='Date', parse_dates=['Date'], infer_datetime_format=True)
    return df

def cleanData(df):
    """Cleans the dataset by handling missing values and removing unnecessary columns."""
    # Assuming 'Dividends' and 'Stock Splits' are less relevant for price prediction, you may drop them
    # You can adjust this based on your model's requirements
    df.drop(['Dividends', 'Stock Splits'], axis=1, inplace=True)
    
    # Fill or drop missing values. Here we're filling missing values with the previous day's data
    df.fillna(method='ffill', inplace=True)
    
    return df

def addMovingAverages(df, windowSizes=[5, 10, 20]):
    """Adds moving averages for specified window sizes."""
    for window in windowSizes:
        df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
    return df

def addRelativeStrengthIndex(df, periods=14):
    """Adds Relative Strength Index (RSI) for given period."""
    delta = df['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()

    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def addBollingerBands(df, window=20, noOfStd=2):
    """Adds Bollinger Bands."""
    rollingMean = df['Close'].rolling(window=window).mean()
    rollingStd = df['Close'].rolling(window=window).std()
    
    df['Bollinger Upper'] = rollingMean + (rollingStd * noOfStd)
    df['Bollinger Lower'] = rollingMean - (rollingStd * noOfStd)
    return df

def addVolumeChangeRate(df):
    """Adds Volume Change Rate."""
    df['Volume Change Rate'] = df['Volume'].pct_change()
    return df

def featureEngineering(df):
    """Adds new features to the dataset, which could be helpful for the models."""
    df = addMovingAverages(df)
    df = addRelativeStrengthIndex(df)
    df = addBollingerBands(df)
    df = addVolumeChangeRate(df)
    # Add a feature for the daily price change
    df['Daily Change'] = df['Close'] - df['Open']

    return df

def preprocessData(filePath):
    """Main function to load, clean, and add features to the dataset."""
    df = loadData(filePath)
    df = cleanData(df)
    df = featureEngineering(df)
    
    # Save the preprocessed data to a new CSV file
    df.to_csv('data/preprocessedMasterDF.csv')

if __name__ == '__main__':
    filePath = 'data/masterDF.csv' 
    preprocessData(filePath)
