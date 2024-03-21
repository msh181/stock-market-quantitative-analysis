import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

# Load the dataset
df = pd.read_csv('data/preprocessedMasterDF.csv')
df = df.dropna()

# Use OneHotEncoder for the 'ticker' column
encoder = OneHotEncoder(handle_unknown='ignore')
tickerEncoded = encoder.fit_transform(df[['ticker']]).toarray()

# Create the DataFrame for one-hot encoded variables
tickerEncodedDf = pd.DataFrame(tickerEncoded, columns=encoder.get_feature_names_out(['ticker']))

# Concatenate the one-hot encoded DataFrame with the original DataFrame minus the 'ticker' column
dfEncoded = pd.concat([df.drop('ticker', axis=1).reset_index(drop=True), tickerEncodedDf], axis=1)

# Create sequences for RNN and LSTM
def createSequences(features, targets, sequenceLength):
    sequenceList = []
    targetList = []
    for i in range(len(features) - sequenceLength):
        sequenceList.append(features[i:i + sequenceLength])
        targetList.append(targets[i + sequenceLength])
    return np.array(sequenceList), np.array(targetList)

# Transform 'ticker' column into a numerical ID
tickerIds = {ticker: i for i, ticker in enumerate(df['ticker'].unique())}
df['tickerId'] = df['ticker'].map(tickerIds)

# Create sequences
sequenceLength = 10
features = dfEncoded.drop(['Close', 'Date'], axis=1).values
targets = dfEncoded['Close'].values

sequences, sequenceTargets = createSequences(features, targets, sequenceLength)

# Scale features for non-sequential models
def scaleFeatures(features, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    else:
        features = scaler.transform(features)
    return features, scaler

# Split data
def splitData(features, targets, testSize=0.2, sequence=False):
    if sequence:
        splitIndex = int((1 - testSize) * len(features))
        xTrain, xTest = features[:splitIndex], features[splitIndex:]
        yTrain, yTest = targets[:splitIndex], targets[splitIndex:]
    else:
        xTrain, xTest, yTrain, yTest = train_test_split(features, targets, test_size=testSize, shuffle=not sequence)
    return xTrain, xTest, yTrain, yTest

# Scale non-sequential features
nonSequenceTargets = dfEncoded['Close'].values
scaledFeatures, scaler = scaleFeatures(features)

# Split non-sequential data
xTrainNs, xTestNs, yTrainNs, yTestNs = splitData(scaledFeatures, nonSequenceTargets)

# Split sequential data
xTrainSeq, xTestSeq, yTrainSeq, yTestSeq = splitData(sequences, sequenceTargets, sequence=True)

# Save the non-sequential features and targets
xTrainNsdf = pd.DataFrame(xTrainNs)
xTestNsdf = pd.DataFrame(xTestNs)
yTrainNsdf = pd.DataFrame(yTrainNs, columns=['Close'])
yTestNsdf = pd.DataFrame(yTestNs, columns=['Close'])

xTrainNsdf.to_csv('modelData/xTrainNsdf.csv', index=False)
xTestNsdf.to_csv('modelData/xTestNsdf.csv', index=False)
yTrainNsdf.to_csv('modelData/yTrainNsdf.csv', index=False)
yTestNsdf.to_csv('modelData/yTestNsdf.csv', index=False)

# Save the sequential features and targets
np.save('modelData/xTrainSeq.npy', xTrainSeq)
np.save('modelData/xTestSeq.npy', xTestSeq)
np.save('modelData/yTrainSeq.npy', yTrainSeq)
np.save('modelData/yTestSeq.npy', yTestSeq)

# Save the StandardScaler and OneHotEncoder
joblib.dump(scaler, 'models/scaler.joblib')
joblib.dump(encoder, 'models/encoder.joblib')