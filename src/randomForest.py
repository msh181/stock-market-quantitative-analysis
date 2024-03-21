import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Load the datasets
xTrain = pd.read_csv('modelData/xTrainNsdf.csv')
xTest = pd.read_csv('modelData/xTestNsdf.csv')
yTrain = pd.read_csv('modelData/yTrainNsdf.csv')
yTest = pd.read_csv('modelData/yTestNsdf.csv')

# Initialize the Random Forest Regressor
randomForestModel = RandomForestRegressor(n_estimators=90, random_state=42)  # n_estimators can be adjusted

# Fit the model on the training data
randomForestModel.fit(xTrain, yTrain)

# Making predictions
yTrainPred = randomForestModel.predict(xTrain)
yTestPred = randomForestModel.predict(xTest)

# Evaluating the model
trainMSE = mean_squared_error(yTrain, yTrainPred)
testMSE = mean_squared_error(yTest, yTestPred)

print(f"Train MSE: {trainMSE}")
print(f"Test MSE: {testMSE}")

# Save the model
joblib.dump(randomForestModel, 'models/randomForest.joblib')

print("Random Forest model has been saved.")