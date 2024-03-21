import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Load the datasets
xTrain = pd.read_csv('modelData/xTrainNsdf.csv')
xTest = pd.read_csv('modelData/xTestNsdf.csv')
yTrain = pd.read_csv('modelData/yTrainNsdf.csv')
yTest = pd.read_csv('modelData/yTestNsdf.csv')


# Initialise the Linear Regression model
linearRegressor = LinearRegression()

# Train the model
linearRegressor.fit(xTrain, yTrain)

# Predict on the testing set
y_pred = linearRegressor.predict(xTest)

# Evaluate the model
mse = mean_squared_error(yTest, y_pred)
print(f'Mean Squared Error: {mse}')

# Save the trained model for later use
joblib.dump(linearRegressor, 'models/linearRegressor.joblib')

print("Model trained and saved successfully.")
