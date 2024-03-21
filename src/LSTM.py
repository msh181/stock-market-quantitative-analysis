import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Load the sequential datasets
xTrainSeq = np.load('modelData/xTrainSeq.npy')
xTestSeq = np.load('modelData/xTestSeq.npy')
yTrainSeq = np.load('modelData/yTrainSeq.npy')
yTestSeq = np.load('modelData/yTestSeq.npy')

# Normalize your data - This step assumes data is not normalized
# Here's a simple example with TensorFlow, adjust according to your data's needs
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(xTrainSeq)

# Define the LSTM model
model = Sequential()
model.add(normalizer)  # Apply normalization as the first layer
model.add(LSTM(50, activation='relu', input_shape=(xTrainSeq.shape[1], xTrainSeq.shape[2])))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# Callback to stop training when no improvement is observed
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')

history = model.fit(xTrainSeq, yTrainSeq, epochs=100, validation_split=0.2, callbacks=[earlyStopping], batch_size=64)

testLoss = model.evaluate(xTestSeq, yTestSeq, verbose=0)
print(f"Test Loss: {testLoss}")

model.save('models/LSTM.keras')


import pickle
with open('models/LSTM_history.pkl', 'wb') as file:
    pickle.dump(history.history, file)

print("LSTM model and training history have been saved.")
