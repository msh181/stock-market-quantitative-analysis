import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import pickle

# Load the sequential datasets
xTrainSeq = np.load('modelData/xTrainSeq.npy')
xTestSeq = np.load('modelData/xTestSeq.npy')
yTrainSeq = np.load('modelData/yTrainSeq.npy')
yTestSeq = np.load('modelData/yTestSeq.npy')

# Normalize your data - This step assumes data is not normalized
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(xTrainSeq)

# Define the RNN model
model = Sequential()
model.add(normalizer)  # Apply normalization as the first layer
model.add(SimpleRNN(50, activation='relu', input_shape=(xTrainSeq.shape[1], xTrainSeq.shape[2])))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# Callback to stop training early if validation loss stops improving
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')

# Train the model
history = model.fit(xTrainSeq, yTrainSeq, epochs=100, validation_split=0.2, callbacks=[earlyStopping], batch_size=64)

# Evaluate the model
testLoss = model.evaluate(xTestSeq, yTestSeq, verbose=0)

print(f"Test Loss: {testLoss}")


model.save('models/RNN.keras')

# Optional: Save the training history
with open('models/RNN_history.pkl', 'wb') as file:
    pickle.dump(history.history, file)

print("RNN model and training history have been saved.")
