import numpy as np
import pandas as pd
import math
from sklearn.metrics import mean_squared_error

# Set the random seed for reproducibility
from numpy.random import seed
seed(1)
from tensorflow import random
random.set_seed(2)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau
# Load data
df = pd.read_csv('D.csv', infer_datetime_format=True, parse_dates=True)

# This function accepts the column number for the features (X) and the target (y)
# It returns a numpy array of X any y
def window_data(df, window, feature_col_number, target_col_number):
    X = []
    y = []
    for i in range(len(df) - window - 1):
        features = df.iloc[i:(i + window), feature_col_number]
        target = df.iloc[(i + window), target_col_number]
        X.append(features)
        y.append(target)
    return np.array(X), np.array(y).reshape(-1, 1)

# Predict RoR using a 4 day window of previous RoR
window_size = 4

# Column index 1 is the `Effiency (RoR)` column
feature_column = 1
target_column = 1
X, y = window_data(df, window_size, feature_column, target_column)

# Use 80% of the data for training and the remaineder for testing
split = int(.8*len(X))
X_train = X[:split-1]
X_test = X[split:]
y_train = y[:split-1]
y_test = y[split:]

from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()
scaler.fit(X)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
scaler.fit(y)
y_train = scaler.transform(y_train)
y_test = scaler.transform(y_test)


# Reshape the features for the model
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1],1))


# Build the LSTM model. 
model = Sequential()
dropout_fraction = 0.2 

#layer1

model.add(LSTM(units=512, return_sequences = True, input_shape=(X_train.shape[1],1)))
model.add(Dropout(dropout_fraction))

#Layer2

model.add(LSTM(units=128, return_sequences=True))
model.add(Dropout(dropout_fraction))

#Layer3

model.add(LSTM(units=64))
model.add(Dropout(dropout_fraction))


#Output layer

model.add(Dense(1))


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5, min_lr=0.000001, verbose=1)

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# Train the model

model.fit(X_train, y_train, epochs=5, batch_size=1, verbose=1, validation_data = (X_test, y_test), callbacks=[reduce_lr])


model.evaluate(X_test, y_test, verbose=False)

# Make some predictions
predicted = model.predict(X_test,batch_size=1)

# Recover the original prices instead of the scaled version
predicted_RoR = scaler.inverse_transform(predicted)
real_RoR = scaler.inverse_transform(y_test.reshape(-1, 1))


testScore = math.sqrt(mean_squared_error(real_RoR[:, 0], predicted_RoR[:, 0]))
print('Test Score: %.2f RMSE' % (testScore))


# Create a DataFrame of Real and Predicted values
stocks = pd.DataFrame({
    "Real": real_RoR.ravel(),
    "Predicted": predicted_RoR.ravel()
})
stocks.head()

# Plot the real vs predicted test values as a line chart
from matplotlib import pyplot as plt    

fig = plt.figure()
plt.plot(stocks)
fig.suptitle('Rate of return: Predicted verse Actual, Window 4', fontsize=10)
plt.ylabel('RoR')



