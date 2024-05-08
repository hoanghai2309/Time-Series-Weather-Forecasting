import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

df=pd.read_csv('dataset/temperature.csv', index_col='datetime', parse_dates=['datetime'])
df = df[['Houston']]
df.interpolate(inplace=True)
df.dropna(inplace=True)
df = df.resample('M').mean()
df.to_csv('preprocessed_data.csv')


train = df.iloc[:40]
test = df.iloc[40:]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

from keras.preprocessing.sequence import TimeseriesGenerator
# define generator
n_features = 1
n_input = 12
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
# define model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(generator,epochs=50)


loss_per_epoch = model.history.history['loss']
#plt.plot(range(len(loss_per_epoch)),loss_per_epoch)

test_predictions = []

first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(len(test)):
    # get the prediction value for the first batch
    current_pred = model.predict(current_batch)[0]

    # append the prediction into the array
    test_predictions.append(current_pred)

    # use the prediction to update the batch and remove the first value
    current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

true_predictions = scaler.inverse_transform(test_predictions)
test['Predictions'] = true_predictions
test.plot(figsize=(14,5))


from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from math import sqrt
mae = mean_absolute_error(test['Houston'],test['Predictions'])
rmse=sqrt(mean_squared_error(test['Houston'],test['Predictions']))
print(f'mae - manual: {mae}')
print(f'rmse - manual: {rmse}')


plt.show()