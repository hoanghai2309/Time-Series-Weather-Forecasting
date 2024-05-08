import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from keras.models import Sequential
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam

temp=pd.read_csv('dataset/temperature.csv', index_col='datetime', parse_dates=['datetime'])
temp=temp[['Houston']]
temp.columns=['Temp']
press=pd.read_csv('dataset/pressure.csv', index_col='datetime', parse_dates=['datetime'])
press=press[['Houston']]
press.columns=['Press']
df=press.join(temp,on='datetime',how='outer')
df.interpolate(inplace=True)
df.dropna(inplace=True)
df.to_csv('preprocessed_data.csv')


day = 24*60*60
year = (365.2425)*day
df['Seconds'] = df.index.map(pd.Timestamp.timestamp)
df['Day sin'] = np.sin(df['Seconds'] * (2 * np.pi / day))
df['Day cos'] = np.cos(df['Seconds'] * (2 * np.pi / day))
df['Year sin'] = np.sin(df['Seconds'] * (2 * np.pi / year))
df['Year cos'] = np.cos(df['Seconds'] * (2 * np.pi / year))
df.drop('Seconds',axis=1,inplace=True)

# print(df.head())
def df_to_X_y3(df, window_size=7):
  df_as_np = df.to_numpy()
  X = []
  y = []
  for i in range(len(df_as_np)-window_size):
    row = [r for r in df_as_np[i:i+window_size]]
    X.append(row)
    label = [df_as_np[i+window_size][0], df_as_np[i+window_size][1]]
    y.append(label)
  return np.array(X), np.array(y)

X3, y3 = df_to_X_y3(df)
print(X3.shape, y3.shape)


X3_train, y3_train = X3[:35000], y3[:35000]
X3_val, y3_val = X3[35000:40000], y3[35000:40000]
X3_test, y3_test = X3[40000:], y3[40000:]
print(X3_train.shape, y3_train.shape, X3_val.shape, y3_val.shape, X3_test.shape, y3_test.shape)

p_training_mean3 = np.mean(X3_train[:, :, 0])
p_training_std3 = np.std(X3_train[:, :, 0])

temp_training_mean3 = np.mean(X3_train[:, :, 1])
temp_training_std3 = np.std(X3_train[:, :, 1])

def preprocess3(X):
  X[:, :, 0] = (X[:, :, 0] - p_training_mean3) / p_training_std3
  X[:, :, 1] = (X[:, :, 1] - temp_training_mean3) / temp_training_std3
  return X

def preprocess_output(y):
  y[:, 0] = (y[:, 0] - p_training_mean3) / p_training_std3
  y[:, 1] = (y[:, 1] - temp_training_mean3) / temp_training_std3
  return y

X3_train=preprocess3(X3_train)
X3_val=preprocess3(X3_val)
X3_test=preprocess3(X3_test)
y3_train=preprocess_output(y3_train)
y3_val=preprocess_output(y3_val)
y3_test=preprocess_output(y3_test)


model5 = Sequential()
model5.add(InputLayer((7, 6)))
model5.add(LSTM(64))
model5.add(Dense(8, 'relu'))
model5.add(Dense(2, 'linear'))

model5.summary()

cp5 = ModelCheckpoint('checkpoint.hdf5', save_best_only=True)
model5.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])


#model5.fit(X3_train, y3_train, validation_data=(X3_val, y3_val), epochs=10, callbacks=[cp5])
from keras.models import load_model
model5=load_model('checkpoint.hdf5')



def plot_predictions2(model, X, y, start=0, end=100):
  predictions = model.predict(X)
  p_preds, temp_preds = predictions[:, 0], predictions[:, 1]
  p_actuals, temp_actuals = y[:, 0], y[:, 1]
  df = pd.DataFrame(data={'Temperature Predictions': temp_preds,
                          'Temperature Actuals':temp_actuals,
                          'Pressure Predictions': p_preds,
                          'Pressure Actuals': p_actuals
                          })
  plt.plot(df['Temperature Predictions'][start:end])
  plt.plot(df['Temperature Actuals'][start:end])
  plt.plot(df['Pressure Predictions'][start:end])
  plt.plot(df['Pressure Actuals'][start:end])
  return df

# plot_predictions2(model5, X3_test, y3_test)

def postprocess_temp(arr):
  arr = (arr*temp_training_std3) + temp_training_mean3
  return arr

def postprocess_p(arr):
  arr = (arr*p_training_std3) + p_training_mean3
  return arr

def get_predictions_postprocessed(model, X, y):
  predictions = model.predict(X)
  p_preds, temp_preds = postprocess_p(predictions[:, 0]), postprocess_temp(predictions[:, 1])
  p_actuals, temp_actuals = postprocess_p(y[:, 0]), postprocess_temp(y[:, 1])
  df = pd.DataFrame(data={'Temperature Predictions': temp_preds,
                          'Temperature Actuals':temp_actuals,
                          'Pressure Predictions': p_preds,
                          'Pressure Actuals': p_actuals
                          })
  return df

# temp=X3_test[0]
# print(model5.predict(temp.reshape(1, temp.shape[0], temp.shape[1])))
def multi_step_predict(model, X, steps):
  input_data = np.array(X)
  predictions = np.zeros((0, 2))

  for _ in range(steps):
    temp=input_data[0]
    pred = model.predict(temp.reshape(1, input_data.shape[1], input_data.shape[2]))
    predictions=np.append(predictions,pred,axis=0)
    input_data = input_data[1:]
    input_data[0,0,-1], input_data[0,1,-1] = pred[0,0], pred[0,1]


  return np.array(predictions)
post_processed_df = multi_step_predict(model5, X3_test, 100)
print(post_processed_df.shape)
p_preds, temp_preds = postprocess_p(post_processed_df[:, 0]), postprocess_temp(post_processed_df[:, 1])
p_actuals, temp_actuals = postprocess_p(y3_test[:, 0]), postprocess_temp(y3_test[:, 1])

start, end = 0, 100
plt.plot(temp_preds[start:end], label='Temperature Predictions')
plt.plot(temp_actuals[start:end], label='Temperature Actuals')

plt.plot(p_preds[start:end],label='Pressure Predictions')
plt.plot(p_actuals[start:end], label='Pressure Actuals')

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from math import sqrt
mae = mean_absolute_error(temp_preds,temp_actuals[0:100])
rmse=sqrt(mean_squared_error(temp_preds,temp_actuals[0:100]))
print(f'mae - manual: {mae}')
print(f'rmse - manual: {rmse}')
mae = mean_absolute_error(p_preds,p_actuals[start:end])
rmse=sqrt(mean_squared_error(p_preds,p_actuals[start:end]))
print(f'mae - manual: {mae}')
print(f'rmse - manual: {rmse}')

plt.legend()
plt.show()
