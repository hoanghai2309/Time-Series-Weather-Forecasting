import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from pandas.plotting import register_matplotlib_converters
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
register_matplotlib_converters()
from time import time

# Load the data
df=pd.read_csv('dataset/temperature.csv', index_col='datetime', parse_dates=['datetime'])
df = df[['Houston']]
df.interpolate(inplace=True)
df.dropna(inplace=True)
df = df.resample('M').mean()
df.to_csv('preprocessed_data.csv')

plt.figure(figsize=(10,2))
plt.plot(df)
plt.title('Temperature', fontsize=20)

import statsmodels.api as sm
decomposed_google_volume = sm.tsa.seasonal_decompose(df) # The frequncy is annual
figure = decomposed_google_volume.plot()

from statsmodels.tsa.stattools import adfuller
adf_test = adfuller(df)
print(f'p-value: {adf_test[1]}')

acf_diff = plot_acf(df)
pacf_diff = plot_pacf(df)

first_diff = df.diff()[1:]
decomposed_google_volume = sm.tsa.seasonal_decompose(first_diff) # The frequncy is annual
figure = decomposed_google_volume.plot()
adf_test = adfuller(first_diff)
print(f'p-value: {adf_test[1]}')
acf_diff = plot_acf(first_diff)
pacf_diff = plot_pacf(first_diff)

train_end = datetime(2016,1,31)
test_end = datetime(2017,11,30)
train_data = df[:train_end]
test_data = df[train_end + timedelta(days=1):test_end]



my_order = (3,1,1)
my_seasonal_order = (1, 0, 1, 12)
# define model
model = SARIMAX(train_data, order=my_order, seasonal_order=my_seasonal_order)
model_fit = model.fit()
print(model_fit.summary())

start_date = datetime(2012,10,31)
end_date = datetime(2017,11,30)

predictions = model_fit.forecast(len(test_data))
predictions = pd.Series(predictions, index=test_data.index)



plt.figure(figsize=(10,4))

plt.plot(df)
plt.plot(predictions)

plt.legend(('Data', 'Predictions'), fontsize=16)

plt.title('Temperature', fontsize=20)
for year in range(start_date.year,end_date.year):
    plt.axvline(pd.to_datetime(str(year)+'-01-01'), color='k', linestyle='--', alpha=0.2)


from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
mae = mean_absolute_error(predictions, test_data)
mape = mean_absolute_percentage_error(predictions, test_data)
rmse = np.sqrt(mean_squared_error(predictions, test_data))
print(f'mae - manual: {mae}')
print(f'mape - manual: {mape}')
print(f'rmse - manual: {rmse}')


rolling_predictions = predictions.copy()
for train_end in test_data.index:
    train_data = df[:train_end-timedelta(days=1)]
    model = SARIMAX(train_data, order=my_order, seasonal_order=my_seasonal_order)
    model_fit = model.fit()

    pred = model_fit.forecast()
    rolling_predictions[train_end] = pred

plt.figure(figsize=(10,4))

plt.plot(df)
plt.plot(rolling_predictions)

plt.legend(('Data', 'Predictions'), fontsize=16)

plt.title('Temperature', fontsize=20)
for year in range(start_date.year,end_date.year):
    plt.axvline(pd.to_datetime(str(year)+'-01-01'), color='k', linestyle='--', alpha=0.2)


mae = mean_absolute_error(rolling_predictions, test_data)
mape = mean_absolute_percentage_error(rolling_predictions, test_data)
rmse = np.sqrt(mean_squared_error(rolling_predictions, test_data))
print(f'mae - manual: {mae}')
print(f'mape - manual: {mape}')
print(f'rmse - manual: {rmse}')
plt.show()