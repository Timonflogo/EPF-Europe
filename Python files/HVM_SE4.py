#%% import libraries
import pandas as pd
import numpy as np 
import seaborn as sns

#%% Setup plotting environment
from pylab import rcParams
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters

mpl.rcParams['figure.figsize'] = (12, 6)
mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.dpi'] = 300
# set styles

# set seaborn style
register_matplotlib_converters()

# set seaborn style
sns.set(style='whitegrid', palette='deep', font_scale=1.8)

# set plotting parameters
rcParams['figure.figsize'] = 20, 12  
rcParams['font.family'] = "sans-serif"
rc('lines', linewidth=4, linestyle='-')
# rcParams['text.usetex'] = True

#%% Create Class for colour coding of ADF test results to enhance readability in the terminal
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[36m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[31m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
#%% import dataset
df = pd.read_csv('NP-HMV.csv', index_col='HourDK', parse_dates=True)
SE4 = df[['SE4']]
# SE4.index = pd.DatetimeIndex(SE4.index).to_period('H')

###############################################################################
# STATISTICAL MODELS
###############################################################################

#%% define adf test
from statsmodels.tsa.stattools import adfuller
def adf_test(series,title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis{bcolors.ENDC}")
        print("Reject the null hypothesis")
        print(bcolors.OKBLUE + series.name + " Series Data has no unit root and is stationary" + bcolors.ENDC)
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print(bcolors.WARNING + series.name + " Series Data has a unit root and is non-stationary " + bcolors.ENDC)

#%% run adf test on data
adf_test(SE4["SE4"])


#%% import libraries for AR-based models
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf # for determining (p,q) orders
from statsmodels.tsa.seasonal import seasonal_decompose 
from statsmodels.tools.eval_measures import mse,rmse     # for ETS Plots
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from pmdarima import auto_arima 

#%% reduce DE series load to enable auto arima
SE4_autoarima = SE4['2022-01-01':'2022-03-01']

 #%% run auto arima on dataset
print(auto_arima(SE4_autoarima))
# optimal model is ARIMA(1,1,2)(0,0,0)[0]

#%% reset index 
SE4.reset_index(drop=False, inplace=True)
SE4 = SE4[['SE4']]
#%% define forecasting horizon
horizon = 24

#%% train test split
# we will go with a train-test split such that our test set represents 168 Hours worth of data
train =  SE4[:len(SE4)-horizon]
test = SE4[len(SE4)-horizon:]
len(SE4) == len(train) + len(test) # True

# forecast start and end
# obtain predicted results
start = len(train)
end = len(train)+len(test)-1

#%% create dataframe for time computed
time_compute = pd.DataFrame()

#%% Fit AR model
model_AR = SARIMAX(train['SE4'],order=(1,0,0),enforce_invertibility=False)

import time
start_time = time.time()
results_AR = model_AR.fit()
end_time_AR = time.time()
time_SE4_AR = (time.time() - start_time)
results_AR.summary()

# predict
predictions_AR = results_AR.predict(start=start, end=end).rename('AR(1) Predictions')


#%% Fit ARIMA model
model_ARIMA = SARIMAX(train['SE4'],order=(1,1,2),enforce_invertibility=False)

import time
start_time = time.time()
results_ARIMA = model_ARIMA.fit()
end_time_AR = time.time()
time_SE4_ARIMA = (time.time() - start_time)
results_ARIMA.summary()

# predict
predictions_ARIMA = results_ARIMA.predict(start=start, end=end).rename('ARIMA(1,1,2) Predictions')


#%% run auto arima on dataset with seasonal is TRUE
#print(auto_arima(SE4_autoarima, seasonal=True, m=24))
# optimal model is ARIMA(4,1,0)(2,0,0)[24]

#%% reduce train data
train = train[-15000:]

#%% Fit SARIMA model
model_SARIMA = SARIMAX(train['SE4'],order=(4,1,0),seasonal_order=(2,0,0,24),enforce_invertibility=False)

import time
start_time = time.time()
results_SARIMA = model_SARIMA.fit()
end_time_AR = time.time()
time_SE4_SARIMA = (time.time() - start_time)
results_SARIMA.summary()

start = len(train)
end = len(train)+len(test)-1

# predict
predictions_SARIMA = results_SARIMA.predict(start=start, end=end).rename('SARIMA(4,1,0)(2,0,0,24) Predictions')


#%% get evaluation metrics
error_SE4_AR = mape(test['SE4'], predictions_AR)*100
error_SE4_ARIMA = mape(test['SE4'], predictions_ARIMA)*100
error_SE4_SARIMA = mape(test['SE4'], predictions_SARIMA)*100

print(error_SE4_AR, error_SE4_ARIMA, error_SE4_SARIMA)

#%%test_eval wrangling
test_eval = test
# reset index
#test_eval.drop('HourDK',axis=1, inplace=True)
#test_eval

#%% append stat predictions
test_eval["AR(1)"] = predictions_AR
test_eval["ARIMA(1,1,2)"] = predictions_ARIMA
test_eval["SARIMA(4,1,0)(2,0,0,24)"] = predictions_SARIMA

#%% reset index for plotting
test_eval.reset_index(inplace=True)
test_eval.drop('index',axis=1, inplace=True)

#%% plot predictions
# define plot parameters
title='Statistical forecasting performance DE'
ylabel='Electricity Price'
xlabel=''

ax = test_eval['SE4'].plot(legend=True,figsize=(20,6),title=title)
test_eval['AR(1)'].plot(linestyle = '--', legend=True, color='orange')
test_eval['ARIMA(1,1,2)'].plot(linestyle = '--', legend=True, color='green')
test_eval['SARIMA(4,1,0)(2,0,0,24)'].plot(linestyle = '--', legend=True,color='purple')
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)

#%% save to statistical forecast to dataframe
# test_eval.to_csv('DE-stat-models-24')

###############################################################################
# MACHINE LEARNING MODELS
###############################################################################
#%% XGBOOST MODEL
#%% import libraries
from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from xgboost import XGBRegressor
from matplotlib import pyplot

#%% Setup plotting environment
from pylab import rcParams
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.dpi'] = 300
# set styles

# set seaborn style
register_matplotlib_converters()

# set seaborn style
sns.set(style='whitegrid', palette='deep', font_scale=1.8)

# set plotting parameters
rcParams['figure.figsize'] = 20, 12  
rcParams['font.family'] = "sans-serif"
# rcParams['text.usetex'] = True



#%% create series_to_supervised function to enable supervised training for XGBoost
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols = list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
	# put it all together
	agg = concat(cols, axis=1)
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg.values

#%% define train test split function for univariate time series
def train_test_split(data, n_test):
	return data[:-n_test, :], data[-n_test:, :]

#%% create xgboost forecaster
def xgboost_forecast(train, testX):
	# transform list into array
	train = asarray(train)
	# split into input and output columns
	trainX, trainy = train[:, :-1], train[:, -1]
	# fit model
	model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
	model.fit(trainX, trainy)
	# make a one-step prediction
	yhat = model.predict(asarray([testX]))
	return yhat[0]

#%% Create autoregressive forecasting loop
def walk_forward_validation(data, n_test):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# split test row into input and output columns
		testX, testy = test[i, :-1], test[i, -1]
		# fit model on history and make a prediction
		yhat = xgboost_forecast(history, testX)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
		# summarize progress
		print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
	# estimate prediction error
	error = mape(test[:, -1], predictions)
	return error, test[:, -1], predictions

#%% Run XGBoost on SE4 series
values_HMV_SE4 = df[['SE4']].values
# transform the time series data into supervised learning
data_HMV_SE4 = series_to_supervised(values_HMV_SE4, n_in=24)
# evaluate
import time
start_time = time.time()
mape_SE4_XGBoost, y_SE4, yhat_SE4 = walk_forward_validation(data_HMV_SE4, 24)
time_SE4_XGBoost = (time.time() - start_time)
print('MAPE: %.3f' % mape_SE4_XGBoost)
error_SE4_XGBoost = mape_SE4_XGBoost *100

#%% append XGBOOST metrics to test and time dataframes
test_eval['XGBoost'] = yhat_SE4

print(error_SE4_XGBoost)

#%% SVR MODEL
#%% import libraries for SVR
import sys
sys.path.append('../../')
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
import math

from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error as mape

#%% import dataset
HMV = pd.read_csv('NP-HMV.csv', index_col='HourDK', parse_dates=True) 
SE4 = HMV[['SE4']]

#%% define forecasting horizon
horizon = 28

#%% train test split
# we will go with a train-test split such that our test set represents 168 Hours worth of data
train =  SE4[:len(SE4)-horizon]
test = SE4[len(SE4)-horizon:]
len(SE4) == len(train) + len(test) # True

# forecast start and end
# obtain predicted results
start = len(train)
end = len(train)+len(test)-1

#%% print shapes
print('Training data shape: ', train.shape)
print('Test data shape: ', test.shape)

#%% scale data
scaler = MinMaxScaler()
train['SE4'] = scaler.fit_transform(train)

test['SE4'] = scaler.transform(test)

#%% Create data with time-steps

# Converting to numpy arrays
train_data = train.values
test_data = test.values

# determine number of timesteps. If 5 then inputs will be 4 and output will be the 5th timestep
timesteps=5

# Converting training data to 2D tensor using nested list comprehension:
train_data_timesteps=np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]
print("train data shape",train_data_timesteps.shape)

# Converting testing data to 2D tensor:
test_data_timesteps=np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]
print("test data shape", test_data_timesteps.shape)

#%% Selecting inputs and outputs from training and testing data:
x_train, y_train = train_data_timesteps[:,:timesteps-1],train_data_timesteps[:,[timesteps-1]]
x_test, y_test = test_data_timesteps[:,:timesteps-1],test_data_timesteps[:,[timesteps-1]]

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# Implement SVR 
model = SVR(kernel='rbf',
            gamma=0.5,
            C=10,
            epsilon = 0.05,
            degree=3,
            shrinking=True)

#%% run SVR
import time
start_time = time.time()
# fit the model on training data
print("running: ", model.fit(x_train, y_train[:,0]))
time_SE4_SVR = (time.time() - start_time)

#%% make model predictions
y_train_pred = model.predict(x_train).reshape(-1,1)
y_test_pred = model.predict(x_test).reshape(-1,1)

print(y_train_pred.shape, y_test_pred.shape)

#%% inverse transform data
y_train_pred = scaler.inverse_transform(y_train_pred)
y_test_pred = scaler.inverse_transform(y_test_pred)

print(len(y_train_pred), len(y_test_pred))

y_train = scaler.inverse_transform(y_train)
y_test = scaler.inverse_transform(y_test)

print(y_train_pred.shape, y_test_pred.shape)

#%% plot predicitons for testing data
plt.figure(figsize=(12,6))
plt.plot(y_test, color = 'blue', linewidth=2.0, alpha = 0.6)
plt.plot(y_test_pred, color = 'red', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.title("Test data prediction")
plt.show()

#%% get MAPE for test data
print('MAPE for testing data: ', mape(y_test_pred, y_test)*100, '%')

# assign to variable
error_SE4_SVR = mape(y_test_pred, y_test)*100

#%% append SVR metrics to evaluation dataframes
test_eval['SVR'] = y_test_pred

print(error_SE4_SVR)

#%% plot Statistical and ml
title='ML forecasting performance SE4'
ylabel='Electricity Price'
xlabel=''

ax = test_eval['SE4'].plot(legend=True,figsize=(20,6),title=title)
test_eval['AR(5)'].plot(linestyle = '--', legend=True, color='orange')
test_eval['ARIMA(5,1,3)'].plot(linestyle = '--', legend=True, color='green')
test_eval['SARIMA(3,1,1)(2,0,1,24)'].plot(linestyle = '--', legend=True,color='purple')
test_eval['SVR'].plot(linestyle = '--', legend=True, color='magenta')
test_eval['XGBoost'].plot(linestyle = '--', legend=True, color='cyan')
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)

#%% plot test eval_ml


###############################################################################
# DEEP LEARNING MODELS
###############################################################################
#LSTM MODEL
#%% import libraries
import tensorflow as tf
import os
import pandas as pd
import numpy as np
import seaborn as sns

#%% Setup plotting environment
from pylab import rcParams
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters

mpl.rcParams['figure.figsize'] = (12, 6)
mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.dpi'] = 300
# set styles

# set seaborn style
register_matplotlib_converters()

# set seaborn style
sns.set(style='whitegrid', palette='deep', font_scale=1.8)

# set plotting parameters
rcParams['figure.figsize'] = 20, 12  
rcParams['font.family'] = "sans-serif"
rc('lines', linewidth=4, linestyle='-')
# rcParams['text.usetex'] = True

#%% import dataset
HMV = pd.read_csv('NP-HMV.csv', index_col='HourDK', parse_dates=True) 
SE4 = HMV['SE4']

#%% create windowing function
def df_to_supervised(df, window_size=5):
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np)-window_size):
        row = [[a] for a in df_as_np[i:i+window_size]]
        X.append(row)
        label = df_as_np[i+window_size]
        y.append(label)
    return np.array(X), np.array(y)

#%% use df_to_X_y function
WINDOW_SIZE = 24
X, y = df_to_supervised(SE4, WINDOW_SIZE)
print(X.shape, y.shape)

#%% create train, val, and test set
val_horizon = 48
test_horizon = 24

X_train, y_train = X[:-(val_horizon + test_horizon)], y[:-(val_horizon + test_horizon)]
X_val, y_val = X[len(X_train): (len(X_train) + val_horizon)], y[len(X_train): (len(X_train) + val_horizon)]
X_test, y_test = X[(len(X_train) + val_horizon):], y[(len(X_train) + val_horizon):]
    
# get shapes
print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)

#%% import libraries for Deep learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

#%% BUILD LSTM
model_LSTM = Sequential()
model_LSTM.add(InputLayer((X_train.shape[1], X_train.shape[2])))
model_LSTM.add(LSTM(64))
model_LSTM.add(Dense(8, activation='relu'))
model_LSTM.add(Dense(1, activation='linear'))

model_LSTM.summary()

#%% callbacks
cp = ModelCheckpoint('model_SE4_LSTM/', save_best_only=True)

#%% compile
model_LSTM.compile(loss='mse',
               optimizer=Adam(learning_rate=0.0001),
               metrics=[MeanAbsoluteError()])


#%% fit model
import time
start_time = time.time()

history = model_LSTM.fit(X_train, y_train,
                     validation_data=(X_val, y_val),
                     epochs=30,
                     callbacks=[cp])

time_SE4_LSTM = (time.time() - start_time)


#%% plot history loss 
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title('DK1 LSTM train and validation loss')
plt.legend()
plt.show()

#%% load the model
from tensorflow.keras.models import load_model
model_LSTM = load_model('model_SE4_LSTM/')

#%% prediction Train and plot train results
train_predictions = model_LSTM.predict(X_train).flatten()
train_results = pd.DataFrame(data={'Train Predictions': train_predictions, 'Actuals': y_train})
print(train_results)

import matplotlib.pyplot as plt
plt.plot(train_results['Train Predictions'][-100:])
plt.plot(train_results['Actuals'][-100:])

#%% predict Val and plot val results
val_predictions = model_LSTM.predict(X_val).flatten()
val_results = pd.DataFrame(data={'Val Predictions':val_predictions, 'Actuals':y_val})
print(val_results)

plt.plot(val_results['Val Predictions'])
plt.plot(val_results['Actuals'])

#%% predict test and plot test results
test_predictions = model_LSTM.predict(X_test).flatten()
test_results = pd.DataFrame(data={'Test Predictions':test_predictions, 'Actuals':y_test})
print(test_results)

plt.plot(test_results['Test Predictions'])
plt.plot(test_results['Actuals'])

#%% get error metrics 
from sklearn.metrics import mean_absolute_percentage_error as mape
error_SE4_LSTM = mape(test_results['Actuals'], test_results['Test Predictions'])*100
print(error_SE4_LSTM)

#%% append LSTM metrics to evaluation dataframes
test_eval['LSTM'] = test_predictions

#%% GRU MODEL
#%% import libraries
import tensorflow as tf
import os
import pandas as pd
import numpy as np
import seaborn as sns

#%% Setup plotting environment
from pylab import rcParams
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters

mpl.rcParams['figure.figsize'] = (12, 6)
mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.dpi'] = 300
# set styles

# set seaborn style
register_matplotlib_converters()

# set seaborn style
sns.set(style='whitegrid', palette='deep', font_scale=1.8)

# set plotting parameters
rcParams['figure.figsize'] = 20, 12  
rcParams['font.family'] = "sans-serif"
rc('lines', linewidth=4, linestyle='-')
# rcParams['text.usetex'] = True

#%% import dataset
HMV = pd.read_csv('NP-HMV.csv', index_col='HourDK', parse_dates=True) 
SE4 = HMV['SE4']

#%% create windowing function
def df_to_supervised(df, window_size=5):
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np)-window_size):
        row = [[a] for a in df_as_np[i:i+window_size]]
        X.append(row)
        label = df_as_np[i+window_size]
        y.append(label)
    return np.array(X), np.array(y)

#%% use df_to_X_y function
WINDOW_SIZE = 24
X, y = df_to_supervised(SE4, WINDOW_SIZE)
print(X.shape, y.shape)

#%% create train, val, and test set
val_horizon = 48
test_horizon = 24

X_train, y_train = X[:-(val_horizon + test_horizon)], y[:-(val_horizon + test_horizon)]
X_val, y_val = X[len(X_train): (len(X_train) + val_horizon)], y[len(X_train): (len(X_train) + val_horizon)]
X_test, y_test = X[(len(X_train) + val_horizon):], y[(len(X_train) + val_horizon):]
    
# get shapes
X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape

#%% import libraries for Deep learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

#%% BUILD LSTM
model_GRU = Sequential()
model_GRU.add(InputLayer((X_train.shape[1], X_train.shape[2])))
model_GRU.add(GRU(64))
model_GRU.add(Dense(8, 'relu'))
model_GRU.add(Dense(1, 'linear'))

model_GRU.summary()

#%% callbacks
cp = ModelCheckpoint('model_SE4_GRU/', save_best_only=True)

#%% compile
model_GRU.compile(loss='mse',
               optimizer=Adam(learning_rate=0.0001),
               metrics=[RootMeanSquaredError()])

#%% fit model
import time
start_time = time.time()

history = model_GRU.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=30,
                    callbacks=[cp])

time_SE4_GRU = (time.time() - start_time)

#%% plot history loss 
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title('SE4 GRU train and validation loss')
plt.legend()
plt.show()

#%% load the model
from tensorflow.keras.models import load_model
model_GRU = load_model('model_SE4_GRU/')

#%% prediction Train and plot train results
train_predictions = model_GRU.predict(X_train).flatten()
train_results = pd.DataFrame(data={'Train Predictions': train_predictions, 'Actuals': y_train})
print(train_results)

import matplotlib.pyplot as plt
plt.plot(train_results['Train Predictions'][-100:])
plt.plot(train_results['Actuals'][-100:])

#%% predict Val and plot val results
val_predictions = model_GRU.predict(X_val).flatten()
val_results = pd.DataFrame(data={'Val Predictions':val_predictions, 'Actuals':y_val})
print(val_results)

plt.plot(val_results['Val Predictions'])
plt.plot(val_results['Actuals'])

#%% predict test and plot test results
test_predictions = model_GRU.predict(X_test).flatten()
test_results = pd.DataFrame(data={'Test Predictions':test_predictions, 'Actuals':y_test})
print(test_results)

plt.plot(test_results['Test Predictions'])
plt.plot(test_results['Actuals'])

#%% get error metrics 
from sklearn.metrics import mean_absolute_percentage_error as mape
error_SE4_GRU = mape(test_results['Actuals'], test_results['Test Predictions'])*100
print(error_SE4_GRU)

#%% append GRU metrics to evaluation dataframes
test_eval['GRU'] = test_predictions

#%% save test_eval dataframe for plotting 
test_eval.to_csv('test_eval_SE4.csv')

#%% get dataframe with all mape values
evaluation_metrics = pd.DataFrame()

evaluation_metrics = evaluation_metrics.append({'AR': error_SE4_AR,
                                                'ARIMA': error_SE4_ARIMA,
                                                'SARIMA': error_SE4_SARIMA,
                                                'XGBoost': error_SE4_XGBoost,
                                                'SVR': error_SE4_SVR,
                                                'LSTM': error_SE4_LSTM,
                                                'GRU': error_SE4_GRU},
                                               ignore_index=True
                                               )

evaluation_metrics.to_csv('evaluation-metrics-SE4.csv')

#%% get computation time for all algorithms 
# append compute time to list
time_compute = time_compute.append({'time_SE4_AR': time_SE4_AR,
                                    'time_SE4_ARIMA': time_SE4_ARIMA,
                                    'time_SE4_SARIMA': time_SE4_SARIMA,
                                    'time_SE4_XGBoost': time_SE4_XGBoost,
                                    'time_SE4_SVR': time_SE4_SVR,
                                    'time_SE4_LSTM': time_SE4_LSTM,
                                    'time_SE4_GRU': time_SE4_GRU}, 
                                    ignore_index=True)

time_compute.to_csv('time-compute-SE4.csv')

