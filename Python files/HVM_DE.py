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
HMV = pd.read_csv('NP-HMV.csv', index_col='HourDK', parse_dates=True) 
DE = HMV[['DE']]

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
adf_test(DE["DE"])


#%% import libraries for AR-based models
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf # for determining (p,q) orders
from statsmodels.tsa.seasonal import seasonal_decompose 
from statsmodels.tools.eval_measures import mse,rmse     # for ETS Plots
from pmdarima import auto_arima 

#%% reduce DE series load to enable auto arima
# DE_autoarima = DE['2021-12-01':'2022-03-01']

#%% run auto arima on dataset
# print(auto_arima(DE_autoarima))
# optimal model is ARIMA(4,1,3)(0,0,0)[0]

#%% define forecasting horizon
horizon = 24

#%% train test split
# we will go with a train-test split such that our test set represents 168 Hours worth of data
train =  DE[:len(DE)-horizon]
test = DE[len(DE)-horizon:]
len(DE) == len(train) + len(test) # True

# forecast start and end
# obtain predicted results
start = len(train)
end = len(train)+len(test)-1

#%% create dataframe for evaluation metrics
test_eval = test

#%% Fit AR model
model_AR = SARIMAX(train[['DE']],order=(4,0,0),enforce_invertibility=False)

import time
start_time_AR = time.time()
results_AR = model_AR.fit()
end_time_AR = time.time()
time_DE_AR = (end_time_AR - start_time_AR)
results_AR.summary()

# predict
predictions_AR = results_AR.predict(start=start, end=end).rename('AR(4) Predictions')

# append predictions to test_eval dataframe
test_eval['AR(4) Predictions'] = predictions_AR

#%% Fit ARIMA model
model_ARIMA = SARIMAX(train['DE'],order=(4,1,3),enforce_invertibility=False)

import time
start_time_ARIMA = time.time()
results_ARIMA = model_ARIMA.fit()
end_time_ARIMA = time.time()
time_DE_ARIMA = (end_time_ARIMA - start_time_ARIMA)
results_ARIMA.summary()

# predict
predictions_ARIMA = results_ARIMA.predict(start=start, end=end).rename('ARIMA(4,1,3) Predictions')

# append predictions to test_eval dataframe
test_eval['ARIMA(4,1,3) Predictions'] = predictions_ARIMA

#%% run auto arima on dataset with seasonal is TRUE
# print(auto_arima(DE_autoarima, seasonal=True, m=24))
# optimal model is ARIMA(1,1,0)(2,0,0)[24]

#%% Fit SARIMA model
model_SARIMA = SARIMAX(train['DE'],order=(1,1,0),seasonal_order=(2,0,0,24),enforce_invertibility=False)

import time
start_time_SARIMA = time.time()
results_SARIMA = model_SARIMA.fit()
end_time_SARIMA = time.time()
time_DE_SARIMA = (end_time_SARIMA - start_time_SARIMA)
results_SARIMA.summary()

# predict
predictions_SARIMA = results_SARIMA.predict(start=start, end=end).rename('SARIMA(1,1,0)(2,0,0,24) Predictions')

# append predictions to test_eval dataframe
test_eval['SARIMA(1,1,0)(2,0,0,24) Predictions'] = predictions_SARIMA

#%% get evaluation metrics
error_AR = mape(test['DE'], predictions_AR)
error_ARIMA = mape(test['DE'], predictions_ARIMA)
error_SARIMA = maper(test['DE'], predictions_SARIMA)

###############################################################################
# MACHINE LEARNING MODELS
###############################################################################
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

#%% Run XGBoost on DE series
values_HMV_DE = HMV[['DE']].values
# transform the time series data into supervised learning
data_HMV_DE = series_to_supervised(values_HMV_DE, n_in=24)
# evaluate
import time
start_time = time.time()
mape_DE, y_DE, yhat_DE = walk_forward_validation(data_HMV_DE, 24)
time_DE_XGBoost = (time.time() - start_time)
print('MAPE: %.3f' % mape_DE)
# plot expected vs preducted
pyplot.plot(y_DE, label='Expected')
pyplot.plot(yhat_DE, label='Predicted')
pyplot.legend()
pyplot.show() 

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
from common.utils import load_data, mape
   
#%% Create training and test data
# we will go with a train-test split such that our test set represents 168 Hours worth of data
train =  DE[:len(DE)-168]
test = DE[len(DE)-168:]
len(DE) == len(train) + len(test) # True

# forecast start and end
# obtain predicted results
start = len(train)
end = len(train)+len(test)-1
